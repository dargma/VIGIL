"""
VIGIL Vision Drift Analyzer — track vision head activation across token positions.

Analyzes thinking chain token-position vs activation to show:
1. Vision attention decay over generation length
2. Effect of steering on preventing decay
3. "Lookback" events where vision attention spikes
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .rewards import VisionHeadActivationCollector


class VisionDriftAnalyzer:
    """Track vision head activations across all generated tokens."""

    def __init__(
        self,
        model_info: Dict[str, Any],
        vision_heads: List[Tuple[int, int]],
    ):
        self.model_info = model_info
        self.vision_heads = vision_heads
        self.collector = VisionHeadActivationCollector(model_info, vision_heads)
        self.trajectory: List[Dict[Tuple[int, int], float]] = []

    def install(self):
        self.collector.install()

    def remove(self):
        self.collector.remove()

    def record_step(self):
        """Call after each forward pass during generation."""
        norms = self.collector.get_activation_norms()
        if norms:
            self.trajectory.append(norms)
        self.collector.clear()

    def clear(self):
        self.trajectory.clear()
        self.collector.clear()

    def get_mean_trajectory(self) -> np.ndarray:
        """Return mean vision head activation at each token position."""
        if not self.trajectory:
            return np.array([])
        return np.array([
            np.mean(list(step.values())) for step in self.trajectory
        ])

    def get_per_head_trajectory(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return per-head activation trajectory."""
        if not self.trajectory:
            return {}
        result = {}
        for key in self.vision_heads:
            result[key] = np.array([
                step.get(key, 0.0) for step in self.trajectory
            ])
        return result

    def compute_drift_metrics(self) -> Dict[str, float]:
        """Compute drift metrics from trajectory.

        Returns dict with:
            decay_ratio: second_half_mean / first_half_mean (< 1 = decay)
            slope: linear regression slope (negative = decay)
            min_activation: minimum activation in trajectory
            lookback_count: number of spikes > 1.5 * local mean
        """
        traj = self.get_mean_trajectory()
        if len(traj) < 4:
            return {"decay_ratio": 1.0, "slope": 0.0, "min_activation": 0.0, "lookback_count": 0}

        mid = len(traj) // 2
        first_half = traj[:mid].mean()
        second_half = traj[mid:].mean()
        decay_ratio = float(second_half / (first_half + 1e-8))

        # Linear regression slope
        x = np.arange(len(traj))
        slope = float(np.polyfit(x, traj, 1)[0])

        # Lookback detection: spikes > 1.5× rolling mean
        window = max(3, len(traj) // 10)
        lookback_count = 0
        for i in range(window, len(traj)):
            local_mean = traj[max(0, i - window):i].mean()
            if traj[i] > 1.5 * local_mean:
                lookback_count += 1

        return {
            "decay_ratio": decay_ratio,
            "slope": slope,
            "min_activation": float(traj.min()),
            "lookback_count": lookback_count,
            "mean_activation": float(traj.mean()),
            "trajectory_length": len(traj),
        }


def analyze_thinking_drift(
    model_info: Dict[str, Any],
    vision_heads: List[Tuple[int, int]],
    samples: List[Dict],
    generate_fn,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Run drift analysis on thinking-mode generation.

    Args:
        model_info: loaded model
        vision_heads: calibrated vision head list
        samples: eval samples with images
        generate_fn: callable(model_info, question, image, analyzer) -> str
            Must call analyzer.record_step() after each forward pass
        max_samples: samples to analyze

    Returns:
        aggregate metrics + per-sample trajectories
    """
    analyzer = VisionDriftAnalyzer(model_info, vision_heads)
    analyzer.install()

    all_metrics = []
    all_trajectories = []

    for i, sample in enumerate(samples[:max_samples]):
        try:
            analyzer.clear()
            _ = generate_fn(model_info, sample["question"], sample.get("image"), analyzer)
            metrics = analyzer.compute_drift_metrics()
            all_metrics.append(metrics)
            all_trajectories.append(analyzer.get_mean_trajectory().tolist())

            if (i + 1) % 10 == 0:
                avg_decay = np.mean([m["decay_ratio"] for m in all_metrics])
                avg_slope = np.mean([m["slope"] for m in all_metrics])
                print(f"[drift] {i+1}/{min(len(samples), max_samples)}: "
                      f"avg_decay={avg_decay:.3f}, avg_slope={avg_slope:.4f}")

        except Exception as e:
            print(f"[drift] Skip sample {i}: {e}")
            continue

    analyzer.remove()

    if not all_metrics:
        return {"error": "no valid samples"}

    summary = {
        "avg_decay_ratio": float(np.mean([m["decay_ratio"] for m in all_metrics])),
        "avg_slope": float(np.mean([m["slope"] for m in all_metrics])),
        "avg_lookback_count": float(np.mean([m["lookback_count"] for m in all_metrics])),
        "avg_mean_activation": float(np.mean([m["mean_activation"] for m in all_metrics])),
        "n_samples": len(all_metrics),
        "per_sample_metrics": all_metrics,
    }

    print(f"\n[drift] SUMMARY:")
    print(f"  Decay ratio: {summary['avg_decay_ratio']:.3f} (< 1 = attention decays)")
    print(f"  Slope: {summary['avg_slope']:.4f} (negative = decay)")
    print(f"  Lookbacks: {summary['avg_lookback_count']:.1f} per sample")

    return summary
