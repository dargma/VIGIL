"""
VIGIL Steering Calibrator — compute steering vectors as (correct_mean - incorrect_mean)
per head, filter to top-K by Cohen's d. Save/load calibration results.

Supports GQA and MHA. Uses o_proj pre-hooks for activation capture.
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class CalibrationResult:
    steering_vectors: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    head_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    top_heads: List[Tuple[int, int]] = field(default_factory=list)
    n_correct: int = 0
    n_incorrect: int = 0

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        vec_dict = {f"{li}_{hi}": v.cpu() for (li, hi), v in self.steering_vectors.items()}
        torch.save(vec_dict, out / "steering_vectors.pt")
        meta = {
            "head_scores": {f"{li}_{hi}": s for (li, hi), s in self.head_scores.items()},
            "top_heads": [[li, hi] for li, hi in self.top_heads],
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }
        with open(out / "calibration_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[calibrator] Saved to {out}: {len(self.top_heads)} heads, "
              f"{self.n_correct} correct, {self.n_incorrect} incorrect")

    @staticmethod
    def load(calibration_dir: str) -> "CalibrationResult":
        d = Path(calibration_dir)
        vec_dict = torch.load(d / "steering_vectors.pt", map_location="cpu", weights_only=True)
        with open(d / "calibration_meta.json") as f:
            meta = json.load(f)
        result = CalibrationResult(
            n_correct=meta["n_correct"],
            n_incorrect=meta["n_incorrect"],
        )
        for key_str, vec in vec_dict.items():
            li, hi = map(int, key_str.split("_"))
            result.steering_vectors[(li, hi)] = vec
        for key_str, score in meta["head_scores"].items():
            li, hi = map(int, key_str.split("_"))
            result.head_scores[(li, hi)] = score
        result.top_heads = [tuple(pair) for pair in meta["top_heads"]]
        print(f"[calibrator] Loaded from {d}: {len(result.top_heads)} heads")
        return result


class SteeringCalibrator:
    """Compute steering vectors from correct vs incorrect activations."""

    def __init__(
        self,
        model_info: Dict[str, Any],
        top_k: int = 20,
        min_per_bucket: int = 5,
        confidence_split_threshold: int = 5,
    ):
        self.model_info = model_info
        self.model = model_info["model"]
        self.num_layers = model_info["num_layers"]
        self.num_heads = model_info["num_heads"]
        self.head_dim = model_info["head_dim"]
        self.top_k = top_k
        self.min_per_bucket = min_per_bucket
        self.confidence_split_threshold = confidence_split_threshold
        self._captured: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def _install_hooks(self):
        self._remove_hooks()
        layers = self.model_info["get_layers_fn"]()
        for li, layer in enumerate(layers):
            o_proj = layer.self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def _extract_head_vectors(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Extract per-head vectors from last token of captured o_proj inputs."""
        vectors = {}
        for li in range(self.num_layers):
            inp = self._captured.get(li)
            if inp is None:
                continue
            last = inp[0, -1, :].view(self.num_heads, self.head_dim)
            for hi in range(self.num_heads):
                vectors[(li, hi)] = last[hi].clone()
        return vectors

    @torch.no_grad()
    def calibrate(
        self,
        samples: List[Dict],
        process_fn,
        max_samples: int = 1000,
    ) -> CalibrationResult:
        """Run calibration.

        Args:
            samples: list of dicts with question/answer/image
            process_fn: callable(model_info, sample) -> (input_dict, ground_truth_str)
            max_samples: max samples to process
        """
        self.model.eval()
        self._install_hooks()

        correct_vecs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        incorrect_vecs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        n_correct = n_incorrect = 0
        use_confidence_split = False

        # First pass: try correctness-based split
        for i, sample in enumerate(samples[:max_samples]):
            try:
                inputs, gt = process_fn(self.model_info, sample)
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                pred_id = logits.argmax(dim=-1).item()
                tokenizer = self.model_info["tokenizer"]
                pred_str = tokenizer.decode([pred_id]).strip().lower()
                is_correct = pred_str.startswith(gt.strip().lower()[:3])

                head_vecs = self._extract_head_vectors()
                self._captured.clear()

                bucket = correct_vecs if is_correct else incorrect_vecs
                for key, vec in head_vecs.items():
                    bucket.setdefault(key, []).append(vec)

                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

            except Exception as e:
                print(f"[calibrator] Skip sample {i}: {e}")
                continue

        # If too few incorrect, switch to confidence split
        if n_incorrect < self.confidence_split_threshold:
            print(f"[calibrator] Only {n_incorrect} incorrect. Switching to confidence split.")
            use_confidence_split = True
            # Re-run with confidence-based bucketing
            correct_vecs.clear()
            incorrect_vecs.clear()
            n_correct = n_incorrect = 0

            for i, sample in enumerate(samples[:max_samples]):
                try:
                    inputs, gt = process_fn(self.model_info, sample)
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs.max().item()

                    head_vecs = self._extract_head_vectors()
                    self._captured.clear()

                    # High confidence = "correct" bucket, low = "incorrect"
                    if confidence > 0.7:
                        bucket = correct_vecs
                        n_correct += 1
                    else:
                        bucket = incorrect_vecs
                        n_incorrect += 1

                    for key, vec in head_vecs.items():
                        bucket.setdefault(key, []).append(vec)

                except Exception as e:
                    continue

        self._remove_hooks()

        # Compute steering vectors
        result = CalibrationResult(n_correct=n_correct, n_incorrect=n_incorrect)
        for li in range(self.num_layers):
            for hi in range(self.num_heads):
                key = (li, hi)
                c_list = correct_vecs.get(key, [])
                w_list = incorrect_vecs.get(key, [])

                if len(c_list) < self.min_per_bucket or len(w_list) < self.min_per_bucket:
                    continue

                c_mean = torch.stack(c_list).mean(dim=0)
                w_mean = torch.stack(w_list).mean(dim=0)
                diff = c_mean - w_mean
                result.steering_vectors[key] = diff

                # Cohen's d
                c_norms = torch.stack(c_list).float().norm(dim=-1).cpu().numpy()
                w_norms = torch.stack(w_list).float().norm(dim=-1).cpu().numpy()
                pooled_std = np.sqrt(
                    (c_norms.var() * (len(c_norms) - 1) + w_norms.var() * (len(w_norms) - 1))
                    / (len(c_norms) + len(w_norms) - 2)
                )
                if pooled_std > 1e-8:
                    result.head_scores[key] = float(diff.norm().item() / pooled_std)
                else:
                    result.head_scores[key] = 0.0

        # Select top-K
        ranked = sorted(result.head_scores.items(), key=lambda x: x[1], reverse=True)
        result.top_heads = [k for k, _ in ranked[:self.top_k]]

        print(f"[calibrator] Done: {n_correct} correct, {n_incorrect} incorrect, "
              f"top-{self.top_k} heads selected"
              f"{' (confidence split)' if use_confidence_split else ''}")
        if result.top_heads:
            top3 = [(k, f"{result.head_scores[k]:.3f}") for k in result.top_heads[:3]]
            print(f"  Top-3 heads: {top3}")

        return result
