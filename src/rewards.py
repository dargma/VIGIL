"""
VIGIL Reward Functions — R_vhad + R_asi (core novelty).

R_total = w1*R_correct + w2*R_visual_grounding + w3*R_fluency
R_visual_grounding = α*R_vhad + (1-α)*R_asi

Two configurations:
- Full reward: R_vhad (extra forward with black image) + R_asi (extra generation)
- Lightweight: in-situ vision head activation during normal generation (zero overhead)
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional


class VisionHeadActivationCollector:
    """Collect per-head activations from o_proj during forward pass.
    Used for both R_vhad computation and in-situ lightweight reward.
    """

    def __init__(self, model_info: Dict[str, Any], vision_heads: List[Tuple[int, int]]):
        self.model_info = model_info
        self.vision_heads = vision_heads
        self.num_heads = model_info["num_heads"]
        self.head_dim = model_info["head_dim"]
        self._captured: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def install(self):
        self.remove()
        layers = self.model_info["get_layers_fn"]()
        target_layers = set(li for li, _ in self.vision_heads)
        for li in target_layers:
            o_proj = layers[li].self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def get_activations(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Return per-head activation vectors at last token for vision heads."""
        results = {}
        for (li, hi) in self.vision_heads:
            inp = self._captured.get(li)
            if inp is None:
                continue
            last = inp[0, -1, :].view(self.num_heads, self.head_dim)
            results[(li, hi)] = last[hi].clone()
        return results

    def get_activation_norms(self) -> Dict[Tuple[int, int], float]:
        acts = self.get_activations()
        return {k: v.norm().item() for k, v in acts.items()}

    def clear(self):
        self._captured.clear()


def make_black_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a black image for blind testing / R_vhad computation."""
    return Image.new("RGB", size, (0, 0, 0))


@torch.no_grad()
def compute_r_vhad(
    model_info: Dict[str, Any],
    vision_heads: List[Tuple[int, int]],
    real_inputs: Dict[str, torch.Tensor],
    black_inputs: Dict[str, torch.Tensor],
) -> float:
    """R_vhad: activation differential between real and black image.

    Forward with real image, collect vision head activations.
    Forward with black image, collect vision head activations.
    Return normalized difference.
    """
    model = model_info["model"]
    collector = VisionHeadActivationCollector(model_info, vision_heads)
    collector.install()

    # Forward with real image
    model(**real_inputs)
    act_real = collector.get_activation_norms()
    collector.clear()

    # Forward with black image
    model(**black_inputs)
    act_black = collector.get_activation_norms()
    collector.clear()
    collector.remove()

    if not act_real or not act_black:
        return 0.0

    # Sum of absolute differences across vision heads
    total_diff = sum(
        abs(act_real.get(k, 0.0) - act_black.get(k, 0.0))
        for k in vision_heads
    )
    # Normalize by number of heads and typical activation scale
    norm_factor = len(vision_heads) * max(
        np.mean(list(act_real.values())) if act_real else 1.0, 1e-6
    )
    return min(total_diff / norm_factor, 1.0)


def compute_r_asi(
    answer_with_image: str,
    answer_without_image: str,
    similarity_metric: str = "token_jaccard",
) -> float:
    """R_asi: Answer Sensitivity to Image.

    If answers are the same, model is blind → low reward.
    Higher difference = higher reward.
    """
    a = answer_with_image.strip().lower()
    b = answer_without_image.strip().lower()

    if not a or not b:
        return 0.5  # can't tell

    if similarity_metric == "exact_match":
        return 0.0 if a == b else 1.0

    elif similarity_metric == "token_jaccard":
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a and not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union) if union else 1.0
        return 1.0 - jaccard  # high jaccard = same answers = blind = low reward

    elif similarity_metric == "f1":
        tokens_a = a.split()
        tokens_b = b.split()
        common = set(tokens_a) & set(tokens_b)
        if not common:
            return 1.0
        precision = len(common) / len(tokens_a) if tokens_a else 0
        recall = len(common) / len(tokens_b) if tokens_b else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return 1.0 - f1

    return 0.5


def compute_r_fluency(completion: str, max_tokens: int = 512) -> float:
    """Fluency reward: penalize degenerate outputs and excessive length."""
    if not completion or len(completion.strip()) < 3:
        return 0.0

    # Check for degenerate repetition
    unique_chars = len(set(completion))
    if unique_chars < 3:
        return 0.0

    # Length penalty near max_tokens
    words = completion.split()
    # Rough token estimate
    approx_tokens = len(words) * 1.3
    if approx_tokens > max_tokens * 0.8:
        excess = (approx_tokens - max_tokens * 0.8) / (max_tokens * 0.2)
        return max(0.0, 1.0 - min(excess, 1.0) * 0.5)

    return 1.0


def compute_r_correct(prediction: str, ground_truth: str, question_type: str = "yesno") -> float:
    """Tiered accuracy reward."""
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    if question_type == "yesno":
        pred_yn = "yes" if "yes" in pred[:10] else ("no" if "no" in pred[:10] else "")
        return 1.0 if pred_yn == gt else 0.0

    elif question_type == "mc":
        # First letter match
        pred_letter = pred[0] if pred else ""
        gt_letter = gt[0] if gt else ""
        return 1.0 if pred_letter == gt_letter else 0.0

    elif question_type == "numeric":
        try:
            p = float("".join(c for c in pred if c.isdigit() or c == "."))
            g = float("".join(c for c in gt if c.isdigit() or c == "."))
            return max(0.0, 1.0 - abs(p - g) / max(abs(g), 1.0))
        except (ValueError, ZeroDivisionError):
            return 0.0

    elif question_type == "short_answer":
        # F1 word overlap
        pred_tokens = set(pred.split())
        gt_tokens = set(gt.split())
        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0
        common = pred_tokens & gt_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return 0.0


def compute_composite_reward(
    r_correct: float,
    r_vhad: float,
    r_asi: float,
    r_fluency: float,
    w_correct: float = 0.3,
    w_visual: float = 0.5,
    w_fluency: float = 0.2,
    alpha_vhad: float = 0.6,
) -> float:
    """VIGIL composite reward.

    R_total = w1*R_correct + w2*(α*R_vhad + (1-α)*R_asi) + w3*R_fluency
    """
    r_visual = alpha_vhad * r_vhad + (1 - alpha_vhad) * r_asi
    return w_correct * r_correct + w_visual * r_visual + w_fluency * r_fluency


class InSituVisionReward:
    """Lightweight reward: collect vision head activations during normal generation.
    Zero extra cost — hooks collect during the model's own forward passes.
    Less precise than full R_vhad but free.
    """

    def __init__(self, model_info: Dict[str, Any], vision_heads: List[Tuple[int, int]]):
        self.collector = VisionHeadActivationCollector(model_info, vision_heads)
        self.vision_heads = vision_heads
        self._running_norms: List[Dict[Tuple[int, int], float]] = []

    def install(self):
        self.collector.install()

    def remove(self):
        self.collector.remove()

    def step(self):
        """Call after each forward pass during generation to record activations."""
        norms = self.collector.get_activation_norms()
        if norms:
            self._running_norms.append(norms)
        self.collector.clear()

    def compute(self) -> float:
        """Compute reward from collected activation trajectory.

        Higher mean activation of vision heads = model is attending to visual features.
        """
        if not self._running_norms:
            return 0.0

        mean_norms = []
        for step_norms in self._running_norms:
            mean_norms.append(np.mean(list(step_norms.values())))

        # Reward based on: did vision head activation stay high throughout generation?
        # Penalize decay (the drift problem)
        overall_mean = np.mean(mean_norms)
        if len(mean_norms) > 1:
            # Check if activation decays (bad) vs stays stable (good)
            first_half = np.mean(mean_norms[:len(mean_norms) // 2])
            second_half = np.mean(mean_norms[len(mean_norms) // 2:])
            decay_ratio = second_half / (first_half + 1e-8)
            # Sigmoid mapping: ratio 1.0 → 0.5 reward, >1 → higher, <1 → lower
            return float(1.0 / (1.0 + np.exp(-5.0 * (decay_ratio - 0.8))))
        return float(1.0 / (1.0 + np.exp(-0.1 * overall_mean)))

    def clear(self):
        self._running_norms.clear()
        self.collector.clear()
