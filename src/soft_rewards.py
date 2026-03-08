"""
VIGIL Soft Thresholding Rewards — smooth, differentiable reward functions.

Replaces all hard thresholds with sigmoid-based soft gates:
- soft_iig: sigmoid(IIG / tau) instead of max(IIG, 0)
- soft_gate: sigmoid((threshold - agreement) / tau) instead of binary gate
- soft_correct: token overlap instead of exact match
- soft_alpha: continuous steering strength from agreement + activation

Temperature annealing: tau decreases over training (soft → hard curriculum).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def soft_iig(iig_value: float, tau: float = 0.5) -> float:
    """Smooth IIG gate. Replaces max(IIG, 0).

    tau controls sharpness:
        tau=0.1: near-hard (sigmoid very steep)
        tau=0.5: moderate (default, good balance)
        tau=1.0: very soft (gradual transition)

    Returns: [0, 1] — smooth probability that image helps.
    """
    return float(torch.sigmoid(torch.tensor(iig_value / tau)))


def soft_gate(agreement: float, threshold: float = 0.65, tau: float = 0.1) -> float:
    """Smooth agreement gate. Replaces binary if agreement < threshold.

    Returns: [0, 1] — how much to steer.
        agreement << threshold → ~1.0 (steer strongly)
        agreement == threshold → 0.5
        agreement >> threshold → ~0.0 (don't steer)
    """
    return float(torch.sigmoid(torch.tensor((threshold - agreement) / tau)))


def soft_correct(pred: str, gt: str) -> float:
    """Soft correctness score via normalized token overlap.

    Replaces binary 0/1 exact match.
    Returns: [0, 1] — partial credit for near-correct answers.
    """
    pred_lower = pred.strip().lower()
    gt_lower = gt.strip().lower()

    # Exact match → 1.0
    if pred_lower == gt_lower:
        return 1.0

    # For yes/no: check containment with position weighting
    if gt_lower in ("yes", "no"):
        if gt_lower in pred_lower[:20]:
            # Correct answer present early → high score
            pos = pred_lower.index(gt_lower)
            return max(0.5, 1.0 - pos * 0.05)  # penalize late mention
        # Wrong answer
        other = "no" if gt_lower == "yes" else "yes"
        if other in pred_lower[:20]:
            return 0.0
        # Neither → uncertain → small partial credit
        return 0.1

    # For free-form: character-level similarity
    pred_set = set(pred_lower.split())
    gt_set = set(gt_lower.split())
    if not gt_set:
        return 0.0
    overlap = len(pred_set & gt_set) / len(gt_set)
    return min(1.0, overlap)


def soft_alpha(
    agreement: float,
    vision_activation: float,
    threshold: float = 0.65,
    tau_gate: float = 0.1,
    tau_act: float = 5.0,
) -> float:
    """Continuous steering strength from agreement + vision activation.

    Returns: [0, 1] — how strongly to steer.
        Low agreement + high vision activation → steer strongly
        High agreement → don't steer (model is confident)
    """
    gate = soft_gate(agreement, threshold, tau_gate)
    strength = float(torch.sigmoid(torch.tensor(vision_activation / tau_act)))
    return gate * strength


class SoftVIGILReward:
    """Complete soft-thresholded reward for DAPO/GRPO training.

    All components are smooth and differentiable w.r.t. their inputs.
    Temperature annealing: soft early → harder late (curriculum).
    """

    def __init__(
        self,
        w_correct: float = 0.35,
        w_visual: float = 0.45,
        w_gate: float = 0.20,
        tau_iig: float = 0.5,
        tau_gate: float = 0.1,
        tau_act: float = 5.0,
        anneal_rate: float = 0.8,  # how much to anneal (0 = no anneal, 1 = full)
        agreement_threshold: float = 0.65,
    ):
        self.w_correct = w_correct
        self.w_visual = w_visual
        self.w_gate = w_gate
        self.tau_iig = tau_iig
        self.tau_gate = tau_gate
        self.tau_act = tau_act
        self.anneal_rate = anneal_rate
        self.agreement_threshold = agreement_threshold

    def get_annealed_tau(self, step: int, total_steps: int) -> Tuple[float, float]:
        """Anneal temperature: soft early → harder late."""
        progress = step / max(total_steps, 1)
        factor = 1.0 - self.anneal_rate * progress
        return self.tau_iig * factor, self.tau_gate * factor

    def compute_single(
        self,
        pred: str,
        gt: str,
        iig_value: float,
        agreement: float,
        vision_activation: float,
        step: int = 0,
        total_steps: int = 100,
    ) -> Dict[str, float]:
        """Compute soft reward for a single candidate.

        Returns dict with total reward and per-component breakdown.
        """
        tau_i, tau_g = self.get_annealed_tau(step, total_steps)

        r_correct = soft_correct(pred, gt)
        r_iig = soft_iig(iig_value, tau=tau_i)
        r_gate = soft_gate(agreement, self.agreement_threshold, tau=tau_g)
        r_vision_act = float(torch.sigmoid(
            torch.tensor(vision_activation / self.tau_act)
        ))

        # Visual grounding = IIG × vision activation × gate
        # Multiplicative: gradient flows through all three
        r_visual = r_iig * r_vision_act * r_gate

        # Composite reward
        r_total = (
            self.w_correct * r_correct
            + self.w_visual * r_visual
            + self.w_gate * r_gate
        )

        return {
            "total": r_total,
            "correct": r_correct,
            "visual": r_visual,
            "iig": r_iig,
            "gate": r_gate,
            "vision_act": r_vision_act,
            "tau_iig": tau_i,
            "tau_gate": tau_g,
        }

    def compute_group(
        self,
        candidates: List[Dict[str, Any]],
        step: int = 0,
        total_steps: int = 100,
    ) -> List[Dict[str, float]]:
        """Compute soft rewards for a group of candidates.

        Each candidate dict should have:
            pred: str, gt: str, iig: float, agreement: float, vision_act: float
        """
        return [
            self.compute_single(
                pred=c["pred"],
                gt=c["gt"],
                iig_value=c.get("iig", 0.0),
                agreement=c.get("agreement", 0.5),
                vision_activation=c.get("vision_act", 0.0),
                step=step,
                total_steps=total_steps,
            )
            for c in candidates
        ]

    def compute_advantage(
        self,
        candidates: List[Dict[str, Any]],
        step: int = 0,
        total_steps: int = 100,
        eps: float = 1e-8,
    ) -> List[float]:
        """Compute GRPO/DAPO advantage from soft rewards.

        Advantage = (R_i - mean(R)) / (std(R) + eps)

        Soft thresholding ensures:
        - Reward variance is almost always > 0 (no cliff edges)
        - Gradient is defined everywhere
        - Temperature annealing increases discrimination over time
        """
        rewards = self.compute_group(candidates, step, total_steps)
        totals = [r["total"] for r in rewards]
        mean_r = np.mean(totals)
        std_r = np.std(totals)

        advantages = [(r - mean_r) / (std_r + eps) for r in totals]
        return advantages


def demo_soft_vs_hard():
    """Demonstrate soft vs hard thresholding behavior."""
    print("=== Soft vs Hard Thresholding Demo ===\n")

    # IIG comparison
    print("IIG values → reward contribution:")
    for iig in [-2.0, -0.5, -0.01, 0.0, 0.01, 0.5, 2.0]:
        hard = max(iig, 0)
        soft = soft_iig(iig, tau=0.5)
        print(f"  IIG={iig:+.2f}  hard={hard:.3f}  soft={soft:.3f}")

    print("\nAgreement → steering gate:")
    for agr in [0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]:
        hard = 1.0 if agr < 0.65 else 0.0
        soft = soft_gate(agr, threshold=0.65, tau=0.1)
        print(f"  agreement={agr:.2f}  hard={hard:.1f}  soft={soft:.3f}")

    print("\nTemperature annealing (tau_iig):")
    for step in [0, 25, 50, 75, 100]:
        reward = SoftVIGILReward()
        tau_i, tau_g = reward.get_annealed_tau(step, 100)
        print(f"  step={step:3d}  tau_iig={tau_i:.3f}  tau_gate={tau_g:.4f}")


if __name__ == "__main__":
    demo_soft_vs_hard()
