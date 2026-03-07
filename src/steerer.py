"""
VIGIL Activation Steerer — inject steering vectors into vision-specialized heads.

Pre-hook on o_proj input where individual Q head outputs are separable as
[num_Q_heads × head_dim] before projection. Supports GQA and MHA.
Agreement-gated: only steers when model is uncertain.
"""

import torch
from typing import Dict, List, Tuple, Any, Optional
from .calibrator import CalibrationResult


class SteeringHook:
    """Single hook on one o_proj for one head."""

    def __init__(
        self,
        layer_module,
        layer_idx: int,
        head_idx: int,
        steering_vector: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.steering_vector = steering_vector
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.alpha = 0.0
        self.active = False
        self._handle = layer_module.register_forward_pre_hook(self._pre_hook_fn)

    def _pre_hook_fn(self, module, args):
        if not self.active or self.alpha == 0.0:
            return args
        inp = args[0]  # (batch, seq, hidden)
        modified = inp.clone()
        batch = modified.shape[0]
        # Reshape last token: (batch, hidden) -> (batch, num_heads, head_dim)
        last_token = modified[:, -1, :].view(batch, self.num_heads, self.head_dim)
        vec = self.steering_vector.to(last_token.device, last_token.dtype)
        last_token[:, self.head_idx, :] += self.alpha * vec
        modified[:, -1, :] = last_token.reshape(batch, self.num_heads * self.head_dim)
        return (modified,) + args[1:]

    def activate(self, alpha: float = 1.0):
        self.alpha = alpha
        self.active = True

    def deactivate(self):
        self.active = False
        self.alpha = 0.0

    def remove(self):
        self._handle.remove()


class ActivationSteerer:
    """Manage multiple SteeringHooks across layers/heads."""

    def __init__(
        self,
        model_info: Dict[str, Any],
        calibration: CalibrationResult,
        steer_layers_start: int = 0,
    ):
        self.model_info = model_info
        self.calibration = calibration
        self.hooks: List[SteeringHook] = []
        self.steer_layers_start = steer_layers_start
        self._install(calibration)

    def _install(self, calibration: CalibrationResult):
        layers = self.model_info["get_layers_fn"]()
        num_heads = self.model_info["num_heads"]
        head_dim = self.model_info["head_dim"]

        for (li, hi) in calibration.top_heads:
            if li < self.steer_layers_start:
                continue
            if (li, hi) not in calibration.steering_vectors:
                continue
            o_proj = layers[li].self_attn.o_proj
            hook = SteeringHook(
                layer_module=o_proj,
                layer_idx=li,
                head_idx=hi,
                steering_vector=calibration.steering_vectors[(li, hi)],
                num_heads=num_heads,
                head_dim=head_dim,
            )
            self.hooks.append(hook)

        print(f"[steerer] Installed {len(self.hooks)} hooks "
              f"(layers {self.steer_layers_start}+, {len(calibration.top_heads)} top heads)")

    def steer(self, alpha: float = 1.0):
        for hook in self.hooks:
            hook.activate(alpha)

    def steer_proportional(self, global_alpha: float = 1.0):
        """Steer each head proportionally to its Cohen's d score.

        Heads with higher Cohen's d (stronger correct/incorrect separation)
        get steered more aggressively. This is more surgical than uniform alpha.
        """
        scores = self.calibration.head_scores
        if not scores:
            self.steer(global_alpha)
            return

        max_score = max(scores.values()) if scores else 1.0
        for hook in self.hooks:
            key = (hook.layer_idx, hook.head_idx)
            score = scores.get(key, 0.0)
            # Normalize to [0, 1] relative to max score, then scale by global_alpha
            per_head_alpha = (score / max_score) * global_alpha if max_score > 0 else global_alpha
            hook.activate(per_head_alpha)

    def steer_adaptive(self, agreement: float, threshold: float = 0.7):
        """Steer with strength inversely proportional to agreement."""
        if agreement < threshold:
            alpha = 1.0 - agreement
            self.steer(alpha)
        else:
            self.release()

    def release(self):
        for hook in self.hooks:
            hook.deactivate()

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class AgreementMonitor:
    """Monitor layer agreement for gated steering. Uses 1-step lag."""

    def __init__(self, model_info: Dict[str, Any], n_sample_layers: int = 5):
        self.model_info = model_info
        self.n_sample_layers = n_sample_layers
        self._prev_agreement: Optional[float] = None

    def compute_agreement(self, hidden_states: Tuple[torch.Tensor, ...]) -> float:
        """Compute agreement across sampled layers.

        Args:
            hidden_states: tuple of (batch, seq, hidden) from model output

        Returns:
            agreement score in [0, 1]
        """
        if hidden_states is None or len(hidden_states) < 2:
            return 1.0

        n = len(hidden_states)
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        indices = sorted(set(min(i, n - 1) for i in indices))

        norm_fn = self.model_info["get_norm_fn"]()
        lm_head = self.model_info["get_lm_head_fn"]()

        # Get the overall prediction from the last layer
        last_hs = hidden_states[-1][:, -1:, :]
        overall_logits = lm_head(norm_fn(last_hs))
        overall_pred = overall_logits.argmax(dim=-1).item()

        # Check each sampled layer
        n_agree = 0
        for idx in indices:
            hs = hidden_states[idx][:, -1:, :]
            logits = lm_head(norm_fn(hs))
            pred = logits.argmax(dim=-1).item()
            if pred == overall_pred:
                n_agree += 1

        return n_agree / len(indices)

    def get_gated_alpha(self, hidden_states, threshold: float = 0.7) -> float:
        """Return steering alpha with 1-step lag."""
        current = self.compute_agreement(hidden_states)
        # Use previous step's agreement (1-step lag)
        if self._prev_agreement is not None:
            alpha = max(0.0, 1.0 - self._prev_agreement) if self._prev_agreement < threshold else 0.0
        else:
            alpha = max(0.0, 1.0 - current) if current < threshold else 0.0
        self._prev_agreement = current
        return alpha

    def reset(self):
        self._prev_agreement = None
