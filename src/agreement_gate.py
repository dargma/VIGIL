"""
VIGIL Agreement Gate — monitor layer agreement for conditional steering.

Only steer when the model is uncertain (low agreement across layers).
Uses 1-step lag to prevent self-reinforcing feedback loops.
"""

import torch
from typing import Dict, Tuple, Any, Optional


class AgreementGate:
    """Gate steering based on inter-layer prediction agreement.

    Samples predictions from multiple layers. If layers agree on the next token,
    the model is confident → don't steer. If they disagree → steer.

    Uses 1-step lag: steering decision at step t is based on agreement at step t-1.
    """

    def __init__(
        self,
        model_info: Dict[str, Any],
        threshold: float = 0.7,
        sample_layers: int = 5,
    ):
        self.model_info = model_info
        self.threshold = threshold
        self.sample_layers = sample_layers
        self._prev_agreement: Optional[float] = None
        self._agreement_history: list = []

    def _get_sample_indices(self, n_layers: int) -> list:
        """Select layer indices to sample: evenly spaced including first and last."""
        if n_layers <= self.sample_layers:
            return list(range(n_layers))
        step = (n_layers - 1) / (self.sample_layers - 1)
        return sorted(set(int(round(i * step)) for i in range(self.sample_layers)))

    @torch.no_grad()
    def compute_agreement(self, hidden_states: Tuple[torch.Tensor, ...]) -> float:
        """Compute agreement score from model hidden states.

        Args:
            hidden_states: tuple of (batch, seq, hidden) from all layers

        Returns:
            agreement in [0, 1] — 1.0 = all layers agree
        """
        if hidden_states is None or len(hidden_states) < 2:
            return 1.0

        n = len(hidden_states)
        indices = self._get_sample_indices(n)
        norm_fn = self.model_info["get_norm_fn"]()
        lm_head = self.model_info["get_lm_head_fn"]()

        # Overall prediction from final layer
        last_hs = hidden_states[-1][:, -1:, :]
        overall_pred = lm_head(norm_fn(last_hs)).argmax(dim=-1).item()

        n_agree = 0
        for idx in indices:
            hs = hidden_states[idx][:, -1:, :]
            pred = lm_head(norm_fn(hs)).argmax(dim=-1).item()
            if pred == overall_pred:
                n_agree += 1

        agreement = n_agree / len(indices)
        self._agreement_history.append(agreement)
        return agreement

    def should_steer(self, hidden_states: Tuple[torch.Tensor, ...]) -> Tuple[bool, float]:
        """Decide whether to steer and at what strength. Uses 1-step lag.

        Returns:
            (should_steer: bool, alpha: float)
        """
        current_agreement = self.compute_agreement(hidden_states)

        # 1-step lag: use previous agreement for decision
        if self._prev_agreement is not None:
            ref = self._prev_agreement
        else:
            ref = current_agreement

        self._prev_agreement = current_agreement

        if ref < self.threshold:
            alpha = 1.0 - ref  # lower agreement → stronger steering
            return True, alpha
        return False, 0.0

    def reset(self):
        self._prev_agreement = None
        self._agreement_history.clear()

    def get_history(self) -> list:
        return self._agreement_history.copy()
