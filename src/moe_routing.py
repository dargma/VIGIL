"""
VIGIL MoE Routing Analyzer — track expert selection in DeepSeek-VL2-Tiny.

Measures whether steering redirects expert selection toward vision-related experts.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class MoERoutingTracker:
    """Track which experts are activated per token in MoE layers."""

    def __init__(self, model_info: Dict[str, Any]):
        if not model_info.get("is_moe"):
            raise ValueError("MoERoutingTracker requires a MoE model")
        self.model_info = model_info
        self.model = model_info["model"]
        self._hooks = []
        self._routing_log: List[Dict[int, List[int]]] = []  # per-step: layer -> expert_ids

    def install(self):
        """Install hooks on MoE gate/router modules to capture routing decisions."""
        self.remove()
        layers = self.model_info["get_layers_fn"]()

        for li, layer in enumerate(layers):
            # DeepSeek-VL2 MoE: layer.mlp.gate or layer.mlp.router
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
            if gate is None:
                continue

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    # Gate output is typically (batch, seq, num_experts) logits
                    # or routing weights. We want the top-k expert indices.
                    if isinstance(output, tuple):
                        routing_weights = output[0]
                    else:
                        routing_weights = output

                    if routing_weights.dim() >= 2:
                        # Take last token routing
                        last = routing_weights[:, -1] if routing_weights.dim() == 3 else routing_weights
                        top_experts = last.topk(min(6, last.shape[-1]), dim=-1).indices
                        expert_ids = top_experts[0].cpu().tolist()
                    else:
                        expert_ids = []

                    if not self._routing_log or len(self._routing_log[-1]) >= len(layers):
                        self._routing_log.append({})
                    self._routing_log[-1][layer_idx] = expert_ids

                return hook_fn

            handle = gate.register_forward_hook(make_hook(li))
            self._hooks.append(handle)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self._routing_log.clear()

    def get_expert_frequency(self) -> Dict[int, Dict[int, int]]:
        """Return per-layer expert activation frequency.

        Returns: {layer_idx: {expert_id: count}}
        """
        freq = defaultdict(lambda: defaultdict(int))
        for step in self._routing_log:
            for li, experts in step.items():
                for eid in experts:
                    freq[li][eid] += 1
        return {li: dict(counts) for li, counts in freq.items()}

    def get_routing_distribution(self) -> Dict[int, np.ndarray]:
        """Return per-layer routing distribution as probability vectors."""
        num_experts = self.model_info.get("spec", None)
        n_exp = 64  # DeepSeek-VL2-Tiny default
        if num_experts and hasattr(num_experts, "num_experts"):
            n_exp = num_experts.num_experts

        freq = self.get_expert_frequency()
        dist = {}
        for li, counts in freq.items():
            vec = np.zeros(n_exp)
            for eid, count in counts.items():
                if eid < n_exp:
                    vec[eid] = count
            total = vec.sum()
            if total > 0:
                vec /= total
            dist[li] = vec
        return dist

    def compare_distributions(
        self,
        baseline_dist: Dict[int, np.ndarray],
        steered_dist: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Compare routing distributions before/after steering.

        Returns KL divergence and top shifted experts per layer.
        """
        results = {}
        for li in set(list(baseline_dist.keys()) + list(steered_dist.keys())):
            b = baseline_dist.get(li, np.zeros(64))
            s = steered_dist.get(li, np.zeros(64))
            # KL divergence (with smoothing)
            eps = 1e-8
            b_smooth = b + eps
            s_smooth = s + eps
            b_smooth /= b_smooth.sum()
            s_smooth /= s_smooth.sum()
            kl = float(np.sum(s_smooth * np.log(s_smooth / b_smooth)))

            # Top shifted experts
            diff = s - b
            top_up = np.argsort(diff)[-3:][::-1]  # experts that increased most
            top_down = np.argsort(diff)[:3]  # experts that decreased most

            results[f"layer_{li}"] = {
                "kl_divergence": kl,
                "top_increased": [(int(e), float(diff[e])) for e in top_up],
                "top_decreased": [(int(e), float(diff[e])) for e in top_down],
            }

        return results
