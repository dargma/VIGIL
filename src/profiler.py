"""
VIGIL Vision Head Profiler — extract per-head activations from o_proj input.

Supports GQA (Qwen3-VL, InternVL3) and MHA (DeepSeek-VL2).
Profiles heads by collecting activation norms on correct vs incorrect predictions,
then computes Cohen's d to rank vision-specialized heads.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class HeadProfile:
    layer_idx: int
    head_idx: int
    correct_mean_norm: float = 0.0
    incorrect_mean_norm: float = 0.0
    cohen_d: float = 0.0
    is_vision_head: bool = False


@dataclass
class ProfileResult:
    head_profiles: Dict[Tuple[int, int], HeadProfile] = field(default_factory=dict)
    top_vision_heads: List[Tuple[int, int]] = field(default_factory=list)
    num_correct: int = 0
    num_incorrect: int = 0


class VisionHeadProfiler:
    """Profile attention heads to identify vision-specialized ones via Cohen's d."""

    def __init__(self, model_info: Dict[str, Any]):
        self.model_info = model_info
        self.model = model_info["model"]
        self.num_layers = model_info["num_layers"]
        self.num_heads = model_info["num_heads"]
        self.num_kv_heads = model_info["num_kv_heads"]
        self.head_dim = model_info["head_dim"]
        self.hidden_size = model_info["hidden_size"]
        self.gqa = model_info["gqa"]
        self._captured: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def _install_hooks(self):
        """Register forward hooks on o_proj for all layers."""
        self._remove_hooks()
        layers = self.model_info["get_layers_fn"]()
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            o_proj = attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    # o_proj input: (batch, seq, num_heads * head_dim)
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def _get_per_head_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Extract per-Q-head activations from captured o_proj input.

        Returns: (batch, num_heads, head_dim) at last token position.
        For GQA: num_heads = num_attention_heads (Q heads), not KV heads.
        """
        inp = self._captured.get(layer_idx)
        if inp is None:
            return None
        # inp shape: (batch, seq, hidden_size)
        # For GQA: hidden_size = num_Q_heads * head_dim
        last = inp[:, -1, :]  # (batch, hidden_size)
        return last.view(-1, self.num_heads, self.head_dim)  # (batch, num_Q_heads, head_dim)

    @torch.no_grad()
    def profile(
        self,
        samples: List[Dict],
        process_fn,
        max_samples: int = 500,
    ) -> ProfileResult:
        """Run profiling on samples.

        Args:
            samples: list of {question, answer, image, ...}
            process_fn: callable(model_info, sample) -> (input_dict, ground_truth_str)
            max_samples: max samples to process
        """
        self.model.eval()
        self._install_hooks()

        correct_acts = {}  # (li, hi) -> list of norms
        incorrect_acts = {}
        n_correct = n_incorrect = 0

        for i, sample in enumerate(samples[:max_samples]):
            try:
                inputs, gt = process_fn(self.model_info, sample)
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                pred_id = logits.argmax(dim=-1).item()
                tokenizer = self.model_info["tokenizer"]
                pred_str = tokenizer.decode([pred_id]).strip().lower()
                is_correct = pred_str.startswith(gt.strip().lower()[:3])

                bucket = correct_acts if is_correct else incorrect_acts
                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                for li in range(self.num_layers):
                    head_acts = self._get_per_head_activations(li)
                    if head_acts is None:
                        continue
                    for hi in range(self.num_heads):
                        key = (li, hi)
                        norm = head_acts[0, hi, :].norm().item()
                        bucket.setdefault(key, []).append(norm)

                self._captured.clear()
            except Exception as e:
                print(f"[profiler] Skip sample {i}: {e}")
                continue

        self._remove_hooks()

        result = ProfileResult(num_correct=n_correct, num_incorrect=n_incorrect)
        for li in range(self.num_layers):
            for hi in range(self.num_heads):
                key = (li, hi)
                c_norms = correct_acts.get(key, [])
                w_norms = incorrect_acts.get(key, [])
                profile = HeadProfile(layer_idx=li, head_idx=hi)

                if len(c_norms) >= 5 and len(w_norms) >= 5:
                    c_arr = np.array(c_norms)
                    w_arr = np.array(w_norms)
                    profile.correct_mean_norm = float(c_arr.mean())
                    profile.incorrect_mean_norm = float(w_arr.mean())
                    pooled_std = np.sqrt(
                        (c_arr.var() * (len(c_arr) - 1) + w_arr.var() * (len(w_arr) - 1))
                        / (len(c_arr) + len(w_arr) - 2)
                    )
                    if pooled_std > 1e-8:
                        profile.cohen_d = abs(c_arr.mean() - w_arr.mean()) / pooled_std

                result.head_profiles[key] = profile

        ranked = sorted(result.head_profiles.items(), key=lambda x: x[1].cohen_d, reverse=True)
        result.top_vision_heads = [k for k, _ in ranked[:20]]
        for k in result.top_vision_heads:
            result.head_profiles[k].is_vision_head = True

        return result
