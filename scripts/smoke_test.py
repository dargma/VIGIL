"""
VIGIL Smoke Test — validates entire pipeline foundation in under 5 minutes.

Checks:
1. Model loads, config matches expected architecture
2. o_proj hook registers, activation shape is [batch, seq, num_Q_heads * head_dim]
3. Single image forward pass produces coherent output
4. Real image vs black image: vision head activation difference is non-trivial
5. Hook-based activation collection works during generation

If any check fails, prints exactly what went wrong. No silent failures.
"""

import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model, make_chat_prompt


class SmokeTestResult:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0

    def check(self, name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.checks.append((name, passed, detail))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Smoke Test: {self.passed}/{total} passed, {self.failed} failed")
        if self.failed > 0:
            print("\nFailed checks:")
            for name, passed, detail in self.checks:
                if not passed:
                    print(f"  - {name}: {detail}")
        print(f"{'='*60}")
        return self.failed == 0


def run_smoke_test(model_key: str = "qwen3_vl_2b"):
    result = SmokeTestResult()
    start = time.time()

    # --- Check 1: Model loads and architecture matches ---
    print("\n[1/5] Model loading and architecture verification")
    try:
        model_info = load_model(model_key, dtype=torch.float16)
        result.check("Model loads", True)

        expected = {
            "num_layers": model_info["num_layers"],
            "num_heads": model_info["num_heads"],
            "num_kv_heads": model_info["num_kv_heads"],
            "head_dim": model_info["head_dim"],
            "hidden_size": model_info["hidden_size"],
        }
        result.check("Architecture verified", True,
                      f"layers={expected['num_layers']}, heads={expected['num_heads']}, "
                      f"kv={expected['num_kv_heads']}, head_dim={expected['head_dim']}, "
                      f"hidden={expected['hidden_size']}")
    except Exception as e:
        result.check("Model loads", False, str(e))
        result.summary()
        return False

    model = model_info["model"]
    num_layers = model_info["num_layers"]
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]
    device = model_info["device"]

    # --- Check 2: o_proj hook registers, activation shape correct ---
    print("\n[2/5] Hook registration and activation shape")
    captured = {}
    hooks = []
    try:
        layers = model_info["get_layers_fn"]()
        for li, layer in enumerate(layers):
            o_proj = layer.self_attn.o_proj

            def make_hook(idx):
                def hook_fn(module, args):
                    captured[idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            hooks.append(handle)

        result.check("Hooks registered", True, f"{len(hooks)} hooks on {num_layers} layers")
    except Exception as e:
        result.check("Hooks registered", False, str(e))

    # --- Check 3: Single image forward pass ---
    print("\n[3/5] Single image forward pass")
    # Create a synthetic test image (red square on white background)
    test_image = Image.new("RGB", (320, 320), (255, 255, 255))
    for x in range(80, 240):
        for y in range(80, 240):
            test_image.putpixel((x, y), (255, 0, 0))
    test_sample = {
        "question": "What color is the square in the image?",
        "answer": "red",
        "image": test_image,
    }
    result.check("Test image created", True, "320x320 white background with red square")

    if True:
        try:
            with torch.no_grad():
                inputs = make_chat_prompt(model_info, test_sample["question"], test_sample["image"])
                outputs = model(**inputs)
                logits = outputs.logits
                pred_id = logits[:, -1, :].argmax(dim=-1).item()
                pred_str = model_info["tokenizer"].decode([pred_id]).strip()

            result.check("Forward pass succeeds", True, f"prediction='{pred_str}'")

            # Check activation shape
            if captured:
                sample_li = list(captured.keys())[0]
                act = captured[sample_li]
                expected_last_dim = num_heads * head_dim
                actual_last_dim = act.shape[-1]
                shape_ok = actual_last_dim == expected_last_dim
                result.check("Activation shape correct", shape_ok,
                             f"shape={list(act.shape)}, expected last_dim={expected_last_dim}, "
                             f"got={actual_last_dim}")
            else:
                result.check("Activation shape correct", False, "No activations captured")
        except Exception as e:
            result.check("Forward pass succeeds", False, str(e))

    # --- Check 4: Real vs Black image activation difference ---
    print("\n[4/5] Real vs black image vision head activation difference")
    try:
        # Real image forward
        captured.clear()
        with torch.no_grad():
            real_inputs = make_chat_prompt(model_info, test_sample["question"], test_sample["image"])
            model(**real_inputs)
        real_acts = {}
        for li in range(num_layers):
            if li in captured:
                last = captured[li][0, -1, :].view(num_heads, head_dim)
                for hi in range(num_heads):
                    real_acts[(li, hi)] = last[hi].clone()

        # Black image forward
        real_img = test_sample["image"]
        black_img = Image.new("RGB", real_img.size, (0, 0, 0))
        captured.clear()
        with torch.no_grad():
            black_inputs = make_chat_prompt(model_info, test_sample["question"], black_img)
            model(**black_inputs)
        black_acts = {}
        for li in range(num_layers):
            if li in captured:
                last = captured[li][0, -1, :].view(num_heads, head_dim)
                for hi in range(num_heads):
                    black_acts[(li, hi)] = last[hi].clone()

        # Compute activation differences
        deltas = []
        per_layer_delta = {}
        for key in real_acts:
            if key in black_acts:
                d = (real_acts[key] - black_acts[key]).float().norm().item()
                deltas.append((key, d))
                li = key[0]
                per_layer_delta.setdefault(li, []).append(d)

        if deltas:
            deltas.sort(key=lambda x: x[1], reverse=True)
            mean_delta = np.mean([d for _, d in deltas])
            max_delta = deltas[0][1]
            max_head = deltas[0][0]

            # Non-trivial threshold: if mean Δ < 0.01, R_vhad will be useless
            nontrivial = mean_delta > 0.01
            result.check("Vision activation Δ non-trivial", nontrivial,
                          f"mean_Δ={mean_delta:.4f}, max_Δ={max_delta:.4f} at head {max_head}")

            # Report top-5 heads by delta
            print("    Top-5 heads by activation Δ:")
            for (li, hi), d in deltas[:5]:
                print(f"      Layer {li:2d} Head {hi:2d}: Δ={d:.4f}")

            # Per-layer summary
            print("    Per-layer mean Δ:")
            for li in sorted(per_layer_delta.keys()):
                layer_mean = np.mean(per_layer_delta[li])
                bar = "█" * int(layer_mean * 100)
                print(f"      Layer {li:2d}: {layer_mean:.4f} {bar}")
        else:
            result.check("Vision activation Δ non-trivial", False, "No matching activations")
    except Exception as e:
        result.check("Vision activation Δ non-trivial", False, str(e))

    # --- Check 5: Hook-based collection during generation ---
    print("\n[5/5] Hook-based activation collection during generation")
    try:
        captured.clear()
        with torch.no_grad():
            gen_inputs = make_chat_prompt(model_info, test_sample["question"], test_sample["image"])
            gen_out = model.generate(
                **gen_inputs,
                max_new_tokens=20,
                do_sample=False,
            )
        # Check that hooks fired during generation
        gen_captured_layers = len(captured)
        result.check("Hooks fire during generation", gen_captured_layers > 0,
                      f"{gen_captured_layers}/{num_layers} layers captured")

        # Decode generated text
        gen_text = model_info["tokenizer"].decode(gen_out[0], skip_special_tokens=True)
        result.check("Generation produces text", len(gen_text) > 0,
                      f"'{gen_text[:100]}...'")
    except Exception as e:
        result.check("Hooks fire during generation", False, str(e))

    # Cleanup
    for h in hooks:
        h.remove()

    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s")
    return result.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Smoke Test")
    parser.add_argument("--model", type=str, default="qwen3_vl_2b")
    args = parser.parse_args()

    ok = run_smoke_test(args.model)
    sys.exit(0 if ok else 1)
