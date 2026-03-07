"""
VIGIL Block 1 v4 — GRPO on open-ended VQA (non-binary).

v1-v3 all collapsed on binary yes/no VQA. The hypothesis:
GRPO fails specifically on binary VQA because output diversity is too low.

v4 test: Use TextVQA (open-ended answers) instead of binary VQA.
If GRPO is stable on open-ended → the issue is binary-specific.
If it still collapses → GRPO is fundamentally broken with TRL on this model.

Reward: R_correct uses exact string match (partial credit for substring).
"""
import sys, os, gc, json, time, random
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from datasets import Dataset, load_from_disk

RESULTS_DIR = Path("lab/results/block1_v4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_yesno(pred, gt):
    p = pred.strip().lower()
    g = gt.strip().lower()
    has_yes = "yes" in p
    has_no = "no" in p
    if has_yes and has_no:
        yn = "yes" if p.index("yes") < p.index("no") else "no"
    elif has_yes:
        yn = "yes"
    elif has_no:
        yn = "no"
    else:
        yn = ""
    return yn == g


def reward_openended(prompts, completions, answer, **kwargs):
    """Soft reward for open-ended VQA: exact match + substring partial credit."""
    rewards = []
    for comp, gt in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        pred = text.strip().lower()
        target = gt.strip().lower()
        if target in pred or pred in target:
            r = 1.0  # exact or substring match
        elif any(w in pred for w in target.split()):
            r = 0.5  # partial word match
        else:
            r = 0.0
        rewards.append(r)
    return rewards


def prepare_textvqa_for_grpo(limit=1000, seed=42):
    """Load TextVQA train for GRPO — open-ended, NOT binary."""
    ds = load_from_disk("data/training/textvqa_train")
    records = list(ds)
    random.seed(seed)
    random.shuffle(records)
    selected = records[:min(limit, len(records))]
    print(f"TextVQA: {len(selected)} samples (open-ended)")

    out = []
    for row in selected:
        q = row["question"]
        # TextVQA may have multiple answers
        ans = str(row.get("answers", row.get("answer", ""))).strip().lower()
        if isinstance(row.get("answers"), list):
            ans = row["answers"][0].strip().lower() if row["answers"] else ""
        out.append({
            "prompt": [{"role": "user", "content": q}],
            "image": row.get("image"),
            "answer": ans,
            "question": q,
        })
    return Dataset.from_list(out)


def eval_pope(model, processor, device, n_samples=100):
    from src.data_loader import load_pope
    from src.model_registry import make_chat_prompt
    model_info = {"model": model, "processor": processor, "tokenizer": processor.tokenizer, "model_type": "qwen3_vl", "device": device}
    samples = load_pope("adversarial", limit=n_samples)
    correct = total = yes_count = no_count = 0
    for s in samples:
        try:
            inputs = make_chat_prompt(model_info, s["question"], s.get("image"))
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            pred = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            p = pred.strip().lower()
            if "yes" in p: yes_count += 1
            elif "no" in p: no_count += 1
            if check_yesno(pred, s["answer"]): correct += 1
            total += 1
        except Exception:
            continue
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"    POPE: {correct}/{total} = {acc:.1f}% (yes={yes_count}, no={no_count})")
    return acc, {"yes": yes_count, "no": no_count, "total": total}


def main():
    from trl import GRPOTrainer, GRPOConfig
    from peft import LoraConfig
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    t0 = time.time()
    print(f"{'='*60}\nVIGIL Block 1 v4: Open-ended VQA GRPO\n{datetime.now().isoformat()}\n{'='*60}")

    train_ds = prepare_textvqa_for_grpo(limit=1000)

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "o_proj", "k_proj"], task_type="CAUSAL_LM")

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    device = next(model.parameters()).device

    grpo_config = GRPOConfig(
        output_dir="checkpoints/block1_v4",
        num_generations=4,
        temperature=1.0,
        beta=0.01,
        max_completion_length=64,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=10,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
        bf16=False,
        fp16=True,
        report_to="none",
    )

    # Baseline
    print("\n--- Baseline POPE eval ---")
    pope_acc_before, dist_before = eval_pope(model, processor, device, n_samples=100)
    print(f"  Baseline POPE: {pope_acc_before:.1f}%")

    # Train
    print(f"\n--- Training (10 steps on TextVQA, open-ended) ---")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_openended,
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=processor,
    )
    trainer.train()

    # Final eval
    print("\n--- Post-training POPE eval ---")
    pope_acc_after, dist_after = eval_pope(model, processor, device, n_samples=100)
    print(f"  Post-training POPE: {pope_acc_after:.1f}%")

    pope_drop = pope_acc_before - pope_acc_after
    if pope_drop > 10:
        print(f"  COLLAPSE: Even open-ended GRPO breaks the model (-{pope_drop:.1f}pp)")
        print(f"  >>> TRL GRPOTrainer is fundamentally broken for this model")
    elif pope_drop > 3:
        print(f"  MODERATE DAMAGE: Open-ended GRPO hurts ({pope_drop:+.1f}pp)")
    else:
        print(f"  STABLE: POPE delta = {-pope_drop:+.1f}pp")
        print(f"  >>> Binary VQA is the problem, not GRPO itself!")
        print(f"  >>> Proceed to Block 2 with open-ended data")

    results = {
        "version": "v4_openended",
        "baseline_pope": pope_acc_before, "baseline_dist": dist_before,
        "final_pope": pope_acc_after, "final_dist": dist_after,
        "pope_drop": pope_drop,
        "train_log": trainer.state.log_history,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rpath = RESULTS_DIR / f"openended_grpo_{ts}.json"
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {rpath}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
