"""
VIGIL Block 1 v3 — Ultra-conservative GRPO.

v1 collapsed (always-no, 50 steps). v2 collapsed (always-yes, 20 steps).
Both with group_size 4-8, temp 1.0-1.2, LoRA r=8-16.

v3 hypothesis: GRPO on binary VQA is inherently unstable at 2B scale.
Fix with extreme conservatism:
  - beta=0.1 (10x higher KL penalty — keep model close to reference)
  - 5 steps only (minimal drift)
  - lr=5e-7 (10x lower)
  - LoRA r=4 (minimal parameters)
  - eval every 1 step to catch collapse immediately
  - If collapse still happens at 5 steps: GRPO is not viable for this task.
    → Pivot to KTO or DPO with IIG-based preferences.
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

RESULTS_DIR = Path("lab/results/block1_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IIG_LAMBDA = 0.0615
IIG_EPS = 0.1


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


def reward_correct_only(prompts, completions, answer, **kwargs):
    rewards = []
    for comp, gt in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        r = 1.0 if check_yesno(text, gt) else 0.0
        rewards.append(r)
    return rewards


def prepare_vqav2_for_grpo(limit=2000, seed=42):
    ds = load_from_disk("data/training/vqav2_train")
    yesno = [r for r in ds if str(r.get("multiple_choice_answer", "")).strip().lower() in ("yes", "no")]
    random.seed(seed)
    random.shuffle(yesno)
    selected = yesno[:min(limit, len(yesno))]
    yes_count = sum(1 for s in selected if str(s.get("multiple_choice_answer", "")).strip().lower() == "yes")
    no_count = len(selected) - yes_count
    print(f"VQAv2: {len(selected)} samples ({yes_count} yes, {no_count} no)")
    records = []
    for row in selected:
        q = row["question"]
        ans = str(row["multiple_choice_answer"]).strip().lower()
        records.append({
            "prompt": [{"role": "user", "content": q}],
            "image": row["image"],
            "answer": ans,
            "question": q,
        })
    return Dataset.from_list(records)


def eval_pope(model, processor, device, n_samples=200):
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


def eval_blind_test(model, processor, device, n_samples=100):
    from src.data_loader import load_pope
    from src.model_registry import make_chat_prompt
    model_info = {"model": model, "processor": processor, "tokenizer": processor.tokenizer, "model_type": "qwen3_vl", "device": device}
    samples = load_pope("adversarial", limit=n_samples)
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))
    correct_real = correct_black = total = 0
    for s in samples:
        try:
            inputs_r = make_chat_prompt(model_info, s["question"], s.get("image"))
            with torch.no_grad():
                out_r = model.generate(**inputs_r, max_new_tokens=30, do_sample=False)
            pred_r = processor.tokenizer.decode(out_r[0][inputs_r["input_ids"].shape[1]:], skip_special_tokens=True)
            if check_yesno(pred_r, s["answer"]): correct_real += 1
            inputs_b = make_chat_prompt(model_info, s["question"], black_img)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=30, do_sample=False)
            pred_b = processor.tokenizer.decode(out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=True)
            if check_yesno(pred_b, s["answer"]): correct_black += 1
            total += 1
        except Exception:
            continue
    acc_real = correct_real / total * 100 if total > 0 else 0.0
    acc_black = correct_black / total * 100 if total > 0 else 0.0
    return acc_real, acc_black, acc_real - acc_black


def main():
    from trl import GRPOTrainer, GRPOConfig
    from peft import LoraConfig
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from src.iig import compute_iig, vigil_reward

    t0 = time.time()
    print(f"{'='*60}\nVIGIL Block 1 v3: Ultra-conservative GRPO\n{datetime.now().isoformat()}\n{'='*60}")

    train_ds = prepare_vqav2_for_grpo(limit=2000)
    all_results = {}

    # Ultra-conservative LoRA
    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    for setting_name, use_iig in [("A_correct_only", False), ("B_correct_iig", True)]:
        print(f"\n{'='*60}\n  Setting: {setting_name}\n{'='*60}")

        print("Loading model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        device = next(model.parameters()).device

        if use_iig:
            iig_model_info = {
                "model": model, "processor": processor,
                "tokenizer": processor.tokenizer, "model_type": "qwen3_vl", "device": device,
            }
            iig_log = []

            def reward_fn(prompts, completions, answer, image, question, **kwargs):
                rewards = []
                for comp, gt, img, q in zip(completions, answer, image, question):
                    text = comp[0]["content"] if isinstance(comp, list) else str(comp)
                    r_correct = 1.0 if check_yesno(text, gt) else 0.0
                    try:
                        iig = compute_iig(iig_model_info, q, img, text)
                    except Exception:
                        iig = 0.0
                    iig_log.append(iig)
                    r_total = vigil_reward(r_correct, iig, IIG_LAMBDA, IIG_EPS)
                    rewards.append(r_total)
                return rewards
            reward_fn.__name__ = "reward_correct_iig"
        else:
            iig_log = None
            reward_fn = reward_correct_only

        output_dir = Path("checkpoints/block1_v3") / setting_name
        grpo_config = GRPOConfig(
            output_dir=str(output_dir),
            num_generations=4,
            temperature=1.0,
            beta=0.1,                   # 10x higher KL penalty
            max_completion_length=64,
            learning_rate=5e-7,          # 10x lower lr
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=5,                 # minimal steps
            logging_steps=1,
            save_steps=5,
            remove_unused_columns=False,
            bf16=False,
            fp16=True,
            report_to="none",
        )

        # Baseline eval (smaller for speed)
        print("\n--- Baseline eval ---")
        pope_acc, pope_dist = eval_pope(model, processor, device, n_samples=100)
        real, black, gap = eval_blind_test(model, processor, device, n_samples=100)
        baseline_pope = pope_acc
        evals = [{"step": 0, "pope_acc": pope_acc, "pope_dist": pope_dist,
                  "blind_real": real, "blind_black": black, "blind_gap": gap}]
        print(f"  Step 0: POPE={pope_acc:.1f}%, Gap={gap:.1f}pp")

        # Train
        print(f"\n--- Training (5 steps, beta=0.1, lr=5e-7, r=4) ---")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=train_ds,
            peft_config=lora_config,
            processing_class=processor,
        )
        trainer.train()

        # Final eval
        print("\n--- Final eval (step 5) ---")
        pope_acc, pope_dist = eval_pope(model, processor, device, n_samples=100)
        real, black, gap = eval_blind_test(model, processor, device, n_samples=100)
        evals.append({"step": 5, "pope_acc": pope_acc, "pope_dist": pope_dist,
                      "blind_real": real, "blind_black": black, "blind_gap": gap})
        print(f"  Step 5: POPE={pope_acc:.1f}%, Gap={gap:.1f}pp")

        pope_drop = baseline_pope - pope_acc
        if pope_drop > 10:
            print(f"  COLLAPSE at 5 steps! GRPO is not viable for binary VQA at 2B.")
        elif pope_drop > 3:
            print(f"  CAUTION: POPE dropped {pope_drop:.1f}pp at just 5 steps")
        else:
            print(f"  STABLE: POPE delta = {-pope_drop:+.1f}pp")

        results = {
            "setting": setting_name, "version": "v3",
            "config": {"group_size": 4, "temp": 1.0, "max_completion": 64,
                       "lr": 5e-7, "lora_r": 4, "lora_alpha": 8, "beta": 0.1, "steps": 5},
            "eval_results": evals,
            "train_log": trainer.state.log_history,
            "iig_values": iig_log,
            "lambda": IIG_LAMBDA if use_iig else None,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpath = RESULTS_DIR / f"{setting_name}_{ts}.json"
        with open(rpath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved: {rpath}")
        all_results[setting_name] = results

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    # Comparison
    elapsed = time.time() - t0
    print(f"\n{'='*60}\nBLOCK 1 v3 COMPARISON\n{'='*60}")

    for label, key in [("(A) R_correct only", "A_correct_only"), ("(B) R_correct + IIG", "B_correct_iig")]:
        res = all_results[key]
        print(f"\n{label}:")
        for ev in res["eval_results"]:
            d = ev.get("pope_dist", {})
            print(f"  Step {ev['step']:3d}: POPE={ev['pope_acc']:.1f}% (y={d.get('yes','?')},n={d.get('no','?')}), Gap={ev['blind_gap']:.1f}pp")

    gap_a = all_results["A_correct_only"]["eval_results"][-1]["blind_gap"]
    gap_b = all_results["B_correct_iig"]["eval_results"][-1]["blind_gap"]
    pope_a = all_results["A_correct_only"]["eval_results"][-1]["pope_acc"]
    pope_b = all_results["B_correct_iig"]["eval_results"][-1]["pope_acc"]
    baseline_a = all_results["A_correct_only"]["eval_results"][0]["pope_acc"]

    iig_vals = all_results["B_correct_iig"].get("iig_values", [])
    if iig_vals:
        iig_arr = np.array(iig_vals)
        print(f"\n--- IIG: mean={iig_arr.mean():.3f}, std={iig_arr.std():.3f}, "
              f"positive={int((iig_arr>0).sum())}/{len(iig_arr)} ---")

    drop_a = baseline_a - pope_a
    drop_b = all_results["B_correct_iig"]["eval_results"][0]["pope_acc"] - pope_b

    print(f"\n--- Go/No-Go ---")
    print(f"  POPE drop (A): {drop_a:+.1f}pp, (B): {drop_b:+.1f}pp")
    print(f"  Gap (A): {gap_a:.1f}pp, (B): {gap_b:.1f}pp, Delta: {gap_b-gap_a:+.1f}pp")

    if drop_a > 10 or drop_b > 10:
        print(f"  >>> COLLAPSE even at 5 steps. GRPO NOT VIABLE for binary VQA at 2B.")
        print(f"  >>> PIVOT: Use KTO or DPO with IIG-based preferences instead.")
    elif gap_b >= gap_a:
        print(f"  >>> PROCEED: Gap(B) >= Gap(A), model stable")
    elif gap_b < gap_a - 2:
        print(f"  >>> STOP: IIG hurts")
    else:
        print(f"  >>> MARGINAL")

    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
