"""
Phase 5: Three experiments for improved LSR integration.

Experiment A (vppo): VPPO token masking — gradients weighted by per-token LSR
Experiment B (dpo): DPO with LSR-ranked preference pairs
Experiment C (mult_gate): Multiplicative gated GDPO — R_lsr_gated = R_correct * R_lsr

All experiments share: model loading, data, generation, eval from phase4_gdpo.py.

Usage:
    # Experiment A: VPPO token masking
    python scripts/phase5_experiments.py --experiment vppo \
        --steps 50 --model-path Qwen/Qwen3-VL-2B-Thinking \
        --output-dir checkpoints/phase5/vppo

    # Experiment B: DPO with LSR pairs
    python scripts/phase5_experiments.py --experiment dpo \
        --steps 50 --model-path Qwen/Qwen3-VL-2B-Thinking \
        --output-dir checkpoints/phase5/dpo

    # Experiment C: Multiplicative gated GDPO
    python scripts/phase5_experiments.py --experiment mult_gate \
        --steps 50 --model-path Qwen/Qwen3-VL-2B-Thinking \
        --output-dir checkpoints/phase5/mult_gate
"""

import os, sys, gc, json, re, time, random, argparse, string
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase4_gdpo import (
    split_thinking, extract_yes_no, extract_answer,
    find_think_token_range, compute_format_reward,
    load_training_data, load_pope_eval, load_model,
    prepare_inputs, generate_candidates, compute_r_correct,
    compute_logprobs, compute_gdpo_advantages, compute_policy_loss,
    evaluate_pope, evaluate_blind,
    DEFAULT_MODEL, HF_ID,
)

PROJECT_ROOT = Path(__file__).parent.parent


# ══════════════════════════════════════════════════════════════════════
#  Shared: Per-token LSR
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_lsr_per_token(model, processor, sample, candidate_ids,
                          think_range, device):
    """Like compute_thinking_lsr but returns per-token KL array.

    Returns:
        per_token_kl: np.ndarray of shape [think_len]
        mean_lsr: float
        think_len: int
    """
    if candidate_ids.numel() == 0:
        return np.array([]), 0.0, 0

    t_start, t_end = think_range
    if t_end <= t_start:
        return np.array([]), 0.0, 0

    image = sample["image"]
    question = sample["question"]
    candidate_ids = candidate_ids.clone().detach()

    # Real image forward
    real_inputs = prepare_inputs(processor, image, question, device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"],
                    candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    logits_real = model(**real_inputs).logits[0]

    # Black image forward
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"],
                    candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    logits_black = model(**black_inputs).logits[0]

    # Align logits to candidate tokens
    lr = logits_real[rpl - 1: rpl - 1 + len(candidate_ids)]
    lb = logits_black[bpl - 1: bpl - 1 + len(candidate_ids)]

    ml = min(lr.shape[0], lb.shape[0], len(candidate_ids))
    t_end_safe = min(t_end, ml)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe

    if think_len <= 0:
        return np.array([]), 0.0, 0

    lr_think = lr[t_start_safe:t_end_safe].float()
    lb_think = lb[t_start_safe:t_end_safe].float()

    kl = F.kl_div(
        F.log_softmax(lb_think, dim=-1),
        F.softmax(lr_think, dim=-1),
        reduction='none'
    ).sum(dim=-1)

    per_token_kl = kl.cpu().numpy()
    mean_lsr = float(per_token_kl.mean())

    del logits_real, logits_black, lr, lb, lr_think, lb_think, kl
    del real_inputs, black_inputs
    return per_token_kl, mean_lsr, think_len


# ══════════════════════════════════════════════════════════════════════
#  Shared: Report Generation
# ══════════════════════════════════════════════════════════════════════

def generate_report(history, report_dir, experiment_name):
    """Generate markdown report + training dynamics plot."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save history
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(report_dir / f"history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Report
    cfg = history["config"]
    pre = history.get("pre_eval", {})
    pre_pope = pre.get("pope", {}).get("acc", 0)
    pre_gap = pre.get("blind", {}).get("gap", 0)

    steps = [s for s in history["steps"] if not s.get("skipped", False)]
    skipped = [s for s in history["steps"] if s.get("skipped", False)]

    evals = history.get("evals", [])
    best_pope = max([e["pope"]["acc"] for e in evals], default=pre_pope)

    lines = [
        f"# Phase 5: {experiment_name}",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model**: {history.get('base_model', 'unknown')}",
        f"\n## Config\n",
        f"| Parameter | Value |",
        f"|-----------|-------|",
    ]
    for k, v in cfg.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"\n## Results\n",
        f"| Metric | Pre | Best | Delta |",
        f"|--------|-----|------|-------|",
        f"| POPE | {pre_pope:.1%} | {best_pope:.1%} | {best_pope-pre_pope:+.1%} |",
        f"| Gap | {pre_gap:.1%} | — | — |",
        f"\n**Steps**: {len(steps)} effective / {len(steps)+len(skipped)} total "
        f"({len(skipped)} skipped, {len(skipped)/(len(steps)+len(skipped))*100:.0f}%)\n",
    ]

    if steps:
        losses = [s["loss"] for s in steps]
        lines.append(f"**Loss**: mean={np.mean(losses):.4f}, "
                      f"min={np.min(losses):.4f}, max={np.max(losses):.4f}\n")

    with open(report_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines))

    # Plot training dynamics
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Phase 5: {experiment_name}", fontsize=14)

        if steps:
            step_nums = [s["step"] for s in steps]
            losses = [s["loss"] for s in steps]
            axes[0].plot(step_nums, losses, 'b-', alpha=0.7)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(True, alpha=0.3)

        if evals:
            eval_steps = [e["step"] for e in evals]
            eval_accs = [e["pope"]["acc"] for e in evals]
            axes[1].plot(eval_steps, eval_accs, 'ro-', markersize=8)
            axes[1].axhline(y=pre_pope, color='gray', linestyle='--', label='Pre-train')
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("POPE Accuracy")
            axes[1].set_title("POPE Accuracy")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / "training_dynamics.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [report] Plot failed: {e}")

    print(f"  [report] {report_dir}")


# ══════════════════════════════════════════════════════════════════════
#  Experiment A: VPPO Token Masking
# ══════════════════════════════════════════════════════════════════════

def compute_vppo_token_mask(per_token_kl, think_range, gen_len, low_weight=0.1):
    """Create token-level weight mask based on per-token LSR.

    Tokens with above-median LSR get weight=1.0, below get low_weight.
    Answer tokens (after thinking) always get weight=1.0.
    """
    mask = torch.ones(gen_len) * low_weight

    t_start, t_end = think_range

    if len(per_token_kl) > 0 and per_token_kl.std() > 1e-6:
        median_kl = np.median(per_token_kl)
        for i, kl_val in enumerate(per_token_kl):
            idx = t_start + i
            if idx < gen_len:
                mask[idx] = 1.0 if kl_val >= median_kl else low_weight
    else:
        # No variance in KL — don't mask (all 1.0)
        mask[t_start:min(t_end, gen_len)] = 1.0

    # Answer tokens always get full weight
    if t_end < gen_len:
        mask[t_end:] = 1.0

    return mask


def compute_vppo_policy_loss(model, inputs, cand_ids_list, prompt_len,
                             advantages, token_masks, cfg):
    """REINFORCE with per-token VPPO masking + entropy bonus."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"entropy": [], "mask_density": []}

    for cand_ids, adv, mask in zip(cand_ids_list, advantages, token_masks):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        token_lp, entropy = compute_logprobs(model, inputs, cand_ids, prompt_len)

        # Apply VPPO mask
        mask_t = mask[:len(token_lp)].to(token_lp.device)
        weighted_lp = (token_lp * mask_t).sum() / (mask_t.sum() + 1e-8)

        policy_loss = -weighted_lp * adv
        loss = policy_loss - cfg["beta_entropy"] * entropy

        total_loss = total_loss + loss
        n_valid += 1
        stats["entropy"].append(entropy.item())
        stats["mask_density"].append(float((mask_t > 0.5).float().mean()))

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss, stats


def run_vppo(cfg, train_data, eval_data, model_path):
    """Experiment A: VPPO token masking — weight gradients by per-token LSR."""
    output_dir = Path(cfg["output_dir"])
    report_dir = PROJECT_ROOT / "lab" / "reports" / "phase5" / "vppo"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    weights = {"correct": 0.7, "format": 0.3}

    print(f"\n{'='*70}")
    print(f"  Phase 5A: VPPO Token Masking")
    print(f"  {cfg['num_steps']} steps | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    print(f"  Low weight: {cfg['vppo_low_weight']}")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_pope = evaluate_pope(model, processor, eval_data, device, 60)
    pre_blind = evaluate_blind(model, processor, eval_data, device, 50)
    print(f"  POPE: {pre_pope['acc']:.1%} | Gap: {pre_blind['gap']:.1%} | "
          f"Think: {pre_pope['avg_think_words']:.0f}w")

    history = {
        "config": cfg, "experiment": "vppo",
        "base_model": model_path or DEFAULT_MODEL,
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "steps": [], "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate candidates
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        try:
            candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg.get("top_p", 0.95),
                    cfg.get("max_new_tokens", 512),
                    cfg.get("min_think_tokens", 32), device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM gen, skip")
            history["steps"].append({"step": step+1, "skipped": True})
            continue

        # Compute rewards + per-token LSR for masking
        r_corrects, r_formats = [], []
        token_masks = []
        details = []
        gt = sample["answer"]
        qtype = sample.get("type", "short_answer")

        for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
            pred = extract_answer(cand, qtype)
            r_correct = compute_r_correct(pred, gt, qtype)
            r_format = compute_format_reward(cand)
            r_corrects.append(r_correct)
            r_formats.append(r_format)

            # Per-token LSR for mask
            try:
                per_tok_kl, mean_lsr, tlen = compute_lsr_per_token(
                    model, processor, sample, cand_ids, t_range, device)
            except Exception:
                per_tok_kl, mean_lsr, tlen = np.array([]), 0.0, 0

            mask = compute_vppo_token_mask(
                per_tok_kl, t_range, len(cand_ids), cfg["vppo_low_weight"])
            token_masks.append(mask)

            thinking, _ = split_thinking(cand)
            details.append({
                "correct": r_correct, "format": r_format,
                "lsr_raw": mean_lsr,
                "think_words": len(thinking.split()) if thinking else 0,
                "mask_density": float((mask > 0.5).float().mean()),
            })

        reward_components = {
            "correct": np.array(r_corrects),
            "format": np.array(r_formats),
        }

        advantages, has_variance, reward_stats = compute_gdpo_advantages(
            reward_components, weights)

        if not has_variance:
            elapsed = time.time() - step_t0
            mt = np.mean([d["think_words"] for d in details])
            print(f"  [step {step+1}/{cfg['num_steps']}] "
                  f"SKIP (zero var) think={mt:.0f}w ({elapsed:.1f}s)")
            history["steps"].append({"step": step+1, "skipped": True})
            del candidates, cand_ids_list, inputs, token_masks
            continue

        # VPPO policy loss with token masks
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        try:
            loss, lstats = compute_vppo_policy_loss(
                model, inputs, cand_ids_list, prompt_len,
                advantages.tolist(), token_masks, cfg)
            (loss / cfg["grad_accum"]).backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM loss, skip")
            history["steps"].append({"step": step+1, "skipped": True})
            continue

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        mf = np.mean([d["format"] for d in details])
        ml = np.mean([d["lsr_raw"] for d in details])
        mt = np.mean([d["think_words"] for d in details])
        md = np.mean([d["mask_density"] for d in details])
        var_str = " | ".join(f"{k}:{v['std']:.3f}" for k, v in reward_stats.items())

        history["steps"].append({
            "step": step+1, "loss": loss.item(),
            "mean_correct": float(mc), "mean_format": float(mf),
            "mean_lsr_raw": float(ml), "mean_think_words": float(mt),
            "mask_density": float(md),
            "reward_stats": reward_stats, "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} correct={mc:.2f} fmt={mf:.2f} "
              f"LSR={ml:.3f} mask={md:.0%} [{var_str}] "
              f"think={mt:.0f}w ({elapsed:.1f}s)", flush=True)

        # Eval
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["num_steps"]:
            pope_res = evaluate_pope(model, processor, eval_data, device, 60)
            blind_res = evaluate_blind(model, processor, eval_data, device, 50)
            print(f"  === Eval step {step+1}: "
                  f"POPE={pope_res['acc']:.1%} Gap={blind_res['gap']:.1%} "
                  f"Think={pope_res['avg_think_words']:.0f}w ===")
            history["evals"].append({
                "step": step+1, "pope": pope_res, "blind": blind_res,
            })
            if pope_res["acc"] > best_acc:
                best_acc = pope_res["acc"]
                print(f"  * New best: {best_acc:.1%}")
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")

        del candidates, cand_ids_list, inputs, token_masks
        torch.cuda.empty_cache()

    # Final save
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    generate_report(history, report_dir, "VPPO Token Masking")

    final_pope = history["evals"][-1]["pope"]["acc"] if history["evals"] else pre_pope["acc"]
    final_gap = history["evals"][-1]["blind"]["gap"] if history["evals"] else pre_blind["gap"]
    print(f"\n{'='*70}")
    print(f"  VPPO Complete")
    print(f"  POPE: {pre_pope['acc']:.1%} -> {final_pope:.1%} "
          f"({final_pope - pre_pope['acc']:+.1%})")
    print(f"  Gap:  {pre_blind['gap']:.1%} -> {final_gap:.1%} "
          f"({final_gap - pre_blind['gap']:+.1%})")
    print(f"  Best: {best_acc:.1%}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════
#  Experiment B: DPO with LSR-ranked Pairs
# ══════════════════════════════════════════════════════════════════════

def compute_dpo_loss(model, inputs, chosen_ids, rejected_ids, prompt_len, beta):
    """DPO loss: -log(sigmoid(beta * (logP(chosen) - logP(rejected))))

    Reference-free DPO (SimPO-style). beta controls implicit KL regularization.
    """
    chosen_lp, chosen_ent = compute_logprobs(model, inputs, chosen_ids, prompt_len)
    chosen_mean = chosen_lp.mean()

    rejected_lp, rejected_ent = compute_logprobs(model, inputs, rejected_ids, prompt_len)
    rejected_mean = rejected_lp.mean()

    logit_diff = beta * (chosen_mean - rejected_mean)
    loss = -F.logsigmoid(logit_diff)

    stats = {
        "chosen_lp": chosen_mean.item(),
        "rejected_lp": rejected_mean.item(),
        "margin": (chosen_mean - rejected_mean).item(),
        "entropy": (chosen_ent.item() + rejected_ent.item()) / 2,
    }
    return loss, stats


def run_dpo(cfg, train_data, eval_data, model_path):
    """Experiment B: DPO with LSR-ranked preference pairs."""
    output_dir = Path(cfg["output_dir"])
    report_dir = PROJECT_ROOT / "lab" / "reports" / "phase5" / "dpo"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Phase 5B: DPO with LSR-ranked Pairs")
    print(f"  {cfg['num_steps']} steps | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    print(f"  beta={cfg['dpo_beta']} | "
          f"LSR weight={cfg['dpo_lsr_weight']} | "
          f"format weight={cfg['dpo_format_weight']}")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_pope = evaluate_pope(model, processor, eval_data, device, 60)
    pre_blind = evaluate_blind(model, processor, eval_data, device, 50)
    print(f"  POPE: {pre_pope['acc']:.1%} | Gap: {pre_blind['gap']:.1%} | "
          f"Think: {pre_pope['avg_think_words']:.0f}w")

    history = {
        "config": cfg, "experiment": "dpo",
        "base_model": model_path or DEFAULT_MODEL,
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "steps": [], "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate candidates
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        try:
            candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg.get("top_p", 0.95),
                    cfg.get("max_new_tokens", 512),
                    cfg.get("min_think_tokens", 32), device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM gen, skip")
            history["steps"].append({"step": step+1, "skipped": True})
            continue

        # Score each candidate
        gt = sample["answer"]
        qtype = sample.get("type", "short_answer")
        scored = []

        for i, (cand, cand_ids, t_range) in enumerate(
                zip(candidates, cand_ids_list, think_ranges)):
            pred = extract_answer(cand, qtype)
            r_correct = compute_r_correct(pred, gt, qtype)
            r_format = compute_format_reward(cand)

            r_lsr = 0.0
            if cfg.get("dpo_use_lsr", True):
                try:
                    _, r_lsr_raw, _ = compute_lsr_per_token(
                        model, processor, sample, cand_ids, t_range, device)
                    r_lsr = min(r_lsr_raw / cfg.get("lsr_scale", 2.0), 1.0)
                except Exception:
                    pass

            score = (r_correct
                     + cfg["dpo_lsr_weight"] * r_lsr
                     + cfg["dpo_format_weight"] * r_format)

            thinking, _ = split_thinking(cand)
            scored.append({
                "idx": i, "score": score,
                "correct": r_correct, "format": r_format, "lsr": r_lsr,
                "think_words": len(thinking.split()) if thinking else 0,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        worst = scored[-1]

        # Skip ties
        if best["score"] <= worst["score"] + 1e-6:
            elapsed = time.time() - step_t0
            mt = np.mean([s["think_words"] for s in scored])
            print(f"  [step {step+1}/{cfg['num_steps']}] "
                  f"SKIP (tie) score={best['score']:.3f} think={mt:.0f}w ({elapsed:.1f}s)")
            history["steps"].append({"step": step+1, "skipped": True, "reason": "tie"})
            del candidates, cand_ids_list, inputs
            continue

        # DPO loss
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        try:
            loss, dpo_stats = compute_dpo_loss(
                model, inputs,
                cand_ids_list[best["idx"]],
                cand_ids_list[worst["idx"]],
                prompt_len, cfg["dpo_beta"])
            (loss / cfg["grad_accum"]).backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM loss, skip")
            history["steps"].append({"step": step+1, "skipped": True})
            continue

        elapsed = time.time() - step_t0
        mt = np.mean([s["think_words"] for s in scored])

        history["steps"].append({
            "step": step+1, "loss": loss.item(),
            "best_score": best["score"], "worst_score": worst["score"],
            "best_correct": best["correct"], "worst_correct": worst["correct"],
            "margin": dpo_stats["margin"],
            "mean_think_words": float(mt),
            "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} "
              f"best={best['score']:.2f}(c={best['correct']:.0f}) "
              f"worst={worst['score']:.2f}(c={worst['correct']:.0f}) "
              f"margin={dpo_stats['margin']:.3f} "
              f"think={mt:.0f}w ({elapsed:.1f}s)", flush=True)

        # Eval
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["num_steps"]:
            pope_res = evaluate_pope(model, processor, eval_data, device, 60)
            blind_res = evaluate_blind(model, processor, eval_data, device, 50)
            print(f"  === Eval step {step+1}: "
                  f"POPE={pope_res['acc']:.1%} Gap={blind_res['gap']:.1%} "
                  f"Think={pope_res['avg_think_words']:.0f}w ===")
            history["evals"].append({
                "step": step+1, "pope": pope_res, "blind": blind_res,
            })
            if pope_res["acc"] > best_acc:
                best_acc = pope_res["acc"]
                print(f"  * New best: {best_acc:.1%}")
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")

        del candidates, cand_ids_list, inputs
        torch.cuda.empty_cache()

    # Final save
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    generate_report(history, report_dir, "DPO with LSR-ranked Pairs")

    final_pope = history["evals"][-1]["pope"]["acc"] if history["evals"] else pre_pope["acc"]
    final_gap = history["evals"][-1]["blind"]["gap"] if history["evals"] else pre_blind["gap"]
    print(f"\n{'='*70}")
    print(f"  DPO Complete")
    print(f"  POPE: {pre_pope['acc']:.1%} -> {final_pope:.1%} "
          f"({final_pope - pre_pope['acc']:+.1%})")
    print(f"  Gap:  {pre_blind['gap']:.1%} -> {final_gap:.1%} "
          f"({final_gap - pre_blind['gap']:+.1%})")
    print(f"  Best: {best_acc:.1%}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════
#  Experiment C: Multiplicative Gated GDPO
# ══════════════════════════════════════════════════════════════════════

def run_mult_gate(cfg, train_data, eval_data, model_path):
    """Experiment C: GDPO with R_lsr_gated = R_correct * R_lsr."""
    output_dir = Path(cfg["output_dir"])
    report_dir = PROJECT_ROOT / "lab" / "reports" / "phase5" / "mult_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    weights = {
        "correct": cfg.get("w_correct", 0.5),
        "format": cfg.get("w_format", 0.2),
        "lsr_gated": cfg.get("w_lsr", 0.3),
    }

    print(f"\n{'='*70}")
    print(f"  Phase 5C: Multiplicative Gated GDPO")
    print(f"  {cfg['num_steps']} steps | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    print(f"  Rewards: correct={weights['correct']}, "
          f"format={weights['format']}, lsr_gated={weights['lsr_gated']}")
    print(f"  Gate: R_lsr_gated = R_correct * R_lsr_norm")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_pope = evaluate_pope(model, processor, eval_data, device, 60)
    pre_blind = evaluate_blind(model, processor, eval_data, device, 50)
    print(f"  POPE: {pre_pope['acc']:.1%} | Gap: {pre_blind['gap']:.1%} | "
          f"Think: {pre_pope['avg_think_words']:.0f}w")

    history = {
        "config": cfg, "experiment": "mult_gate",
        "base_model": model_path or DEFAULT_MODEL,
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "steps": [], "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate candidates
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        try:
            candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg.get("top_p", 0.95),
                    cfg.get("max_new_tokens", 512),
                    cfg.get("min_think_tokens", 32), device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM gen, skip")
            history["steps"].append({"step": step+1, "skipped": True})
            continue

        # Compute rewards with multiplicative LSR gating
        gt = sample["answer"]
        qtype = sample.get("type", "short_answer")
        r_corrects, r_formats, r_lsrs_gated = [], [], []
        details = []

        for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
            pred = extract_answer(cand, qtype)
            r_correct = compute_r_correct(pred, gt, qtype)
            r_format = compute_format_reward(cand)

            r_lsr_raw = 0.0
            try:
                _, r_lsr_raw, _ = compute_lsr_per_token(
                    model, processor, sample, cand_ids, t_range, device)
            except Exception:
                pass

            r_lsr_norm = min(r_lsr_raw / cfg.get("lsr_scale", 2.0), 1.0)
            # MULTIPLICATIVE GATE: correct * lsr
            r_lsr_gated = r_correct * r_lsr_norm

            r_corrects.append(r_correct)
            r_formats.append(r_format)
            r_lsrs_gated.append(r_lsr_gated)

            thinking, _ = split_thinking(cand)
            details.append({
                "correct": r_correct, "format": r_format,
                "lsr_raw": r_lsr_raw, "lsr_gated": r_lsr_gated,
                "think_words": len(thinking.split()) if thinking else 0,
            })

        reward_components = {
            "correct": np.array(r_corrects),
            "format": np.array(r_formats),
            "lsr_gated": np.array(r_lsrs_gated),
        }

        advantages, has_variance, reward_stats = compute_gdpo_advantages(
            reward_components, weights)

        if not has_variance:
            elapsed = time.time() - step_t0
            mt = np.mean([d["think_words"] for d in details])
            print(f"  [step {step+1}/{cfg['num_steps']}] "
                  f"SKIP (zero var) think={mt:.0f}w ({elapsed:.1f}s)")
            history["steps"].append({"step": step+1, "skipped": True})
            del candidates, cand_ids_list, inputs
            continue

        # Policy loss
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        try:
            loss, lstats = compute_policy_loss(
                model, inputs, cand_ids_list, prompt_len,
                advantages.tolist(), cfg)
            (loss / cfg["grad_accum"]).backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM loss, skip")
            history["steps"].append({"step": step+1, "skipped": True})
            continue

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        mf = np.mean([d["format"] for d in details])
        ml = np.mean([d["lsr_raw"] for d in details])
        mg = np.mean([d["lsr_gated"] for d in details])
        mt = np.mean([d["think_words"] for d in details])
        var_str = " | ".join(f"{k}:{v['std']:.3f}" for k, v in reward_stats.items())

        history["steps"].append({
            "step": step+1, "loss": loss.item(),
            "mean_correct": float(mc), "mean_format": float(mf),
            "mean_lsr_raw": float(ml), "mean_lsr_gated": float(mg),
            "mean_think_words": float(mt),
            "reward_stats": reward_stats, "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} correct={mc:.2f} fmt={mf:.2f} "
              f"LSR={ml:.3f} gated={mg:.3f} [{var_str}] "
              f"think={mt:.0f}w ({elapsed:.1f}s)", flush=True)

        # Eval
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["num_steps"]:
            pope_res = evaluate_pope(model, processor, eval_data, device, 60)
            blind_res = evaluate_blind(model, processor, eval_data, device, 50)
            print(f"  === Eval step {step+1}: "
                  f"POPE={pope_res['acc']:.1%} Gap={blind_res['gap']:.1%} "
                  f"Think={pope_res['avg_think_words']:.0f}w ===")
            history["evals"].append({
                "step": step+1, "pope": pope_res, "blind": blind_res,
            })
            if pope_res["acc"] > best_acc:
                best_acc = pope_res["acc"]
                print(f"  * New best: {best_acc:.1%}")
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")

        del candidates, cand_ids_list, inputs
        torch.cuda.empty_cache()

    # Final save
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    generate_report(history, report_dir, "Multiplicative Gated GDPO")

    final_pope = history["evals"][-1]["pope"]["acc"] if history["evals"] else pre_pope["acc"]
    final_gap = history["evals"][-1]["blind"]["gap"] if history["evals"] else pre_blind["gap"]
    print(f"\n{'='*70}")
    print(f"  Mult Gate Complete")
    print(f"  POPE: {pre_pope['acc']:.1%} -> {final_pope:.1%} "
          f"({final_pope - pre_pope['acc']:+.1%})")
    print(f"  Gap:  {pre_blind['gap']:.1%} -> {final_gap:.1%} "
          f"({final_gap - pre_blind['gap']:+.1%})")
    print(f"  Best: {best_acc:.1%}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 5 Experiments")
    parser.add_argument("--experiment", required=True,
                        choices=["vppo", "dpo", "mult_gate"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lsr-scale", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=999)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=32)

    # VPPO-specific
    parser.add_argument("--vppo-low-weight", type=float, default=0.1,
                        help="Weight for below-median tokens (Exp A)")

    # DPO-specific
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="DPO temperature (Exp B)")
    parser.add_argument("--dpo-lsr-weight", type=float, default=0.3,
                        help="LSR weight in DPO scoring (Exp B)")
    parser.add_argument("--dpo-format-weight", type=float, default=0.1,
                        help="Format weight in DPO scoring (Exp B)")
    parser.add_argument("--dpo-use-lsr", action="store_true", default=True,
                        help="Include LSR in DPO pair scoring (Exp B)")
    parser.add_argument("--no-dpo-lsr", dest="dpo_use_lsr", action="store_false",
                        help="Exclude LSR from DPO scoring")

    # Mult gate weights
    parser.add_argument("--w-correct", type=float, default=0.5)
    parser.add_argument("--w-format", type=float, default=0.2)
    parser.add_argument("--w-lsr", type=float, default=0.3)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_data = load_training_data(args.train_samples, args.seed)
    eval_data = load_pope_eval(args.eval_samples)

    cfg = {
        "experiment": args.experiment,
        "num_steps": args.steps,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_new_tokens": args.max_new_tokens,
        "min_think_tokens": args.min_think_tokens,
        "lr": args.lr,
        "beta_entropy": 0.01,
        "grad_accum": 2,
        "max_grad_norm": 1.0,
        "lsr_scale": args.lsr_scale,
        "eval_every": args.eval_every,
        "output_dir": args.output_dir,
        # VPPO
        "vppo_low_weight": args.vppo_low_weight,
        # DPO
        "dpo_beta": args.dpo_beta,
        "dpo_lsr_weight": args.dpo_lsr_weight,
        "dpo_format_weight": args.dpo_format_weight,
        "dpo_use_lsr": args.dpo_use_lsr,
        # Mult gate
        "w_correct": args.w_correct,
        "w_format": args.w_format,
        "w_lsr": args.w_lsr,
    }

    if args.experiment == "vppo":
        run_vppo(cfg, train_data, eval_data, args.model_path)
    elif args.experiment == "dpo":
        run_dpo(cfg, train_data, eval_data, args.model_path)
    elif args.experiment == "mult_gate":
        run_mult_gate(cfg, train_data, eval_data, args.model_path)


if __name__ == "__main__":
    main()
