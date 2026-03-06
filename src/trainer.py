"""
VIGIL Trainer — GRPO/DAPO training with visually-grounded reward.

R_total = w1*R_correct + w2*(α*R_vhad + (1-α)*R_asi) + w3*R_fluency

Two modes:
- Full reward: extra forward+generation per sample (~25% overhead)
- Lightweight: in-situ activation collection (zero overhead)
"""

import os
import json
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .rewards import (
    compute_r_vhad, compute_r_asi, compute_r_fluency,
    compute_r_correct, compute_composite_reward,
    InSituVisionReward, make_black_image,
    VisionHeadActivationCollector,
)
from .calibrator import CalibrationResult


def load_training_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "training.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class VIGILRewardFunction:
    """Reward function factory for TRL GRPOTrainer.

    Creates a reward_fn compatible with TRL's signature:
        reward_fn(prompts, completions, **kwargs) -> List[float]
    """

    def __init__(
        self,
        model_info: Dict[str, Any],
        calibration: CalibrationResult,
        mode: str = "full",  # "full" or "lightweight"
        w_correct: float = 0.3,
        w_visual: float = 0.5,
        w_fluency: float = 0.2,
        alpha_vhad: float = 0.6,
        max_tokens: int = 512,
    ):
        self.model_info = model_info
        self.calibration = calibration
        self.mode = mode
        self.w_correct = w_correct
        self.w_visual = w_visual
        self.w_fluency = w_fluency
        self.alpha_vhad = alpha_vhad
        self.max_tokens = max_tokens
        self.vision_heads = calibration.top_heads

        if mode == "lightweight":
            self.insitu = InSituVisionReward(model_info, self.vision_heads)
            self.insitu.install()
        else:
            self.insitu = None

        self._reward_log: List[Dict] = []

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> List[float]:
        """Compute VIGIL composite reward for each completion."""
        rewards = []
        batch_size = len(completions)

        for i in range(batch_size):
            completion = completions[i]
            gt = ground_truths[i] if ground_truths else ""
            q_type = question_types[i] if question_types else "yesno"
            image = images[i] if images else None

            # R_correct
            r_correct = compute_r_correct(completion, gt, q_type) if gt else 0.0

            # R_fluency
            r_fluency = compute_r_fluency(completion, self.max_tokens)

            # R_visual_grounding
            if self.mode == "lightweight" and self.insitu:
                r_vhad = self.insitu.compute()
                r_asi = 0.5  # not available in lightweight mode
                self.insitu.clear()
            else:
                r_vhad = 0.0
                r_asi = 0.5

            # Composite
            r_total = compute_composite_reward(
                r_correct, r_vhad, r_asi, r_fluency,
                self.w_correct, self.w_visual, self.w_fluency, self.alpha_vhad,
            )
            rewards.append(r_total)

            self._reward_log.append({
                "r_total": r_total,
                "r_correct": r_correct,
                "r_vhad": r_vhad,
                "r_asi": r_asi,
                "r_fluency": r_fluency,
            })

        return rewards

    def cleanup(self):
        if self.insitu:
            self.insitu.remove()

    def get_reward_stats(self) -> Dict[str, float]:
        if not self._reward_log:
            return {}
        keys = ["r_total", "r_correct", "r_vhad", "r_asi", "r_fluency"]
        return {k: float(np.mean([r[k] for r in self._reward_log])) for k in keys}


def setup_grpo_training(
    model_info: Dict[str, Any],
    calibration: CalibrationResult,
    train_samples: List[Dict],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Set up GRPO training with TRL.

    Returns dict with trainer, reward_fn, dataset ready for trainer.train().
    """
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset

    if config is None:
        config = load_training_config()

    grpo_cfg = config["grpo"]
    reward_cfg = config["reward"]

    # Build reward function
    reward_fn = VIGILRewardFunction(
        model_info=model_info,
        calibration=calibration,
        mode="lightweight",  # default to lightweight for training
        w_correct=reward_cfg["w_correct"],
        w_visual=reward_cfg["w_visual_grounding"],
        w_fluency=reward_cfg["w_fluency"],
        alpha_vhad=reward_cfg["alpha_vhad"],
        max_tokens=grpo_cfg["max_new_tokens"],
    )

    # Build dataset
    ds_records = []
    for s in train_samples:
        ds_records.append({
            "prompt": s["question"],
            "ground_truth": s.get("answer", ""),
            "question_type": s.get("type", "yesno"),
        })
    dataset = Dataset.from_list(ds_records)

    # TRL config
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    trl_config = GRPOConfig(
        output_dir=f"checkpoints/grpo_{model_info['model_key']}",
        learning_rate=grpo_cfg["learning_rate"],
        per_device_train_batch_size=grpo_cfg["batch_size"],
        gradient_accumulation_steps=grpo_cfg["grad_accum"],
        num_train_epochs=1,
        max_steps=grpo_cfg["num_steps"],
        warmup_ratio=grpo_cfg["warmup_ratio"],
        logging_steps=1,
        save_steps=grpo_cfg["save_every"],
        seed=grpo_cfg["seed"],
        bf16=True,
        # GRPO specific
        num_generations=grpo_cfg["group_size"],
        max_new_tokens=grpo_cfg["max_new_tokens"],
        temperature=grpo_cfg["temperature"],
        beta=grpo_cfg.get("beta", 0.01),
    )

    # Handle InternVL beta=0.0 (no ref model)
    spec = model_info.get("spec")
    if spec and hasattr(spec, "trl_beta"):
        trl_config.beta = spec.trl_beta

    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    return {
        "trainer": trainer,
        "reward_fn": reward_fn,
        "dataset": dataset,
        "config": trl_config,
    }


def setup_dapo_training(
    model_info: Dict[str, Any],
    calibration: CalibrationResult,
    train_samples: List[Dict],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Set up DAPO training (TRL GRPOTrainer with loss_type='dapo')."""
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset

    if config is None:
        config = load_training_config()

    dapo_cfg = config["dapo"]
    reward_cfg = config["reward"]

    reward_fn = VIGILRewardFunction(
        model_info=model_info,
        calibration=calibration,
        mode="lightweight",
        w_correct=reward_cfg["w_correct"],
        w_visual=reward_cfg["w_visual_grounding"],
        w_fluency=reward_cfg["w_fluency"],
        alpha_vhad=reward_cfg["alpha_vhad"],
        max_tokens=dapo_cfg["max_new_tokens"],
    )

    ds_records = [{"prompt": s["question"], "ground_truth": s.get("answer", ""),
                   "question_type": s.get("type", "yesno")} for s in train_samples]
    dataset = Dataset.from_list(ds_records)

    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    trl_config = GRPOConfig(
        output_dir=f"checkpoints/dapo_{model_info['model_key']}",
        learning_rate=dapo_cfg["learning_rate"],
        per_device_train_batch_size=dapo_cfg["batch_size"],
        gradient_accumulation_steps=dapo_cfg["grad_accum"],
        num_train_epochs=1,
        max_steps=dapo_cfg["num_steps"],
        warmup_ratio=dapo_cfg["warmup_ratio"],
        logging_steps=1,
        save_steps=dapo_cfg["save_every"],
        seed=dapo_cfg["seed"],
        bf16=True,
        num_generations=dapo_cfg["group_size"],
        max_new_tokens=dapo_cfg["max_new_tokens"],
        temperature=dapo_cfg["temperature"],
        beta=dapo_cfg.get("beta", 0.0),  # DAPO: no KL
        loss_type="dapo",  # TRL native DAPO
    )

    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    return {
        "trainer": trainer,
        "reward_fn": reward_fn,
        "dataset": dataset,
        "config": trl_config,
    }
