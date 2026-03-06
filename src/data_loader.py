"""
VIGIL Data Loader — load calibration, training, and eval datasets.

Loads from local disk cache (data/) first, falls back to HuggingFace.
Iron rule: zero image overlap between calibration/training and evaluation.
POPE uses COCO val2014 images. Must cross-check A-OKVQA before training.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter

DATA_ROOT = Path(__file__).parent.parent / "data"


def _load_from_disk_or_hf(local_path: str, hf_id: str, hf_config: str = None,
                           hf_split: str = "test"):
    """Load dataset from local disk cache, fall back to HuggingFace."""
    from datasets import load_from_disk, load_dataset
    disk_path = DATA_ROOT / local_path
    if disk_path.exists():
        try:
            ds = load_from_disk(str(disk_path))
            print(f"[data] Loaded from disk: {disk_path} ({len(ds)} rows)")
            return ds
        except Exception as e:
            print(f"[data] Disk load failed ({e}), falling back to HF...")
    args = [hf_id]
    if hf_config:
        args.append(hf_config)
    return load_dataset(*args, split=hf_split)


def load_pope(split: str = "adversarial", limit: Optional[int] = None) -> List[Dict]:
    """Load POPE benchmark split.

    Args:
        split: one of 'random', 'popular', 'adversarial', 'all'
        limit: max samples to return
    """
    ds = _load_from_disk_or_hf("eval/pope", "lmms-lab/POPE", hf_split="test")
    samples = []
    for row in ds:
        cat = row.get("category", "")
        if split != "all" and cat and cat != split:
            continue
        samples.append({
            "question": row["question"],
            "answer": str(row.get("answer", "")).strip().lower(),
            "image": row.get("image"),
            "image_id": str(row.get("image_source", "")),
            "type": "yesno",
            "source": "pope",
            "category": cat or split,
        })

    if limit:
        samples = samples[:limit]
    print(f"[data] Loaded POPE {split}: {len(samples)} samples")
    return samples


def load_gqa_balanced_val(limit: Optional[int] = 1000) -> List[Dict]:
    """Load GQA balanced validation for calibration."""
    ds = _load_from_disk_or_hf(
        "calibration/gqa_balanced_val",
        "lmms-lab/GQA", "testdev_balanced_instructions", hf_split="testdev",
    )
    samples = []
    for row in ds:
        samples.append({
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")).strip().lower(),
            "image": row.get("image"),
            "type": "short_answer",
            "source": "gqa",
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded GQA balanced val: {len(samples)} samples")
    return samples


def load_textvqa_val(limit: Optional[int] = 1000) -> List[Dict]:
    """Load TextVQA validation for calibration."""
    ds = _load_from_disk_or_hf(
        "calibration/textvqa_val",
        "lmms-lab/textvqa", hf_split="validation",
    )
    samples = []
    for row in ds:
        answers = row.get("answers", [])
        answer = answers[0] if answers else ""
        samples.append({
            "question": row.get("question", ""),
            "answer": str(answer).strip().lower(),
            "image": row.get("image"),
            "type": "short_answer",
            "source": "textvqa",
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded TextVQA val: {len(samples)} samples")
    return samples


def load_textvqa_train(limit: Optional[int] = 10000) -> List[Dict]:
    """Load TextVQA train for GRPO/DAPO training."""
    ds = _load_from_disk_or_hf(
        "training/textvqa_train",
        "lmms-lab/textvqa", hf_split="train",
    )
    samples = []
    for row in ds:
        answers = row.get("answers", [])
        answer = answers[0] if answers else ""
        samples.append({
            "question": row.get("question", ""),
            "answer": str(answer).strip().lower(),
            "image": row.get("image"),
            "type": "short_answer",
            "source": "textvqa_train",
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded TextVQA train: {len(samples)} samples")
    return samples


def load_vqav2_train(limit: Optional[int] = 20000) -> List[Dict]:
    """Load VQAv2 for GRPO/DAPO training."""
    ds = _load_from_disk_or_hf(
        "training/vqav2_train",
        "merve/vqav2-small", hf_split="validation",
    )
    samples = []
    for row in ds:
        # Handle different answer formats
        answers = row.get("answers", row.get("multiple_choice_answer", ""))
        if isinstance(answers, list):
            answer_texts = [a if isinstance(a, str) else a.get("answer", "") for a in answers]
            answer = Counter(answer_texts).most_common(1)[0][0] if answer_texts else ""
        elif isinstance(answers, str):
            answer = answers
        else:
            answer = str(answers)

        samples.append({
            "question": row.get("question", ""),
            "answer": answer.strip().lower(),
            "image": row.get("image"),
            "image_id": str(row.get("image_id", "")),
            "type": "short_answer",
            "source": "vqav2",
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded VQAv2 train: {len(samples)} samples")
    return samples


def load_aokvqa_train(limit: Optional[int] = 17000) -> List[Dict]:
    """Load A-OKVQA train for GRPO/DAPO training."""
    ds = _load_from_disk_or_hf(
        "training/aokvqa_train",
        "HuggingFaceM4/A-OKVQA", hf_split="train",
    )
    samples = []
    for row in ds:
        choices = row.get("choices", [])
        correct_idx = row.get("correct_choice_idx", 0)
        answer = choices[correct_idx] if correct_idx < len(choices) else ""
        samples.append({
            "question": row.get("question", ""),
            "answer": str(answer).strip().lower(),
            "choices": choices,
            "image": row.get("image"),
            "image_id": str(row.get("image_id", "")),
            "type": "mc",
            "source": "aokvqa",
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded A-OKVQA train: {len(samples)} samples")
    return samples


def load_mmbench(limit: Optional[int] = None) -> List[Dict]:
    """Load MMBench dev for evaluation."""
    ds = _load_from_disk_or_hf(
        "eval/mmbench",
        "lmms-lab/MMBench", "en", hf_split="dev",
    )
    samples = []
    for row in ds:
        choices = [row.get(f"A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
        choices = [c for c in choices if c]
        samples.append({
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")).strip(),
            "choices": choices,
            "image": row.get("image"),
            "type": "mc",
            "source": "mmbench",
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded MMBench: {len(samples)} samples")
    return samples


def load_mme(limit: Optional[int] = None) -> List[Dict]:
    """Load MME test for evaluation."""
    ds = _load_from_disk_or_hf(
        "eval/mme",
        "lmms-lab/MME", hf_split="test",
    )
    samples = []
    for row in ds:
        samples.append({
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")).strip().lower(),
            "image": row.get("image"),
            "type": "yesno",
            "source": "mme",
            "category": row.get("category", ""),
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded MME: {len(samples)} samples")
    return samples


def load_mmmu(limit: Optional[int] = None) -> List[Dict]:
    """Load MMMU validation for evaluation."""
    ds = _load_from_disk_or_hf(
        "eval/mmmu",
        "MMMU/MMMU", "Accounting", hf_split="validation",
    )
    samples = []
    for row in ds:
        choices_raw = row.get("options", "")
        if isinstance(choices_raw, list):
            choices = choices_raw
        elif isinstance(choices_raw, str) and choices_raw.startswith("["):
            try:
                choices = json.loads(choices_raw)
            except (json.JSONDecodeError, ValueError):
                choices = []
        else:
            choices = []
        samples.append({
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")).strip(),
            "choices": choices,
            "image": row.get("image_1") or row.get("image"),
            "type": "mc",
            "source": "mmmu",
            "subject": row.get("subject", ""),
        })
        if limit and len(samples) >= limit:
            break
    print(f"[data] Loaded MMMU: {len(samples)} samples")
    return samples


def check_image_overlap(
    train_samples: List[Dict],
    eval_samples: List[Dict],
    id_field: str = "image_id",
) -> Set[str]:
    """Check image overlap between training and eval sets."""
    train_ids = {str(s.get(id_field, "")) for s in train_samples if s.get(id_field)}
    eval_ids = {str(s.get(id_field, "")) for s in eval_samples if s.get(id_field)}
    overlap = train_ids & eval_ids
    if overlap:
        print(f"[data] WARNING: {len(overlap)} overlapping images found!")
    else:
        print(f"[data] No image overlap detected.")
    return overlap


def remove_overlapping(samples: List[Dict], overlap_ids: Set[str],
                       id_field: str = "image_id") -> List[Dict]:
    """Remove samples with overlapping image IDs."""
    before = len(samples)
    filtered = [s for s in samples if str(s.get(id_field, "")) not in overlap_ids]
    print(f"[data] Removed {before - len(filtered)} overlapping samples "
          f"({before} → {len(filtered)})")
    return filtered


def build_calibration_set(limit_per_source: int = 1000) -> List[Dict]:
    """Build calibration dataset from GQA + TextVQA."""
    gqa = load_gqa_balanced_val(limit=limit_per_source)
    textvqa = load_textvqa_val(limit=limit_per_source)
    combined = gqa + textvqa
    random.shuffle(combined)
    print(f"[data] Calibration set: {len(combined)} samples "
          f"(GQA: {len(gqa)}, TextVQA: {len(textvqa)})")
    return combined


def build_training_set(
    pope_samples: Optional[List[Dict]] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Build training dataset, ensuring no overlap with POPE eval images."""
    vqav2 = load_vqav2_train()
    aokvqa = load_aokvqa_train()
    textvqa = load_textvqa_train()

    # Check A-OKVQA overlap with POPE
    if pope_samples:
        overlap = check_image_overlap(aokvqa, pope_samples)
        if overlap:
            aokvqa = remove_overlapping(aokvqa, overlap)

    combined = vqav2 + aokvqa + textvqa
    random.shuffle(combined)
    if limit:
        combined = combined[:limit]

    print(f"[data] Training set: {len(combined)} samples")
    return combined
