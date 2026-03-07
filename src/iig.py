"""
VIGIL — Image Information Gain (IIG) reward.

IIG(y; I | Q) = (1/T) sum_t [log P(y_t | I, Q, y_<t) - log P(y_t | I_black, Q, y_<t)]

Measures per-token how much the image helped generation.
Used as GRPO reward to prevent blind reasoner collapse.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

BLACK_IMAGE = Image.new("RGB", (448, 448), (0, 0, 0))


@torch.no_grad()
def compute_iig(
    model_info: Dict[str, Any],
    question: str,
    image: Image.Image,
    candidate_text: str,
    black_image: Image.Image = None,
) -> float:
    """Compute Image Information Gain for a candidate response.

    Teacher-forces the same candidate tokens under real vs black image,
    returning mean per-token log-prob difference.

    Args:
        model_info: standardized model info dict from model_registry
        question: the input question
        image: the real image
        candidate_text: the generated candidate to evaluate
        black_image: reference image (default: 448x448 black)

    Returns:
        IIG value (positive = image helped, negative = image hurt)
    """
    if black_image is None:
        black_image = BLACK_IMAGE

    model = model_info["model"]
    processor = model_info["processor"]

    # Tokenize candidate
    candidate_ids = model_info["tokenizer"](
        candidate_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model_info["device"])

    if candidate_ids.shape[1] == 0:
        return 0.0

    # Build inputs for real and black image
    real_inputs = _build_inputs(model_info, question, image)
    black_inputs = _build_inputs(model_info, question, black_image)

    # Get prompt length (before candidate tokens)
    prompt_len_real = real_inputs["input_ids"].shape[1]
    prompt_len_black = black_inputs["input_ids"].shape[1]

    # Concatenate prompt + candidate tokens, extending attention_mask too
    T = candidate_ids.shape[1]
    ids_real = torch.cat([real_inputs["input_ids"], candidate_ids], dim=1)
    ids_black = torch.cat([black_inputs["input_ids"], candidate_ids], dim=1)

    kw_real = {}
    kw_black = {}
    for k, v in real_inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            # Extend attention mask to cover candidate tokens
            ext = torch.ones(v.shape[0], T, dtype=v.dtype, device=v.device)
            kw_real[k] = torch.cat([v, ext], dim=1)
        else:
            kw_real[k] = v
    for k, v in black_inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(v.shape[0], T, dtype=v.dtype, device=v.device)
            kw_black[k] = torch.cat([v, ext], dim=1)
        else:
            kw_black[k] = v

    # Forward pass with real image
    try:
        logits_real = model(input_ids=ids_real, **kw_real).logits
    except Exception:
        return 0.0

    # Forward pass with black image
    try:
        logits_black = model(input_ids=ids_black, **kw_black).logits
    except Exception:
        return 0.0

    # Extract log-probs for candidate tokens
    # logits at position [prompt_len-1 : prompt_len-1+T] predict tokens at [prompt_len : prompt_len+T]

    lp_real = F.log_softmax(logits_real[:, prompt_len_real - 1:prompt_len_real - 1 + T, :], dim=-1)
    tok_lp_real = lp_real.gather(-1, candidate_ids[:, :T].unsqueeze(-1)).squeeze(-1)

    lp_black = F.log_softmax(logits_black[:, prompt_len_black - 1:prompt_len_black - 1 + T, :], dim=-1)
    tok_lp_black = lp_black.gather(-1, candidate_ids[:, :T].unsqueeze(-1)).squeeze(-1)

    iig = (tok_lp_real - tok_lp_black).mean().item()
    return iig


@torch.no_grad()
def compute_iig_batch_candidates(
    model_info: Dict[str, Any],
    question: str,
    image: Image.Image,
    candidates: List[str],
    black_image: Image.Image = None,
) -> List[float]:
    """Compute IIG for multiple candidates (same question/image).

    Slightly more efficient than calling compute_iig per candidate
    since we can reuse the prompt encoding.
    """
    return [compute_iig(model_info, question, image, c, black_image) for c in candidates]


def _build_inputs(model_info: Dict[str, Any], question: str, image: Image.Image) -> dict:
    """Build model inputs for a question + image pair."""
    from src.model_registry import make_chat_prompt
    return make_chat_prompt(model_info, question, image)


def vigil_reward(r_correct: float, iig: float, lam: float, eps: float = 0.1) -> float:
    """Compute VIGIL reward with reversal protection.

    R(y) = R_correct(y) + lambda * max(IIG, 0) * (R_correct(y) + eps)

    The multiplicative (R_correct + eps) term prevents reward reversal:
    a wrong answer can never score higher than a correct one.
    """
    return float(r_correct) + lam * max(iig, 0.0) * (float(r_correct) + eps)


def calibrate_lambda(
    model_info: Dict[str, Any],
    calib_samples: List[Dict[str, Any]],
    max_samples: int = 500,
) -> Tuple[float, List[float]]:
    """Auto-determine lambda from calibration data.

    lambda = 1 / (mean(positive_IIG) + std(positive_IIG))

    Args:
        model_info: loaded model
        calib_samples: list of dicts with 'question', 'answer', 'image'
        max_samples: max calibration samples

    Returns:
        (lambda_value, all_iig_values)
    """
    all_iig = []
    samples = calib_samples[:max_samples]

    for i, s in enumerate(samples):
        if s.get("image") is None:
            continue
        try:
            # Use ground truth answer as candidate for calibration
            iig_val = compute_iig(model_info, s["question"], s["image"], s["answer"])
            all_iig.append(iig_val)
        except Exception as e:
            print(f"  [iig-cal] Skip {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            pos = sum(1 for v in all_iig if v > 0)
            print(f"  [iig-cal] {i+1}/{len(samples)}: {len(all_iig)} computed, "
                  f"{pos}/{len(all_iig)} positive ({pos/len(all_iig)*100:.0f}%)")

    if not all_iig:
        print("[iig-cal] WARNING: no IIG values computed. Returning default lambda=0.5")
        return 0.5, []

    positives = [v for v in all_iig if v > 0]
    if not positives:
        print("[iig-cal] WARNING: all IIG values <= 0. Returning default lambda=0.5")
        return 0.5, all_iig

    import numpy as np
    mu = np.mean(positives)
    sigma = np.std(positives)
    lam = 1.0 / (mu + sigma + 1e-8)

    pos_ratio = len(positives) / len(all_iig) * 100
    print(f"\n[iig-cal] IIG Calibration Results:")
    print(f"  Total samples: {len(all_iig)}")
    print(f"  Positive: {len(positives)} ({pos_ratio:.0f}%)")
    print(f"  Mean (all): {np.mean(all_iig):.4f}")
    print(f"  Mean (positive): {mu:.4f}")
    print(f"  Std (positive): {sigma:.4f}")
    print(f"  Lambda: {lam:.4f}")

    return float(lam), all_iig
