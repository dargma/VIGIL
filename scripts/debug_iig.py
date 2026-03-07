"""Debug IIG computation on a single sample."""
import torch, sys
import torch.nn.functional as F
sys.path.insert(0, ".")
from src.model_registry import load_model, make_chat_prompt
from src.data_loader import load_pope
from PIL import Image

model_info = load_model("qwen3_vl_2b", dtype=torch.float16, device="auto")
pope = load_pope("adversarial", limit=5)
s = pope[0]
print("Q:", s["question"])
print("A:", s["answer"])

tok = model_info["tokenizer"]
ans_ids = tok(s["answer"], return_tensors="pt", add_special_tokens=False).input_ids
print("Answer token ids:", ans_ids)
print("Answer tokens:", [tok.decode([t]) for t in ans_ids[0]])
print("Answer id shape:", ans_ids.shape)

# Build inputs
real_inputs = make_chat_prompt(model_info, s["question"], s["image"])
black_img = Image.new("RGB", (448, 448), (0, 0, 0))
black_inputs = make_chat_prompt(model_info, s["question"], black_img)

prompt_len_r = real_inputs["input_ids"].shape[1]
prompt_len_b = black_inputs["input_ids"].shape[1]
print(f"Prompt len: real={prompt_len_r}, black={prompt_len_b}")

candidate_ids = ans_ids.to(model_info["device"])
T = candidate_ids.shape[1]
print(f"Candidate tokens: {T}")

# Check if prompt lengths differ (they might due to different image tokens)
print(f"Real input_ids[:5]: {real_inputs['input_ids'][0, :5]}")
print(f"Black input_ids[:5]: {black_inputs['input_ids'][0, :5]}")

# Concat
ids_r = torch.cat([real_inputs["input_ids"], candidate_ids], dim=1)
ids_b = torch.cat([black_inputs["input_ids"], candidate_ids], dim=1)

with torch.no_grad():
    # Must extend attention_mask to cover candidate tokens
    kw_r = {}
    kw_b = {}
    for k, v in real_inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(v.shape[0], T, dtype=v.dtype, device=v.device)
            kw_r[k] = torch.cat([v, ext], dim=1)
        else:
            kw_r[k] = v
    for k, v in black_inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(v.shape[0], T, dtype=v.dtype, device=v.device)
            kw_b[k] = torch.cat([v, ext], dim=1)
        else:
            kw_b[k] = v

    logits_r = model_info["model"](input_ids=ids_r, **kw_r).logits
    logits_b = model_info["model"](input_ids=ids_b, **kw_b).logits
    print(f"Logits: real={logits_r.shape}, black={logits_b.shape}")

    # Log-probs at position predicting candidate tokens
    lp_r = F.log_softmax(logits_r[:, prompt_len_r - 1:prompt_len_r - 1 + T, :], dim=-1)
    lp_b = F.log_softmax(logits_b[:, prompt_len_b - 1:prompt_len_b - 1 + T, :], dim=-1)

    tok_lp_r = lp_r.gather(-1, candidate_ids[:, :T].unsqueeze(-1)).squeeze(-1)
    tok_lp_b = lp_b.gather(-1, candidate_ids[:, :T].unsqueeze(-1)).squeeze(-1)

    print(f"Log-prob real:  {tok_lp_r}")
    print(f"Log-prob black: {tok_lp_b}")
    print(f"Diff: {tok_lp_r - tok_lp_b}")
    print(f"IIG: {(tok_lp_r - tok_lp_b).mean().item():.6f}")

    # Also check: what does the model predict at the last prompt position?
    top5_r = torch.topk(logits_r[0, prompt_len_r - 1, :], 5)
    top5_b = torch.topk(logits_b[0, prompt_len_b - 1, :], 5)
    print("\nTop-5 predictions (real image):")
    for i in range(5):
        print(f"  {tok.decode([top5_r.indices[i]])!r}: {top5_r.values[i].item():.2f}")
    print("Top-5 predictions (black image):")
    for i in range(5):
        print(f"  {tok.decode([top5_b.indices[i]])!r}: {top5_b.values[i].item():.2f}")

    # Try with a longer candidate (model's own generation)
    print("\n--- Testing with model's own generation ---")
    gen_r = model_info["model"].generate(**real_inputs, max_new_tokens=20, do_sample=False)
    gen_text = tok.decode(gen_r[0][prompt_len_r:], skip_special_tokens=True)
    print(f"Generated (real): {gen_text!r}")

    gen_b = model_info["model"].generate(**black_inputs, max_new_tokens=20, do_sample=False)
    gen_text_b = tok.decode(gen_b[0][prompt_len_b:], skip_special_tokens=True)
    print(f"Generated (black): {gen_text_b!r}")

    # IIG on generated text
    gen_ids = gen_r[0][prompt_len_r:].unsqueeze(0)
    T2 = gen_ids.shape[1]
    ids_r2 = torch.cat([real_inputs["input_ids"], gen_ids], dim=1)
    ids_b2 = torch.cat([black_inputs["input_ids"], gen_ids], dim=1)

    # Re-build kw with extended attention masks for T2
    kw_r2 = {}
    kw_b2 = {}
    for k, v in real_inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(v.shape[0], T2, dtype=v.dtype, device=v.device)
            kw_r2[k] = torch.cat([v, ext], dim=1)
        else:
            kw_r2[k] = v
    for k, v in black_inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(v.shape[0], T2, dtype=v.dtype, device=v.device)
            kw_b2[k] = torch.cat([v, ext], dim=1)
        else:
            kw_b2[k] = v

    logits_r2 = model_info["model"](input_ids=ids_r2, **kw_r2).logits
    logits_b2 = model_info["model"](input_ids=ids_b2, **kw_b2).logits

    lp_r2 = F.log_softmax(logits_r2[:, prompt_len_r - 1:prompt_len_r - 1 + T2, :], dim=-1)
    lp_b2 = F.log_softmax(logits_b2[:, prompt_len_b - 1:prompt_len_b - 1 + T2, :], dim=-1)
    tok_lp_r2 = lp_r2.gather(-1, gen_ids[:, :T2].unsqueeze(-1)).squeeze(-1)
    tok_lp_b2 = lp_b2.gather(-1, gen_ids[:, :T2].unsqueeze(-1)).squeeze(-1)

    per_tok_iig = (tok_lp_r2 - tok_lp_b2).squeeze(0)
    print(f"\nPer-token IIG on generated text:")
    for j in range(min(T2, 20)):
        t = tok.decode([gen_ids[0, j]])
        print(f"  [{j}] {t!r}: {per_tok_iig[j].item():.4f}")
    print(f"Mean IIG (generated): {per_tok_iig.mean().item():.6f}")
