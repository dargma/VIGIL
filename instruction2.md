

VIGIL — Lab Instruction: Steering Validation → IIG GRPO Training

> 이 파일 하나로 VIGIL 전체 실험을 수행한다. 외부 instruction 참조 없음.
> Calibration → Steering 검증 → IIG GRPO → Post-Training Analysis 까지 일관된 파이프라인.
> Philosophy: mission + theory + direction. You read the codebase, you figure out execution.

---

## Scope

**One model, two modes.**

- **Model**: `Qwen/Qwen3-VL-2B-Instruct` (short answer) and `Qwen/Qwen3-VL-2B-Thinking` (reasoning)
- All other models (InternVL, DeepSeek, 7B) are deferred to a future instruction. Do not load them.

**Architecture note**: Qwen3-VL-2B uses GQA (16 Q heads, 8 KV heads, head_dim=128, 28 layers, hidden=2048). Profiling and steering operate on **Q head output** — reshape o_proj input from `[batch, seq, 2048]` to `[batch, seq, 16, 128]` to isolate individual heads. DeepStack injects ViT features into LLM layers 1–3 → steer/measure layer 4+ only.

---

## Mission

**IIG(Image Information Gain)를 GRPO reward로 사용하여, 모델이 이미지를 보는 습관을 영구적으로 학습하게 한다.**

하지만 IIG GRPO를 바로 돌리면 안 된다. 먼저 **steering으로 "activation ↑ → accuracy ↑" 인과관계를 증명**해야 한다. 이 인과관계가 없으면 IIG reward는 무의미한 signal이다.

파이프라인:
```
Step 1: Calibration — vision head를 찾는다
Step 2: Steering — vision head를 강제로 키우면 성능이 오르는가? (인과 증명)
Step 3: IIG GRPO — 이 인과관계를 모델이 스스로 학습하게 한다 (영구화)
Step 4: 분석 — RL이 steering보다 나은 전략을 발견했는가?
```

**Steering Δ는 GRPO의 상한이 아니다.** Steering은 고정 모델에 고정 벡터를 더하는 국소 개입이지만, GRPO는 모델 weight 자체를 바꾸는 전역 최적화다. GRPO가 steering을 넘는 경로:
1. **새 회로 생성**: calibration에서 발견 안 된 head가 새로 vision head로 활성화
2. **생성 전략 변경**: "이미지를 참조하며 추론하는" 새로운 reasoning pattern 발견
3. **ViT-LLM connector 개선**: LoRA target에 포함 시 visual feature 전달 품질 향상
4. **전역 협력**: FFN, LayerNorm, non-vision head 등이 함께 최적화

목표 결과:
1. POPE accuracy가 baseline보다 올라간다 (R_correct 효과)
2. Blind Test Gap이 baseline보다 **커진다** (IIG 효과 — blind reasoner 방지)
3. Steering OFF 상태에서도 baseline보다 높다 (영구적 개선)
4. MME Cognition이 baseline 대비 -5% 이내다 (language 능력 보존)
5. GRPO Δ ≥ Steering Δ (RL이 수동 개입 이상의 전략을 발견)

2번이 핵심. Gap이 줄어들면 실패.

---

## Background: 왜 이것을 하는가

Small VLMs (1–3B)은 생성이 진행될수록 이미지를 무시한다 (Visual Attention Drift). GRPO로 R_correct만 최적화하면 **blind reasoner**가 된다 — 이미지 없이도 정답을 맞추는 shortcut을 학습 (DVRP, 2026). IIG는 이미지가 답변에 제공한 정보량을 직접 측정하여, "이미지를 보고 맞춘 정답"에 더 높은 reward를 부여한다.

---

## The Reward — IIG (Image Information Gain)

### 한 문장

**각 토큰을 생성할 때, 이미지가 얼마나 도움이 되었는가.**

### 수식

```
IIG(y; I | Q) = (1/T) Σ_t [log P(y_t | I, Q, y_<t) - log P(y_t | I_black, Q, y_<t)]
```

### Total Reward

```
R(y) = R_correct(y) + λ × max(IIG, 0) × (R_correct(y) + ε)
```

- λ: calibration 데이터에서 자동 결정 (hyperparameter-free)
- ε = 0.1 (고정)
- max(·, 0): 이미지가 방해되는 edge case 제거
- (R_correct + ε) 항: 보상 역전 방지

### 설계 근거

| 설계 결정 | 근거 |
|----------|------|
| Head activation이 아닌 log-prob | Head activation은 group 내 variance ≈ 0 → GRPO advantage에서 소멸 |
| Candidate-specific (teacher-forced) | 같은 Q+I에서도 "Yes" vs "No"가 다른 IIG → advantage 생존 |
| 곱셈형 (R_correct + ε) | 덧셈형에서 보상 역전 발생. 곱셈형은 불가능 |
| λ auto-calibration | IIG 스케일이 모델/데이터마다 다름 |
| Black image reference | "이미지 없음"은 VLM에서 OOD. Black이 in-distribution 최소 정보 |
| R_fluency 제거 | GRPO의 β (KL penalty)가 대체 |
| R_asi 제거 | IIG가 per-token answer sensitivity를 자연 포함 |

### 한계

1. Binary VQA에서 IIG within-group variance가 작다. Thinking mode에서 효과 극대.
2. Black image ≠ "이미지 없음". Relative information gain 근사치.
3. OCR/Chart 같은 태스크에서 IIG는 중립. Feature이지 bug가 아니다.

---

## Experiment Blocks

### Block 0A — Head Profiling / Calibration (GPU 15분)

모델 로드. 모든 o_proj에 hook (28 layers × 16 Q heads = 448 heads). GQA balanced val 500 samples.

| Data | Size | 용도 |
|------|------|------|
| GQA balanced val | 500 | Calibration + λ auto-cal |
| POPE Adversarial | 500 (block), 3000 (final) | Main eval |
| POPE Random | 100 | Sanity check |

추출:
- Activation difference heatmap (real − black)
- Vision attention ratio per head
- Cohen's d per head
- **IIG λ calibration**: 같은 500 samples에서 IIG 계산, λ auto-determine
- **Blind Test Gap baseline**: acc(real) - acc(black)

출력: top-K heads, steering vectors, λ 값, IIG 분포 히스토그램

### Block 0B — Steering Validation (GPU 20분)

**IIG GRPO 진행 여부를 결정하는 gate.**

Top-K heads (K=5, α=2.0)로:
- Baseline vs steered on POPE-A 500 (Instruct)
- Baseline vs steered on POPE-A 500 (Thinking)
- Per-sample helped/hurt/neutral
- Blind Test Gap (steered vs unsteered)
- Vision drift curve (Thinking): 토큰 위치별 activation — 논문 figure 후보

**Δ ≥ 0.5 → PROCEED. Δ < 0.5 → DEBUG.**

### Block 1 — Minimal IIG GRPO (50 steps, GPU 30분)

Signal 존재 확인:
- Qwen3-VL-2B-Instruct, VQAv2 2000, 50 steps
- group_size=4, temp=1.0, max_completion=64, LoRA r=16
- (A) R_correct only vs (B) R_correct + IIG
- 매 10 steps: POPE-A 200 + Blind Test Gap

Go/No-Go: IIG 변동 있고 Gap이 (A) ≤ (B)이면 PROCEED.

### Block 2 — Full GRPO (200 steps, GPU 2시간)

- VQAv2 5000 + A-OKVQA 3000, LoRA r=64, group_size=8
- 3설정: (A) R_correct, (B) IIG auto-λ, (C) IIG 2×λ
- 매 50 steps: POPE-A 500 + MME + MMBench 200 + OCRBench 100

성공: (B) Gap > (A) Gap AND (B) POPE ≥ (A) POPE.

### Block 3 — Thinking Mode (200 steps, GPU 2시간)

- Qwen3-VL-2B-Thinking, group_size=4, max_completion=256
- Vision drift curve: baseline / steered / IIG-trained

### Block 4 — DAPO Variant (100 steps, GPU 1시간)

beta=0.0, loss_type="dapo". GRPO+IIG vs DAPO+IIG.

### Block 5 — Post-Training Analysis (GPU 30분)

1. **Blind Test 최종**: POPE 3000 × {real, black}
2. **Steering OFF**: IIG-trained, steering 없이 eval (영구성)
3. **Steering ON**: IIG-trained + steering (orthogonality)
4. **Head 분석**: calibration 재적용. 학습 전/후 top-K overlap. 신규 head 탐색.
5. **GRPO Δ vs Steering Δ**: Block 0B 결과와 직접 대조

핵심 table:
```
| Method              | POPE-A | POPE Δ | Gap  | 영구? | 해석 |
|---------------------|--------|--------|------|-------|------|
| Baseline            |   ?    |  —     |  ?   | —     | —    |
| Steering (Block 0B) |   ?    | +?.?   |  ?   | ✗     | 국소 개입 |
| (A) R_correct GRPO  |   ?    | +?.?   |  ?   | ✓     | blind reasoner 위험 |
| (B) IIG GRPO        |   ?    | +?.?   |  ?   | ✓     | vs Steering Δ? |
| (B) + Steering      |   ?    | +?.?   |  ?   | ✓     | orthogonal? |
```

GRPO Δ > Steering Δ이면: 신규 head 분석 + 생성 패턴 비교 추가.

### Block 6 — Ablation (선택적)

λ sweep, ε sweep, black image 유형, IIG variant 비교.

---

## Data Pipeline

| Dataset | Size | Source | POPE Overlap |
|---------|------|--------|-------------|
| VQAv2 train | 5000 | COCO train2014 | ❌ |
| A-OKVQA train | 3000 | COCO train2017 | ⚠️ cross-check |

```python
pope_image_ids = load_pope_image_ids()
aokvqa_clean = [s for s in aokvqa_train if s['image_id'] not in pope_image_ids]
```

---

## GRPO Config

```python
config_instruct = {
    "num_generations": 8, "temperature": 1.2, "beta": 0.01,
    "max_completion_length": 128, "learning_rate": 5e-6,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "lora_r": 64, "lora_alpha": 128,
    "lora_target_modules": ["q_proj", "v_proj", "o_proj", "k_proj"],
    "remove_unused_columns": False, "logging_steps": 1, "save_steps": 50,
}
config_thinking = {
    **config_instruct,
    "num_generations": 4, "temperature": 1.0, "max_completion_length": 256,
}
```

---

## IIG 구현

```python
import torch, torch.nn.functional as F
from PIL import Image

BLACK_IMAGE = Image.new('RGB', (448, 448), (0, 0, 0))

def compute_iig(model, processor, image, question, candidate_tokens):
    with torch.no_grad():
        real = processor(images=image, text=question, return_tensors="pt").to(model.device)
        n = real.input_ids.shape[1]
        ids_r = torch.cat([real.input_ids, candidate_tokens.unsqueeze(0)], dim=1)
        kw = {k: v for k, v in real.items() if k != 'input_ids'}
        lp_r = F.log_softmax(model(input_ids=ids_r, **kw).logits[:, n-1:-1], dim=-1)
        tok_lp_r = lp_r.gather(-1, candidate_tokens[None, :, None]).squeeze(-1)

        blk = processor(images=BLACK_IMAGE, text=question, return_tensors="pt").to(model.device)
        ids_b = torch.cat([blk.input_ids, candidate_tokens.unsqueeze(0)], dim=1)
        kw_b = {k: v for k, v in blk.items() if k != 'input_ids'}
        lp_b = F.log_softmax(model(input_ids=ids_b, **kw_b).logits[:, n-1:-1], dim=-1)
        tok_lp_b = lp_b.gather(-1, candidate_tokens[None, :, None]).squeeze(-1)

        iig = (tok_lp_r - tok_lp_b).mean().item()
    return iig

def vigil_reward(r_correct, iig, lam, eps=0.1):
    return float(r_correct) + lam * max(iig, 0.0) * (float(r_correct) + eps)

def calibrate_lambda(model, processor, calib_data):
    positives = [compute_iig(model, processor, img, q, t) for img, q, t in calib_data]
    positives = [v for v in positives if v > 0]
    if not positives: return 0.5
    mu, sigma = sum(positives)/len(positives), (sum((x-sum(positives)/len(positives))**2 for x in positives)/len(positives))**0.5
    return 1.0 / (mu + sigma + 1e-8)
```

---

## Adversarial Design Loop

모든 reward/실험 설계 변경에 대해 수행. **파일을 분리하라.**

```
Step 1: PROPOSE → lab/design/{name}_v{N}_propose.md
Step 2: ATTACK → lab/design/{name}_v{N}_attack.md (최소 5개, 숫자 반례 필수)
  체크리스트: gaming? group advantage 생존? 스케일? 역전? 태스크별? novelty? overhead? 수학?
Step 3: REBUT → lab/design/{name}_v{N}_rebuttal.md
Step 4: VERIFY → lab/design/{name}_v{N+1}_verify.md (PASS / NEEDS ANOTHER ROUND)
최대 4 라운드.
```

2-Agent 강화 (선택적): Anthropic API로 별도 instance에 공격 위임.

---

## Ralph Loop Phases

### Ralph 0 — Calibration + Steering + λ
```
/ralph-loop "
Qwen3-VL-2B-Instruct 로드. 

A) Head Profiling: o_proj hook, GQA 500, real/black forward, heatmap+Cohen's d.
B) IIG λ: 같은 500 samples에서 IIG 계산, λ auto-determine.
C) Steering: top-5, α=2.0, POPE-A 500 (Instruct+Thinking), drift curve.

lab/reports/calibration_steering/에 저장. Git commit.
Vision heads 존재 + Steering Δ≥0.5 + IIG양수≥60% → <promise>VALIDATION_DONE</promise>
" --max-iterations 15 --completion-promise "VALIDATION_DONE"
```

### Ralph 1 — Minimal GRPO
```
/ralph-loop "
calibration 결과 읽기. Block 1: 50 steps, (A) R_correct vs (B) IIG.
매 10 steps POPE-A 200 + Gap. <promise>MINIMAL_GRPO_DONE</promise>
" --max-iterations 15 --completion-promise "MINIMAL_GRPO_DONE"
```

### Ralph 2 — Full GRPO
```
/ralph-loop "
Block 2: 200 steps, 3설정, full eval.
<promise>FULL_GRPO_DONE</promise>
" --max-iterations 12 --completion-promise "FULL_GRPO_DONE"
```

### Ralph 3 — Thinking Mode
```
/ralph-loop "
Block 3: Thinking, drift curve (baseline/steered/IIG-trained).
<promise>THINKING_DONE</promise>
" --max-iterations 12 --completion-promise "THINKING_DONE"
```

### Ralph 4 — Post-Training Analysis
```
/ralph-loop "
Block 5: Blind Test 3000, Steering OFF/ON, Head 분석, GRPO vs Steering Δ.
RESEARCH_JOURNAL.md 기록. <promise>ANALYSIS_DONE</promise>
" --max-iterations 8 --completion-promise "ANALYSIS_DONE"
```

---

## Decision Tree

```
Block 0A: Vision heads? No → STOP. Yes → continue.
Block 0A: IIG 양수 < 60%? → STOP. ≥ 60% → continue.
Block 0B: Steering Δ < 0.5? → DEBUG. ≥ 0.5 → PROCEED.

Block 1: Gap (B) < (A)? → STOP. ≈ → λ 조정. ≥ → PROCEED.
Block 2: Gap↑ + POPE↑ → ★. Gap↓ → FAIL. MME-C↓↓ → λ↓.
Block 3: Drift 개선 → ★ 핵심. 없음 → Instruct만으로.

Block 5:
├── Steering OFF > baseline → ★ 영구적.
├── Head overlap > 70% → ★ Mechanistic.
├── 신규 head → ★★ RL이 새 pathway 발견.
├── GRPO Δ > Steering Δ → ★★ RL > manual. 핵심 결과.
├── GRPO Δ ≈ Steering Δ → RL = steering + 영구성.
└── 결합 > 각각 → ★ Orthogonal.
```

---

## 논문 서사

```
문제: Small VLM → Visual Attention Drift
  ↓
진단: GRPO R_correct → Blind Reasoner (DVRP)
  ↓
발견: Steering (vision head 강제 증폭) → 성능 ↑, 하지만 일시적
  ↓
해결: IIG reward GRPO → 영구적 + Blind Test Gap ↑
  → GRPO Δ ≥ Steering Δ: RL이 수동 개입보다 나은 전략 자력 발견
  ↓
Contributions:
  C1. IIG — 정보 이론 기반 visual grounding GRPO reward
  C2. Head-level conditional steering
  C3. Blind Test Gap 프레임워크
  C4. Thinking mode vision drift 분석
  C5. Steering-Training orthogonality + RL > steering 메커니즘
```

---

## Iteration Protocol

```
Experiment → Measure (2m) → Compare (3m) → Diagnose (5m) → Decide (2m) → Execute
```

Anti-Patterns: research-before-run, over-analysis, TODO.md-instead-of-code.

---

## What This Instruction Does NOT Specify

- Hook implementation details (read the codebase)
- TRL API specifics (read TRL docs)
- Dataset loading (HuggingFace standard)
- Visualization (matplotlib, readable)

**Next**: Block 2 양성이면 InternVL3.5-1B, DeepSeek-VL2-Tiny 확장 instruction 별도 작성.
