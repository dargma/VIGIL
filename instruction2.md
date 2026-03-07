

VIGIL — Lab Instruction Phase 2: IIG GRPO Training

> Phase 1 (INSTRUCTION.md)에서 steering으로 "activation ↑ → accuracy ↑" 인과관계를 증명했다.
> Phase 2는 그 인과관계를 **모델이 스스로 학습하도록** GRPO reward에 녹인다.
> Steering이 일시적 개입이라면, IIG GRPO는 영구적 체질 변화다.

---

## Phase 1 → Phase 2 연결

Phase 1이 제공한 것:
- Calibration 결과: top-K vision heads, Cohen's d, steering vectors
- Steering 효과: POPE Δ, Blind Test Gap 변화, vision drift curve
- 최적 hyperparameter: K, α, layer range

Phase 2가 이것을 쓰는 방법:
- Calibration 데이터 → **λ auto-calibration** (IIG 분포 측정)
- Steering 결과 → **실험 기대치 설정** (steering Δ가 GRPO Δ의 상한)
- Vision drift curve → **thinking mode에서 IIG 효과 검증** 비교 대상

Phase 1이 음성이었다면(steering이 안 먹혔다면) 이 instruction을 실행하지 마라.

---

## Mission

**IIG(Image Information Gain)를 GRPO reward로 사용하여, 모델이 이미지를 보는 습관을 영구적으로 학습하게 한다.**

목표 결과:
1. POPE accuracy가 baseline보다 올라간다 (R_correct 효과)
2. Blind Test Gap이 baseline보다 **커진다** (IIG 효과 — blind reasoner 방지)
3. Steering OFF 상태에서도 baseline보다 높다 (영구적 개선)
4. MME Cognition이 baseline 대비 -5% 이내다 (language 능력 보존)

2번이 핵심이다. Gap이 줄어들면 실패.

---

## The Reward — IIG (Image Information Gain)

### 한 문장

**각 토큰을 생성할 때, 이미지가 얼마나 도움이 되었는가.**

### 수식

```
IIG(y; I | Q) = (1/T) Σ_t [log P(y_t | I, Q, y_<t) - log P(y_t | I_black, Q, y_<t)]
```

- y = 생성된 candidate (teacher-forced)
- T = 생성된 토큰 수
- I_black = 검정 이미지 (visual content 없는 reference condition)
- 두 조건에서 **동일한 토큰 시퀀스를 강제**하므로, 순수 이미지 효과만 측정

### Total Reward

```
R(y) = R_correct(y) + λ × max(IIG, 0) × (R_correct(y) + ε)
```

- λ: calibration 데이터에서 자동 결정 (hyperparameter-free)
- ε = 0.1 (고정): 오답에서도 IIG의 10%가 gradient signal로 남음
- max(·, 0): 이미지가 방해되는 edge case 제거
- (R_correct + ε) 항: 보상 역전 방지 — 오답이 정답보다 높은 reward를 받는 것 불가능

### λ 자동 결정

```python
def calibrate_lambda(model, processor, calib_data, black_image):
    iig_values = [compute_iig(sample) for sample in calib_data if compute_iig(sample) > 0]
    return 1.0 / (mean(iig_values) + std(iig_values) + 1e-8)
```

Phase 1의 calibration 데이터(GQA 500)를 그대로 사용. 추가 데이터 불필요.

### 왜 이것인가

| 설계 결정 | 근거 |
|----------|------|
| Head activation이 아닌 log-prob | Head activation은 group 내 variance ≈ 0 → GRPO advantage에서 소멸 |
| Candidate-specific (teacher-forced) | 같은 Q+I에서도 "Yes" vs "No" candidate가 다른 IIG → advantage 생존 |
| 곱셈형 (R_correct + ε) | 덧셈형에서 오답+고IIG > 정답+저IIG 역전 발생. 곱셈형은 불가능 |
| λ auto-calibration | IIG 스케일이 모델/데이터마다 다름. 수동 λ는 brittle |
| Black image reference | "이미지 없음" 조건은 VLM에서 정의 불가 (OOD). Black이 in-distribution 최소 정보 |
| max(IIG, 0) | IIG<0 = 이미지가 방해. R_correct가 이미 벌칙하므로 이중 벌칙 불필요 |
| R_fluency 제거 | GRPO의 β (KL penalty)가 이미 fluency 붕괴 방지 |
| R_asi 제거 | IIG가 per-token 수준에서 answer sensitivity를 자연 포함 |

### 한계 — 이것을 알고 시작하라

1. **Binary VQA ("Yes"/"No")에서 IIG의 within-group variance가 작다.** 같은 "Yes" 토큰이면 IIG가 동일. Thinking mode의 긴 reasoning chain에서 효과가 가장 크다.
2. **Black image ≠ "이미지 없음".** P(y|I_black, Q) ≠ P(y|Q). 정확한 PMI가 아닌 relative information gain의 근사치.
3. **OCR/Chart 같은 이미 이미지를 잘 보는 태스크에서는 IIG가 중립.** Group 내 모든 candidate의 IIG가 높으므로 advantage에서 상쇄. 이것은 feature이지 bug가 아니다.

---

## Experiment Blocks

### Block 0 — λ Calibration (GPU 5분)

Phase 1의 GQA calibration 데이터 재사용. 각 sample에 대해 IIG를 계산하고 λ를 결정.

출력:
- IIG 분포 히스토그램 (음수/양수 비율, 평균, std)
- 결정된 λ 값
- IIG vs Cohen's d 산점도 (Phase 1의 head-level 결과와 IIG의 상관관계 확인)

이 블록이 **IIG가 실제로 non-zero signal을 가지는지** 최종 확인한다.
IIG 양수 비율이 < 60%이면 stop. Phase 1 steering이 성공했더라도 IIG 수식에 문제가 있다는 뜻.

### Block 1 — Minimal GRPO (50 steps, GPU 30분)

**IIG가 학습에 실제로 기여하는지 빠르게 확인.** 성능 향상이 아닌 signal 존재 확인.

Training:
- 모델: Qwen3-VL-2B-Instruct (Thinking은 Block 3)
- 데이터: VQAv2 train subset 2000 samples
- GRPO config: group_size=4, temp=1.0, max_completion=64, beta=0.01
- LoRA: r=16, alpha=32 (빠른 실험용 — 나중에 키움)
- Reward: R_correct + λ × max(IIG, 0) × (R_correct + 0.1)
- Steps: 50
- 비교군: R_correct only (동일 설정, IIG 없이)

Eval (매 10 steps):
- POPE Adversarial 200
- Blind Test Gap (POPE 200, black image)

관찰할 것:
- [ ] IIG가 step마다 변하는가? (flat이면 gradient 미전달 — 실패)
- [ ] R_correct only 대비 Blind Test Gap이 다른가? (핵심 metric)
- [ ] R_correct가 급락하지 않는가? (IIG가 정확도를 해치는 징후)
- [ ] Loss curve가 정상적인가? (NaN, 발산 체크)

**Go/No-Go**:
- IIG가 학습 중 변동 없음 → STOP. 수식 또는 구현 디버그.
- Blind Test Gap이 R_correct only와 동일 → IIG signal이 약함. λ 조정 후 재시도.
- Gap이 R_correct only보다 **줄어듦** → 심각. IIG가 blind reasoner를 악화. 수식 재검토.
- Gap이 baseline 이상 유지/증가 → PROCEED.

### Block 2 — Full GRPO (200 steps, GPU 2시간)

Block 1 통과 후. 본격 학습.

Training:
- 모델: Qwen3-VL-2B-Instruct
- 데이터: VQAv2 train 5000 + A-OKVQA train 3000
- GRPO config: group_size=8, temp=1.2, max_completion=128, beta=0.01
- LoRA: r=64, alpha=128
- Steps: 200
- 3개 설정 병렬:
  - (A) R_correct only
  - (B) R_correct + IIG (λ auto)
  - (C) R_correct + IIG (λ = 2 × auto) — IIG 강화 ablation

Eval (매 50 steps, 마지막에 full):
- POPE Adversarial 500 (real + black → Gap)
- MME (Perception + Cognition 분리)
- MMBench_DEV_EN 200
- OCRBench 100 (IIG가 OCR을 해치지 않는지 확인)

핵심 결과 table:

```
| Setting          | POPE-A | Gap  | MME-P | MME-C | MMBench | OCR  |
|------------------|--------|------|-------|-------|---------|------|
| Baseline (no RL) |   ?    |  ?   |   ?   |   ?   |    ?    |  ?   |
| (A) R_correct    |   ?    |  ?   |   ?   |   ?   |    ?    |  ?   |
| (B) IIG (auto λ) |   ?    |  ?   |   ?   |   ?   |    ?    |  ?   |
| (C) IIG (2× λ)   |   ?    |  ?   |   ?   |   ?   |    ?    |  ?   |
```

**성공 조건**: (B)의 Gap > (A)의 Gap AND (B)의 POPE ≥ (A)의 POPE.

### Block 3 — Thinking Mode (200 steps, GPU 2시간)

IIG의 진정한 가치는 thinking mode에서.

Training:
- 모델: Qwen3-VL-2B-Thinking
- GRPO config: group_size=4 (긴 generation, memory 제약), temp=1.0, max_completion=256
- 나머지 Block 2와 동일

추가 분석:
- Vision drift curve: thinking chain 내 토큰 위치별 IIG 분포
  - 초반 토큰: IIG 높을 것 (이미지 직후)
  - 중반: IIG 감소 (language reasoning 지배)
  - IIG training 후: 감소 폭이 줄어들었는가? → 핵심 figure

### Block 4 — DAPO Variant (100 steps, GPU 1시간)

DAPO = GRPO with beta=0.0, 비대칭 clipping, token-level loss.

- 동일 데이터, Block 2 설정
- loss_type="dapo" (TRL 지원 시), 아니면 beta=0.0만
- 비교: GRPO+IIG vs DAPO+IIG

### Block 5 — Post-Training Analysis (GPU 30분)

학습 완료된 모델에서:

1. **Blind Test 최종**: POPE full 3000 (Random + Popular + Adversarial) × {real, black}
2. **Steering OFF 테스트**: IIG-trained 모델에 steering 없이 eval → baseline보다 나은가? (영구성 증명)
3. **Steering ON 테스트**: IIG-trained + steering → 추가 향상? (orthogonality 증명)
4. **Head 분석**: Phase 1과 동일한 calibration을 학습 후 모델에 적용
   - 학습 전 top-K heads vs 학습 후 top-K heads → overlap 비율
   - IIG-trained 모델에서 새로 강화된 head가 있는가?

### Block 6 — Ablation (선택적, GPU 3시간)

시간이 남으면:
- λ sweep: [0.5×auto, auto, 2×auto, 4×auto]
- ε sweep: [0, 0.05, 0.1, 0.2]
- Black image 유형: pure black vs gray vs noise
- IIG variant: max(IIG,0) vs raw IIG (음수 포함)
- IIG vs v1 (head activation magnitude) 직접 비교

---

## Data Pipeline

### Training Data

| Dataset | Size | Source | POPE Overlap |
|---------|------|--------|-------------|
| VQAv2 train | 5000 (random subset) | COCO train2014 | ❌ 없음 |
| A-OKVQA train | 3000 (random subset) | COCO train2017 | ⚠️ image_id cross-check 필요 |

**A-OKVQA cross-check**: COCO 2014와 2017은 같은 image pool. A-OKVQA train의 image_id와 POPE eval의 image_id를 비교하여 겹치는 이미지를 제거하라. 안 하면 data contamination.

```python
# 반드시 학습 전에 실행
pope_image_ids = load_pope_image_ids()
aokvqa_clean = [s for s in aokvqa_train if s['image_id'] not in pope_image_ids]
```

### Eval Data

| Benchmark | Size | 측정 대상 |
|-----------|------|----------|
| POPE Adversarial | 500 (block eval), 3000 (final) | Hallucination + blind test |
| MME | Full | Perception vs Cognition 분리 |
| MMBench_DEV_EN | 200 | General capability |
| OCRBench | 100 | OCR 보존 확인 |
| MMMU_DEV_VAL | 200 (optional) | Hard reasoning |

---

## GRPO Config

```python
# Qwen3-VL-2B-Instruct
config_instruct = {
    "num_generations": 8,
    "temperature": 1.2,
    "beta": 0.01,
    "max_completion_length": 128,
    "learning_rate": 5e-6,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_target_modules": ["q_proj", "v_proj", "o_proj", "k_proj"],
    "remove_unused_columns": False,  # reward function이 image, answer 접근 필요
    "logging_steps": 1,
    "save_steps": 50,
}

# Qwen3-VL-2B-Thinking
config_thinking = {
    **config_instruct,
    "num_generations": 4,      # 긴 generation, memory 제약
    "temperature": 1.0,        # thinking mode는 낮은 temp
    "max_completion_length": 256,
}
```

---

## IIG 구현

```python
import torch, torch.nn.functional as F
from PIL import Image

BLACK_IMAGE = Image.new('RGB', (448, 448), (0, 0, 0))

def compute_iig(model, processor, image, question, candidate_tokens):
    """Image Information Gain: per-token average log-prob ratio."""
    with torch.no_grad():
        # Real image
        real = processor(images=image, text=question, return_tensors="pt").to(model.device)
        n = real.input_ids.shape[1]
        ids_r = torch.cat([real.input_ids, candidate_tokens.unsqueeze(0)], dim=1)
        kw = {k: v for k, v in real.items() if k != 'input_ids'}
        logits_r = model(input_ids=ids_r, **kw).logits[:, n-1:-1, :]
        lp_r = F.log_softmax(logits_r, dim=-1)
        tok_lp_r = lp_r.gather(-1, candidate_tokens[None, :, None]).squeeze(-1)

        # Black image
        blk = processor(images=BLACK_IMAGE, text=question, return_tensors="pt").to(model.device)
        ids_b = torch.cat([blk.input_ids, candidate_tokens.unsqueeze(0)], dim=1)
        kw_b = {k: v for k, v in blk.items() if k != 'input_ids'}
        logits_b = model(input_ids=ids_b, **kw_b).logits[:, n-1:-1, :]
        lp_b = F.log_softmax(logits_b, dim=-1)
        tok_lp_b = lp_b.gather(-1, candidate_tokens[None, :, None]).squeeze(-1)

        iig = (tok_lp_r - tok_lp_b).mean().item()
    return iig


def vigil_reward(r_correct, iig, lam, eps=0.1):
    """Final reward with reversal protection."""
    return float(r_correct) + lam * max(iig, 0.0) * (float(r_correct) + eps)


def calibrate_lambda(model, processor, calib_data):
    """Auto-determine λ from calibration data."""
    positives = []
    for img, q, tokens in calib_data:
        v = compute_iig(model, processor, img, q, tokens)
        if v > 0:
            positives.append(v)
    if not positives:
        return 0.5
    mu = sum(positives) / len(positives)
    sigma = (sum((x - mu)**2 for x in positives) / len(positives)) ** 0.5
    return 1.0 / (mu + sigma + 1e-8)
```

코드를 `vigil/rewards/iig.py`에 넣어라. 기존 rewards/ 코드가 있으면 구조를 따르되 위 로직을 적용.

---

## Adversarial Design Loop

모든 reward/실험 설계 변경에 대해 이 루프를 수행하라.
**같은 context에서 제안과 공격을 하면 자기 확증 편향에 빠진다. 반드시 파일을 분리하라.**

### 프로토콜

```
Phase 1: PROPOSE → lab/design/{name}_v{N}_propose.md
  - 100% 확신하는 것처럼 써라. 자기 의심 금지.

Phase 2: ATTACK → lab/design/{name}_v{N}_attack.md
  - 역할 전환. "ICLR의 가장 까다로운 리뷰어"로서 최소 5개 공격.
  - 반드시 수식에 구체적 숫자를 넣어 반례를 만들어라.
  - 체크리스트 전항목 검토:
    □ reward gaming shortcut 존재?
    □ GRPO group advantage에서 signal 생존?
    □ 수식 스케일 맞는가? (한 항이 다른 항 압도?)
    □ 보상 역전 가능? (나쁜 candidate > 좋은 candidate?)
    □ 모든 태스크에서 작동? (OCR, binary VQA, open-ended, thinking)
    □ 기존 방법과 정말 다른가?
    □ overhead claim 정확?
    □ 이론적 근거 수학적으로 정확?

Phase 3: REBUT → lab/design/{name}_v{N}_rebuttal.md
  - 유효한 공격은 인정하고 수정. 무시 금지.
  - 수정이 새 문제를 만들지 않는지 자체 점검.

Phase 4: VERIFY → lab/design/{name}_v{N+1}_verify.md
  - 수정본의 논리적 일관성만 검증 (좋고 나쁨 판단 금지)
  - 극단값 (0, ∞, 음수) 대입 → 수식 깨지는지
  - 논리 체인에 끊어진 고리 없는지
  - 판정: PASS / NEEDS ANOTHER ROUND

최대 4 라운드. 4회 후에도 PASS 안 나면 → pivot.md 작성.
```

### 2-Agent 강화 (선택적)

Anthropic API를 호출하여 별도 instance에게 공격을 시키면 편향이 줄어든다.

```python
import anthropic
client = anthropic.Anthropic()

proposal = open("lab/design/iig_reward_v1_propose.md").read()
attack = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    system="""너는 ML 논문 리뷰어다. 최대한 가혹하게 비판하라.
모든 수식에 구체적 숫자를 넣어 반례를 만들어라.
"괜찮은데" 같은 관용 금지. reject 사유만 지적. 최소 5개.""",
    messages=[{"role": "user", "content": f"이 연구 제안을 공격하라:\n\n{proposal}"}]
)
open("lab/design/iig_reward_v1_attack.md", "w").write(attack.content[0].text)
```

GPU 코드를 수정할 때마다 이 API 호출로 red-team 검증을 추가하라. 비용은 미미하다 (Sonnet 1회 호출 < $0.05).

---

## Execution — Ralph Loop Phases

### Phase 0 — IIG Calibration

```
/ralph-loop "
Phase 1의 calibration 결과를 읽어라 (lab/reports/).
GQA calibration 데이터를 사용하여 모든 sample의 IIG를 계산하라.
IIG 분포 히스토그램을 저장하고, λ를 auto-calibrate하라.
IIG vs Phase 1 Cohen's d 산점도를 그려라.
결과를 lab/reports/iig_calibration/에 저장. Git commit.

성공 조건: IIG 양수 비율 ≥ 60%.
성공 시 <promise>IIG_CALIBRATED</promise>.
실패 시 원인 분석 후 <promise>IIG_CALIBRATED</promise> (음성 결과도 결과).
" --max-iterations 8 --completion-promise "IIG_CALIBRATED"
```

### Phase 1 — Minimal GRPO (Block 1)

```
/ralph-loop "
IIG calibration 결과를 읽어라 (lab/reports/iig_calibration/).

Block 1 실행:
- Qwen3-VL-2B-Instruct, VQAv2 2000 samples, 50 steps
- 2개 설정: (A) R_correct only, (B) R_correct + IIG
- 매 10 steps: POPE-A 200 + Blind Test Gap
- 학습 곡선 (R_correct, IIG, total reward) 저장

관찰:
- IIG가 step마다 변하는가?
- (B)의 Gap > (A)의 Gap인가?
- R_correct가 급락하지 않는가?

iteration protocol: measure → compare → diagnose → decide → execute.
문제 발생 시: Adversarial Loop (PROPOSE→ATTACK→REBUT→VERIFY)로 수정안 도출.
매 iteration git commit.

성공: IIG가 변동하고, Gap이 (A)보다 (B)에서 크거나 같음.
<promise>MINIMAL_GRPO_DONE</promise>
" --max-iterations 15 --completion-promise "MINIMAL_GRPO_DONE"
```

### Phase 2 — Full GRPO (Block 2)

```
/ralph-loop "
Block 1 결과를 읽어라. Block 2 실행:
- 3개 설정: (A) R_correct only, (B) IIG auto-λ, (C) IIG 2×λ
- VQAv2 5000 + A-OKVQA 3000, 200 steps
- LoRA r=64, group_size=8
- 매 50 steps: POPE-A 500 + MME + MMBench 200 + OCRBench 100
- 마지막에 full eval

결과 table 작성 (POPE, Gap, MME-P, MME-C, MMBench, OCR).

핵심 판단:
- (B)의 Gap > (A)의 Gap → IIG 효과 증명
- (B)의 POPE ≥ (A)의 POPE → 정확도 유지/향상
- (B)의 MME-C ≥ (A)의 MME-C × 0.95 → language 보존
- (B)의 OCR ≈ baseline → OCR 중립

문제 발생 시 Adversarial Loop로 수정.
<promise>FULL_GRPO_DONE</promise>
" --max-iterations 12 --completion-promise "FULL_GRPO_DONE"
```

### Phase 3 — Thinking Mode (Block 3)

```
/ralph-loop "
Block 2 결과를 읽어라. Block 3 실행:
- Qwen3-VL-2B-Thinking, group_size=4, max_completion=256
- 동일 데이터/설정, 200 steps
- 추가: vision drift curve (토큰 위치별 IIG) — steered vs unsteered vs IIG-trained

핵심 figure: thinking chain 내 visual grounding 유지도
- x축: 토큰 위치 (0 ~ 256)
- y축: 해당 위치에서의 평균 IIG
- 3개 선: baseline / Phase 1 steered / IIG-trained
- IIG-trained 선이 baseline보다 덜 감소하면 성공

<promise>THINKING_DONE</promise>
" --max-iterations 12 --completion-promise "THINKING_DONE"
```

### Phase 4 — Post-Training Analysis (Block 5)

```
/ralph-loop "
모든 학습 완료. Block 5 실행:
1. Blind Test 최종: POPE 3000 × {real, black}
2. Steering OFF: IIG-trained 모델, steering 없이 eval
3. Steering ON: IIG-trained + Phase 1 steering 결합
4. Head 분석: Phase 1 calibration을 IIG-trained 모델에 재적용
   - 학습 전 top-K vs 학습 후 top-K overlap
5. 논문 figure 생성: drift curve, Gap comparison bar chart, head heatmap

종합 결과를 RESEARCH_JOURNAL.md에 기록.
<promise>ANALYSIS_DONE</promise>
" --max-iterations 8 --completion-promise "ANALYSIS_DONE"
```

---

## Decision Tree

```
Block 0: IIG 양수 비율?
├── < 60% → STOP. IIG 수식 디버그 또는 black image 변경.
└── ≥ 60% → continue

Block 1: 50-step Gap 변화?
├── (B) Gap < (A) Gap → STOP. IIG가 blind reasoner 악화.
│   → Adversarial Loop로 수식 재검토.
├── (B) Gap ≈ (A) Gap → IIG signal 약함.
│   → λ × 2 재시도. 여전히 동일하면 STOP.
└── (B) Gap ≥ (A) Gap → PROCEED.

Block 2: 200-step 결과?
├── Gap 증가 + POPE 향상 → ★ 성공. Block 3로.
├── Gap 증가 + POPE 동일 → 부분 성공. λ 조정 후 재시도.
├── Gap 동일 + POPE 향상 → IIG 중립, R_correct만으로 충분.
│   → IIG의 가치 재평가. Binary VQA 한계일 수 있음.
│   → Block 3 (thinking mode)에서 효과가 나타날 수 있으므로 진행.
├── Gap 감소 → FAIL. Adversarial Loop로 전면 재검토.
└── MME-C 급락 → λ 줄이기 또는 conditional IIG (정답에서만) 시도.

Block 3: Thinking mode?
├── Drift curve가 IIG-trained에서 개선 → ★ 핵심 contribution.
├── 개선 없음 → Thinking mode에서는 효과 없음. Instruct 결과만으로 논문.
└── 악화 → 심각. 근본 문제 있음.

Block 5: Post-training
├── Steering OFF에서 baseline 이상 → ★ 영구적 개선 증명.
├── Steering OFF에서 baseline 미만 → 영구성 없음. Steering과 결합해야만 효과.
└── Head overlap > 70% → ★ RL이 calibration과 같은 head를 강화 = mechanistic validation.
```

---

## 논문 서사 — 전체 VIGIL Story

```
문제: 소형 VLM은 이미지를 점점 무시한다 (Visual Attention Drift)
  ↓
진단: GRPO로 학습하면 더 심해진다 (Blind Reasoner, DVRP)
  ↓
Phase 1 발견: Vision head를 찾아 강제로 키우면 성능이 오른다 (Steering)
  → 하지만 끄면 원래대로 돌아간다 (일시적)
  ↓
Phase 2 해결: IIG reward로 GRPO 학습하면 모델이 스스로 이미지를 보는 법을 배운다
  → 끄더라도 효과가 유지된다 (영구적)
  → Blind Test Gap이 줄어들지 않고 오히려 커진다 (blind reasoner 방지)
  ↓
Mechanistic validation:
  - RL이 강화한 head = calibration에서 발견한 vision head (overlap)
  - Steering + IIG training은 상보적 (orthogonal)
  ↓
Contribution:
  C1. IIG — 정보 이론 기반 visual grounding GRPO reward (candidate-level, hyperparameter-free)
  C2. Head-level conditional steering (inference-time)
  C3. Blind Test Gap 분석 프레임워크
  C4. Thinking mode vision drift 분석
  C5. Steering-Training orthogonality 증명
```

---

## What This Instruction Does NOT Specify

- TRL GRPOTrainer의 exact import 경로나 API (버전마다 다름 — 읽고 적응하라)
- Dataset loading 코드 (HuggingFace datasets 표준 패턴을 따르라)
- Checkpoint 관리 방법 (save_steps=50으로 충분)
- 시각화 세부사항 (matplotlib, 읽기 좋게)

**What comes after this instruction**: Block 2 결과가 양성이면, InternVL3.5-1B와 DeepSeek-VL2-Tiny로 확장하는 Phase 3 instruction을 작성한다. 그 instruction은 이 Phase 2 결과에 의존한다.
