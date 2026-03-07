Adversarial Self-Refinement Protocol for Claude Code

> 이 파일을 CLAUDE.md 또는 AGENTS.md에 포함시킨다.
> Claude Code가 연구 설계를 자율적으로 개선하는 루프를 정의한다.

---

## 핵심 원칙

사람과의 대화에서 연구가 개선되는 이유:
1. **역할 분리** — 제안자와 공격자가 다른 사람
2. **강제된 전환** — "공격해봐" 같은 명시적 지시
3. **비가역적 진행** — 발견된 약점은 무시할 수 없음

Claude Code 혼자서 하면 "제안 → 자기가 좋다고 함 → 끝"이 된다.
이를 방지하려면 **persona를 강제 전환하는 구조화된 프로토콜**이 필요하다.

---

## The ADVERSARIAL LOOP

모든 연구 설계/알고리즘/실험 계획에 대해 다음 5단계를 **순서대로, 각 단계를 별도 파일로** 수행한다.

### Phase 1: PROPOSE (Blue Team)

역할: 최선을 다해 제안한다.
출력: `lab/design/{name}_v{N}_propose.md`

```markdown
## [PROPOSE] {제목}
### 핵심 주장
### 수식/알고리즘
### 왜 이것이 작동하는가
### 예상 결과
```

규칙:
- 최대한 강하게 주장하라
- 자기 의심 금지 — 이 단계에서는 100% 확신하는 것처럼 써라
- 구현 코드까지 포함

### Phase 2: ATTACK (Red Team)

역할: Phase 1의 제안을 **최대한 파괴**하라.
출력: `lab/design/{name}_v{N}_attack.md`

```markdown
## [ATTACK] {제목} 에 대한 공격

### Attack 1: {이름}
**메커니즘**: 구체적으로 어떻게 실패하는가
**심각도**: 치명적 / 높음 / 중간 / 낮음
**발생 조건**: 어떤 상황에서 이 문제가 나타나는가
**증거**: 수식, 구체적 수치 예시, 또는 반례

### Attack 2: ...
(최소 5개)
```

규칙:
- **Phase 1을 쓴 사람이 바보인 것처럼 공격하라**
- "이것도 괜찮은데..."류의 관용 금지
- 모든 가정을 의심하라
- 수식에 구체적 숫자를 넣어서 반례를 만들어라
- 리뷰어가 reject 사유로 쓸 만한 것만 포함
- 사소한 문제도 치명적으로 포장하지 말 것 — 심각도를 정직하게

**핵심 질문 체크리스트** (반드시 각각 검토):
- [ ] 이 reward/loss를 gaming하는 shortcut이 있는가?
- [ ] GRPO group-relative advantage에서 이 signal이 살아남는가?
- [ ] 수식의 스케일이 맞는가? (한 항이 다른 항을 압도하지 않는가?)
- [ ] 보상 역전이 가능한가? (나쁜 candidate > 좋은 candidate?)
- [ ] 모든 벤치마크에서 작동하는가? (특정 태스크에서만 효과?)
- [ ] 이미 존재하는 방법과 정말 다른가? (novelty 재검증)
- [ ] Compute overhead가 claim과 일치하는가?
- [ ] 주장한 이론적 근거가 수학적으로 정확한가?

### Phase 3: REBUT & FIX (Blue Team, informed)

역할: Attack을 인정하거나 반박하고, 수정된 버전을 제안한다.
출력: `lab/design/{name}_v{N}_rebuttal.md`

```markdown
## [REBUTTAL] {제목} v{N} → v{N+1}

### Attack 1에 대한 대응
**판정**: 유효 / 부분 유효 / 무효
**유효한 경우 수정 내용**: ...
**무효한 경우 반박 근거**: ...

### 수정된 수식/알고리즘
### 수정으로 인해 새로 발생할 수 있는 문제
```

규칙:
- Attack이 유효하면 **반드시 인정**하라. 무시 금지.
- 수정이 새로운 문제를 만드는지 자체 점검
- "Attack이 유효하지만 실제로는 문제 안 됨"은 허용 — 단, 구체적 근거 필수

### Phase 4: VERIFY (Neutral)

역할: Phase 3의 수정본에 대해 **논리적 일관성만** 검증한다.
출력: `lab/design/{name}_v{N+1}_verify.md`

```markdown
## [VERIFY] {제목} v{N+1}

### 논리 체인 검증
Step 1: [주장] → [근거] → [OK / GAP]
Step 2: ...

### 수식 검증
- 차원 일치: ...
- 범위 일치: ...
- 극단값 검증: ...

### 서사 일관성
- 논문 제목과 방법이 일치하는가?
- Contribution 주장과 실제 메커니즘이 일치하는가?

### 판정: PASS / NEEDS ANOTHER ROUND
```

규칙:
- 좋고 나쁨을 판단하지 말 것 — 논리적 일관성만
- 극단값(0, ∞, 음수)을 넣어서 수식이 깨지는지 확인
- "이것이 왜 작동하는가"의 논리 체인에 끊어진 고리가 있는지 확인

### Phase 5: DECISION

Phase 4가 PASS면 → 채택. `lab/design/{name}_v{N+1}_final.md`로 저장.
Phase 4가 NEEDS ANOTHER ROUND면 → Phase 2로 돌아가되 N+1 버전에 대해.

**최대 반복 횟수: 4회.** 4회 후에도 PASS가 안 나면 → 근본적 접근 변경 필요. `lab/design/{name}_pivot.md`에 왜 이 접근이 안 되는지 기록하고 다른 방향을 탐색.

---

## 파일 구조 예시

```
lab/design/
├── r_vhad_v1_propose.md       ← 최초 제안 (head activation)
├── r_vhad_v1_attack.md        ← 10개 공격 발견
├── r_vhad_v1_rebuttal.md      ← Attack #5 치명적 → v2로 전환
├── r_vhad_v2_propose.md       ← log-prob ratio
├── r_vhad_v2_attack.md        ← Gap 3 (보상 역전) 발견
├── r_vhad_v2_rebuttal.md      ← soft multiplicative + λ auto
├── r_vhad_v3_verify.md        ← PASS
├── r_vhad_v3_final.md         ← 채택
└── r_vhad_changelog.md        ← 전체 진화 기록
```

---

## Claude Code 프롬프트 템플릿

### 자율 실행 시

```
너는 지금부터 연구 설계 개선 루프를 수행한다.

대상: {파일 경로 또는 주제}

다음 순서를 반드시 따라라:

1. 현재 설계를 읽고 lab/design/{name}_v{N}_propose.md로 정리
2. 즉시 역할을 전환하여 최소 5개의 공격을 lab/design/{name}_v{N}_attack.md에 작성
   - 수식에 구체적 숫자를 넣어 반례를 만들 것
   - 리뷰어가 reject할 만한 수준의 공격만 포함
   - 체크리스트의 모든 항목을 검토할 것
3. 공격에 대한 대응을 lab/design/{name}_v{N}_rebuttal.md에 작성
   - 유효한 공격은 반드시 인정하고 수정할 것
4. 수정본의 논리적 일관성을 lab/design/{name}_v{N+1}_verify.md에 검증
5. PASS면 _final.md로 저장, 아니면 2번으로 돌아갈 것

각 파일을 작성한 후 반드시 git commit하라.
최대 4 라운드. 4 라운드 후에도 PASS가 안 나면 pivot.md를 작성하라.
```

### Ralph Loop 연동

```bash
/ralph-loop "
대상: lab/design/reward_design_v1_propose.md

Phase 2 (Attack)를 수행하라.
- 최소 5개 공격
- 체크리스트 전항목 검토
- 구체적 수치 반례 포함
- 심각도 정직하게 평가
결과를 lab/design/reward_design_v1_attack.md에 저장하라.

완료 후 Phase 3 (Rebuttal)을 수행하라.
- 유효한 공격은 인정하고 수정
- 수정된 v2 수식 제시
결과를 lab/design/reward_design_v1_rebuttal.md에 저장하라.

완료 후 Phase 4 (Verify)를 수행하라.
- v2의 논리 체인 검증
- 극단값 검증
- PASS/NEEDS ANOTHER ROUND 판정
결과를 lab/design/reward_design_v2_verify.md에 저장하라.

NEEDS ANOTHER ROUND이면 v2에 대해 Phase 2부터 반복하라.
PASS면 _final.md로 저장하고 종료하라.
" --max-iterations 12 --completion-promise "DESIGN_VERIFIED"
```

---

## 왜 이것이 작동하는가

### 우리 대화에서 실제로 일어난 것의 재구성

```
Turn 1: "R_vhad GRPO 구체화 해줘"
  → Phase 1: PROPOSE (SPEC 1068줄)

Turn 2: "공격해봐"
  → Phase 2: ATTACK (10개 공격, 753줄)
  → 발견: Attack #5가 치명적

Turn 3: "반박 → novelty 확보 → 단순화 → 재제안"
  → Phase 3: REBUTTAL (v1 → v3)
  → Phase 1: PROPOSE v3 (623줄)

Turn 4: "최신 연구 찾아 → 한번 더"
  → Phase 2: 경쟁 환경 업데이트 + ATTACK
  → Phase 3: REBUTTAL (v3 → v4)
  → Phase 1: PROPOSE v4 (302줄)

Turn 5: "논리성 보완"
  → Phase 4: VERIFY (10개 Gap, 475줄)
  → 발견: Gap 3 (보상 역전) 치명적
  → Phase 3: REBUTTAL (수식 수정)

Turn 6: "yes no만?" / "OCR은?"
  → Phase 2: 추가 공격 (태스크별 효과)
  → Phase 3: 반박 (IIG의 태스크별 투명성)
```

**6턴 만에 1068줄 → 302줄 + 475줄 패치.** 알고리즘이 4컴포넌트/4하이퍼파라미터에서 2컴포넌트/0하이퍼파라미터로 단순화되었고, 치명적 버그 2개(Attack #5, Gap 3)가 수정되었다.

이 효과를 Claude Code가 자율적으로 달성하려면:
- **파일 분리가 핵심** — 같은 파일에서 제안과 공격을 하면 편향됨
- **체크리스트가 핵심** — "숫자 넣어서 반례 만들어라"가 없으면 추상적 공격만 함
- **반복 횟수 제한이 핵심** — 없으면 무한루프. 4회면 충분.

---

## INSTRUCTION.md에 추가할 한 줄

```markdown
## 연구 설계 개선 프로토콜

모든 reward 설계, 실험 설계, 알고리즘 변경에 대해 
`lab/design/` 디렉토리에서 Adversarial Self-Refinement Loop 
(PROPOSE → ATTACK → REBUT → VERIFY)를 수행하라. 
프로토콜 상세는 ADVERSARIAL_LOOP.md를 참조. 
최대 4라운드. PASS 없이 4라운드 초과 시 pivot.md 작성.
```
