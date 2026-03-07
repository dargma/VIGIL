VIGIL — Supplementary Instruction: Visual Grounding Analysis

> INSTRUCTION2.md가 이미 실행 중이다. 이 파일은 **추가 분석 지시**다.
> 각 Block의 eval이 끝난 후, 해당 checkpoint에 대해 아래 분석을 수행하라.
> 기존 Block 흐름을 중단하지 마라. Eval 결과 저장 후 이 분석을 추가로 돌려라.

---

## 목적

숫자(POPE accuracy, Blind Test Gap)만으로는 "모델이 정말 이미지를 더 보게 되었는가"를 증명할 수 없다. **이미지 위의 heatmap으로 시각적 증거를 만든다.**

논문 figure 후보:
- Fig A: 단답/추론 × baseline/IIG-trained의 4-panel attention heatmap
- Fig B: Thinking mode에서 토큰 위치별 visual attention 시계열
- Fig C: Spatial IIG map (어떤 이미지 영역이 답변에 정보를 제공했는가)

---

## Analysis 1: Attention Heatmap (모든 Block eval 후 실행)

### 1.1 Attention 추출

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def extract_visual_attention(model, processor, image, question, generated_tokens=None,
                             generate_kwargs=None):
    """
    생성된 토큰이 이미지의 어떤 영역에 attention하는지 추출.
    
    Returns:
        attn_map: [H_img, W_img] array — 이미지 공간에서의 attention 강도
        generated_text: str
        per_token_attn: list of [H_img, W_img] — 토큰별 attention (thinking mode용)
    """
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
    
    if generated_tokens is not None:
        # Teacher-forced: 이미 생성된 토큰에 대한 attention
        full_ids = torch.cat([inputs.input_ids, generated_tokens.unsqueeze(0)], dim=1)
        with torch.no_grad():
            outputs = model(
                input_ids=full_ids,
                **{k: v for k, v in inputs.items() if k != 'input_ids'},
                output_attentions=True
            )
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
    else:
        # Generation mode
        if generate_kwargs is None:
            generate_kwargs = {"max_new_tokens": 256, "temperature": 0.0}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                output_attentions=True,
                return_dict_in_generate=True,
                **generate_kwargs
            )
        generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
    
    # --- Visual token 위치 식별 ---
    # Qwen3-VL: processor가 image token 위치를 알려줌
    # 일반적으로 input_ids에서 image_token_id의 범위를 찾는다
    input_ids = inputs.input_ids[0]
    
    # 모델별로 visual token 범위를 찾는 방법이 다르다.
    # Qwen3-VL의 경우 processor.image_token_id 또는 <|vision_start|>...<|vision_end|> 사이
    # 아래는 범용적 접근: image_grid_thw 정보 활용
    visual_token_start, visual_token_end = find_visual_token_range(inputs)
    num_visual_tokens = visual_token_end - visual_token_start
    
    # 이미지 grid 크기 추론 (정사각형 가정, 아니면 processor에서 가져옴)
    grid_h = grid_w = int(np.sqrt(num_visual_tokens))
    if grid_h * grid_w != num_visual_tokens:
        # Dynamic resolution인 경우 processor의 image_grid_thw 사용
        if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
            grid_info = inputs.image_grid_thw[0]  # [T, H, W]
            grid_h, grid_w = grid_info[1].item(), grid_info[2].item()
        else:
            grid_h = grid_w = int(np.ceil(np.sqrt(num_visual_tokens)))
    
    # --- Attention 집계 ---
    # outputs.attentions: tuple of [batch, heads, seq, seq] per layer
    # 마지막 N개 layer의 평균 사용 (N=4)
    num_layers_to_use = min(4, len(outputs.attentions))
    
    prefix_len = inputs.input_ids.shape[1]
    gen_positions = range(prefix_len, prefix_len + len(generated_tokens))
    
    per_token_attn = []
    
    for t_pos in gen_positions:
        # 각 생성 토큰 → visual token에 대한 attention
        token_visual_attn = []
        for layer_attn in outputs.attentions[-num_layers_to_use:]:
            # layer_attn: [1, num_heads, seq_len, seq_len]
            # t_pos번째 토큰이 visual_token_start:visual_token_end에 주는 attention
            attn_to_visual = layer_attn[0, :, t_pos, visual_token_start:visual_token_end]
            # 모든 head 평균
            attn_to_visual = attn_to_visual.mean(dim=0)  # [num_visual_tokens]
            token_visual_attn.append(attn_to_visual.cpu().numpy())
        
        # layer 평균
        avg_attn = np.mean(token_visual_attn, axis=0)  # [num_visual_tokens]
        
        # grid로 reshape
        if len(avg_attn) >= grid_h * grid_w:
            attn_grid = avg_attn[:grid_h * grid_w].reshape(grid_h, grid_w)
        else:
            attn_grid = np.zeros((grid_h, grid_w))
            attn_grid.flat[:len(avg_attn)] = avg_attn
        
        per_token_attn.append(attn_grid)
    
    # 전체 토큰 평균 → 요약 heatmap
    attn_map = np.mean(per_token_attn, axis=0)  # [grid_h, grid_w]
    
    return {
        'attn_map': attn_map,
        'per_token_attn': per_token_attn,
        'generated_text': generated_text,
        'grid_size': (grid_h, grid_w),
    }


def find_visual_token_range(inputs):
    """모델별 visual token 위치 찾기. 구현은 모델에 맞게 조정 필요."""
    input_ids = inputs.input_ids[0]
    
    # Qwen3-VL: <|vision_start|> = 151652, <|vision_end|> = 151653 (확인 필요)
    # 또는 image_token_id 사용
    # 범용: 연속된 동일 token id 구간 찾기
    
    # 방법 1: processor에 image_token_id가 있으면
    # image_token_id = processor.image_token_id  
    # mask = (input_ids == image_token_id)
    # indices = mask.nonzero(as_tuple=True)[0]
    # return indices[0].item(), indices[-1].item() + 1
    
    # 방법 2: 특수 토큰 사이 찾기 (Qwen3-VL)
    # 실제 구현에서 모델 로드 후 확인하여 적절히 수정할 것
    
    raise NotImplementedError(
        "모델 로드 후 visual token 범위를 확인하고 이 함수를 구현하라. "
        "processor.image_token_id 또는 special token id로 찾을 수 있다."
    )
```

### 1.2 4-Panel Heatmap Figure

```python
def plot_4panel_heatmap(image, result_base, result_iig, question, answer_gt, save_path):
    """
    [Original Image] [Baseline Attention] [IIG-trained Attention] [Difference]
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    img_array = np.array(image)
    
    # (a) Original
    axes[0].imshow(img_array)
    axes[0].set_title(f"Q: {question[:50]}...\nGT: {answer_gt}", fontsize=9)
    axes[0].axis('off')
    
    # (b) Baseline attention overlay
    attn_base = result_base['attn_map']
    attn_base_resized = np.array(
        Image.fromarray((attn_base / attn_base.max() * 255).astype(np.uint8)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
        )
    ) / 255.0
    
    axes[1].imshow(img_array)
    axes[1].imshow(attn_base_resized, alpha=0.5, cmap='jet')
    axes[1].set_title(
        f"Baseline\nans: {result_base['generated_text'][:30]}\n"
        f"IIG: {result_base.get('iig', 'N/A'):.2f}",
        fontsize=9
    )
    axes[1].axis('off')
    
    # (c) IIG-trained attention overlay
    attn_iig = result_iig['attn_map']
    attn_iig_resized = np.array(
        Image.fromarray((attn_iig / attn_iig.max() * 255).astype(np.uint8)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
        )
    ) / 255.0
    
    axes[2].imshow(img_array)
    axes[2].imshow(attn_iig_resized, alpha=0.5, cmap='jet')
    axes[2].set_title(
        f"IIG-trained\nans: {result_iig['generated_text'][:30]}\n"
        f"IIG: {result_iig.get('iig', 'N/A'):.2f}",
        fontsize=9
    )
    axes[2].axis('off')
    
    # (d) Difference (IIG - Baseline)
    diff = attn_iig_resized - attn_base_resized
    axes[3].imshow(img_array, alpha=0.3)
    im = axes[3].imshow(diff, cmap='RdBu_r', alpha=0.7, vmin=-0.5, vmax=0.5)
    axes[3].set_title("Δ Attention\n(red=IIG↑, blue=IIG↓)", fontsize=9)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
```

### 1.3 Dramatic Sample 자동 선택

```python
def find_dramatic_samples(model_base, model_iig, processor, eval_data, top_n=10):
    """
    둘 다 정답이지만 attention 분포가 가장 크게 달라진 sample 찾기.
    "같은 답인데 보는 곳이 다르다" = 가장 설득력 있는 figure.
    """
    candidates = []
    
    for sample in eval_data:
        image, question, answer = sample['image'], sample['question'], sample['answer']
        
        # 두 모델 모두 greedy generation
        result_base = extract_visual_attention(model_base, processor, image, question)
        result_iig = extract_visual_attention(model_iig, processor, image, question)
        
        # 둘 다 정답인 경우만
        correct_base = is_correct(result_base['generated_text'], answer)
        correct_iig = is_correct(result_iig['generated_text'], answer)
        
        if not (correct_base and correct_iig):
            # "baseline 오답 → IIG 정답" 케이스도 별도로 수집
            if not correct_base and correct_iig:
                candidates.append({
                    'sample': sample,
                    'result_base': result_base,
                    'result_iig': result_iig,
                    'category': 'flipped',  # 오답→정답
                    'attn_shift': compute_attn_kl(result_iig['attn_map'], result_base['attn_map']),
                })
            continue
        
        # Attention 분포 차이 (KL divergence)
        shift = compute_attn_kl(result_iig['attn_map'], result_base['attn_map'])
        
        # IIG 차이
        iig_base = compute_iig(model_base, processor, image, question,
                               tokenize(result_base['generated_text']))
        iig_iig_val = compute_iig(model_iig, processor, image, question,
                                  tokenize(result_iig['generated_text']))
        
        result_base['iig'] = iig_base
        result_iig['iig'] = iig_iig_val
        
        candidates.append({
            'sample': sample,
            'result_base': result_base,
            'result_iig': result_iig,
            'category': 'both_correct',
            'attn_shift': shift,
            'iig_delta': iig_iig_val - iig_base,
        })
    
    # 카테고리별 정렬
    both_correct = sorted(
        [c for c in candidates if c['category'] == 'both_correct'],
        key=lambda x: x['attn_shift'], reverse=True
    )
    flipped = sorted(
        [c for c in candidates if c['category'] == 'flipped'],
        key=lambda x: x['attn_shift'], reverse=True
    )
    
    return {
        'both_correct_top': both_correct[:top_n],
        'flipped_top': flipped[:top_n],
    }


def compute_attn_kl(attn_a, attn_b):
    """두 attention map 사이의 KL divergence."""
    # Flatten + normalize to probability distribution
    a = attn_a.flatten().astype(np.float64)
    b = attn_b.flatten().astype(np.float64)
    a = a / (a.sum() + 1e-10) + 1e-10
    b = b / (b.sum() + 1e-10) + 1e-10
    return float(np.sum(a * np.log(a / b)))
```

---

## Analysis 2: Thinking Mode Temporal Attention (Block 3 후 실행)

### 2.1 토큰 위치별 Visual Attention 시계열

```python
def plot_temporal_attention(results_base, results_iig, save_path, 
                            window_size=10):
    """
    x축: 생성 토큰 위치 (0 ~ T)
    y축: 해당 위치에서 visual token에 대한 평균 attention
    2개 선: baseline vs IIG-trained
    
    Thinking mode에서 visual attention이 어떻게 감소하는지,
    IIG training이 그 감소를 완화하는지 보여준다.
    """
    def aggregate_visual_attention(per_token_attn):
        """각 토큰 위치에서 visual attention의 총합."""
        return [grid.sum() for grid in per_token_attn]
    
    attn_base = aggregate_visual_attention(results_base['per_token_attn'])
    attn_iig = aggregate_visual_attention(results_iig['per_token_attn'])
    
    # Smoothing
    def smooth(values, w):
        if len(values) < w:
            return values
        return np.convolve(values, np.ones(w)/w, mode='valid').tolist()
    
    attn_base_smooth = smooth(attn_base, window_size)
    attn_iig_smooth = smooth(attn_iig, window_size)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x_base = range(len(attn_base_smooth))
    x_iig = range(len(attn_iig_smooth))
    
    ax.plot(x_base, attn_base_smooth, color='#d62728', alpha=0.8, 
            linewidth=2, label='Baseline')
    ax.plot(x_iig, attn_iig_smooth, color='#2ca02c', alpha=0.8, 
            linewidth=2, label='IIG-trained')
    
    ax.set_xlabel('Generated Token Position', fontsize=12)
    ax.set_ylabel('Visual Attention (sum over patches)', fontsize=12)
    ax.set_title('Visual Attention During Thinking Chain', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Drift 영역 표시
    if len(attn_base_smooth) > 20:
        early = np.mean(attn_base_smooth[:10])
        late = np.mean(attn_base_smooth[-10:])
        drift_pct = (early - late) / (early + 1e-10) * 100
        ax.annotate(f'Baseline drift: {drift_pct:.0f}% decay',
                   xy=(len(attn_base_smooth)*0.7, late),
                   fontsize=10, color='#d62728')
    
    if len(attn_iig_smooth) > 20:
        early_iig = np.mean(attn_iig_smooth[:10])
        late_iig = np.mean(attn_iig_smooth[-10:])
        drift_pct_iig = (early_iig - late_iig) / (early_iig + 1e-10) * 100
        ax.annotate(f'IIG drift: {drift_pct_iig:.0f}% decay',
                   xy=(len(attn_iig_smooth)*0.7, late_iig),
                   fontsize=10, color='#2ca02c')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_temporal_attention_averaged(model_base, model_iig, processor, 
                                     eval_samples, save_path, max_samples=50):
    """
    여러 sample의 temporal attention을 평균내어 안정적인 곡선을 만든다.
    단일 sample은 noisy할 수 있으므로 50개 이상 평균.
    """
    all_base = []
    all_iig = []
    
    for sample in eval_samples[:max_samples]:
        image, question = sample['image'], sample['question']
        
        rb = extract_visual_attention(model_base, processor, image, question,
                                      generate_kwargs={"max_new_tokens": 256, "temperature": 0.0})
        ri = extract_visual_attention(model_iig, processor, image, question,
                                      generate_kwargs={"max_new_tokens": 256, "temperature": 0.0})
        
        base_curve = [g.sum() for g in rb['per_token_attn']]
        iig_curve = [g.sum() for g in ri['per_token_attn']]
        
        all_base.append(base_curve)
        all_iig.append(iig_curve)
    
    # 길이 맞추기 (가장 짧은 것 기준 truncate)
    min_len = min(min(len(c) for c in all_base), min(len(c) for c in all_iig))
    min_len = min(min_len, 200)  # 최대 200 토큰
    
    base_matrix = np.array([c[:min_len] for c in all_base])
    iig_matrix = np.array([c[:min_len] for c in all_iig])
    
    base_mean = base_matrix.mean(axis=0)
    base_std = base_matrix.std(axis=0)
    iig_mean = iig_matrix.mean(axis=0)
    iig_std = iig_matrix.std(axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(min_len)
    
    ax.plot(x, base_mean, color='#d62728', linewidth=2, label='Baseline')
    ax.fill_between(x, base_mean - base_std, base_mean + base_std, 
                    color='#d62728', alpha=0.15)
    
    ax.plot(x, iig_mean, color='#2ca02c', linewidth=2, label='IIG-trained')
    ax.fill_between(x, iig_mean - iig_std, iig_mean + iig_std,
                    color='#2ca02c', alpha=0.15)
    
    ax.set_xlabel('Generated Token Position', fontsize=12)
    ax.set_ylabel('Visual Attention (mean ± std)', fontsize=12)
    ax.set_title(f'Visual Attention Drift — Averaged over {len(all_base)} Samples', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
```

---

## Analysis 3: Spatial IIG Map (Block 5 후, dramatic sample에만)

### 3.1 패치별 Information Gain

```python
def compute_spatial_iig(model, processor, image, question, candidate_tokens,
                        grid_h=14, grid_w=14):
    """
    각 이미지 패치를 마스킹했을 때 IIG가 얼마나 감소하는지.
    = 해당 패치가 답변에 제공한 정보량.
    
    비용: (grid_h × grid_w) 회 forward pass. 논문 figure용 5~10장에만 사용.
    """
    from vigil.rewards.iig import compute_iig  # 또는 직접 import
    
    base_iig = compute_iig(model, processor, image, question, candidate_tokens)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    patch_h, patch_w = h // grid_h, w // grid_w
    
    contributions = np.zeros((grid_h, grid_w))
    
    for i in range(grid_h):
        for j in range(grid_w):
            # 해당 패치를 black으로 마스킹
            masked = img_array.copy()
            y1, y2 = i * patch_h, (i + 1) * patch_h
            x1, x2 = j * patch_w, (j + 1) * patch_w
            masked[y1:y2, x1:x2] = 0
            masked_image = Image.fromarray(masked)
            
            masked_iig = compute_iig(model, processor, masked_image, question, candidate_tokens)
            
            # 이 패치가 빠지면 IIG가 얼마나 줄어드는가
            contributions[i, j] = base_iig - masked_iig
    
    return contributions, base_iig


def plot_spatial_iig_comparison(image, spatial_base, spatial_iig, 
                                 base_iig_total, iig_iig_total,
                                 question, save_path):
    """Baseline vs IIG-trained의 spatial information gain 비교."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_array = np.array(image)
    
    # Resize spatial maps to image size
    def resize_map(smap):
        return np.array(
            Image.fromarray(smap.astype(np.float32)).resize(
                (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
            )
        )
    
    vmax = max(spatial_base.max(), spatial_iig.max())
    
    # (a) Baseline spatial IIG
    axes[0].imshow(img_array, alpha=0.4)
    im0 = axes[0].imshow(resize_map(spatial_base), cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
    axes[0].set_title(f'Baseline\nTotal IIG: {base_iig_total:.2f}', fontsize=11)
    axes[0].axis('off')
    
    # (b) IIG-trained spatial IIG
    axes[1].imshow(img_array, alpha=0.4)
    im1 = axes[1].imshow(resize_map(spatial_iig), cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
    axes[1].set_title(f'IIG-trained\nTotal IIG: {iig_iig_total:.2f}', fontsize=11)
    axes[1].axis('off')
    
    # (c) Difference
    diff = spatial_iig - spatial_base
    axes[2].imshow(img_array, alpha=0.3)
    im2 = axes[2].imshow(resize_map(diff), cmap='RdBu_r', alpha=0.7,
                          vmin=-vmax/2, vmax=vmax/2)
    axes[2].set_title('Δ Information Gain\n(red=IIG↑)', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.suptitle(f'Q: {question[:80]}...', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
```

---

## 실행 타이밍 — 어떤 Block 후에 어떤 Analysis를 돌리는가

```
Block 1 (50-step) 완료 후:
  → Analysis 1만 실행: 5개 dramatic sample에 대한 4-panel heatmap
  → 목적: IIG가 실제로 attention을 바꾸는지 빠르게 확인 (sanity check)
  → 시간: ~10분

Block 2 (200-step) 완료 후:
  → Analysis 1: 10개 dramatic sample heatmap (both_correct 5 + flipped 5)
  → 목적: 논문 figure 후보 생성
  → 시간: ~20분

Block 3 (Thinking mode) 완료 후:
  → Analysis 1: thinking mode sample 5개
  → Analysis 2: temporal attention 곡선 (50개 sample 평균)  ← 핵심 figure
  → 목적: vision drift 시각화
  → 시간: ~40분 (temporal은 generation + attention 추출이 오래 걸림)

Block 5 (Post-training) 완료 후:
  → Analysis 3: dramatic sample 5개에 대한 spatial IIG map
  → 목적: "어디를 봄으로써 답이 달라지는가"의 직관적 증거
  → 시간: ~30분 (패치별 forward pass)

모든 Analysis 완료 후:
  → 최종 figure 선별: 논문에 들어갈 3~4장 선택
  → lab/figures/ 디렉토리에 고해상도 저장
  → RESEARCH_JOURNAL.md에 figure 설명 추가
```

---

## 저장 구조

```
lab/reports/visual_analysis/
├── block1_sanity/
│   ├── dramatic_samples.json         # 선택된 sample 목록 + metadata
│   ├── heatmap_sample_{N}.png        # 4-panel heatmap
│   └── summary.md                    # "attention이 이동했는가?" 한 줄 판단
│
├── block2_heatmaps/
│   ├── both_correct/
│   │   └── heatmap_{N}.png
│   ├── flipped/
│   │   └── heatmap_{N}.png
│   ├── dramatic_ranking.json
│   └── summary.md
│
├── block3_temporal/
│   ├── temporal_attention_averaged.png    ← 핵심 figure 후보
│   ├── temporal_individual_{N}.png
│   ├── drift_stats.json                  # decay % 비교
│   └── summary.md
│
├── block5_spatial_iig/
│   ├── spatial_iig_{N}.png
│   └── summary.md
│
└── final_figures/
    ├── fig_4panel_best.png
    ├── fig_temporal_drift.png
    ├── fig_spatial_iig.png
    └── figure_selection.md               # 왜 이 figure들을 선택했는가
```

---

## 주의사항

1. **`output_attentions=True`는 메모리를 많이 먹는다.** 모든 layer의 attention matrix를 저장하므로, long sequence에서는 OOM 가능. 메모리 부족 시:
   - 마지막 4개 layer만 사용 (이미 코드에 반영)
   - Batch size 1 유지
   - `torch.cuda.empty_cache()` 각 sample 후 호출

2. **`find_visual_token_range()`는 모델별로 다르다.** Qwen3-VL을 로드한 후 `processor`와 `model.config`를 확인하여 구현하라. 보통 `input_ids`에서 특수 토큰 (`<|vision_start|>`, `<|vision_end|>` 등)의 위치를 찾으면 된다.

3. **Spatial IIG (Analysis 3)는 비용이 높다.** 14×14 = 196회 forward pass per sample. 반드시 dramatic sample 5~10개에만 적용하라. 전체 eval set에 적용하지 마라.

4. **Temporal attention (Analysis 2)은 generation이 필요하다.** Teacher-forcing이 아닌 실제 generation을 해야 자연스러운 thinking chain이 나온다. `temperature=0.0` (greedy)으로 재현 가능하게.

5. **Heatmap 해석 주의**: attention이 "올바른 영역"에 있다는 것은 사람의 판단이다. 정량적 근거를 추가하라: 정답과 관련된 bounding box (있으면)와 attention hotspot의 overlap(IoU)을 계산.

---

## Ralph Loop Prompt (기존 Phase에 추가)

각 Block 완료 후 다음 ralph-loop를 추가로 실행:

```
/ralph-loop "
{Block N}의 학습된 모델 checkpoint를 로드하라.
Baseline 모델과 IIG-trained 모델을 모두 로드.

INSTRUCTION_VISUAL.md의 Analysis {N}을 수행하라:
1. find_dramatic_samples로 극적인 sample 선택
2. 해당 sample에 대해 heatmap/temporal/spatial 분석 생성
3. lab/reports/visual_analysis/ 에 저장
4. summary.md에 핵심 발견 기록

output_attentions=True에서 OOM 발생 시: layer 수를 줄이거나 sequence를 truncate.
find_visual_token_range()가 NotImplementedError이면: 모델의 special token을 조사하여 구현.

<promise>VISUAL_ANALYSIS_{N}_DONE</promise>
" --max-iterations 10 --completion-promise "VISUAL_ANALYSIS_{N}_DONE"
```
