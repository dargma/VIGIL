# Block 2 Creative Ideas: Novel Approaches to Visual Grounding via RL

**Date**: 2026-03-08
**Context**: Block 1 GRPO collapsed on binary VQA (3/3 attempts). Block 2 custom GRPO achieved stability but no improvement (+/-1.5pp oscillation). Existing ideation covers soft rewards, R_vhad training, BoN+SFT, KTO, curriculum, steering-augmented generation, SimPO, and group size scaling. This document explores 15 fundamentally different approaches not yet covered.

---

## Idea 1: Contrastive Visual Training with Image Pairs

**Description**: Train on paired samples where two visually similar but semantically different images are shown (e.g., "cat on sofa" vs "dog on sofa"), and the correct answer depends on distinguishing them. During GRPO/BoN, present both images for the same question; candidates that give the same answer for both images receive zero R_vhad (they are not looking). Candidates that correctly differentiate receive maximum reward. This forces the model to attend to fine-grained visual features rather than relying on text priors.

**Why novel**: Existing VQA training treats each (image, question) pair independently. No prior work uses image-pair contrastive training as an RL reward signal for visual grounding. DVRP (2026) uses external perturbation of the question, not the image. VIGIL's existing R_vhad uses real vs black (total absence), but image-pair contrastive uses real vs similar-real, which is a much harder and more informative signal. The closest prior work is counterfactual VQA (Niu et al., 2021), but that operates on dataset curation, not RL reward design.

**Implementation complexity**: 3/5. Requires generating or sourcing visually similar image pairs. COCO has multiple images per category which can serve as natural pairs. Need a pairing heuristic (same object category, different instance) and a dual-forward-pass reward function. The GRPO/BoN loop itself is unchanged.

**Expected impact on Blind Test Gap**: +4-6pp. This directly teaches the model that "looking at the image matters for getting the right answer" in a much stronger way than real-vs-black, because the model cannot distinguish similar images without genuine visual processing.

**GPU hours estimate**: 8-12 hours (image pair mining: 1h CPU, generation + scoring: 6-10h, SFT: 1h).

---

## Idea 2: Self-Play Visual Grounding

**Description**: The model plays two roles: (1) Questioner -- given an image, generate a question whose answer requires looking at the image; (2) Answerer -- answer the question with and without the image. The reward is the product of question quality (answer changes when image is removed) and answer correctness (verified against the self-generated expected answer or a reference). This creates an automatic curriculum where the model learns to ask increasingly vision-dependent questions and then learns to answer them.

**Why novel**: Self-play has been used in code generation (AlphaCode) and math (AlphaProof) but never for visual grounding in VLMs. The key insight is that the Questioner role naturally generates training data with high vision salience (because the reward penalizes questions answerable without looking), which solves the curriculum problem identified in the existing ideation. No external dataset curation is needed. This is also distinct from VQG (Visual Question Generation) literature which generates questions for data augmentation, not as an RL self-play loop.

**Implementation complexity**: 4/5. Requires two-phase generation per training step (question generation, then answer generation x2). Need to handle the Questioner reward (is the question answerable? does the answer change without image?) and the Answerer reward (is the answer correct?). The verification of question quality requires an extra generation without image.

**Expected impact on Blind Test Gap**: +5-8pp. Self-play generates maximally vision-dependent training signal by construction. The Questioner learns to probe the model's visual weaknesses, creating an adversarial curriculum.

**GPU hours estimate**: 15-20 hours (double generation cost per sample, 2-3 rounds of self-play iteration).

---

## Idea 3: Progressive Image Masking Curriculum

**Description**: Start training with full images, then progressively mask random regions (25%, 50%, 75%) while keeping the question and expected answer the same. The reward includes a masking-robustness bonus: if the model maintains the correct answer with 25% of the image masked, it gets a bonus; if it fails, R_vhad is weighted higher to force it to attend to the remaining visible regions. The masking percentage increases over training, forcing the model to extract information from increasingly sparse visual input. In the final phase, unmask fully and evaluate -- the model should now attend to visual features more efficiently.

**Why novel**: Image masking is used in MAE (Masked Autoencoders) for pretraining vision encoders but has never been applied as a curriculum strategy for RL-based visual grounding in VLMs. The key novelty is using masking as a training difficulty knob that forces the model to develop robust visual attention patterns. VIGIL's existing approach uses binary image presence (real vs black); progressive masking is a continuous interpolation between the two extremes.

**Implementation complexity**: 2/5. Image masking is trivial (random patch masking or grid masking on pixel space before vision encoder). The curriculum scheduler is a simple step function. No changes to the RL algorithm itself.

**Expected impact on Blind Test Gap**: +3-5pp. Models trained with masking should develop more robust visual features, making them more sensitive to image content and less likely to default to text-only reasoning.

**GPU hours estimate**: 6-8 hours (same as standard BoN+SFT, masking adds negligible CPU cost).

---

## Idea 4: Adversarial Image Perturbation for Robustness

**Description**: During training, apply imperceptible perturbations (Gaussian noise sigma=0.01-0.05, JPEG compression, color jitter, spatial transforms) to the input image. The reward has two components: (1) consistency bonus -- the model gives the same answer for the original and perturbed image (proving it extracts semantic features, not pixel artifacts); (2) sensitivity bonus -- the model gives different answers for the original vs a black/irrelevant image (proving it actually uses the image). Models that are both consistent-under-perturbation AND sensitive-to-image-removal are maximally visually grounded.

**Why novel**: Adversarial robustness in VLMs has been studied for safety (jailbreak resistance) but not as a visual grounding training signal. The dual objective (robust to noise, sensitive to image removal) is novel. This is the visual analogue of paraphrase consistency in NLP, applied at the image level. DVRP uses textual perturbation; this uses visual perturbation, which is more natural for vision grounding.

**Implementation complexity**: 2/5. Standard torchvision transforms for perturbation. Two extra forward passes per candidate (perturbed image + black image). The consistency and sensitivity rewards are simple.

**Expected impact on Blind Test Gap**: +2-4pp. Robustness training prevents the model from developing brittle visual features that work on clean images but fail on slightly different inputs.

**GPU hours estimate**: 8-10 hours (2x forward passes for perturbation scoring, otherwise standard).

---

## Idea 5: Cross-Modal Consistency Loss via Caption Grounding

**Description**: For each training image, generate (or retrieve) a caption using a separate captioning model (or use COCO captions). During RL training, add a cross-modal consistency reward: the generated answer should be more semantically consistent with the image caption when the image is present than when it is absent. Compute this as cosine similarity between the answer embedding and caption embedding (using a frozen sentence encoder like all-MiniLM-L6-v2). R_consistency = sim(answer_with_image, caption) - sim(answer_without_image, caption).

**Why novel**: Caption-grounded rewards have not been used in VLM RL training. The closest is CLIP-based reward in image generation (DALL-E), but that operates in the image->text direction. This is text->text consistency mediated by the image, which is a fundamentally different signal. It does not require any visual encoder evaluation -- just text similarity -- making it lightweight. The reward is continuous and provides rich gradient signal even for binary VQA (the confidence and phrasing change with visual grounding).

**Implementation complexity**: 2/5. Requires a frozen sentence encoder (tiny, ~25MB) and pre-computed captions (COCO already has them). Two generations per candidate (with and without image), two embedding lookups, one cosine similarity. No changes to the RL loop.

**Expected impact on Blind Test Gap**: +3-5pp. This reward directly measures whether the image changes the model's reasoning, which is exactly what the Blind Test Gap measures.

**GPU hours estimate**: 6-8 hours (sentence encoding is negligible; main cost is the extra generation without image, but this is already done for R_asi/IIG).

---

## Idea 6: Vision Head Distillation from 7B Teacher

**Description**: Use Qwen2.5-VL-7B (the 7B reference model) as a teacher. For each training sample, record the 7B model's vision head activation patterns (which heads fire, with what magnitude, at which token positions). Then add a distillation reward to the 2B student: R_distill = -MSE(student_vision_activations, teacher_vision_activations) after alignment (project student heads to teacher head space via a small learned linear map). The 2B model learns to mimic the 7B model's visual attention patterns, which are presumably higher quality.

**Why novel**: Knowledge distillation for VLMs exists (LLaVA-distill, TinyGPT-V) but operates on output logits or hidden states. Distilling specifically the vision head activation pattern is novel. The insight is that the 7B model does not have the visual attention drift problem (it has enough capacity), so its attention patterns represent "ideal" visual grounding that the 2B model should aspire to. This sidesteps the problem of defining what "good visual grounding" looks like -- the 7B model defines it implicitly.

**Implementation complexity**: 4/5. Requires loading both 7B and 2B models (7B in 8-bit: ~8GB, 2B in fp16: ~4.3GB, total ~12GB, fits on A100 40GB). Need a head alignment layer (small MLP to map 2B head activations to 7B head space). Pre-compute 7B activations offline to avoid running both models simultaneously during training.

**Expected impact on Blind Test Gap**: +4-7pp. If the 7B model's vision patterns are indeed better (likely, given its higher accuracy), distilling them directly into the 2B model should produce a large grounding improvement.

**GPU hours estimate**: 12-15 hours (7B activation extraction: 4h, alignment training: 2h, BoN+SFT with distill reward: 6-8h).

---

## Idea 7: Sparse Activation Reward (Efficiency-Grounded Training)

**Description**: Instead of maximizing vision head activation (R_vhad), reward the model for achieving correct answers while using the minimum necessary vision head activation. R_sparse = R_correct * (1 / (1 + sum(vision_head_activations))). This encourages the model to develop efficient, targeted visual attention rather than diffuse activation across all vision heads. The hypothesis is that focused attention on the most relevant visual features produces more robust grounding than high activation everywhere.

**Why novel**: All prior visual grounding work (including VIGIL's R_vhad) implicitly assumes "more visual activation = better." This inverts the assumption: efficient, targeted visual attention is better than noisy, high-magnitude activation. This is inspired by the sparse attention literature (Longformer, BigBird) but applied to the reward function rather than the architecture. It also connects to the neuroscience concept of "attentional efficiency" -- experts use less neural activation than novices for the same task.

**Implementation complexity**: 2/5. Simple modification of R_vhad computation. Gate on R_correct to avoid rewarding blindness.

**Expected impact on Blind Test Gap**: +1-3pp (conservative). The primary benefit is efficiency rather than raw grounding. However, sparse attention may generalize better to unseen images, producing indirect Gap improvement.

**GPU hours estimate**: Same as R_vhad training (6-8 hours). No extra compute.

---

## Idea 8: Temporal Vision Attention for Video VLMs

**Description**: Extend VIGIL to video understanding by tracking vision head activation across video frames. For a video QA task, measure which frames receive the most vision head activation and whether this correlates with the frames containing the answer-relevant content. R_temporal = correlation(vision_activation_per_frame, frame_relevance). Frame relevance can be approximated by CLIP similarity between the frame and the question, or by human annotations.

**Why novel**: VIGIL is currently image-only. Video VLMs suffer from an even more severe form of visual attention drift because the temporal dimension adds a second axis of potential neglect (the model may attend to the first or last frame but ignore the middle). No prior work applies head-level steering to video VLMs. This extends VIGIL's contribution to a modality where the problem is arguably more severe and commercially relevant (video understanding is a major industry focus).

**Implementation complexity**: 5/5. Requires a video-capable VLM (e.g., Qwen2-VL or LLaVA-Video), video datasets (ActivityNet-QA, NExT-QA), per-frame activation tracking, and a temporal reward function. Significant engineering effort.

**Expected impact on Blind Test Gap**: N/A for image benchmarks. For video QA benchmarks, potentially +5-10pp as temporal attention drift is even worse than spatial.

**GPU hours estimate**: 30-40 hours (video models are larger, datasets require more processing).

---

## Idea 9: Multi-Image Comparative Reasoning

**Description**: Train on tasks that require comparing two or more images to answer a question (e.g., "Which image shows a larger dog?", "What is different between these two photos?"). The reward structure requires the model to attend to BOTH images (measured by R_vhad on each image independently). This forces the model to develop robust visual comparison abilities that cannot be faked by text-only reasoning. Use Spot-the-Difference datasets, NLVR2, or synthetically paired COCO images.

**Why novel**: Multi-image tasks are the strongest test of visual grounding because text-only reasoning is provably insufficient (the model must look at both images to compare them). Existing VLM RL training uses single images exclusively. NLVR2 exists but has not been used as an RL training signal for visual grounding. The key insight is that multi-image comparison is a natural adversarial task for blind reasoners -- they literally cannot answer without looking.

**Implementation complexity**: 3/5. Requires multi-image input support (Qwen3-VL already supports this). Need multi-image datasets (NLVR2, or synthetic pairs from COCO). The RL loop needs modification to handle multi-image inputs and compute R_vhad per image.

**Expected impact on Blind Test Gap**: +5-8pp. Multi-image training is the strongest possible forcing function for visual grounding.

**GPU hours estimate**: 10-14 hours (multi-image generation is ~2x single-image, scoring is ~3x due to per-image R_vhad).

---

## Idea 10: Synthetic Hard Negatives via Image Editing

**Description**: For each training sample, use an image editing model (InstructPix2Pix, or simpler transforms like object removal/replacement) to create a "hard negative" image where the correct answer changes. For "Is there a cat in the image?" with answer "yes," create an edited image where the cat is removed. During RL, present both original and edited images; the model must give different answers for the two. R_hardneg = 1.0 if model answers correctly for both, 0.5 if correct for original only, 0.0 if same answer for both (blind).

**Why novel**: Synthetic hard negatives are used in contrastive learning for vision (SimCLR, MoCo) but not as RL training signals for VLM grounding. The closest work is counterfactual image editing for VQA explanation (Chen et al., 2022), but that was for interpretability, not training. The key novelty is using image editing as a reward construction tool: the edited image creates a contrastive pair where the correct answer is known by construction.

**Implementation complexity**: 4/5. Requires an image editing pipeline (InstructPix2Pix or object segmentation + removal). Quality control is needed to ensure edits are semantically valid. The RL integration is straightforward once the pairs exist.

**Expected impact on Blind Test Gap**: +4-7pp. Hard negatives are the gold standard for contrastive learning; applying them to visual grounding should produce strong results.

**GPU hours estimate**: 15-20 hours (image editing: 5-8h on GPU, training: 8-10h, eval: 2h).

---

## Idea 11: Layer-Wise Steering Annealing During Training

**Description**: Instead of a fixed set of steered heads throughout training, anneal the steering targets from late layers (feature heads) to early layers (decision heads) over the course of training. The hypothesis is that the model first needs to learn to extract visual features (late layers), then learns to use them for decisions (early layers). Start by steering only layers 24-27 (feature heads), then at step 30 add layers 14-20 (mid layers), then at step 60 add layers 4-6 (decision heads), then remove all steering by step 80.

**Why novel**: Existing steering approaches (VISTA, VIGIL) use a fixed set of heads throughout. Layer-wise annealing is inspired by the "layer-wise learning rate" technique in fine-tuning but applied to steering targets rather than optimization parameters. The connection to the two-types-of-heads finding (Idea 1 in RESEARCH_IDEAS.md) makes this a natural extension: steer features first, then decisions.

**Implementation complexity**: 2/5. Simple scheduler that modifies the steered head set over training steps. No algorithmic changes.

**Expected impact on Blind Test Gap**: +2-4pp. Better initialization of visual feature extraction should cascade into better decision-making.

**GPU hours estimate**: Same as standard steering-augmented training (8-12 hours). No extra compute.

---

## Idea 12: Attention Entropy Regularization

**Description**: During training, compute the entropy of the attention distribution over visual tokens at each vision head. Reward models that maintain high attention entropy over visual tokens (attending broadly to the image) rather than collapsing to a single patch or ignoring all patches. R_entropy = entropy(attention_weights_on_visual_tokens) / max_entropy. This prevents the "tunnel vision" failure mode where the model fixates on one patch and the "blind" failure mode where attention to visual tokens approaches uniform-over-all (including text tokens).

**Why novel**: Attention entropy has been studied analytically in transformers (Merrill et al., 2022) but never used as an explicit RL reward for visual grounding. The insight is that healthy visual attention has moderate entropy (attending to relevant image regions, not too focused, not too diffuse), and this can be directly optimized. This is distinct from R_vhad which measures magnitude, not distribution.

**Implementation complexity**: 3/5. Requires extracting attention weights (not just pre-o_proj activations) during training. Qwen3-VL can output attention weights with output_attentions=True, but this increases memory. May need to compute on a subset of layers.

**Expected impact on Blind Test Gap**: +2-4pp. Healthier attention distributions should correlate with more robust visual grounding.

**GPU hours estimate**: 8-10 hours (attention weight extraction adds ~30% memory overhead, may require gradient checkpointing).

---

## Idea 13: Grounding via Spatial Referring Expressions

**Description**: Add a spatial grounding auxiliary task during RL training. For a subset of training samples with bounding box annotations (GQA has these), require the model to also output the image region relevant to its answer (e.g., "The answer is 'red car,' located in the bottom-left quadrant"). Reward R_spatial = IoU(predicted_region, ground_truth_bbox) if the answer also references a location, else 0. This forces the model to not just attend to the image but to explicitly reason about spatial locations, which requires genuine visual processing.

**Why novel**: Spatial grounding has been studied in the referring expression comprehension (REC) literature but never integrated as an RL reward for preventing visual attention drift. The insight is that spatial reasoning is impossible without looking at the image, making it a strong anti-blind-reasoner signal. Unlike R_vhad (which measures hidden activations), R_spatial measures explicit spatial reasoning in the output, which is directly interpretable.

**Implementation complexity**: 3/5. Requires bounding box annotations (GQA has them). Need a simple parser for spatial references in model output. The reward is straightforward (IoU or quadrant matching for coarse grounding). Need to modify the prompt template to request spatial output.

**Expected impact on Blind Test Gap**: +3-6pp. Spatial reasoning is one of the hardest things for blind reasoners to fake.

**GPU hours estimate**: 8-10 hours (GQA bbox data is already downloaded, prompt modification is simple).

---

## Idea 14: Hindsight Vision Replay

**Description**: Inspired by Hindsight Experience Replay (HER) from robotics RL. When a model generates an incorrect answer, re-label the training sample: instead of the original question, create a new question for which the model's incorrect answer would be correct (if one exists for this image). Then check whether the model's vision head activations were consistent with this new question. If yes, the model was visually grounded but answered the wrong question; reward the grounding but penalize the answer. If no, the model was truly blind; penalize both. This provides a denser reward signal by distinguishing "grounded but confused" from "blind."

**Why novel**: HER has never been applied to VLM training. The key insight is that wrong answers in VQA are often not random -- they reflect genuine (but misapplied) visual processing. A model that says "blue" when the car is red is still looking at the car; a model that says "blue" regardless of the image is blind. Current rewards (R_correct, R_vhad) cannot distinguish these cases; HER can.

**Implementation complexity**: 4/5. Requires a question re-labeling mechanism (either template-based for simple questions or LLM-based for complex ones). The hindsight check (were activations consistent with the new question?) needs a way to evaluate consistency, possibly via CLIP or a separate answer verification step.

**Expected impact on Blind Test Gap**: +2-4pp. Denser reward signal should improve training efficiency and prevent discarding partially-grounded behaviors.

**GPU hours estimate**: 10-15 hours (question re-labeling: 2-3h, modified training loop: 8-10h).

---

## Idea 15: Token-Position-Aware R_vhad (Drift-Weighted Reward)

**Description**: Current R_vhad averages vision head activation across all generated token positions. But visual attention drift means early tokens have high activation and late tokens have low activation. Instead of averaging, weight R_vhad by token position: later tokens receive exponentially higher weight. This means the reward cares most about whether the model is still using the image at the END of its response, not just the beginning. R_drift_vhad = sum(exp(gamma * t/T) * vhad_t) / sum(exp(gamma * t/T)) where t is token position, T is sequence length, and gamma controls the exponential weighting.

**Why novel**: This connects the reward design directly to the visual attention drift problem statement. R_vhad treats all token positions equally, but the thesis argues that drift is the core problem -- attention is fine at the start and decays. Weighting by position targets the decay directly. This is complementary to the drift penalty (Idea 3 in RESEARCH_IDEAS.md), which penalizes negative slope; drift-weighted R_vhad rewards maintaining high activation at late positions regardless of slope.

**Implementation complexity**: 1/5. Trivial modification to R_vhad computation. Change one line from mean() to weighted_mean().

**Expected impact on Blind Test Gap**: +2-3pp. Directly addresses the mechanism that causes Gap degradation.

**GPU hours estimate**: Same as standard R_vhad training (6-8 hours). Zero extra compute.

---

## Summary Table

| # | Idea | Complexity | Gap Impact | GPU Hours | Category |
|---|------|-----------|------------|-----------|----------|
| 1 | Contrastive image pairs | 3/5 | +4-6pp | 8-12h | Data augmentation |
| 2 | Self-play visual grounding | 4/5 | +5-8pp | 15-20h | Training paradigm |
| 3 | Progressive image masking | 2/5 | +3-5pp | 6-8h | Curriculum |
| 4 | Adversarial perturbation | 2/5 | +2-4pp | 8-10h | Robustness |
| 5 | Cross-modal caption consistency | 2/5 | +3-5pp | 6-8h | Reward design |
| 6 | Vision head distillation from 7B | 4/5 | +4-7pp | 12-15h | Distillation |
| 7 | Sparse activation reward | 2/5 | +1-3pp | 6-8h | Reward design |
| 8 | Temporal vision attention (video) | 5/5 | N/A (video) | 30-40h | Extension |
| 9 | Multi-image comparative reasoning | 3/5 | +5-8pp | 10-14h | Data augmentation |
| 10 | Synthetic hard negatives | 4/5 | +4-7pp | 15-20h | Data augmentation |
| 11 | Layer-wise steering annealing | 2/5 | +2-4pp | 8-12h | Steering |
| 12 | Attention entropy regularization | 3/5 | +2-4pp | 8-10h | Reward design |
| 13 | Spatial referring expressions | 3/5 | +3-6pp | 8-10h | Auxiliary task |
| 14 | Hindsight vision replay | 4/5 | +2-4pp | 10-15h | Training paradigm |
| 15 | Token-position-aware R_vhad | 1/5 | +2-3pp | 6-8h | Reward design |

---

## Recommended Priority

### Tier 1: Low-hanging fruit (implement in next session)
- **Idea 15** (drift-weighted R_vhad): 1 line change, directly addresses thesis, composable with everything
- **Idea 3** (progressive masking): trivial implementation, strong curriculum signal
- **Idea 5** (caption consistency): cheap, continuous reward, reuses existing dual-forward infrastructure

### Tier 2: High expected impact (implement after Tier 1 results)
- **Idea 9** (multi-image reasoning): strongest forcing function for visual grounding
- **Idea 1** (contrastive image pairs): strong contrastive signal, natural data exists in COCO
- **Idea 6** (7B distillation): large expected Gap improvement, leverages existing 7B baseline

### Tier 3: Novel contributions (implement for paper differentiation)
- **Idea 2** (self-play): most novel, strongest paper narrative, but high implementation cost
- **Idea 14** (hindsight replay): novel connection to robotics RL, addresses dense reward problem
- **Idea 13** (spatial grounding): leverages GQA bbox data, impossible to fake without vision

### Tier 4: Extensions and refinements
- **Idea 4** (adversarial perturbation): solid but incremental
- **Idea 11** (layer-wise annealing): builds on two-types finding
- **Idea 12** (entropy regularization): complements R_vhad with distributional signal
- **Idea 7** (sparse reward): counterintuitive but may aid generalization
- **Idea 8** (video): future work section of paper

---

## Combinatorial Power Plays

The ideas above are not mutually exclusive. The most powerful combinations:

1. **15 + 5 + 3**: Drift-weighted R_vhad + caption consistency + progressive masking. Three complementary reward/curriculum signals, all cheap (complexity 1-2), expected combined Gap: +6-10pp.

2. **6 + 1 + 15**: 7B distillation + contrastive pairs + drift-weighted R_vhad. Distillation sets the vision attention target, contrastive pairs provide hard training data, drift weighting ensures late-token grounding. Expected combined Gap: +7-12pp.

3. **2 + 9**: Self-play + multi-image reasoning. Self-play generates the questions, multi-image provides the hardest possible visual grounding task. This is the "nuclear option" that makes blind reasoning impossible. Expected combined Gap: +8-14pp but at 25-30 GPU hours.

4. **13 + 14**: Spatial grounding + hindsight replay. Spatial grounding provides the explicit visual reasoning signal, hindsight replay rescues partially-grounded failures. Good for paper narrative: "we not only detect visual grounding failures, we diagnose their type."
