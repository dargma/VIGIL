# VIGIL Paper Collection

Papers relevant to the VIGIL (Vision-Grounded Inference via Guided head-Level steering) project.

## Downloaded Papers

### 1. VISTA — Visual Information Steering
- **File**: `VISTA.pdf`
- **Title**: The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering
- **Authors**: Zhuowei Li, Haizhou Shi, Yunhe Gao, Di Liu, Zhenting Wang, Yuxiao Chen, Ting Liu, Long Zhao, Hao Wang, Dimitris N. Metaxas
- **URL**: https://arxiv.org/abs/2502.03628
- **Venue**: ICML 2025
- **Relevance**: Core competitor. Training-free inference-time visual steering via activation space intervention. VIGIL's Stage A is directly inspired by and extends VISTA's approach with agreement-gated head-level steering.

### 2. RLHF-V
- **File**: `RLHF-V.pdf`
- **Title**: RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback
- **Authors**: Tianqi Yu et al.
- **URL**: https://arxiv.org/abs/2312.00849
- **Venue**: CVPR 2024
- **Relevance**: Establishes RLHF for VLMs with segment-level dense DPO. Reduces hallucination by 34.8% with only 1.4K samples. Baseline for comparing VIGIL's RL approach.

### 3. LLaVA-RLHF
- **File**: `LLaVA-RLHF.pdf`
- **Title**: Aligning Large Multimodal Models with Factually Augmented RLHF
- **Authors**: Zhiqing Sun et al.
- **URL**: https://arxiv.org/abs/2309.14525
- **Venue**: ACL 2024 Findings
- **Relevance**: First LMM trained with RLHF. Introduces factually augmented reward to address reward hacking in multimodal RLHF. Key prior work for VIGIL's visual grounding reward design.

### 4. DeepSeekMath / GRPO
- **File**: `DeepSeekMath-GRPO.pdf`
- **Title**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **Authors**: Zhihong Shao et al. (DeepSeek)
- **URL**: https://arxiv.org/abs/2402.03300
- **Relevance**: Introduces Group Relative Policy Optimization (GRPO), the RL algorithm used in VIGIL's Stage B. Eliminates critic model via group-based advantage estimation.

### 5. DAPO
- **File**: `DAPO.pdf`
- **Title**: DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **Authors**: ByteDance Seed & Tsinghua AIR
- **URL**: https://arxiv.org/abs/2503.14476
- **Relevance**: Improves on GRPO with asymmetric clipping, no KL penalty, dynamic sampling, and overlong reward shaping. VIGIL implements DAPO as a Stage B alternative.

### 6. Qwen3-VL
- **File**: `Qwen3-VL.pdf`
- **Title**: Qwen3-VL Technical Report
- **Authors**: Qwen Team
- **URL**: https://arxiv.org/abs/2511.21631
- **Relevance**: Primary target model for VIGIL (Qwen3-VL-2B). Documents DeepStack integration, interleaved MRoPE, and architecture details needed for hook placement.

### 7. InternVL 3.5
- **File**: `InternVL3.5.pdf`
- **Title**: InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency
- **Authors**: OpenGVLab
- **URL**: https://arxiv.org/abs/2508.18265
- **Relevance**: Second target model for VIGIL (InternVL3.5-1B). Cascade RL framework and Visual Resolution Router provide comparison points.

### 8. ReST
- **File**: `ReST.pdf`
- **Title**: Reinforced Self-Training (ReST) for Language Modeling
- **Authors**: Caglar Gulcehre, Tom Le Paine et al. (DeepMind)
- **URL**: https://arxiv.org/abs/2308.08998
- **Relevance**: Theoretical foundation for VIGIL's BoN+SFT approach (Block 2 breakthrough). Offline RL via iterative Grow/Improve loops. VIGIL's best result uses this paradigm.

### 9. RAFT
- **File**: `RAFT.pdf`
- **Title**: RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment
- **Authors**: Hanze Dong et al.
- **URL**: https://arxiv.org/abs/2304.06767
- **Relevance**: Complementary to ReST. Selects high-quality samples via reward model, then fine-tunes. Direct algorithmic basis for VIGIL's BoN+SFT pipeline.

### 10. Head Pursuit (Attention Head Probing)
- **File**: `HeadPursuit.pdf`
- **Title**: Head Pursuit: Probing Attention Specialization in Multimodal Transformers
- **Authors**: Lorenzo Basile, Valentino Maiorca, Diego Doimo, Francesco Locatello, Alberto Cazzaniga
- **URL**: https://arxiv.org/abs/2510.21518
- **Venue**: NeurIPS 2025
- **Relevance**: Methods for identifying specialized attention heads in multimodal transformers. Directly relevant to VIGIL's Cohen's d calibration for vision head identification.

### 11. Visual Forgetting / TVC (Visual Attention Drift)
- **File**: `VisualForgetting-TVC.pdf`
- **Title**: Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning
- **Authors**: Hai-Long Sun, Zhun Sun, Houwen Peng, Han-Jia Ye
- **URL**: https://arxiv.org/abs/2503.13360
- **Venue**: ACL 2025
- **Relevance**: Directly validates VIGIL's core thesis of visual attention drift. Shows ~2% accuracy drop when removing image mid-reasoning, proving models become "blind reasoners". TVC is a complementary solution to VIGIL's head-level steering.

### 12. DMAS (Dynamic Multimodal Activation Steering)
- **File**: `DMAS.pdf`
- **Title**: Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models
- **Authors**: Jianghao Yin, Qin Chen, Kedi Chen, Jie Zhou, Xingjiao Wu, Liang He
- **URL**: https://arxiv.org/abs/2602.21704
- **Relevance**: Training-free dynamic activation steering for VLMs. Competitor to VIGIL Stage A. Uses input-specific steering vectors rather than uniform ones. VIGIL differentiates with agreement gating and RL permanence.

## Papers Not Found

### DVRP — Dynamic Visual Re-Prompting (2026)
- **Status**: Not found on arxiv or other repositories. May be a planned/unpublished paper, or the title/year may be incorrect.

## Summary by Topic

| Topic | Papers |
|-------|--------|
| Visual Steering / Activation Intervention | VISTA, DMAS |
| Visual Attention Drift | VisualForgetting-TVC |
| Attention Head Analysis | HeadPursuit |
| RL for VLMs | RLHF-V, LLaVA-RLHF |
| RL Algorithms (GRPO/DAPO) | DeepSeekMath-GRPO, DAPO |
| Best-of-N / Offline RL | ReST, RAFT |
| Target Model Architecture | Qwen3-VL, InternVL3.5 |
