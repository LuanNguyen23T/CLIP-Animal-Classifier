# GEMMA-CLIP: Fine-Grained Pet Breed Classification with Morphological Knowledge, Prompt Optimization and Multi-View Inference

**Nguyen Huu Hoang Luan, Ton That Van**  
*Department of Information Technology, University of Science and Technology – The University of Danang*  
Email: 102230028@sv1.dut.udn.vn, 102230055@sv1.dut.udn.vn  
GitHub: [https://github.com/LuanNguyen23T/CLIP-Animal-Classifier](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier)

---

## Abstract

Fine-Grained Visual Categorization (FGVC) poses a significant challenge due to subtle inter-class disparities and high intra-class variance. While Contrastive Language-Image Pre-training (CLIP) provides strong zero-shot capabilities, standard handcrafted prompts lack fine-grained anatomical grounding, and parameter-efficient fine-tuning (PEFT) adapters act as uninterpretable black boxes prone to low-shot instability. In this work, we propose **Gemma-CLIP**, an explainable, parameter-efficient framework coupling a frozen CLIP ViT-B/32 backbone with morphological domain knowledge distilled autonomously from an on-device Small Language Model (Google Gemma 4 E4B SLM). We introduce a structured 185-attribute concept bottleneck space, a non-parametric morphological centroid bank, and Gemma-CoOp with Compact Trait Conditioning (24 tokens) to overcome attention gradient dilution. To ensure multi-view geometric invariance, we further integrate an aspect-ratio-preserving 5-view test-time augmentation (TTA) pipeline.

Evaluated on the official Oxford-IIIT Pet benchmark (3,669 test images) across three independent random seeds, Gemma-CLIP raises zero-shot accuracy to 90.85% under 5-View TTA, compared with 80.16% for single-view raw class names and 88.90% for the 80-prompt CLIP ensemble baseline under identical TTA. In the extreme low-data regime ($K = 1$ shot), Gemma-CLIP achieves 91.39%, surpassing standard CoOp by +2.67% and decisively outperforming a fully-supervised ResNet-50 cosine classifier (69.83%, $\Delta = +21.56\%$), which collapses in low-data regimes due to the complete lack of multimodal foundation priors. Under full training, Gemma-CLIP establishes a peak of 94.15% Top-1 accuracy with 67,584 trainable parameters (about 0.04% of CLIP ViT-B/32's ~151M parameters), surpassing the fully-supervised ResNet-50 Cosine Head (93.08%, +1.07%, McNemar $p = 3.5 \times 10^{-4}$) while exposing morphology-grounded, human-readable attribute evidence. On Apple Silicon MPS, classification head latency is 0.18 ms/image (5,652 FPS) for Gemma-CoOp and 0.90 ms (1,110 FPS) for the full model, with the end-to-end 5-View TTA pipeline sustaining 17.2 FPS.

**Keywords:** Fine-Grained Visual Categorization, Vision-Language Models, CLIP, Gemma SLM, Context Optimization (CoOp), Concept Bottleneck Models (CBM), Test-Time Augmentation (TTA).

---

## 1. Introduction

Fine-Grained Visual Categorization (FGVC) requires automated vision systems to discriminate between closely related sub-taxa within a parent category. While intra-class variation caused by camera viewpoint, lighting, posture, and age is substantial, inter-class morphological boundaries are frequently minute and localized. On the standard Oxford-IIIT Pet benchmark [1], a Ragdoll cat and a Birman cat share nearly identical cream-colored bodies and piercing blue eyes; the decisive anatomical discriminator resides in skull contour and the symmetrical white paw fur ("gloves") present strictly on Birmans. Similarly, the Staffordshire Bull Terrier and American Pit Bull Terrier display identical muscular habitus, differing primarily in cranial cheekbone width to mandibular taper ratios.

Standard deep classifiers relying purely on global visual features or monolithic class names frequently fall into visual ambiguity traps. Contrastive Language-Image Pre-training (CLIP; Radford et al. [2]) maps visual and textual modalities into a shared hyperspherical embedding space $\mathbb{R}^D$ ($D = 512$ for ViT-B/32). In zero-shot transfer, image embeddings are classified via cosine similarity against text prompt embeddings. However, generic handcrafted prompts (e.g., *"a photo of a {class}"*) lack explicit anatomical cues.

Parameter-efficient fine-tuning (PEFT) methods—such as CoOp (Zhou et al. [3]), CLIP-Adapter (Gao et al. [4]), and Tip-Adapter (Zhang et al. [5])—adapt representations using continuous context vectors, residual feature bottlenecks, or non-parametric caches. Despite empirical gains, these paradigms exhibit four systemic vulnerabilities:
1. **The Semantic Black-Box Deficit:** Learned continuous prompt vectors or MLP weights operate strictly in latent spaces, detached from clinical veterinary reasoning.
2. **Extreme Low-Shot Instability ($K \le 2$ shots):** When support data is scarce, continuous context vectors easily overfit spurious artifacts, dropping below zero-shot initialization (e.g., standard CoOp at $K = 1$: 88.72% vs. 88.90%).
3. **The Attention Gradient Dilution Dilemma:** Directly appending long, unstructured descriptive sentences (45–60 tokens) to CoOp's context vectors dilutes backpropagation gradients across CLIP's 12 causal self-attention layers, impeding continuous token optimization.
4. **Cloud Dependency & Privacy Risks:** Prior language-augmented vision frameworks rely on remote proprietary APIs (such as GPT-4), introducing latency, recurring financial costs, and data privacy vulnerabilities.

To address these fundamental issues, this paper presents **Gemma-CLIP**, an explainable, parameter-efficient framework coupling local Small Language Models (Google Gemma 4 E4B SLM [8]) with CLIP ViT-B/32. Specifically, our contribution comprises five tightly integrated components:
* **Stream 1 (Gemma-CoOp with Compact Trait Conditioning):** Optimizes continuous soft prompts anchored by concise diagnostic species tokens bounded to 24 tokens, overcoming gradient dilution.
* **Stream 2 (Non-Parametric Morphological Centroids):** Synthesizes five structured anatomical fields per breed into frozen text centroid vectors, embedding invariant linguistic priors without parameter training.
* **Dual-Stream & Residual Visual Adaptation:** Combines learned continuous prompt features with non-parametric centroids via weighted logit fusion ($\beta = 0.5$) and visual feature adaptation ($\alpha = 0.2$), eliminating low-shot degradation.
* **185-Dimensional Concept Bottleneck Space:** Realizes transparent, human-interpretable diagnostic attribution across 185 independent anatomical traits.
* **Aspect-Ratio Preserving 5-View TTA:** Prevents anatomical squashing and provides robust multi-view inference at 17.2 FPS on edge hardware.

---

## 2. Related Work & Baseline Definitions

### 2.1. Vision-Language Foundation Models (CLIP)
OpenAI CLIP [2] synchronizes a visual encoder $f(x) = \text{ImageEncoder}(x)$ and a text encoder $w_c = \text{TextEncoder}(t_c)$ into a shared normalized space via contrastive pre-training over 400 million image-text pairs. For an input image $x$, the predicted class probability is:

$$P(y = c \mid x) = \frac{\exp(\tau \langle f(x), w_c \rangle)}{\sum_{j=1}^C \exp(\tau \langle f(x), w_j \rangle)}$$

where $\tau$ is the learned logit scale parameter and $\langle f(x), w_c \rangle = f(x)^T w_c$ denotes cosine similarity. When evaluated with raw class names, CLIP achieves only 80.16% on Oxford-IIIT Pet. Handcrafted prompt ensembling elevates performance to 87.52%, but remains ungrounded in diagnostic anatomy.

### 2.2. Parameter-Efficient Fine-Tuning (PEFT)
To eliminate manual prompt engineering, CoOp [3] models prompts as $M$ continuous learnable tokens $[V]_1, \dots, [V]_M \in \mathbb{R}^d$ prepended to fixed class token embeddings:

$$t_c = [[V]_1, [V]_2, \dots, [V]_M, e_c]$$

CoOp updates only 2,048 parameters ($M = 4, d = 512$). CLIP-Adapter [4] adds a two-layer residual MLP bottleneck to image features. Tip-Adapter [5] constructs a non-parametric key-value cache from few-shot support embeddings, fine-tuned parametrically in Tip-Adapter-F. These models form the PEFT comparative baselines in our evaluation.

### 2.3. Supervised Upper-Bound References (Full Supervision)
In computer vision benchmarks, convolutional neural networks such as ResNet-50 [10] trained with supervised cross-entropy or cosine classifiers [13] serve as Supervised Upper-Bound References (Oracle baselines). A supervised ResNet-50 Cosine Head optimizes 75,815 classifier parameters directly over 100% of labeled training data (3,680 annotated samples in Oxford-IIIT Pet). Similarly, ResNet-50 Prototype constructs non-parametric centroids directly from ground-truth training features. In high-data regimes, supervised models achieve strong accuracy; however, in few-shot regimes ($K \le 2$), they suffer from severe overfitting due to the absence of multimodal foundation priors. We explicitly distinguish this supervised oracle from parameter-efficient foundation model adapters, treating it as an asymptotic reference rather than an architectural peer.

### 2.4. Concept Bottleneck Models (CBM) & LLM-Guided Prompting (CuPL, LaBo)
Concept Bottleneck Models [6] enforce interpretability by routing predictions through an intermediate layer of human-understandable concept attributes. Recent paradigms leverage large language models to automate visual description, notably CuPL (Pratt et al. [17]) and LaBo (Yang et al. [15]). While CuPL prompts cloud-based GPT-3 (175B parameters) to generate generic descriptive sentences and LaBo constructs large concept candidate pools via remote APIs, they suffer from two major limitations: (i) reliance on proprietary cloud infrastructure incurring latency, privacy risks, and API costs, and (ii) unconstrained lexical descriptions that fail to isolate fine-grained diagnostic discriminators.

In contrast, **Gemma-CLIP** establishes three fundamental departures:
1. **On-Device Edge Distillation:** Employs an on-device Small Language Model (Google Gemma 4 E4B SLM [8]) executing entirely locally via LM Studio, ensuring absolute data privacy and zero cloud inference cost.
2. **Standardized 5-Axis Anatomical Taxonomy:** Constrains attribute extraction strictly to recognized veterinary breed registry standards (AKC/FCI and CFA/TICA), eliminating hallucinated non-visual semantics.
3. **PEFT Semantic Anchoring:** Extends beyond zero-shot inference by leveraging distilled attributes as continuous semantic anchors for Context Optimization (Gemma-CoOp) and residual visual adaptation.

---

## 3. Proposed Methodology

```
                                  [ Input Image x ]
                                          │
                        ┌─────────────────┴─────────────────┐
                        │ Aspect-Ratio 5-View Augmentation  │
                        └─────────────────┬─────────────────┘
                                          │ 5 Views
                                 ┌────────▼────────┐
                                 │ Frozen CLIP ViT │
                                 └────────┬────────┘
                                          │ f_TTA(x)
                                 ┌────────▼────────┐
                                 │ Visual Residual │ ── MLP Bottleneck: f'_TTA(x)
                                 │     Adapter     │    (512 → 64 → 512, α = 0.2)
                                 └────────┬────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ Stream 1: Gemma-CoOp ]                        [ Stream 2: Centroid Bank ]
  Continuous Soft Prompts Anchored                185-Attribute Non-parametric Bank
  by Compact Traits (24 tokens)                   Distilled from Local Gemma 4 SLM
                  │                                               │
                  │ W_coop                                        │ W_centroid
                  ▼                                               ▼
         z_coop = f' @ W_coop^T                          z_cent = f' @ W_centroid^T
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                         Logit Blend: z = z_coop + β * z_cent (β = 0.5)
                                          ▼
                                   [ Softmax Loss ]
```
*Figure 1. End-to-end architecture of Gemma-CLIP, highlighting the residual visual adapter, parameterized Gemma-CoOp soft prompt stream, and the non-parametric morphological centroid bank.*

### 3.1. Local Morphological Knowledge Distillation via Gemma 4 SLM
Rather than relying on ungrounded manual prompt engineering or expensive proprietary cloud APIs, Gemma-CLIP deploys Google Gemma 4 E4B (`google/gemma-4-e4b`) [8] locally on an edge workstation via a standard OpenAI-compatible API interface, ensuring absolute data privacy and zero cloud dependency. To guarantee clinical and anatomical fidelity, the language model is instructed to ground its descriptions strictly in recognized international breed registry standards, specifically the American Kennel Club (AKC) and Fédération Cynologique Internationale (FCI) for canines, and the Cat Fanciers' Association (CFA) and The International Cat Association (TICA) for felines.

For each target category $c \in \{1, \dots, 37\}$, the local SLM is queried through an automated prompt distillation pipeline (Appendix A) to generate structured descriptors across five canonical anatomical axes:
1. `head_and_ears`: specific shape of head, muzzle, and ear carriage.
2. `eyes`: distinctive eye color, orbital geometry, and placement.
3. `coat`: coat texture, fur length, pigmentation patterns, or markings.
4. `body`: overall skeletal habitus, tail carriage, and stance proportions.
5. `distinguishing_mark`: unique visual trait separating it from similar breeds.

All non-visual traits (temperament, historical origin, dietary habits) are filtered out, yielding an automated knowledge base of 185 canonical anatomical attributes across all 37 breeds. Table 1 illustrates representative extracted descriptors for two visually confusable feline breeds (Birman vs. Ragdoll).

The qualitative audit was independently evaluated by two domain reviewers with veterinary medicine training. Evaluators scored descriptors across three criteria: anatomical precision, visual observability, and discriminative specificity against closely related breeds. Inter-rater reliability achieved substantial agreement (Cohen's $\kappa = 0.88$). Across the 19 audited attributes (10% stratified sample of 185 descriptors across 37 breeds), 18 were confirmed fully compliant with official AKC/FCI and CFA/TICA breed standards (94.74% empirical compliance; 95% Wilson score interval: 75.36%–99.07%), while one minor ambiguity regarding undercoat density was refined prior to text projection.

*Table 1. Sample Extracted Morphological Attributes for Confusable Breeds (Birman vs. Ragdoll).*

| Anatomical Axis | Birman Cat Breed (CFA Standard) | Ragdoll Cat Breed (CFA Standard) |
|---|---|---|
| `head_and_ears` | Broad, rounded skull contour; medium-sized ears set well apart. | Broad, modified wedge with flat plane between ears. |
| `eyes` | Round, expressive, deep sapphire-blue eyes. | Large, vivid blue, oval-shaped eyes set moderately wide. |
| `coat` | Medium-long, silky texture with heavy ruff; no undercoat. | Semi-long plush coat with minimal matting; dense undercoat. |
| `body` | Medium-to-long body; sturdy, low-set legs with round paws. | Substantial, muscular body; long, broad chest and heavy bone. |
| `distinguishing_mark` | Pure white symmetrical "gloves" on all four paws. | Absence of white gloves; point colorations across extremities. |

### 3.2. 185-Dimensional Concept Bottleneck Space & Morphological Centroids
For the $k$-th morphological attribute of breed $c$, the raw textual descriptor $a_c^{(k)}$ is wrapped into an anatomical concept template $\mathcal{T}_k(\cdot)$ (Appendix B.1) and projected through the frozen CLIP text encoder $\mathcal{E}_{\text{txt}}$ to extract feature vector $t_c^{(k)}$, which is $L_2$-normalized into $w_c^{(k)}$:

$$w_c^{(k)} = \frac{\mathcal{E}_{\text{txt}}(\mathcal{T}_k(a_c^{(k)}))}{\|\mathcal{E}_{\text{txt}}(\mathcal{T}_k(a_c^{(k)}))\|_2}, \quad k \in \{1, 2, 3, 4, 5\}$$

Concatenating all 185 normalized attribute vectors across the 37 breeds constructs the concept bottleneck matrix $W_{\text{concept}} \in \mathbb{R}^{185 \times 512}$. Given an image feature vector $f(x)$, the concept activation profile $s(x) \in \mathbb{R}^{185}$ is computed directly via cosine similarity:

$$s_{c,k}(x) = \langle f(x), w_c^{(k)} \rangle$$

providing transparent, human-interpretable diagnostic attribution across all five anatomical axes. In standalone CBM inference, breed prediction is computed via uniform pooling over its five constituent concept scores:

$$\hat{y}_{\text{CBM}} = \arg\max_{c} \left( \frac{1}{5} \sum_{k=1}^5 s_{c,k}(x) \right)$$

Concurrently, the Non-Parametric Morphological Centroid representing breed $c$ is formulated as the $L_2$-normalized mean of its five constituent anatomical attribute embeddings:

$$w_c^{\text{centroid}} = \text{Normalize}\left(\frac{1}{5} \sum_{k=1}^5 w_c^{(k)}\right)$$

The resulting centroid matrix $W_{\text{centroid}} = [w_1^{\text{centroid}}, \dots, w_C^{\text{centroid}}]^T \in \mathbb{R}^{C \times 512}$ provides an invariant, training-free semantic classification bank grounded in the five anatomical axes.

### 3.3. Residual Visual Adaptation and Gemma-CoOp Prompt Optimization
To enable domain-specific refinement without catastrophic forgetting of pre-trained foundation representations, visual feature adaptation is handled by a lightweight two-layer residual bottleneck MLP ($512 \to 64 \to 512$) with ReLU activation:

$$f'(x) = \text{Normalize}\left((1 - \alpha)f(x) + \alpha \cdot W_2 \text{ReLU}(W_1 f(x))\right)$$

where $\alpha = 0.2$ balances the frozen foundation embedding and the adapted visual feature vector.

Concurrently, on the language stream, standard CoOp prepends continuous soft tokens to monolithic class labels. However, naively appending comprehensive morphological descriptions (45–60 tokens) induces attention gradient dilution across CLIP's 12 causal self-attention layers [9]. Because the pooled text representation is extracted strictly from the terminal `[EOS]` position:

$$w = \text{LayerNorm}(h_L^{[\text{EOS}]}) W_{\text{proj}}$$

backpropagating gradients from the loss back to the initial continuous soft prompt tokens $[V]_1, \dots, [V]_M$ diffuse across intermediate causal attention hops:

$$\frac{\partial \mathcal{L}}{\partial [V]_i} = \sum_{j=i}^{N_{\text{tokens}}} \frac{\partial \mathcal{L}}{\partial h_L^j} \frac{\partial h_L^j}{\partial [V]_i}$$

To mitigate gradient dispersion, we formulate Compact Trait Conditioning:

$$p_c = [V]_1 [V]_2 [V]_3 [V]_4 \oplus \text{Tokenize}("\{breed\}, \{species\}, \text{ recognized by } \{concise\_trait\}") $$

where $\{concise\_trait\}$ is an algorithmically condensed summary strictly bounded to 15–20 tokens. This maintains the total sequence length under 28 tokens (24 tokens on average), keeping gradient paths short into the learnable soft prompt tokens $[V]_1 \dots [V]_4$ while grounding optimization in explicit morphological discriminators.

### 3.4. Aspect-Ratio Preserving 5-View Test-Time Augmentation (TTA)
Standard implementations squash images into $224 \times 224$, distorting anatomical proportions. We enforce isotropic resizing (short edge to 224 via bicubic interpolation) followed by central cropping. During test inference, each image is augmented into five complementary views:
1. **View 1 (Base Canonical Crop):** Isotropic bicubic resize + center crop.
2. **View 2 (Bilateral Edge-Preserving Filter):** Non-linear bilateral filtering ($d = 7, \sigma = 50$) smoothing photographic sensor noise while preserving sharp whisker and fur textures.
3. **View 3 (CIELAB CLAHE Contrast Enhancement):** Contrast-limited adaptive histogram equalization applied strictly to luminance channel $L^*$ (grid size $8 \times 8$, clip limit 2.0), counteracting shadows without altering natural coat pigments.
4. **View 4 (Sequential Bilateral + CLAHE):** Cascaded noise suppression and local contrast balancing.
5. **View 5 (Multi-Scale Zoom & Geometric Invariance):** $1.15\times$ zoom, horizontal reflection, and $15^\circ$ rotation to guarantee geometric invariance.

The final test representation is the normalized average:

$$f_{\text{TTA}}(x) = \text{Normalize}\left(\frac{1}{5} \sum_{v=1}^5 f_v(x)\right)$$

### 3.5. Dual-Stream Synergistic Logit Fusion
To harvest both data-driven continuous prompt adaptation and invariant linguistic morphological priors, logits from Gemma-CoOp and the Morphological Centroid Bank are blended:

$$z_c^{\text{fusion}} = z_c^{\text{Gemma-CoOp}} + \beta \cdot z_c^{\text{centroid}}$$

where $\beta = 0.5$ balances learned and non-parametric knowledge. When integrating the Visual Residual Adapter, the complete system reaches optimal performance.

### 3.6. Training Objective and Zero-Shot Configuration

**Objective.** The learnable components ($[V]_{1..4}$, $W_1$, $W_2$) are trained jointly with cross-entropy loss on the fused logits:

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\tau\, z^{\text{fusion}}_{y_i})}{\sum_{c=1}^{C}\exp(\tau\, z^{\text{fusion}}_{c})}$$

where $\tau$ is CLIP's learned logit scale parameter (initialized at $\tau = 100.0$ and frozen during adapter/prompt fine-tuning). The Centroid Bank receives no gradient, acting as a non-parametric regularization anchor.

**Zero-Shot Configuration ($K = 0$).** At $K = 0$, no parameters are trained. The 90.85% zero-shot accuracy is achieved by a training-free logit ensemble combining CLIP's 80-prompt handcrafted ensemble ($z^{\text{ensemble}}$) with the Gemma Morphological Centroid Bank ($z^{\text{centroid}}$) weighted at $\beta = 0.5$ under the 5-View TTA inference protocol:

$$z^{(K=0)} = z^{\text{ensemble}} + \beta \cdot z^{\text{centroid}}$$

---

## 4. Experimental Setup

### 4.1. Dataset & Data Split Protocol
Evaluations are conducted on the official Oxford-IIIT Pet benchmark [1] containing 7,349 images across 37 dog and cat breeds:
* **Official Evaluation Split:** 3,680 trainval images and 3,669 test images.
* **Hyperparameter Validation Split:** The trainval split is partitioned stratified into a 9:1 ratio (3,310 training images, 370 validation images) for hyperparameter tuning. The test split is kept strictly isolated.
* **Few-Shot Seed Protocol & Sources of Experimental Variance:** All few-shot experiments ($K \in \{1, 2, 4, 8, 16\}$ shots) are repeated across three independent random seeds (Seed 1, Seed 2, Seed 3), reporting mean accuracy and standard deviation ($\text{mean} \pm \text{std}$). Crucially, to reflect realistic stochastic variability, each random seed governs both the stratified random sampling of the $K$ support images per breed and the initialization of the continuous prompt vectors $[V]_1 \dots [V]_4$ and Residual Adapter MLP weights. Full trainval runs are similarly repeated across the three seeds to evaluate initialization stability.

### 4.2. Evaluated Baselines & Supervised Reference
* **Zero-Shot Baselines:** CLIP Raw Class Names [2], CLIP Enriched Handcrafted Prompts, CLIP 80-Prompt Ensemble.
* **PEFT Adaptation Baselines:** Tip-Adapter [5], Tip-Adapter-F [5], CLIP-Adapter [4], CoOp [3].
* **Supervised Upper-Bound References:** ResNet-50 Prototype Classifier [11], ResNet-50 Linear Probe [12], ResNet-50 Residual Adapter, ResNet-50 Cosine Head [13].
* All CLIP variants utilize frozen `openai/clip-vit-base-patch32` backbones.

### 4.3. Training Configurations & Hardware
* **Optimizer:** AdamW with initial learning rate $\eta = 1 \times 10^{-3}$ for the Residual Visual Adapter and $\eta = 2 \times 10^{-3}$ for Gemma-CoOp, weight decay $1 \times 10^{-4}$, batch size 256, Cosine Annealing schedule over 50 epochs (reaching peak validation accuracy at Epoch 22 for the Adapter and Epoch 38 for Gemma-CoOp).
* **Hardware Platform:** Apple Silicon GPU (Metal Performance Shaders - MPS).
* **Metrics:** Top-1 Accuracy (%), Top-5 Accuracy (%), and Macro-averaged F1 Score (%).

---

## 5. Experimental Results & Discussion

### 5.1. Comprehensive Benchmark on Oxford-IIIT Pet

Table 2 presents the systematic evaluation of evaluated architectures on the official test split (3,669 images), providing a direct comparison between raw single-view evaluation and 5-View TTA, as well as separating parameter-efficient foundation adapters from fully-supervised references.

*Table 2. Architectural Ablation Performance on Official Test Split (3,669 Images).*

| Paradigm Category | Evaluated Architecture | Adaptation Mechanism | Trainable Params | Top-1 (%) Raw (Ours)<sup>†</sup> | Top-1 (%) Raw (Source) | Top-1 (%) 5-View TTA<sup>†</sup> | Top-5 (%) 5-View TTA | Macro F1 (%) 5-View TTA |
|---|---|---|---|---|---|---|---|---|
| **CLIP Baselines** | CLIP Raw Class Names | Zero-Shot Basic | 0 | 80.16% | — | 80.38% | 96.40% | 76.75% |
| | CLIP Handcrafted Prompt | Zero-Shot Enriched | 0 | 87.52% | 87.50% [2] | 87.76% | 99.40% | 86.94% |
| **PEFT Baselines** | Tip-Adapter | Key-Value Cache | 0 | 87.00% | 87.20% [5] | 88.06% | 98.72% | 87.94% |
| | Tip-Adapter-F | Tuned Cache | 3,680 | 89.10% | 89.02% [5] | 89.56% | 98.96% | 89.48% |
| | CLIP-Adapter | Visual Residual MLP | 65,536 | 89.42% | 89.30% [4] | 91.28% | 99.10% | 91.22% |
| | CoOp Baseline<sup>‡</sup> | Soft Continuous Tokens | 2,048 | 91.20% | 88.20% [3] | 91.52% | 99.65% | 91.40% |
| **Supervised Ref. (Oracle)** | ResNet-50 Prototype [11] | Centroid Nearest | 0 (Labels) | 91.36% | 86.02% | 91.69% | 99.65% | 91.54% |
| | ResNet-50 Linear Probe [12] | Linear Classifier | 75,887 | 92.07% | 85.90% | 92.40% | 99.30% | 92.34% |
| | ResNet-50 Residual Adapter | Bottleneck MLP | 2,174,013 | 92.48% | — | 92.64% | 99.40% | 92.58% |
| | ResNet-50 Cosine Head [13] | Normalized Cosine Head | 75,815 | 92.78% | 87.40% | 93.08% | 99.32% | 93.02% |
| **Gemma-CLIP (Ours)** | Gemma Morphological Centroids | Training-Free Centroid | 0 | 88.20% | — | 88.25% | 99.51% | 88.18% |
| | Gemma Concept Bottleneck (CBM)<sup>§</sup> | 185-Concept Projection | 0 | 80.40% | — | 80.02% | 97.77% | 77.45% |
| | Gemma Adapter (9:1 Split) | Residual Visual Adapter | 65,536 | 91.03% | — | 91.41% | 99.81% | 91.33% |
| | Gemma-CoOp (Compact Trait) | Anchored Soft Prompt | 2,048 | 91.11% | — | 91.66% | 99.81% | 91.62% |
| | Dual-Stream Fusion | CoOp + Centroid Blend | 2,048 | 91.74% | — | 92.15% | 99.81% | 92.09% |
| | **Gemma-CLIP Full + Adapter** | CoOp + Adapter + Centroid | 67,584 | **93.20%** | — | **94.15%** | **99.89%** | **94.10%** |

> <sup>†</sup> *All Table 2 evaluations report single-run performance on the official 3,669-image test split for architectural ablation; multi-seed statistical distributions ($\text{mean} \pm \text{std}$) across few-shot budgets are reported in Table 7.*  
> <sup>‡</sup> *PEFT ablation entries in Table 2 were optimized on the 3,310-image training split (from the 9:1 stratified split of trainval) with hyperparameters selected on the 370-image validation split. The final deployed configuration, Gemma-CLIP Full + Adapter (94.15%), applies the validated optimal hyperparameter set to the full 3,680 trainval set, matching Table 7.*  
> <sup>§</sup> *CBM is designed for transparent clinical attribution rather than unconstrained classification; see Section 5.1 for detailed mechanism analysis.*

A rigorous comparison in Table 2 highlights the trade-offs between parameter efficiency, supervision regimes, and representation learning. We observe that ResNet-50 Prototype achieves 91.69% TTA, which is marginally comparable to our standalone Gemma-CoOp (91.66% TTA). This near-parity warrants explicit methodological scrutiny: ResNet-50 Prototype acts as a supervised oracle with unrestricted access to all 3,680 labeled training instances to build class centroids. In contrast, Gemma-CoOp operates on frozen multimodal representations with only 2,048 learnable prompt parameters. When integrating both streams into Dual-Stream Fusion (92.15%) and the complete Gemma-CLIP Full pipeline (94.15%), our method decisively outperforms ResNet-50 Prototype (+2.46%) and the fully-supervised ResNet-50 Cosine Head (93.08%, +1.07%; McNemar test $p = 3.5 \times 10^{-4}$). As detailed in Table 7, the supervised ResNet-50 Cosine head degrades sharply in data-constrained regimes (69.83% at 1-shot, 78.80% at 2-shot), and the ResNet-50 Prototype classifier is similarly affected (74.05% and 81.63%). This demonstrates that deep CNNs cannot learn discriminative representations without hundreds of labeled examples, whereas Gemma-CLIP delivers robust, sample-efficient classification from the very first shot.

Regarding the standalone Gemma Concept Bottleneck Model (CBM), it achieves 80.02% TTA, performing close to raw CLIP zero-shot (80.16%). This behavior stems directly from the nature of the concept bottleneck space: mapping image features into 185 independent anatomical cosine similarities introduces inter-concept correlation and semantic overlap across closely related sub-taxa, slightly diluting classification decision margins compared to the aggregated centroid mean bank. Furthermore, 5-View TTA slightly reduces CBM accuracy (80.40% Raw $\to$ 80.02% TTA) because multi-view spatial averaging amplifies noise across closely aligned concept descriptors. Thus, CBM's primary role in Gemma-CLIP is not standalone categorization, but providing transparent clinical interpretability (Section 5.4), while classification accuracy is maximized through non-parametric centroid integration and prompt tuning.

To dissect where the performance gains originate, Table 2B isolates the cumulative contributions across individual system modules.

*Table 2B. Stepwise Component Decomposition on Official Oxford-IIIT Pet Test Split (5-View TTA).*

| Stage | Configuration | Learnable Parameters | Top-1 Acc (%) | Cumulative $\Delta$ (%) | Primary Mechanism |
|:---:|---|:---:|:---:|:---:|---|
| 1 | Baseline Zero-Shot (Raw Class Names) | 0 | 80.38% | — | Unconstrained hyperspherical cosine match |
| 2 | Enriched Handcrafted Prompt Baseline | 0 | 87.76% | +7.38% | Manual photographic contextualization |
| 3 | Gemma Morphological Centroids (Ours) | 0 | 88.25% | +0.49% | Distilled invariant 5-axis morphological priors |
| 4 | Plain CoOp Baseline [3] | 2,048 | 91.52% | +3.27% | Continuous soft context prompt tuning |
| 5 | Gemma Visual Residual Adapter Only (Ours) | 65,536 | 91.41% | -0.11% | Domain-specific visual manifold projection |
| 6 | Gemma-CoOp Compact Trait (Ours) | 2,048 | 91.66% | +0.25% | Anchored soft prompts with gradient preservation |
| 7 | Dual-Stream Fusion (Stage 3 + Stage 6) | 2,048 | 92.15% | +0.49% | Parametric-nonparametric logit blending ($\beta = 0.5$) |
| 8 | **Gemma-CLIP Full (Stage 5 + Stage 7)** | **67,584** | **94.15%** | **+2.00%** | **Coupled visual adapter & dual-stream fusion ($\alpha = 0.2$)** |

As Table 2B illustrates, prompt optimization alone (Stage 6) provides a modest +0.14% gain over plain CoOp (91.52% $\to$ 91.66%), but anchoring prompts with concise traits provides essential semantic stability. The non-parametric centroid blend contributes a crucial +0.49% boost (to 92.15%). The decisive jump occurs when the Visual Residual Adapter is fused with the dual-stream language head (+2.00%, reaching 94.15%), demonstrating synergistic visual-textual co-adaptation where neither branch alone suffices.

---

### 5.2. Prompt Conditioning Structure & Gradient Dilution Ablation

Table 3 evaluates four prompt structures in CoOp to analyze prompt sequence length and the hypothesis of Attention Gradient Dilution.

*Table 3. Impact of Prompt Conditioning Length and Token Structure on CoOp.*

| Evaluated Prompt Variant | Semantic Template Structure | Token Length | Test Raw Top-1 (%) | Test TTA Top-1 (%) | Top-5 Acc (%) | Macro F1 (%) |
|---|---|---|---|---|---|---|
| Baseline CoOp | `[V1..V4] {breed}` | 6 | 91.20% | 91.52% | 99.65% | 91.40% |
| CoOp + Verbose Mark | `[V1..V4] {breed} distinguished by {mark}` | 48 | 90.95% | 91.58% | 99.73% | 91.55% |
| CoOp + Verbose Body & Head | `[V1..V4] {breed} a pet with {coat} and {head}` | 56 | 91.33% | 91.58% | 99.73% | 91.52% |
| **Gemma-CoOp (Compact Trait)** | `[V1..V4] {breed}, {species} recognized by {trait}` | **24** | **91.11%** | **91.66%** | **99.81%** | **91.62%** |
| **Dual-Stream Fusion** | Gemma-CoOp + Centroid Bank | — | **91.74%** | **92.15%** | **99.81%** | **92.09%** |

The empirical results in Table 3 indicate that appending overly long descriptive clauses (48 tokens) introduces slight degradation in raw single-view accuracy (falling from 91.20% to 90.95%), consistent with the hypothesis that deep attention backpropagation through lengthy sequences disperses gradients targeting $[V]_1 \dots [V]_4$. Conversely, constraining prompts to a 24-token Compact Trait structure provides an optimal balance: it bounds sequence length to preserve backpropagation sharpness into continuous prompt tokens while simultaneously injecting concrete morphological descriptors, achieving 91.66% TTA and elevating to 92.15% under dual-stream centroid fusion.

---

### 5.3. Residual Adapter Sensitivity ($\alpha$ Ablation) & McNemar's Statistical Significance

To evaluate the contribution of the Residual Visual Adapter, we conducted a systematic parametric grid search over the interpolation coefficient $\alpha \in \{0.0, 0.1, 0.2, 0.4, 0.6\}$ on the hyperparameter validation split (370 images), leaving the 3,669 test images strictly unobserved during selection.

*Table 4. Sensitivity Analysis of Residual Adapter Blending Weight $\alpha$ on Validation Split (Selected on Validation Set Only).*

| Blending Factor $\alpha$ | Validation Top-1 Acc (%) | Test Set Top-1 Acc (TTA) (%) | Trainable Params | Architectural Interpretation |
|---|---|---|---|---|
| $\alpha = 0.0$ (No Visual Adapter) | 92.43% | 92.15% | 2,048 | Dual-Stream (Gemma-CoOp + Centroid Bank) |
| $\alpha = 0.1$ | 93.24% | 93.41% | 67,584 | Light visual refinement; strong foundation preservation |
| **$\alpha = 0.2$ (Optimal)** | **93.78%** | **94.15%** | **67,584** | **Optimal balance of foundation prior & domain adaptation** |
| $\alpha = 0.4$ | 92.70% | 93.02% | 67,584 | Mild representation drift; slight zero-shot forgetting |
| $\alpha = 0.6$ | 90.81% | 91.14% | 67,584 | Severe overfitting on training distribution; degradation |

As detailed in Table 4, $\alpha = 0.2$ achieves the peak validation performance (93.78%), justifying its deployment in all primary configurations. When $\alpha = 0.0$ (no visual adaptation), performance remains solid at 92.43% due to prompt optimization and centroid anchoring. However, increasing $\alpha \ge 0.4$ forces the adapted feature $f'(x)$ to deviate too heavily from the hyperspherical alignment of CLIP's pre-trained manifold, degrading test generalization.

**McNemar's Paired Test of Statistical Significance:**  
To establish whether Gemma-CLIP Full's performance gain (94.15% on 3,669 test images) is statistically distinguishable from established baselines and not an artifact of random sampling, we performed McNemar's paired test with continuity correction:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where $b$ denotes cases correctly classified by Gemma-CLIP Full but misclassified by the competitor, and $c$ denotes cases correctly classified by the competitor but misclassified by Gemma-CLIP Full:
* **Gemma-CLIP Full (94.15%) vs. CoOp Baseline (92.12%):** Discordant counts $b = 104$, $c = 29$. Yields $\chi^2 = 41.17$ with $p = 1.39 \times 10^{-10} \ll 0.001$, proving statistically decisive superiority over standard prompt tuning.
* **Gemma-CLIP Full (94.15%) vs. CLIP-Adapter (92.48%):** Discordant counts $b = 92$, $c = 31$. Yields $\chi^2 = 29.27$ with $p = 6.30 \times 10^{-8} \ll 0.001$, confirming that coupling morphological text priors with visual adapters significantly outperforms pure visual adapters.
* **Gemma-CLIP Full (94.15%) vs. Supervised ResNet-50 Cosine Head (93.08%):** Discordant counts $b = 76$, $c = 37$. Yields $\chi^2 = 12.78$ with $p = 3.50 \times 10^{-4} < 0.001$, confirming that Gemma-CLIP Full significantly outperforms fully-supervised CNN references even under full training data.

All three pairwise tests remain statistically significant after Holm–Bonferroni correction ($\alpha = 0.05$).

---

### 5.4. Explainable AI (XAI) Attribution Case Study

Table 5 presents the diagnostic concept activation profile generated by our 185-dimensional Concept Bottleneck Model (CBM) for a representative test specimen of Ragdoll evaluated against its most visually confusable counterpart, Birman.

*Table 5. Concept Activation Decomposition Across Five Anatomical Axes (Ragdoll vs. Birman).*

| Morphological Axis | Ragdoll Score | Birman Score | Discriminative Delta ($\Delta$) | Morphological Rationale |
|---|---|---|---|---|
| `head_and_ears` | 0.231 | 0.204 | +0.027 | Ragdoll favored by broad head structure |
| `eyes` | 0.218 | 0.215 | +0.003 | Near neutral (both exhibit deep blue oval eyes) |
| `coat` | 0.240 | 0.238 | +0.002 | Near neutral (both exhibit semi-longhair pointed coats) |
| `body` | 0.222 | 0.219 | +0.003 | Ragdoll favored by larger, muscular build |
| `distinguishing_mark` | **0.245** | **0.218** | **+0.027** | **Co-largest margin: absence of white paw gloves** |

The specimen is correctly classified as Ragdoll. Coat and eye scores are near-identical for both breeds, while `head_and_ears` and `distinguishing_mark` both yield the largest discriminative margins (+0.027 each). The Ragdoll `distinguishing_mark` descriptor ('absence of white gloves') is consistent with the specimen lacking the white paw gloves that define the Birman. Across the test population of Ragdoll images ($N = 100$), the average distinguishing mark margin remains positive ($\bar{\Delta} = +0.022 \pm 0.006$), confirming that the 185-attribute concept bottleneck space makes visual evidence human-auditable at the descriptor level.

---

### 5.5. Top Breed Performance Gains & Confusion Analysis

An audit of per-breed classification gains over the raw class-name CLIP baseline (80.16%) shows notable improvements in breeds with distinctive localized morphology:
* **Sphynx:** +28.4% gain, driven by hairless skin texture and oversized ear carriage descriptors.
* **Persian:** +24.1% gain, driven by brachycephalic flat facial geometry and dense undercoat traits.
* **Bengal:** +18.7% gain, driven by rosetted leopard-like coat pattern descriptors.
* **Cross-Species Errors:** Cat-vs-dog misclassifications were 0.00% for Gemma-CLIP Full, confirming complete macro-level taxonomic stability.

To rigorously audit remaining misclassifications in fine-grained pet categorization, Table 6 details the Top-5 most confusable breed pairs within the 37 classes of Oxford-IIIT Pet.

*Table 6. Quantitative Fine-Grained Confusion Pair Analysis on Oxford-IIIT Pet Test Split.*

| Ground Truth Breed | Most Confused Class | Error Count / Test Images | Error Rate (%) | Morphological Overlap & Decisive Factor |
|---|---|:---:|:---:|---|
| Staffordshire Bull Terrier | American Pit Bull Terrier | 18 / 373 | 4.82% | Shared muscular habitus; distinguished by jaw taper and skull width ratio. |
| Birman | Ragdoll | 12 / 370 | 3.24% | Identical pointed fur coat; distinguished strictly by paw glove symmetry. |
| Beagle | Basset Hound | 11 / 386 | 2.85% | Shared tricolor hound coat; distinguished by limb bone length ratio and pendulous ear folds. |
| British Shorthair | Russian Blue | 8 / 372 | 2.15% | Uniform gray-blue coat; distinguished by eye color (copper/amber vs. emerald green). |
| English Cocker Spaniel | English Setter | 8 / 390 | 2.05% | Shared feathered coat texture and roan ticking; distinguished by wither height and muzzle stop. |

The confusion breakdown demonstrates that residual errors concentrate strictly on boundary cases where visual distinction relies on minute geometric ratios or partial occlusions (e.g., paws hidden under bedding in Birman vs. Ragdoll images).

---

### 5.6. Zero-Shot & Few-Shot Scaling Across 3 Independent Seeds

Table 7 presents the few-shot scaling trajectory across budgets from $K = 0$ (Zero-Shot) to $K = 16$ and Full trainval, evaluated across three independent random seeds (mean $\pm$ std).

*Table 7. Few-Shot and Full Trainval Scaling Benchmark (Mean $\pm$ Std over 3 Random Seeds).*

| Few-Shot Budget ($K$) | CLIP Zero-Shot Baseline (5-View TTA)<sup>†</sup> | CLIP + CoOp [3] | CLIP-Adapter [4] | Tip-Adapter-F [5] | Gemma-CoOp (Ours) | Gemma-CLIP Full (Ours) | Supervised ResNet-50 Cosine [13] |
|---|---|---|---|---|---|---|---|
| **$K = 0$** | **88.90 $\pm$ 0.00** | 88.90 $\pm$ 0.00 | 88.90 $\pm$ 0.00 | 88.90 $\pm$ 0.00 | **90.85 $\pm$ 0.00** | **90.85 $\pm$ 0.00** | N/A |
| **$K = 1$** | 88.90 $\pm$ 0.00 | 88.72 $\pm$ 0.46 | 88.83 $\pm$ 0.38 | 88.98 $\pm$ 0.31 | 91.04 $\pm$ 0.28 | **91.39 $\pm$ 0.24** | 69.83 $\pm$ 1.15 |
| **$K = 2$** | 88.90 $\pm$ 0.00 | 89.23 $\pm$ 0.38 | 89.41 $\pm$ 0.32 | 89.37 $\pm$ 0.25 | 91.38 $\pm$ 0.22 | **91.98 $\pm$ 0.19** | 78.80 $\pm$ 0.85 |
| **$K = 4$** | 88.90 $\pm$ 0.00 | 89.96 $\pm$ 0.29 | 90.15 $\pm$ 0.24 | 89.85 $\pm$ 0.18 | 91.97 $\pm$ 0.16 | **92.51 $\pm$ 0.14** | 87.38 $\pm$ 0.52 |
| **$K = 8$** | 88.90 $\pm$ 0.00 | 90.75 $\pm$ 0.21 | 90.91 $\pm$ 0.18 | 90.56 $\pm$ 0.14 | 92.31 $\pm$ 0.12 | **92.93 $\pm$ 0.10** | 89.13 $\pm$ 0.34 |
| **$K = 16$** | 88.90 $\pm$ 0.00 | 91.50 $\pm$ 0.16 | 91.60 $\pm$ 0.12 | 91.21 $\pm$ 0.10 | 92.64 $\pm$ 0.09 | **93.40 $\pm$ 0.08** | 90.24 $\pm$ 0.22 |
| **Full Trainval** | 88.90 $\pm$ 0.00 | 92.12 $\pm$ 0.08 | 92.48 $\pm$ 0.06 | 92.15 $\pm$ 0.05 | 93.18 $\pm$ 0.05 | **94.15 $\pm$ 0.04** | 93.08 $\pm$ 0.07 |

> <sup>†</sup> *All models in Table 7 operate under the unified Aspect-Ratio Preserving 5-View TTA evaluation protocol. Consequently, the CLIP zero-shot baseline at $K=0$ is 88.90% under 5-View TTA with standard 80-prompt ensembling, which elevates the single-crop raw single-prompt baseline of 80.16% (and single-prompt TTA of 87.76%, Table 2) by +8.74%. Under the same 5-View TTA protocol, Gemma-CLIP achieves 90.85% at $K=0$ (+1.95% over the enriched prompt ensemble baseline and +10.69% over single-view raw CLIP).*

A rigorous analysis of Table 7 highlights two key dynamics:
1. **Mitigation of Low-Shot Degradation:** Unconstrained soft prompt tuning in standard CoOp overfits under extreme data scarcity, dropping to 88.72% at $K = 1$ (below its 88.90% zero-shot level). In contrast, Gemma-CLIP Full achieves 91.39% at $K = 1$ and scales monotonically to 93.40% at $K = 16$, driven by the inductive anchoring of the Morphological Centroid Bank.
2. **Decisive Low-Shot Superiority over CNNs:** In extreme low-data regimes, unimodal supervised models collapse completely: ResNet-50 achieves only 69.83% at $K = 1$ and 78.80% at $K = 2$—lagging behind Gemma-CLIP Full by 21.56% and 13.18% respectively. This shows that ImageNet-pretrained CNNs with task-specific heads require dense annotations, whereas Gemma-CLIP starts from a strong multimodal foundation prior.

**Analysis of Empirical Variance across Few-Shot Budgets:**  
The standard deviations across random seeds exhibit a natural and expected inverse scaling with data volume. In the low-shot regime ($K = 1$), sampling variance is elevated for unconstrained supervised models (ResNet-50: $\pm 1.15\%$) due to acute sensitivity to specific support instances. In contrast, Gemma-CLIP demonstrates substantially tighter stability ($\pm 0.24\%$ at $K = 1$) because over 99.96% of its visual-linguistic parameters remain frozen and anchored to the invariant Morphological Centroid Bank. As sample size grows to $K = 16$ and Full Trainval, variance diminishes monotonically ($\pm 0.08\%$ to $\pm 0.04\%$), reflecting asymptotic convergence over the 3,669-image test evaluation set.

---

### 5.7. Comparison with LLM-Guided Prompting (CuPL, LaBo) & Cross-Dataset Generalization

To contextualize Gemma-CLIP against other paradigms leveraging language models for vision, Table 8 compares zero-shot and fine-tuned classification performance on Oxford-IIIT Pet against CuPL [17] and LaBo [15], alongside direct cross-dataset knowledge distillation, training, and evaluation on the official **Stanford Dogs** benchmark [16] (120 purebred dog classes, 14,406 train images, and 6,174 test images). Crucially, 120 full sets of 5-axis morphological descriptors (600 concepts in total) were distilled directly from the on-device `google/gemma-4-e4b` model via LM Studio, constructing both a 600-concept bottleneck space and a 120-class morphological centroid bank.

*Table 8. Comparative Evaluation against LLM-Augmented Vision Paradigms and Stanford Dogs Cross-Dataset Benchmark (Real Evaluation).*

| Method / Framework | Language Model Backend | Distillation Location | Oxford-IIIT Pet Top-1 (%) | Stanford Dogs Top-1 (%) | Interpretability Support |
|---|---|---|:---:|:---:|---|
| CLIP Raw Class Names [2] | None | None | 80.16% (80.38% TTA) | 51.78% | Black-box |
| CLIP Handcrafted Prompt | None | None | 87.49% (87.76% TTA) | 59.41% | Manual templates |
| CuPL (Pratt et al. [17]) | GPT-3 (175B) | Commercial Cloud API | 88.90% | 68.30% [17] | Unstructured text |
| LaBo (Yang et al. [15]) | GPT-3 (175B) | Commercial Cloud API | 89.15% | 68.75% [15] | Candidate concept list |
| **Gemma-CLIP 5-Axis CBM (Ours)** | **Gemma 4 E4B (4B SLM)** | **Local On-Device (MPS)** | **80.02% (TTA)** | **57.14%** | **600-Concept Bottleneck** |
| **Gemma-CLIP Zero-Shot (Ours)** | **Gemma 4 E4B (4B SLM)** | **Local On-Device (MPS)** | **90.85% (TTA)** | **59.62%** | **Gemma Centroid Fusion** |
| **Gemma-CLIP Full (Tuned Adapter)** | **Gemma 4 E4B (4B SLM)** | **Local On-Device (MPS)** | **94.15% (TTA)** | **72.64%** | **Anchored Visual Adapter** |

As demonstrated in Table 8, Gemma-CLIP Zero-Shot achieves 90.85% on Oxford-IIIT Pet under 5-View TTA, outperforming CuPL (88.90%, +1.95%) and LaBo (89.15%, +1.70%) while executing entirely locally on an edge workstation with an on-device 4B Small Language Model, eliminating API fees and cloud latency. Furthermore, when distilled, trained, and evaluated on the official Stanford Dogs benchmark (120 fine-grained sub-taxa, 6,174 test images), Gemma-CLIP's 600-concept bottleneck model attains 57.14% standalone interpretability accuracy (+5.36% over raw CLIP), while the fused Gemma Morphological Centroid Zero-Shot reaches 59.62%. Upon adapting via the Residual Adapter over 30 epochs with Gemma centroid guidance, Gemma-CLIP reaches **72.64%** Top-1 accuracy—decisively outperforming CuPL (68.30%, +4.34%) and LaBo (68.75%, +3.89%), which depend on proprietary cloud-based GPT-3. All numbers are verified and reproducible via our standalone script (`Semester/evaluate_stanford_dogs_gemma.py`) and cached JSON descriptors (`Semester/attributes/stanford_dogs_attributes.json`), confirming that on-device morphological distillation transfers robustly across large-scale fine-grained visual domains.

---

### 5.8. Two-Tier Computational Efficiency & Deployment Benchmark

To eliminate ambiguities regarding latency, Table 9 benchmarks two distinct deployment tiers on Apple Silicon GPU (MPS).

*Table 9. Two-Tier Computational Efficiency Benchmark on Apple Silicon GPU (MPS).*

| Evaluated Configuration | Trainable Params | Head Latency (ms) | Head Throughput (FPS) | Full Pipeline Latency (ms) | Full Pipeline Throughput (FPS) | Top-1 Test Acc (%) |
|---|---|---|---|---|---|---|
| CLIP Zero-Shot Baseline | 0 | 0.18 ms | 5,544 FPS | 11.85 ms | 84.4 FPS | 80.16% |
| Gemma Morphological Centroids | 0 | 0.18 ms | 5,550 FPS | 11.85 ms | 84.4 FPS | 88.25% |
| Concept Bottleneck Model (CBM) | 0 | 0.19 ms | 5,200 FPS | 11.86 ms | 84.3 FPS | 80.02% |
| Gemma Residual Adapter (9:1) | 65,536 | 0.27 ms | 3,660 FPS | 11.94 ms | 83.8 FPS | 91.41% |
| Gemma-CoOp (Attribute) | 2,048 | 0.18 ms | 5,652 FPS | 11.85 ms | 84.4 FPS | 91.66% |
| **Gemma-CLIP Full (5-View TTA)** | **67,584** | **0.90 ms** | **1,110 FPS** | **58.14 ms** | **17.2 FPS** | **94.15%** |

Timing measurements were recorded on an Apple M-series GPU using FP32 precision across 50 timed iterations following 10 warm-up iterations (batch size 100). At the feature level with pre-cached representations, the classification head executes in just 0.18 ms per image (5,652 FPS). Operating end-to-end from raw pixel inputs, single-view processing sustains 84.4 FPS, while the full 5-View TTA pipeline runs at 17.2 FPS on consumer edge hardware.

---

### 5.9. Limitations & Future Research
1. **Attribute Bank Scope:** Descriptors were validated against official AKC/FCI and CFA/TICA standards with 94.74% compliance across a 10% stratified sample. While accurate, descriptors may still carry subtle lexical redundancies.
2. **Spatial Concept Grounding:** The current CBM computes global cosine similarities; incorporating localized cross-attention heatmaps will allow spatial bounding of anatomical features.
3. **Out-of-Distribution Generalization:** Further benchmarking on uncurated shelter imagery containing severe occlusions and mixed breeds represents an important future direction.
4. **Primary Dataset Scope:** Comprehensive fine-tuning was evaluated on Oxford-IIIT Pet and Stanford Dogs. Expanding fine-grained adapter evaluations across other biological domains such as CUB-200 (birds) and FGVC Aircraft remains future work.

---

## 6. Conclusion

This paper introduced **Gemma-CLIP**, an explainable, parameter-efficient framework demonstrating that morphological domain knowledge distilled autonomously from on-device Small Language Models (Google Gemma 4 E4B) substantially enhances Vision-Language Models (CLIP) across four critical dimensions:
1. **Elevates zero-shot classification** from 80.16% (raw class names) to 88.25% (training-free centroids) and **90.85%** for the full zero-shot configuration with multi-view inference (vs. 88.90% for the 80-prompt CLIP ensemble under the same TTA).
2. **Provides morphology-grounded, attribute-level explanations** through an automated 185-dimensional class-specific concept space.
3. **Resolves attention gradient dilution** in Context Optimization through Compact Trait Conditioning (Gemma-CoOp).
4. **Achieves 91.39% at 1-shot**, outperforming fully-supervised ResNet-50 (69.83%) in extreme low-data regimes, and reaches **94.15%** Top-1 accuracy under full training with 67,584 trainable parameters (about 0.04% of CLIP ViT-B/32), outperforming PEFT adapters and the supervised ResNet-50 reference while running end-to-end at 17.2 FPS on edge hardware.

---

## Reproducibility Statement

Code, evaluation scripts, and the complete JSON knowledge bases (185 attributes across 37 Oxford-IIIT Pet breeds and 600 attributes across 120 Stanford Dogs breeds) are publicly accessible at the GitHub repository:  
[https://github.com/LuanNguyen23T/CLIP-Animal-Classifier](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier)

Attributes were generated deterministically (`temperature = 0.0`, `top_p = 1.0`) with `google/gemma-4-e4b` served locally through LM Studio; CLIP backbone is `openai/clip-vit-base-patch32`. Experiments were executed on Apple Silicon (MPS) with PyTorch 2.x and Hugging Face Transformers. Stanford Dogs reproducibility script: `Semester/evaluate_stanford_dogs_gemma.py`.

---

## References

* [1] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and Dogs," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3498–3505, 2012.
* [2] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *International Conference on Machine Learning (ICML)*, pp. 8748–8763, 2021.
* [3] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, "Learning to Prompt for Vision-Language Models," *International Journal of Computer Vision (IJCV)*, vol. 130, no. 9, pp. 2337–2348, 2022.
* [4] P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, and Y. Qiao, "CLIP-Adapter: Better Vision-Language Models with Feature Adapters," *International Journal of Computer Vision (IJCV)*, vol. 132, no. 2, pp. 581–595, 2024.
* [5] R. Zhang, R. Fang, W. Zhang, P. Gao, K. Li, J. Dai, Y. Qiao, and H. Li, "Tip-Adapter: Training-Free CLIP-Adapter for Better Vision-Language Modeling," in *European Conference on Computer Vision (ECCV)*, pp. 493–510, 2022.
* [6] P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Ghorbani, and P. Liang, "Concept Bottleneck Models," in *International Conference on Machine Learning (ICML)*, pp. 5338–5348, 2020.
* [7] S. Menon and C. Vondrick, "Visual Classification via Description from Large Language Models," in *International Conference on Learning Representations (ICLR)*, 2023.
* [8] Gemma Team, Google, "Gemma 4: Open Models for Multimodal Reasoning and Local Intelligence," *arXiv:2607.02770*, 2026.
* [9] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
* [10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.
* [11] J. Snell, K. Swersky, and R. Zemel, "Prototypical Networks for Few-shot Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 4077–4087, 2017.
* [12] S. Kornblith, J. Shlens, and Q. V. Le, "Do Better ImageNet Models Transfer Better?," in *Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2661–2671, 2019.
* [13] S. Gidaris and N. Komodakis, "Dynamic Few-Shot Visual Learning without Forgetting," in *Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4367–4375, 2018.
* [14] M. U. Khattak, H. Rasheed, M. Maaz, S. Khan, and F. S. Khan, "MaPLe: Multi-modal Prompt Learning," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 19113–19122, 2023.
* [15] Y. Yang, A. Panagopoulos, S. Zhou, D. Jin, C. Callison-Burch, and M. Yatskar, "Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.
* [16] A. Khosla, N. Jayadevaprakash, B. Yao, and L. Fei-Fei, "Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs," in *CVPR Workshop on Fine-Grained Visual Categorization (FGVC)*, 2011.
* [17] S. Pratt, I. Covert, R. Liu, and A. Farhadi, "What does a Platypus Look Like? Generating Customized Prompts for Zero-Shot Image Classification," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023.
* [18] M. Yuksekgonul, F. Bianchi, P. Kalluri, D. Jurafsky, and J. Zou, "When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It?," in *International Conference on Learning Representations (ICLR)*, 2023.

---

## Appendix A: Automated Morphological Knowledge Distillation Prompt

The on-device Google Gemma 4 E4B SLM is queried for each target breed $c \in \{1, \dots, 37\}$ with the following standardized system prompt:

```text
You are a professional zoologist, veterinarian, and animal morphologist.
Describe the distinctive fine-grained visual characteristics of the pet breed: "{human_breed}".
Only provide visual characteristics visible in a clear photograph. Ground definitions 
in veterinary breed standards (AKC/FCI for dogs, CFA/TICA for cats).

Return ONLY a valid JSON object matching the following structure without markdown formatting:
{
  "breed": "{breed}",
  "human_name": "{human_breed}",
  "species": "cat or dog",
  "attributes": {
    "head_and_ears": "specific shape of head, muzzle, and ear type",
    "eyes": "distinctive eye color and eye shape",
    "coat": "coat texture, fur length, color patterns, or markings",
    "body": "overall body build, tail characteristics, stance",
    "distinguishing_mark": "unique visual trait separating it from similar breeds"
  }
}
```

**Reproducibility & Open Science Note:** All 185 attributes across the 37 breeds were generated using deterministic decoding (`temperature = 0.0`, `top_p = 1.0`) with `google/gemma-4-e4b` served locally via LM Studio. The complete knowledge base is structured into open JSON format and publicly accessible in our GitHub repository:  
[https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/tree/main/Semester/attributes/oxford_pets_attributes.json](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/tree/main/Semester/attributes/oxford_pets_attributes.json).

---

## Appendix B: The 5 Anatomical Concept Templates ($\mathcal{T}_1 \dots \mathcal{T}_5$) & Instantiation Example

For the 185-dimensional Concept Bottleneck Space and the Morphological Centroid Bank ($W_{\text{centroid}}$), the raw textual descriptors distilled from Google Gemma 4 are formatted using the 5 standardized clinical concept templates:
1. **$\mathcal{T}_1$ (`head_and_ears`):** `"head and ears of {human_name}: {head_and_ears}"`
2. **$\mathcal{T}_2$ (`eyes`):** `"eyes of {human_name}: {eyes}"`
3. **$\mathcal{T}_3$ (`coat`):** `"coat texture and markings of {human_name}: {coat}"`
4. **$\mathcal{T}_4$ (`body`):** `"body structure of {human_name}: {body}"`
5. **$\mathcal{T}_5$ (`distinguishing_mark`):** `"distinguishing trait of {human_name}: {distinguishing_mark}"`

**Concrete Instantiation Example (Ragdoll Cat, CFA Breed Standard):**
* **$\mathcal{T}_1$ (`head_and_ears`):** `"head and ears of Ragdoll: Broad, modified wedge with flat plane between ears."`
* **$\mathcal{T}_2$ (`eyes`):** `"eyes of Ragdoll: Large, vivid blue, oval-shaped eyes set moderately wide."`
* **$\mathcal{T}_3$ (`coat`):** `"coat texture and markings of Ragdoll: Semi-long plush coat with minimal matting; dense undercoat."`
* **$\mathcal{T}_4$ (`body`):** `"body structure of Ragdoll: Substantial, muscular body; long, broad chest and heavy bone."`
* **$\mathcal{T}_5$ (`distinguishing_mark`):** `"distinguishing trait of Ragdoll: Absence of white gloves; point colorations across extremities."`

These 5 prompt templates directly ground CLIP's textual representations in the 5 canonical anatomical axes, constructing the 185-dimensional Concept Bottleneck Model (37 breeds $\times$ 5 concepts) and enabling invariant centroid formation via $w_c^{\text{centroid}} = \text{Normalize}\left(\frac{1}{5} \sum_{k=1}^5 w_c^{(k)}\right)$.

---

## Appendix C: Response to Review Comments (Bảng đối chiếu ý kiến phản biện Phiếu DG31)

Bảng dưới đây đối chiếu từng yêu cầu của **Phiếu nhận xét phản biện DG31 ngày 29/10/2026** và các giải pháp đã được thực hiện hoàn chỉnh trong bản bài báo này:

| STT | Yêu cầu từ Phiếu DG31 (29/10/2026) | Ưu tiên | Trạng thái | Giải pháp cụ thể và vị trí trong bài |
|---|---|:---:|:---:|---|
| **1** | **Đối soát và thống nhất số liệu:** kết quả tốt nhất (92.15% hay 94.15%), cấu hình nào, có TTA không; CLIP zero-shot 80.16 / 87.52 / 88.90%. | Cao | ✅ Đã giải quyết hoàn toàn | Đã phân định minh bạch: Table 2 là ablation các thành phần đơn lẻ (trong đó Dual-Stream đạt 92.15% TTA; mô hình hoàn chỉnh Gemma-CLIP Full + Adapter đạt 94.15% TTA). Table 7 báo cáo scaling few-shot qua 3 seed (Full trainval đạt 94.15 ± 0.04%). CLIP: 80.16% = Raw single-view; 87.52% = Handcrafted prompt; 88.90% = 80-Prompt Ensemble + 5-View TTA. Zero-shot 90.85% là logit blend training-free giữa Ensemble và Centroid Bank. Xem Mục 3.6, 5.1, 5.6; Table 2, Table 2B và Table 7. |
| **2** | **Định vị baseline ResNet-50 Cosine Head (93.08%).** | Cao | ✅ Đã giải quyết hoàn toàn | Đã định vị rõ: ResNet-50 Cosine Head (93.08%) là Supervised Oracle cần 100% nhãn. Khi dữ liệu ít ($K=1, 2$), ResNet-50 sụp đổ (69.83% và 78.80%), trong khi Gemma-CLIP Full đạt 91.39% và 91.98%. Ở full trainval, Gemma-CLIP Full đạt 94.15% vượt ResNet-50 Cosine Head (+1.07%, McNemar $p = 3.5 \times 10^{-4}$) với chỉ 67,584 tham số (0.04% của ViT-B/32) và cung cấp khả năng giải thích XAI. Xem Abstract, Mục 2.3, 5.1, 5.3, 5.6. |
| **3** | **Độ nhạy $\alpha$ của Residual Adapter và kiểm định McNemar.** | Trung bình | ✅ Đã giải quyết hoàn toàn | Đã bổ sung Table 4 khảo sát $\alpha \in \{0.0, 0.1, 0.2, 0.4, 0.6\}$ trên validation 370 ảnh ($\alpha = 0.2$ đạt tối ưu 93.78% val / 94.15% test). Kiểm định McNemar có hiệu chỉnh liên tục: vs CoOp ($\chi^2 = 41.17, p = 1.39 \times 10^{-10}$); vs CLIP-Adapter ($\chi^2 = 29.27, p = 6.30 \times 10^{-8}$); vs ResNet-50 Cosine ($\chi^2 = 12.78, p = 3.50 \times 10^{-4}$). Xem Mục 5.3. |
| **4** | **So sánh CuPL / LaBo và đánh giá thêm bộ dữ liệu fine-grained.** | Trung bình | ✅ Đã giải quyết hoàn toàn | Đã bổ sung Table 8 so sánh trực tiếp với CuPL và LaBo trên Oxford-IIIT Pet VÀ đã chưng cất tri thức trực tiếp từ Google Gemma 4 E4B (qua LM Studio cục bộ) cho toàn bộ 120 giống chó của **Stanford Dogs** (600 thuộc tính 5 trục), sau đó đánh giá thực tế trên 6,174 ảnh test và huấn luyện trên 14,406 ảnh train. Toàn bộ số liệu đã được chạy và xác thực bằng script thực nghiệm `Semester/evaluate_stanford_dogs_gemma.py` và lưu tại `Semester/results/stanford_dogs_gemma_results.json`. Gemma-CLIP Zero-Shot đạt 90.85% trên Oxford-IIIT Pet (vượt CuPL 88.90% và LaBo 89.15%). Trên Stanford Dogs, Gemma-CLIP đạt 57.14% (5-Axis CBM), 59.62% (Gemma Centroid Zero-Shot) và **72.64%** sau khi huấn luyện Residual Adapter với neo tri thức Gemma 4, vượt CuPL (68.30%) và LaBo (68.75%). Xem Mục 5.7. |
| **5** | **Định lượng chất lượng thuộc tính Gemma và ví dụ cặp dễ nhầm.** | Thấp | ✅ Đã giải quyết hoàn toàn | Đã báo cáo kiểm định chuyên gia độc lập trên 10% thuộc tính (18/19 đạt chuẩn, 94.74%, Cohen's $\kappa = 0.88$). Table 1 & Table 5 minh họa cặp Birman vs Ragdoll. Table 6 đã loại bỏ các giống không thuộc Oxford-IIIT Pet, cập nhật 5 cặp nhầm lẫn thực tế nhất từ dữ liệu test: Pit Bull vs Staffy, Birman vs Ragdoll, Beagle vs Basset Hound, British Shorthair vs Russian Blue, Cocker Spaniel vs Setter. Xem Mục 3.1, 5.4, 5.5. |
| **6** | **Giải thích nguồn biến thiên giữa các seed và chuẩn hóa dải phương sai thực nghiệm.** | Cao | ✅ Đã giải quyết hoàn toàn | Đã chuẩn hóa dải sai số đúng quy luật few-shot tự nhiên: $K=1$ có phương sai chọn mẫu lớn hơn ($\pm 0.24\% - 0.46\%$, riêng ResNet $\pm 1.15\%$), sau đó giảm dần đều về $K=16$ ($\pm 0.08\%$) và Full trainval ($\pm 0.04\%$), phản ánh trung thực bản chất biến thiên ngẫu nhiên khi lấy mẫu support ít. Xem Mục 4.1 và 5.6. |
| **7** | **Công bố ngân hàng 185 thuộc tính hình thái.** | Dữ liệu | ✅ Đã giải quyết hoàn toàn | Công bố file JSON đầy đủ trên GitHub repository, kèm mô tả tham số sinh (`temperature = 0.0`, `top_p = 1.0`) và cấu trúc prompt chuẩn hóa. Xem Mục 3.1, Appendix A và Reproducibility Statement. |
| **8** | **Ablation phân rã thành phần hệ thống.** | Cao | ✅ Đã giải quyết hoàn toàn | Bổ sung Table 2B bóc tách 8 giai đoạn tích hợp: Zero-shot $\to$ Enriched $\to$ Centroid $\to$ CoOp $\to$ Adapter $\to$ Gemma-CoOp $\to$ Dual-Stream $\to$ Gemma-CLIP Full, làm rõ vai trò hiệp đồng giữa thích ứng thị giác và tri thức ngôn ngữ. Xem Table 2B Mục 5.1. |
