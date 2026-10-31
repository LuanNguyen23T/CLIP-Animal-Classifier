# GEMMA-CLIP: Fine-Grained Pet Breed Classification with Morphological Knowledge, Prompt Optimization and Multi-View Inference

**Nguyen Huu Hoang Luan, Ton That Van**  
*Department of Information Technology, University of Science and Technology – The University of Danang*  
Email: 102230028@sv1.dut.udn.vn, 102230055@sv1.dut.udn.vn  
GitHub: [https://github.com/LuanNguyen23T/CLIP-Animal-Classifier](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier)

---

## Abstract

Fine-Grained Visual Categorization (FGVC) is hindered by minute inter-class differences and large intra-class variation. While Contrastive Language-Image Pre-training (CLIP) provides strong zero-shot capabilities, standard prompts lack explicit anatomical cues, and parameter-efficient fine-tuning (PEFT) adapters act as uninterpretable black boxes prone to low-shot instability. We propose **Gemma-CLIP**, an explainable, parameter-efficient framework coupling a frozen CLIP ViT-B/32 backbone with morphological domain knowledge distilled autonomously from an on-device Small Language Model (Google Gemma 4 E4B SLM). Gemma-CLIP integrates: (i) an interpretable 185-attribute concept bottleneck space (Gemma-CBM), (ii) a non-parametric morphological centroid bank (Gemma-Centroid), (iii) Gemma-CoOp soft prompt tuning with 24-token Compact Trait Conditioning to mitigate attention gradient dilution, (iv) a centroid-anchored residual visual adapter (Gemma-VisAdapter), and (v) an aspect-ratio-preserving 5-view test-time augmentation (TTA) pipeline.

Evaluated on the official Oxford-IIIT Pet benchmark (3,669 test images) across three independent random seeds, Gemma-CLIP demonstrates comprehensive gains across all supervision regimes. In the zero-shot setting ($K=0$), training-free logit blending (Gemma-ZeroShot) reaches **90.85%** under 5-View TTA, outperforming raw class names (80.16%) and the 80-prompt CLIP ensemble (88.90%). In few-shot adaptation ($K=1$), Gemma-CLIP Full achieves **91.39%**, surpassing standard CoOp (88.72%) and fully-supervised ResNet-50 Cosine (69.83%, $\Delta = +21.56\%$). Under full training, Gemma-CLIP Full achieves **94.15%** Top-1 accuracy with 67,584 trainable parameters (0.04% of ViT-B/32), outperforming fully-supervised ResNet-50 Cosine (93.08%, $p = 3.5 \times 10^{-4}$). On Apple Silicon MPS, classification head latency is 0.18 ms (5,652 FPS), with the full 5-View TTA pipeline sustaining 17.2 FPS. On Stanford Dogs (120 breeds, 600 concepts), Gemma-distilled priors elevate zero-shot accuracy to 59.62% and supervised adaptation to 73.06% (Gemma-CLIP Full).

**Keywords:** FGVC, Vision-Language Models (VLM), CLIP, Gemma 4 SLM, CoOp, CBM, TTA.

---

## 1. Introduction

FGVC requires vision systems to resolve minute, localized anatomical boundaries amidst large posture and illumination variations [1]. In Oxford-IIIT Pet, Ragdoll and Birman cats share pointed cream coats and blue eyes; their decisive discriminator lies in Birman's symmetrical white paw "gloves". Similarly, Staffordshire Bull Terriers and American Pit Bull Terriers share muscular builds, separated primarily by cheekbone width to mandibular taper ratios.

Global visual classifiers frequently fail on such localized cues. Contrastive Language-Image Pre-training (CLIP; Radford et al. [2]) enables zero-shot classification via cosine similarity between visual embedding $f(x)$ and text prompt embedding $w_c$, yet generic prompts (*"a photo of a {class}"*) omit diagnostic morphology. While parameter-efficient fine-tuning (PEFT) methods—such as CoOp [3], CLIP-Adapter [4], and Tip-Adapter [5]—learn continuous prompts or residual bottlenecks, they exhibit key vulnerabilities: learned continuous vectors act as uninterpretable black boxes lacking anatomical grounding; unconstrained context optimization overfits limited data ($K \le 2$), dropping below zero-shot baselines (CoOp $K=1$: 88.72% vs. 88.90%); naively appending long descriptive sentences disperses backpropagation gradients across CLIP's 12 causal self-attention layers; and prior LLM-guided vision approaches (e.g., CuPL, LaBo) rely on remote proprietary cloud APIs (GPT-3/4), incurring monetary costs, network latency, and privacy vulnerabilities.

To address these challenges, we propose **Gemma-CLIP**, an explainable, parameter-efficient framework coupling an on-device Gemma 4 SLM [8] with CLIP ViT-B/32. The framework introduces a transparent 185-dimensional concept bottleneck space (Gemma-CBM) for clinical attribution, an invariant non-parametric morphological centroid bank (Gemma-Centroid) across 5 canonical anatomical axes, continuous prompt optimization anchored by 24-token Compact Trait summaries (Gemma-CoOp) to preserve attention gradients, a lightweight residual visual bottleneck MLP ($512 \to 64 \to 512$, $\alpha = 0.2$) anchored to Centroid weights (Gemma-VisAdapter), and dual-stream logit fusion ($\beta = 0.5$). When jointly co-adapted with aspect-ratio-preserving 5-View TTA, Gemma-CLIP Full achieves 94.15% Top-1 accuracy with only 67,584 trainable parameters.

---

## 2. Related Work & Baseline Definitions

### 2.1. Vision-Language Models (CLIP)
CLIP [2] maps visual $f(x)$ and text $w_c$ embeddings into a shared normalized space $\mathbb{R}^D$ ($D=512$). Class probabilities are given by:

$$P(y = c \mid x) = \frac{\exp(\tau \langle f(x), w_c \rangle)}{\sum_{j=1}^C \exp(\tau \langle f(x), w_j \rangle)}$$

where $\tau$ is the learned logit scale. Evaluated on Oxford-IIIT Pet, raw class names achieve 80.16%, while handcrafted 80-prompt ensembles achieve 88.90% under 5-View TTA.

### 2.2. PEFT Baselines
CoOp [3] models prompts as $M$ learnable continuous tokens $[V]_1, \dots, [V]_M \in \mathbb{R}^d$ prepended to fixed class embeddings: $t_c = [[V]_1, \dots, [V]_M, e_c]$ ($M=4, d=512$, 2,048 parameters). CLIP-Adapter [4] inserts a two-layer bottleneck MLP on image features, while Tip-Adapter [5] builds a non-parametric key-value cache fine-tuned in Tip-Adapter-F.

### 2.3. Supervised Upper-Bound References (Oracle Baselines)
Supervised models—such as ResNet-50 Cosine Head [13] (75,815 parameters) and ResNet-50 Prototype [11]—train directly on 100% of labeled training data (3,680 images). While achieving strong full-data accuracy (93.08%), they collapse in few-shot regimes ($K=1$: 69.83%) due to the absence of multimodal foundation priors. We treat them as supervised references rather than architectural peers.

### 2.4. Concept Bottlenecks (CBM) & LLM Prompting (CuPL, LaBo)
CBMs [6] route predictions through interpretable concept scores. While CuPL [17] and LaBo [15] query cloud-based GPT-3 (175B), Gemma-CLIP departs in three ways: (i) executes an on-device Gemma 4 SLM locally with zero API cost; (ii) constrains descriptors to AKC/FCI and CFA/TICA standards across 5 anatomical axes; and (iii) anchors PEFT prompt and visual adapter tuning directly to morphological priors.

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
                                 │ Gemma-VisAdapter│ ── Residual MLP: f'(x)
                                 └────────┬────────┘    (512 → 64 → 512, α = 0.2)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ Stream 1: Gemma-CoOp ]                        [ Stream 2: Gemma-Centroid ]
  Continuous Soft Prompts Anchored                185-Attribute Non-parametric Bank
  by Compact Traits (24 tokens)                   Distilled from Local Gemma 4 SLM
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
*Figure 1. Architecture of Gemma-CLIP: coupling visual residual adaptation, parameterized Gemma-CoOp prompt tuning, and non-parametric morphological centroids.*

### 3.1. Local Morphological Knowledge Distillation via Gemma 4 SLM
Gemma 4 E4B (`google/gemma-4-e4b`) [8] is served locally via LM Studio with deterministic greedy decoding (`temp=0.0`, `seed=42`). For each class $c$, the SLM generates structured descriptors across five anatomical axes: `head_and_ears`, `eyes`, `coat`, `body`, and `distinguishing_mark` (Appendix A). Non-visual descriptors are excluded. A 10% stratified spot-check (19 descriptors) confirmed 18/19 (94.7%) clinical fidelity against AKC/FCI and CFA/TICA breed standards (Appendix A).

*Table 1. Distilled Morphological Descriptors for Primary Confusable Breed Pairs.*

| Confusable Pair | Breed & Registry | Key Anatomical Descriptors (Gemma 4 Distilled) | Decisive Distinguishing Feature |
|---|---|---|---|
| **Feline Pair** | **Birman** *(CFA)* | • `head`: Broad, rounded skull contour.<br>• `eyes`: Round, deep sapphire-blue.<br>• `coat`: Silky semi-longhair, no undercoat. | **Pure white symmetrical "gloves"** on all four paws. |
| | **Ragdoll** *(CFA)* | • `head`: Broad modified wedge, flat plane.<br>• `eyes`: Large, vivid blue, oval.<br>• `coat`: Semi-long plush, dense undercoat. | **Absence of white gloves**; point colorations across extremities. |
| **Canine Pair** | **Staffordshire Bull Terrier** *(AKC)* | • `head`: Broad skull; semi-pricked close ears.<br>• `body`: Compact, low-slung, short tail.<br>• `coat`: Short, dense, sleek double coat. | **Blocky, low-slung silhouette** with extreme muscle density. |
| | **American Pit Bull Terrier** *(UKC)* | • `head`: Broad skull; semi-erect high-set ears.<br>• `body`: Leggier frame, pronounced chest depth.<br>• `coat`: Extremely short, smooth single coat. | **Athletic, leggy taper** with pronounced chest depth. |

### 3.2. 185-Dimensional CBM Space & Morphological Centroids
Each raw attribute $a_c^{(k)}$ ($k \in \{1..5\}$) is formatted via template $\mathcal{T}_k(\cdot)$ (Appendix B) and projected through frozen CLIP text encoder $\mathcal{E}_{\text{txt}}$ into normalized vector $w_c^{(k)} \in \mathbb{R}^{512}$.

In the **Gemma-CBM** stream, stacking all 185 concept vectors forms the clinical concept projection matrix $W_{\text{concept}} \in \mathbb{R}^{185 \times 512}$. For any image embedding $f(x)$, concept activation scores are computed via inner products $s_{c,k}(x) = \langle f(x), w_c^{(k)} \rangle$, and class predictions are produced through uniform pooling across the five anatomical concepts: $\hat{y}_{\text{CBM}} = \arg\max_{c} \frac{1}{5} \sum_{k=1}^5 s_{c,k}(x)$.

In parallel, the **Gemma-Centroid** stream constructs an invariant morphological centroid representation for each class $c$ by normalizing the average of its five concept projections: $w_c^{\text{centroid}} = \text{Normalize}\left(\frac{1}{5} \sum_{k=1}^5 w_c^{(k)}\right)$. These class centroids form the non-parametric text matrix $W_{\text{centroid}} \in \mathbb{R}^{C \times 512}$, serving as fixed semantic anchors for inference and adaptation.

### 3.3. Residual Visual Adaptation & Gemma-CoOp Prompt Optimization

#### 3.3.1. Gemma-VisAdapter with Prior Weight Anchoring
Visual adaptation employs a lightweight two-layer residual MLP ($512 \to 64 \to 512$, 65,536 parameters):
$$\Delta f(x) = W_2 \cdot \text{ReLU}(W_1 f(x))$$
Zero-initialization ($W_2 \leftarrow \mathbf{0}$) guarantees $f'(x) \equiv f(x)$ at step 0, preserving pre-trained representations. The adapted visual embedding is:
$$f'(x) = \text{Normalize}\left((1 - \alpha)f(x) + \alpha \cdot \Delta f(x)\right)$$
where $\alpha = 0.2$ anchors features to the Gemma-Centroid subspace through $z_c^{\text{adapter}} = \tau \langle f'(x), w_c^{\text{centroid}} \rangle$.

#### 3.3.2. Gemma-CoOp with Compact Trait Conditioning
Standard CoOp prepends soft prompt tokens to class names. Appending long descriptive sentences (45–60 tokens) causes attention gradient dilution across CLIP's 12 causal self-attention layers because text representations pool strictly from `[EOS]`:
$$\frac{\partial \mathcal{L}}{\partial [V]_i} = \sum_{j=i}^{N_{\text{tokens}}} \frac{\partial \mathcal{L}}{\partial h_L^j} \frac{\partial h_L^j}{\partial [V]_i}$$
We bound sequence length to 24 tokens average via Compact Trait Conditioning:
$$p_c = [V]_1 [V]_2 [V]_3 [V]_4 \oplus \text{Tokenize}("\{breed\}, \{species\}, \text{ recognized by } \{concise\_trait\}") $$
maintaining short gradient paths to $[V]_1 \dots [V]_4$ (2,048 parameters).

### 3.4. Aspect-Ratio Preserving 5-View TTA Pipeline
To prevent anatomical distortion from naive square resizing, images are isotropically resized (short edge 224) and center-cropped. Five complementary views are generated per test sample: (1) Canonical crop, (2) Bilateral edge-preserving filter ($d=7, \sigma=50$), (3) CIELAB CLAHE contrast enhancement ($8\times 8$ grid, clip limit 2.0), (4) Cascaded Bilateral + CLAHE, and (5) Multi-scale zoom ($1.15\times$, horizontal flip, $15^\circ$ rotation). The test embedding is $f_{\text{TTA}}(x) = \text{Normalize}\left(\frac{1}{5} \sum_{v=1}^5 f_v(x)\right)$, operating at 17.2 FPS.

### 3.5. Dual-Stream Synergistic Logit Fusion & Loss
Logits from Gemma-CoOp and Gemma-Centroid are blended:
$$z_c^{\text{fusion}} = z_c^{\text{Gemma-CoOp}} + \beta \cdot z_c^{\text{centroid}} \quad (\beta = 0.5)$$
Learnable parameters ($[V]_{1..4}, W_1, W_2$) are trained with cross-entropy loss:
$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\tau\, z^{\text{fusion}}_{y_i})}{\sum_{c=1}^{C}\exp(\tau\, z^{\text{fusion}}_{c})}$$

### 3.6. Standardized Architectural Taxonomy
To guarantee unambiguous evaluation across supervision regimes and experimental protocols, we establish a standardized architectural taxonomy spanning training-free and parameterized variants.

In the **training-free regime (0 trainable parameters)**, three configurations are evaluated: *Gemma-CBM* projects image features onto the 185-axis (Pets) or 600-axis (Dogs) concept space for transparent attribution; *Gemma-Centroid* performs standalone cosine matching against the 5-axis morphological centroid bank $W_{\text{centroid}}$; and *Gemma-ZeroShot* combines the standard CLIP 80-prompt ensemble with Gemma-Centroid via training-free logit blending ($\beta = 0.5$).

In the **prompt-tuning regime (2,048 parameters)**, *Gemma-CoOp* optimizes four continuous prompt tokens conditioned on 24-token Compact Traits to mitigate attention gradient dilution. *Gemma-DualStream* blends the adapted logits of Gemma-CoOp with the invariant morphological priors of Gemma-Centroid ($\beta = 0.5$).

In the **visual and joint adaptation regime (65,536 to 67,584 parameters)**, *Gemma-VisAdapter* (65,536 parameters) trains a residual bottleneck MLP ($512 \to 64 \to 512$, $\alpha = 0.2$) anchored to fixed centroid weights. *Gemma-CLIP Lite* (65,536 parameters) couples this visual adapter with dual-stream logit blending of handcrafted prompts and Gemma centroids without optimizing soft prompt tokens, achieving 90.79% (91.17% TTA) on Oxford-IIIT Pet and 72.68% on Stanford Dogs. Finally, *Gemma-CLIP Full* (67,584 parameters) achieves unified multimodal co-adaptation by jointly optimizing Gemma-VisAdapter (65,536 parameters) and Gemma-CoOp prompt tokens (2,048 parameters) combined with Gemma-Centroid priors and 5-View TTA, reaching 93.65% TTA (9:1 partition) and 94.15% TTA (Full Trainval) on Oxford-IIIT Pet, and 73.06% on Stanford Dogs. Performance is reported across both benchmarks under standardized evaluation protocols.

---

## 4. Experimental Setup

### 4.1. Dataset & Partition Protocols
Evaluations are conducted on the official Oxford-IIIT Pet benchmark [1], containing 7,349 images across 37 cat and dog breeds partitioned into 3,680 trainval images and 3,669 test images. To ensure rigorous separation between hyperparameter selection and final testing, the 3,680 trainval images are stratified into a 9:1 training-validation partition (3,310 training images, 370 validation images). Optimal hyperparameters ($\alpha = 0.2$, $\beta = 0.5$) are selected strictly on this 370-image validation split, while the 3,669 test images remain strictly unobserved until final inference. To assess data volume scaling, models configured with these optimal hyperparameters are subsequently retrained on the full 3,680 trainval set.

To account for stochastic experimental variance, all few-shot evaluations ($K \in \{1, 2, 4, 8, 16\}$ support images per class) are repeated across three independent random seeds (Seed 1, Seed 2, Seed 3), reporting mean accuracy and sample standard deviation ($\text{mean} \pm \text{std}$). Each seed jointly dictates the stratified random sampling of the $K$ support images per breed and the parameter initialization of continuous prompt vectors $[V]_{1..4}$ and adapter weights. Full trainval runs are likewise evaluated across all three seeds to verify initialization stability.

### 4.2. Comparative Baselines & Supervised Oracle Reference
We benchmark our framework against three representative paradigm categories using frozen `openai/clip-vit-base-patch32` backbones. First, training-free zero-shot baselines comprise CLIP Raw Class Names [2], Enriched Handcrafted Prompts, and the standard 80-Prompt Ensemble. Second, parameter-efficient adaptation baselines encompass Tip-Adapter [5], Tip-Adapter-F [5], CLIP-Adapter [4], and CoOp [3]. Third, to establish an empirical upper bound, we compare against fully-supervised CNN references trained directly on 100% of labeled training data: the ResNet-50 Prototype Classifier [11], ResNet-50 Linear Probe [12], ResNet-50 Residual Adapter, and ResNet-50 Cosine Head [13]. These supervised CNN models serve as oracle reference baselines rather than architectural peers due to their reliance on complete supervision.

### 4.3. Training Hyperparameters & Implementation Details
Models are optimized using AdamW with an initial learning rate $\eta = 1 \times 10^{-3}$ for Gemma-VisAdapter and $\eta = 2 \times 10^{-3}$ for Gemma-CoOp, weight decay $1 \times 10^{-4}$, batch size 256, and a cosine annealing learning rate schedule over 50 epochs. Peak validation accuracy occurs at Epoch 22 for the visual adapter and Epoch 38 for Gemma-CoOp. All experiments are executed on Apple Silicon GPU via Metal Performance Shaders (MPS). Performance is quantified using Top-1 Accuracy (%), Top-5 Accuracy (%), and Macro-averaged F1 Score (%).

---

## 5. Experimental Results & Discussion

### 5.1. Comprehensive Benchmark on Oxford-IIIT Pet

Table 2 evaluates all architectures on the 3,669-image test split, comparing raw single-view vs. 5-View TTA across zero-shot, PEFT, and supervised oracle baselines.

*Table 2. Architectural Ablation Performance on Official Test Split (3,669 Images) under Hyperparameter Validation Split (3,310 Train / 370 Val Protocol).*

| Paradigm Category | Evaluated Architecture | Adaptation Mechanism | Trainable Params | Top-1 (%) Raw (Ours)<sup>†</sup> | Top-1 (%) Raw (Source) | Top-1 (%) 5-View TTA<sup>†</sup> | Top-5 (%) 5-View TTA | Macro F1 (%) 5-View TTA |
|---|---|---|---|---|---|---|---|---|
| **CLIP Baselines** | CLIP Raw Class Names | Zero-Shot Basic | 0 | 80.16% | — | 80.38% | 96.40% | 76.75% |
| | CLIP Handcrafted Prompt | Zero-Shot Enriched | 0 | 87.52% | 87.50% [2] | 87.76% | 99.40% | 86.94% |
| | CLIP 80-Prompt Ensemble | Zero-Shot Multi-Prompt | 0 | 88.42% | — | 88.90% | 99.48% | 88.62% |
| **PEFT Baselines**<sup>‡</sup> | Tip-Adapter | Key-Value Cache | 0 | 87.00% | 87.20% [5] | 88.06% | 98.72% | 87.94% |
| | Tip-Adapter-F (9:1 Split) | Tuned Cache | 3,680 | 89.10% | 89.02% [5] | 89.56% | 98.96% | 89.48% |
| | CLIP-Adapter (9:1 Split) | Visual Residual MLP | 65,536 | 89.42% | 89.30% [4] | 91.28% | 99.10% | 91.22% |
| | CoOp Baseline (9:1 Split) | Soft Continuous Tokens | 2,048 | 91.20% | 88.20% [3] | 91.52% | 99.65% | 91.40% |
| **Supervised Ref. (Oracle)** | ResNet-50 Prototype [11] | Centroid Nearest | 0 (Labels) | 91.36% | 86.02% | 91.69% | 99.65% | 91.54% |
| | ResNet-50 Linear Probe [12] | Linear Classifier | 75,887 | 92.07% | 85.90% | 92.40% | 99.30% | 92.34% |
| | ResNet-50 Residual Adapter | Bottleneck MLP | 2,174,013 | 92.48% | — | 92.64% | 99.40% | 92.58% |
| | ResNet-50 Cosine Head [13] | Normalized Cosine Head | 75,815 | 92.78% | 87.40% | 93.08% | 99.32% | 93.02% |
| **Gemma-CLIP Family (Ours)** | **Gemma-CBM**<sup>§</sup> | 185-Concept Projection | 0 | 80.40% | — | 80.02% | 97.77% | 77.45% |
| | **Gemma-Centroid** | Training-Free Centroid Bank | 0 | 88.20% | — | 88.25% | 99.51% | 88.18% |
| | **Gemma-ZeroShot** | 80-Prompt + Centroid Blend | 0 | 90.41% | — | 90.85% | 99.70% | 90.72% |
| | **Gemma-VisAdapter (9:1 Split)** | Residual Visual Adapter | 65,536 | 91.03% | — | 91.41% | 99.81% | 91.33% |
| | **Gemma-CoOp (9:1 Split)** | Anchored Soft Prompt | 2,048 | 91.11% | — | 91.66% | 99.81% | 91.62% |
| | **Gemma-DualStream (9:1 Split)** | CoOp + Centroid Blend | 2,048 | 91.74% | — | 92.15% | 99.81% | 92.09% |
| | **Gemma-CLIP Full (9:1 Split)** | CoOp + VisAdapter + Centroid | 67,584 | **92.86%** | — | **93.65%** | **99.85%** | **93.60%** |
| | **Gemma-CLIP Full (Full Trainval)** | CoOp + VisAdapter + Centroid | 67,584 | **93.20%** | — | **94.15%** | **99.89%** | **94.10%** |

> <sup>†</sup> *Table 2 reports performance on the official 3,669 test split. '(9:1 Split)' denotes models trained on 3,310 training images with 370 validation images for hyperparameter tuning. 'Full Trainval' applies optimal hyperparameters ($\alpha=0.2, \beta=0.5$) over all 3,680 images (cross-seed mean: 94.15 ± 0.04%, representative single-run: 94.14%).*  
> <sup>§</sup> *Gemma-CBM is designed for transparent clinical attribution rather than unconstrained categorization.*

A systematic comparison across Table 2 highlights critical architectural trade-offs between supervised learning and foundation adaptation. While the supervised ResNet-50 Prototype baseline reaches 91.69% TTA, it requires 100% labeled supervision across all 3,680 training images, matching the performance of standalone Gemma-CoOp (91.66% TTA) which tunes only 2,048 prompt parameters on frozen representations. When visual residual adaptation and prompt tuning are unified into Gemma-CLIP Full (94.15% TTA), our framework surpasses the fully-supervised ResNet-50 Cosine Head (93.08%, $\Delta = +1.07\%$, McNemar $p = 3.5 \times 10^{-4}$) while utilizing only 67,584 trainable parameters (0.04% of ViT-B/32). Crucially, while supervised CNNs suffer catastrophic performance collapse under data scarcity (plunging to 69.83% at $K=1$ as shown in Table 7), Gemma-CLIP retains robust multimodal zero-shot grounding from the first shot.

Comparing the linguistic representations, standalone Gemma-CBM achieves 80.02% TTA, comparable to raw zero-shot CLIP (80.16%). Projecting images into 185 independent concept axes introduces inter-concept correlations and semantic overlap across closely related breeds, slightly diluting classification margins. Consequently, Gemma-CBM is deployed specifically for transparent morphological attribution (Section 5.4), whereas classification accuracy is maximized through non-parametric centroid integration and prompt optimization.

Beyond baseline comparisons, Table 2 traces the stepwise progression across model configurations under 5-View TTA, isolating the precise contribution of each architectural module from training volume expansion. In the zero-shot regime, transitioning from raw class names (80.38%) to handcrafted prompts yields 87.76% (+7.38%), which is further elevated by our non-parametric Gemma-Centroid prior to 88.25% (+0.49%) and by training-free linguistic fusion in Gemma-ZeroShot to 90.85% (+2.60%). In the parameter-efficient adaptation regime evaluated on the 9:1 training subset (3,310 images), continuous prompt optimization via Gemma-CoOp achieves 91.66%, while Gemma-DualStream logit fusion reaches 92.15% (+0.49%). Coupling the visual residual bottleneck in Gemma-CLIP Full achieves 93.65% TTA (3,436/3,669 correct, +1.50% over dual-stream). Crucially, extending training from the 3,310-image subset to the full 3,680 trainval set reaches 94.15% TTA (+0.50%), strictly isolating this terminal gain as a data volume scaling effect rather than an unverified architectural module.

---

### 5.2. Prompt Conditioning Structure & Empirical Analysis of Prompt Length

Table 3 evaluates prompt structures in CoOp to analyze prompt length and test the Attention Gradient Dilution hypothesis.

*Table 3. Impact of Prompt Conditioning Length and Token Structure on CoOp.*

| Evaluated Prompt Variant | Semantic Template Structure | Token Length | Test Raw Top-1 (%) | Test TTA Top-1 (%) | Top-5 Acc (%) | Macro F1 (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Baseline CoOp | `[V1..V4] {breed}` | 6 | 91.20% | 91.52% | 99.65% | 91.40% |
| CoOp + Verbose Mark | `[V1..V4] {breed} distinguished by {mark}` | 48 | 90.95% | 91.58% | 99.73% | 91.55% |
| CoOp + Verbose Body & Head | `[V1..V4] {breed} a pet with {coat} and {head}` | 56 | 91.33% | 91.58% | 99.73% | 91.52% |
| **Gemma-CoOp (Compact Trait)** | `[V1..V4] {breed}, {species} recognized by {trait}` | **24** | **91.11%** | **91.66%** | **99.81%** | **91.62%** |
| **Gemma-DualStream** | Gemma-CoOp + Gemma-Centroid | — | **91.74%** | **92.15%** | **99.81%** | **92.09%** |

Appending overly long descriptive clauses (48 tokens) introduces slight degradation in raw single-view accuracy (91.20% $\to$ 90.95%), consistent with gradient dispersion across CLIP's 12 causal self-attention layers. Constraining prompts to a 24-token Compact Trait structure provides an empirically effective compromise: it bounds sequence length while injecting concrete morphological descriptors, achieving 91.66% TTA and reaching 92.15% under Gemma-DualStream.

---

### 5.3. Residual Adapter Sensitivity ($\alpha$ Ablation) & McNemar Test

To determine the optimal interpolation between pre-trained visual representations and fine-tuned features, we conduct a parametric sensitivity analysis over blending factor $\alpha \in \{0.0, 0.1, 0.2, 0.4, 0.6\}$. In this evaluation protocol, Gemma-VisAdapter (two-layer bottleneck MLP $512 \to 64 \to 512$, $W_2 = \mathbf{0}$, 65,536 parameters) operates jointly with Gemma-CoOp (2,048 prompt parameters) and is anchored to fixed Gemma-Centroid weights ($\beta = 0.5$). For each candidate $\alpha$, the network is trained from scratch for 35 epochs exclusively on the 3,310-image training partition. Model selection is performed strictly at peak validation accuracy on the 370-image validation split, ensuring the 3,669 test images remain strictly unobserved during hyperparameter tuning. When $\alpha = 0.0$, the visual adapter is disabled ($f'(x) \equiv f(x)$), reverting to Gemma-DualStream (2,048 parameters); for $\alpha > 0$, visual and text prompt parameters are optimized jointly (67,584 parameters).

*Table 4. Sensitivity Analysis of Residual Adapter Blending Weight $\alpha$ on Validation Split (Selected on Validation Set Only).*

| Blending Factor $\alpha$ | Validation Top-1 Acc (%) | Test Set Top-1 Acc (TTA) (%) | Trainable Params | Architectural Interpretation |
|---|:---:|:---:|:---:|---|
| $\alpha = 0.0$ (No Visual Adapter) | 92.43% | 92.15% | 2,048 | Gemma-DualStream (Gemma-CoOp + Gemma-Centroid) |
| $\alpha = 0.1$ | 93.24% | 92.95% | 67,584 | Light visual refinement; strong foundation preservation |
| **$\alpha = 0.2$ (Optimal)** | **93.78%** | **93.65% (94.15% Full)<sup>*</sup>** | **67,584** | **Gemma-CLIP Full (Optimal balance of prior & adaptation)** |
| $\alpha = 0.4$ | 92.70% | 93.02% | 67,584 | Mild representation drift; slight zero-shot forgetting |
| $\alpha = 0.6$ | 90.81% | 91.14% | 67,584 | Severe overfitting on training distribution; degradation |

> <sup>*</sup> *On the 3,310 training subset, the optimal $\alpha = 0.2$ yields 93.65% TTA. Training across the full 3,680 trainval set reaches 94.15% cross-seed mean (representative run: 94.14%), matching Table 2 and Table 7.*

As detailed in Table 4, validation accuracy peaks at $\alpha = 0.2$ (93.78%, 347/370 correct validation images), corresponding to 93.65% TTA on the test split under the 9:1 partition protocol and reaching 94.15% when retrained on the full 3,680 trainval set. Setting $\alpha \ge 0.4$ induces representation drift that deviates excessively from CLIP's pre-trained hyperspherical manifold, degrading generalization to 90.81% validation accuracy at $\alpha = 0.6$.

To establish whether performance gains are statistically significant rather than stochastic artifacts, we conduct paired McNemar tests with continuity correction ($\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$) on representative full-data runs across the 3,669 test samples. When compared against the standard CoOp baseline (3,380 correct predictions, 92.12%), Gemma-CLIP Full achieves 3,455 correct predictions ($b = 104, c = 29, \chi^2 = 41.17, p = 1.39 \times 10^{-10} \ll 0.001$). Compared against CLIP-Adapter (3,393 correct, 92.48%), Gemma-CLIP Full yields 3,454 correct predictions ($b = 92, c = 31, \chi^2 = 29.27, p = 6.30 \times 10^{-8} \ll 0.001$). Furthermore, when tested against the fully-supervised ResNet-50 Cosine Head (3,415 correct, 93.08%), our framework demonstrates significant superiority ($b = 76, c = 37, \chi^2 = 12.78, p = 3.50 \times 10^{-4} < 0.001$). All three pairwise tests remain statistically significant after Holm–Bonferroni correction ($\alpha = 0.05$).

---

### 5.4. XAI Attribution Case Study (Gemma-CBM)

Table 5 details the diagnostic concept activation profile from Gemma-CBM (185 concepts) for a test specimen of Ragdoll evaluated against Birman.

*Table 5. Concept Activation Decomposition Across Five Anatomical Axes (Ragdoll vs. Birman).*

| Morphological Axis | Ragdoll Score | Birman Score | Discriminative Delta ($\Delta$) | Morphological Rationale |
|---|:---:|:---:|:---:|---|
| `head_and_ears` | 0.231 | 0.204 | +0.027 | Ragdoll favored by broad head structure |
| `eyes` | 0.218 | 0.215 | +0.003 | Near neutral (both exhibit deep blue oval eyes) |
| `coat` | 0.240 | 0.238 | +0.002 | Near neutral (both exhibit semi-longhair pointed coats) |
| `body` | 0.222 | 0.219 | +0.003 | Ragdoll favored by larger, muscular build |
| `distinguishing_mark` | **0.245** | **0.218** | **+0.027** | **Co-largest margin: absence of white paw gloves** |

The specimen is correctly classified as Ragdoll. While coat and eye scores are nearly identical, `head_and_ears` and `distinguishing_mark` yield the decisive margins (+0.027 each). Across all test Ragdoll images ($N = 100$), the distinguishing mark margin remains consistently positive ($\bar{\Delta} = +0.022 \pm 0.006$), confirming human-auditable concept-level attribution.

---

### 5.5. Morphological Distinction and Fine-Grained Ambiguity Analysis

Morphological descriptors provide the greatest separation for breeds with distinctive localized anatomy: Sphynx (hairless wrinkle texture, large wide-set ears), Persian (brachycephalic flat facial geometry, compact muzzle), and Bengal (rosetted leopard markings). Cat-vs-dog classification exhibits zero cross-species errors (0.00%).

Table 6 analyzes representative confusable breed pairs identified across the 37 classes of Oxford-IIIT Pet, reporting official test sample counts ($N = 89$ for Staffordshire Bull Terrier, $N = 100$ each for Birman, Beagle, British Shorthair, and English Cocker Spaniel).

*Table 6. Fine-Grained Morphological Boundary and Confusable Breed Pair Analysis on Oxford-IIIT Pet.*

| Ground Truth Breed | Test $N$ | Most Confusable Counterpart | Shared Phenotypic Features | Decisive Morphological Differentiator | Primary Photographic Ambiguity Factor |
|---|:---:|---|---|---|---|
| Staffordshire Bull Terrier | 89 | American Pit Bull Terrier | Muscular build, short smooth coat, broad skull | Cranial cheekbone width to mandibular taper ratio | Frontal perspective obscuring jaw taper |
| Birman | 100 | Ragdoll | Pointed semi-longhair coat, sapphire-blue eyes | Symmetrical white paw gloves on Birmans | Paws hidden under body posture/bedding |
| Beagle | 100 | Basset Hound | Tricolor hound coat, drop ears, white tail tip | Limb bone length ratio and pendulous ear folds | Cropped framing obscuring limb stature |
| British Shorthair | 100 | Russian Blue | Solid gray-blue dense coat, compact head | Eye coloration (amber/copper vs. emerald green) | Low-light sensor noise shifting eye hues |
| English Cocker Spaniel | 100 | English Setter | Feathered silky coat, ticked roan markings | Muzzle stop angle and wither height proportions | Close-up framing eliminating height cues |

Misclassifications concentrate on photographic occlusions (hidden paws, low-light hue shifts) rather than multimodal representation failures.

---

### 5.6. Zero-Shot & Few-Shot Scaling Across 3 Independent Seeds

Table 7 presents scaling trajectories across budgets from $K = 0$ (Zero-Shot) to $K = 16$ and Full Trainval across three independent random seeds (mean $\pm$ std).

*Table 7. Few-Shot and Full Trainval Scaling Benchmark (Mean ± Std over 3 Random Seeds).*

| Budget ($K$) | CLIP Zero-Shot (5-View TTA)<sup>†</sup> | CLIP + CoOp [3] | CLIP-Adapter [4] | Tip-Adapter-F [5] | Gemma-CoOp (Ours) | Gemma-CLIP Full (Ours) | ResNet-50 Cosine [13] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 0$** | **88.90 $\pm$ 0.00** | 88.90 $\pm$ 0.00 | 88.90 $\pm$ 0.00 | 88.90 $\pm$ 0.00 | **90.85 $\pm$ 0.00** | **90.85 $\pm$ 0.00** | N/A |
| **$K = 1$** | 88.90 $\pm$ 0.00 | 88.72 $\pm$ 0.46 | 88.83 $\pm$ 0.38 | 88.98 $\pm$ 0.31 | 91.04 $\pm$ 0.28 | **91.39 $\pm$ 0.24** | 69.83 $\pm$ 1.15 |
| **$K = 2$** | 88.90 $\pm$ 0.00 | 89.23 $\pm$ 0.38 | 89.41 $\pm$ 0.32 | 89.37 $\pm$ 0.25 | 91.38 $\pm$ 0.22 | **91.98 $\pm$ 0.19** | 78.80 $\pm$ 0.85 |
| **$K = 4$** | 88.90 $\pm$ 0.00 | 89.96 $\pm$ 0.29 | 90.15 $\pm$ 0.24 | 89.85 $\pm$ 0.18 | 91.97 $\pm$ 0.16 | **92.51 $\pm$ 0.14** | 87.38 $\pm$ 0.52 |
| **$K = 8$** | 88.90 $\pm$ 0.00 | 90.75 $\pm$ 0.21 | 90.91 $\pm$ 0.18 | 90.56 $\pm$ 0.14 | 92.31 $\pm$ 0.12 | **92.93 $\pm$ 0.10** | 89.13 $\pm$ 0.34 |
| **$K = 16$** | 88.90 $\pm$ 0.00 | 91.50 $\pm$ 0.16 | 91.60 $\pm$ 0.12 | 91.21 $\pm$ 0.10 | 92.64 $\pm$ 0.09 | **93.40 $\pm$ 0.08** | 90.24 $\pm$ 0.22 |
| **Full** | 88.90 $\pm$ 0.00 | 92.12 $\pm$ 0.08 | 92.48 $\pm$ 0.06 | 92.15 $\pm$ 0.05 | 93.18 $\pm$ 0.05 | **94.15 $\pm$ 0.04** | 93.08 $\pm$ 0.07 |

> <sup>†</sup> *All models operate under 5-View TTA across 3 seeds. At $K=0$, both Gemma-CoOp and Gemma-CLIP Full default to training-free **Gemma-ZeroShot** (80-prompt ensemble + Centroid bank at $\beta = 0.5$, 90.85%). At 'Full', all parameterized models are trained across 100% of the 3,680 trainval images.*

Three primary empirical dynamics emerge from the scaling curves in Table 7. First, Gemma-CLIP successfully resolves the severe low-shot overfitting vulnerability observed in unconstrained prompt optimization. While standard CoOp degrades below the zero-shot baseline at $K=1$ (88.72% vs. 88.90%), Gemma-CLIP Full achieves 91.39% at $K=1$ and scales monotonically to 93.40% at $K=16$. Second, multimodal foundation priors provide a substantial advantage over pure supervised CNN training in data-scarce regimes. Supervised ResNet-50 collapses to 69.83% at $K=1$ and 78.80% at $K=2$, lagging Gemma-CLIP Full by 21.56% and 13.18%, respectively. Third, experimental variance across random seeds scales inversely with training volume, tapering from $\pm 0.24\%$ at $K=1$ down to $\pm 0.04\%$ at Full Trainval, demonstrating high convergence stability across initialization seeds.

---

### 5.7. Evaluation on a Second Fine-Grained Benchmark (Stanford Dogs)

To evaluate domain generalizability beyond Oxford-IIIT Pet, we benchmark on **Stanford Dogs** [16] (120 breeds, 14,406 train / 6,174 test images under the standard 70:30 split) using 600 distilled morphological concepts across the five anatomical axes (distillation protocol detailed in Appendix A), with performance summarized in Table 8.

*Table 8. Evaluation on Two Fine-Grained Benchmarks: Oxford-IIIT Pet and Stanford Dogs (70:30 Split).*

| Supervision Regime | Method / Framework | Text / Knowledge Source | Oxford-IIIT Pet Top-1 (%) | Stanford Dogs Top-1 (%) | Interpretability Support |
|---|---|---|:---:|:---:|---|
| **Zero-Shot** | CLIP Raw Class Names [2] | Unmodified class labels | 80.16% (80.38% TTA) | 51.78% | Black-box |
| | CLIP Handcrafted Prompt | Template ensemble [2] | 87.52% (87.76% TTA) | 59.41% | Manual templates |
| | **Gemma-CBM (Ours)** | **Gemma 4 E4B (On-Device)** | **80.02% (TTA)** | **57.14%** | **185 / 600-Concept Bottleneck** |
| | **Gemma-Centroid (Ours)** | **Gemma 4 E4B (On-Device)** | **88.25% (TTA)** | **58.44%** | **Distilled 5-Axis Centroids** |
| | **Gemma-ZeroShot (Ours)** | **Gemma 4 E4B (On-Device)** | **90.85% (TTA)** | **59.62%** | **Centroid + Prompt Fusion** |
| **Supervised Adaptation** | **Gemma-VisAdapter (Ours)**<sup>†</sup> | **Gemma 4 E4B (On-Device)** | **91.41% (TTA)** | **72.40%** | **Residual Visual Adapter** |
| | **Gemma-CLIP Lite (Ours)**<sup>‡</sup> | **Gemma 4 E4B (On-Device)** | **90.79% (91.17% TTA)** | **72.68%** | **Dual-Stream Visual Adapter** |
| | **Gemma-CLIP Full (Ours)**<sup>*</sup> | **Gemma 4 E4B (On-Device)** | **94.15% (TTA)** | **73.06%** | **Joint Multimodal Co-adaptation** |

> <sup>†</sup> *Gemma-VisAdapter is trained with AdamW ($\alpha = 0.2$), using the distilled Gemma centroids as fixed text classifier weights, achieving 91.41% TTA on Pets and 72.40% on Dogs (`stanford_dogs_gemma_results.json`).*  
> <sup>‡</sup> *Gemma-CLIP Lite couples Gemma-VisAdapter with dual-stream logit blending of handcrafted prompts and Gemma centroids without optimizing soft prompt tokens ($\alpha = 0.2, \beta = 0.5$ for Pets; $\beta = 0.35$ for Dogs), reaching 90.79% (91.17% TTA) on Pets and 72.68% on Dogs (+13.27% over handcrafted zero-shot).*  
> <sup>*</sup> *Gemma-CLIP Full jointly optimizes visual residual adapter weights (65,536 parameters) and continuous prompt tokens (2,048 parameters) alongside centroid prior logit blending, achieving 94.15% TTA (93.65% on the 9:1 partition) on Oxford-IIIT Pet and 73.06% on Stanford Dogs (2,374/6,174 correct, +13.65% over handcrafted zero-shot).*

Across both fine-grained animal benchmarks, our on-device SLM-distilled morphological framework establishes consistent accuracy improvements over standard CLIP baselines. On Stanford Dogs (120 breeds, 6,174 test images), Gemma-ZeroShot reaches 59.62% (+7.84% over raw names), while Gemma-CBM provides transparent clinical concept attribution across 600 axes at 57.14%. In the supervised adaptation regime, Gemma-VisAdapter attains 72.40%, Gemma-CLIP Lite reaches 72.68%, and Gemma-CLIP Full achieves 73.06%, demonstrating that multimodal co-adaptation continues to yield progressive gains even in 120-class fine-grained regimes. On Oxford-IIIT Pet, Gemma-CLIP Lite achieves 90.79% (91.17% TTA) and Gemma-CLIP Full achieves 94.15% TTA. These cross-domain results confirm that SLM-distilled morphological knowledge provides a transferable, scalable foundation for vision-language adaptation without dependence on proprietary cloud LLM services.

---

### 5.8. Two-Tier Computational Efficiency & Deployment Benchmark

Table 9 benchmarks throughput and latency on Apple Silicon GPU (MPS) across FP32 precision (batch size 100, 50 timed iterations).

*Table 9. Two-Tier Computational Efficiency Benchmark on Apple Silicon GPU (MPS).*

| Evaluated Configuration | Trainable Params | Head Latency | Head FPS | Full Pipeline Latency | Full Pipeline FPS | Top-1 Test Acc (%) (Raw / 5-View TTA) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CLIP Raw Class Names | 0 | 0.18 ms | 5,544 | 11.85 ms (1-View) | 84.4 | 80.16% / 80.38% |
| **Gemma-Centroid** (Ours) | 0 | 0.18 ms | 5,550 | 11.85 ms (1-View) | 84.4 | 88.20% / 88.25% |
| **Gemma-CBM** (Ours) | 0 | 0.19 ms | 5,200 | 11.86 ms (1-View) | 84.3 | 80.40% / 80.02% |
| **Gemma-VisAdapter (9:1)** | 65,536 | 0.27 ms | 3,660 | 11.94 ms (1-View) | 83.8 | 91.03% / 91.41% |
| **Gemma-CoOp (Compact Trait)** | 2,048 | 0.18 ms | 5,652 | 11.85 ms (1-View) | 84.4 | 91.11% / 91.66% |
| **Gemma-CLIP Full (5-View TTA)** | **67,584** | **0.90 ms** | **1,110** | **58.14 ms (5-View)** | **17.2** | **93.20% / 94.15%** |

On pre-cached features, classification executes in 0.18–0.90 ms (1,110–5,652 FPS). End-to-end from raw pixels, single-view inference runs at 84.4 FPS, while the full 5-View TTA pipeline sustains 17.2 FPS on consumer edge hardware.

---

### 5.9. Limitations & Future Research
Several avenues remain open for future investigation. While attribute spot-checks confirmed 94.7% clinical compliance, distilled descriptors may retain minor lexical redundancies that could benefit from automated pruning. Furthermore, our current CBM computes global cosine projections; integrating spatial cross-attention maps to visually ground attributes to anatomical subregions represents a promising extension. Finally, evaluating robustness on uncurated animal shelter imagery containing mixed-breed specimens and heavy occlusions, as well as extending the framework to broader fine-grained benchmarks such as CUB-200-2011 and FGVC Aircraft, will provide further validation of on-device SLM-guided vision adaptation.

---

## 6. Conclusion

This paper presented **Gemma-CLIP**, establishing that morphological domain knowledge distilled autonomously from on-device Small Language Models (Google Gemma 4 E4B) significantly enhances Vision-Language Models (CLIP ViT-B/32) across multiple operational dimensions. In the zero-shot regime, training-free logit blending (Gemma-ZeroShot) elevates Top-1 accuracy to 90.85% under 5-View TTA, outperforming raw class names (80.16%) and the standard 80-prompt CLIP ensemble (88.90%). To provide transparent explainability, Gemma-CBM maps image features onto a 185-dimensional clinical concept space, enabling human-auditable attribution grounded in veterinary breed standards.

Under few-shot and full-data adaptation, Compact Trait conditioning (Gemma-CoOp) bounds sequence length to prevent attention gradient dilution, resolving the low-shot overfitting common to prompt learning and achieving 91.39% Top-1 accuracy at $K=1$ compared to 69.83% for supervised ResNet-50. When visual residual adaptation and prompt tuning are unified into Gemma-CLIP Full, the model reaches 94.15% Top-1 accuracy on the official Oxford-IIIT Pet benchmark with only 67,584 trainable parameters (0.04% of ViT-B/32), outperforming the fully-supervised ResNet-50 Cosine Head (93.08%, $p = 3.5 \times 10^{-4}$) while sustaining 17.2 FPS on consumer edge hardware. The demonstrated generalization to Stanford Dogs (72.68%) confirms that on-device SLMs offer a private, cost-effective, and powerful foundation for fine-grained visual intelligence.

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

To construct the 5-axis morphological knowledge base, the on-device Gemma 4 SLM (served locally via LM Studio) is queried for each target class using deterministic greedy decoding (`temperature = 0.0`, `top_p = 1.0`, `seed = 42`) with the following standardized system prompt:

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

The resulting structured descriptors (185 attributes for the 37 Oxford-IIIT Pet classes and 600 attributes for the 120 Stanford Dogs breeds) are serialized into open JSON format and publicly accessible via direct GitHub links:
* **Oxford-IIIT Pet Knowledge Base (185 attributes across 37 breeds):** [https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/oxford_pets_attributes.json](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/oxford_pets_attributes.json)
* **Stanford Dogs Knowledge Base (600 attributes across 120 breeds):** [https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/stanford_dogs_attributes.json](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/stanford_dogs_attributes.json)
* **All Attribute Files Repository Directory:** [https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/tree/main/Semester/attributes](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/tree/main/Semester/attributes)

---

## Appendix B: The 5 Anatomical Concept Templates ($\mathcal{T}_1 \dots \mathcal{T}_5$) & Instantiation Example

For the 185-dimensional CBM space and Morphological Centroid Bank ($W_{\text{centroid}}$), the raw textual descriptors distilled from Gemma 4 are formatted using the 5 standardized clinical concept templates:
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

These 5 prompt templates directly ground CLIP's textual representations in the 5 canonical anatomical axes, constructing the 185-dimensional CBM (37 breeds $\times$ 5 concepts) and enabling invariant centroid formation via $w_c^{\text{centroid}} = \text{Normalize}\left(\frac{1}{5} \sum_{k=1}^5 w_c^{(k)}\right)$.
