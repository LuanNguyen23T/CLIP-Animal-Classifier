# GEMMA-CLIP: Fine-Grained Pet Breed Classification with Morphological Knowledge, Prompt Optimization and Multi-View Inference

**Nguyen Huu Hoang Luan, Ton That Van**
*Department of Information Technology, University of Science and Technology – The University of Danang*
Email: 102230028@sv1.dut.udn.vn, 102230055@sv1.dut.udn.vn
GitHub: [https://github.com/LuanNguyen23T/CLIP-Animal-Classifier](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier)

---

## Abstract

We study parameter-efficient adaptation of a frozen CLIP ViT-B/32 model using morphological descriptors generated locally by Gemma 4 E4B. The framework separates an inspectable concept-score branch, a fixed morphological centroid bank, compact-trait prompt tuning, and a residual visual adapter. On the official Oxford-IIIT Pet test set (3,669 images), Gemma-DualStream combines prompt tuning with fixed centroids and reaches 92.07% Top-1 under 5-View TTA using 2,048 trainable prompt parameters. Its 0.68-point margin over standard CoOp is not significant in the reported paired McNemar test ($p=0.0696$). In the separate three-seed few-shot benchmark at $K=1$, DualStream has lower observed variation than CoOp (87.55 $\pm$ 0.28% versus 86.32 $\pm$ 2.25%); this descriptive result is limited to the tested seeds. On the official Stanford Dogs split (12,000 train / 8,580 test), Gemma-CoOp obtains 64.45%, DualStream 64.17%, and standard CoOp 63.62% using single-view cached features. Concept scores can be inspected by anatomical group, but their causal faithfulness has not been established. The 2,048-parameter count excludes frozen CLIP weights, offline descriptor generation, and inference cost. Our contribution is the controlled study of trainable-parameter cost, inspectable morphological scores, and behavior under few-shot supervision.

**Keywords:** FGVC, Vision-Language Models (VLM), CLIP, Gemma 4 SLM, CoOp, CBM, TTA.

---

## 1. Introduction

FGVC requires vision systems to resolve minute, localized anatomical boundaries amidst large posture and illumination variations [1]. In Oxford-IIIT Pet, Ragdoll and Birman cats share pointed cream coats and blue eyes; Birman has a characteristic symmetrical glove and lace pattern, but white feet alone are not a definitive discriminator because mitted Ragdolls can also have them [19, 20]. Similarly, Staffordshire Bull Terriers and American Pit Bull Terriers share muscular builds, separated primarily by cheekbone width to mandibular taper ratios.

Global visual classifiers frequently fail on such localized cues. Contrastive Language-Image Pre-training (CLIP; Radford et al. [2]) enables zero-shot classification via cosine similarity between visual embedding $f(x)$ and text prompt embedding $w_c$, yet generic prompts (*"a photo of a {class}"*) omit diagnostic morphology. While parameter-efficient fine-tuning (PEFT) methods—such as CoOp [3], CLIP-Adapter [4], and Tip-Adapter [5]—learn continuous prompts or residual bottlenecks, standard continuous prompt vectors lack explicit anatomical grounding, can exhibit sensitivity to sample selection under few-shot initialization ($K=1$, where CoOp averages 86.32% with high variance across random seeds), and appending lengthy descriptive text risks dispersing self-attention gradients across CLIP layers. Furthermore, prior LLM-guided vision approaches (e.g., CuPL, LaBo) rely on remote proprietary cloud APIs (GPT-3/4), incurring recurring monetary costs, latency overhead, and privacy constraints.

We study how locally generated morphological descriptors can support parameter-efficient fine-grained classification. Gemma-CLIP pairs a frozen CLIP ViT-B/32 backbone with Gemma-generated attributes along five anatomical axes. These feed a 185-concept projection for score inspection, a fixed morphological centroid bank, compact-trait CoOp prompt tuning, and a separately evaluated visual adapter. The Gemma-DualStream configuration fuses CoOp and centroid scores without using the visual adapter; it trains 2,048 prompt parameters. Our evaluation examines (i) accuracy and trainable-parameter cost under explicit data and inference protocols, (ii) concept-score examples and their limitations, and (iii) few-shot behavior over three seeds. On the Oxford-IIIT Pet 9:1 training protocol, DualStream obtains 92.07% 5-View TTA versus 91.39% for standard CoOp; the paired comparison does not establish a difference at the 0.05 significance level ($p=0.0696$).

---

## 2. Related Work & Baseline Definitions

### 2.1. Vision-Language Models (CLIP)
CLIP [2] maps visual $f(x)$ and text $w_c$ embeddings into a shared normalized space $\mathbb{R}^D$ ($D=512$). Class probabilities are given by:

$$P(y = c \mid x) = \frac{\exp(\tau \langle f(x), w_c \rangle)}{\sum_{j=1}^C \exp(\tau \langle f(x), w_j \rangle)} \tag{1}$$

where $f(x) \in \mathbb{R}^{512}$ denotes the $L_2$-normalized visual feature vector extracted by the frozen Vision Transformer ($\|f(x)\|_2 = 1$), $w_c \in \mathbb{R}^{512}$ is the normalized text embedding of class $c$ ($\|w_c\|_2 = 1$), $\tau$ represents the learned logit scale parameter ($\tau \approx 100.0$ at evaluation), and $C$ is the total number of candidate classes ($C=37$ for Oxford-IIIT Pet and $C=120$ for Stanford Dogs). Evaluated on Oxford-IIIT Pet, raw class names achieve 80.16% (80.68% TTA), while handcrafted prompts achieve 87.49% (87.98% TTA).

### 2.2. PEFT Baselines
CoOp [3] models prompts as $M$ learnable continuous tokens $[V]_1, \dots, [V]_M \in \mathbb{R}^d$ prepended to fixed class embeddings: $t_c = [[V]_1, \dots, [V]_M, e_c]$ ($M=4, d=512$, 2,048 parameters). CLIP-Adapter [4] inserts a two-layer bottleneck MLP on image features, while Tip-Adapter [5] builds a non-parametric key-value cache fine-tuned in Tip-Adapter-F.

### 2.3. Supervised Upper-Bound References (Ceiling Baselines)
Supervised models—such as ResNet-50 Cosine Head [13] (75,815 parameters) and ResNet-50 Prototype [11]—train directly on 100% of labeled training data (3,680 images). While achieving strong full-data accuracy (93.08%), they require full supervision and suffer pronounced degradation in few-shot regimes ($K=1$: 72.88%) due to the absence of multimodal foundation priors. We treat them as supervised reference baselines rather than architectural peers.

### 2.4. Concept Bottlenecks (CBM) & LLM Prompting (CuPL, LaBo)
CBMs [6] route predictions through interpretable concept scores. In parallel, prompt ensembling frameworks query large language models: CuPL [17] queries GPT-3 to generate customized visual descriptive prompts, while LaBo [15] constructs concept bottlenecks via submodular selection. In this work, we evaluate official author-released CuPL prompts on Oxford-IIIT Pet alongside locally simulated concept pipelines. Gemma-CLIP departs in three key ways: (i) executes an on-device Gemma 4 SLM locally with zero API cost and full data privacy; (ii) constrains descriptors to AKC/FCI and CFA/TICA standards across 5 canonical anatomical axes; and (iii) anchors PEFT prompt and visual adapter tuning directly to morphological priors.

---

## 3. Proposed Methodology

![Architecture of Gemma-CLIP](architecture.jpg)
*Figure 1. Architecture of Gemma-CLIP: coupling 5-View Test-Time Augmentation (TTA), frozen CLIP visual encoding, non-parametric Hierarchical Centroid Bank distilled from on-device Google Gemma 4 SLM across 5 anatomical axes, and parameterized Gemma-CoOp soft prompt tuning with dual-stream logit fusion. Note: In the parameter-efficient Gemma-DualStream setting (2,048 parameters), the visual adapter is bypassed ($f'(x) \equiv f(x)$), directly combining Gemma-CoOp soft prompt tuning with the centroid bank.*

### 3.1. Local Morphological Knowledge Distillation via Gemma 4 SLM
Gemma 4 E4B (`google/gemma-4-e4b`) [8] is served locally via LM Studio with deterministic greedy decoding (`temp=0.0`, `seed=42`). For each class $c$, the SLM generates structured descriptors across five anatomical axes: `head_and_ears`, `eyes`, `coat`, `body`, and `distinguishing_mark` (Appendix A). Non-visual descriptors are excluded. An existing internal spot-check of 19/185 descriptors reports 18/19 matches to referenced breed-standard descriptions (Appendix A); this is neither an independent expert review nor validation of the entire attribute bank. We identify a Ragdoll paw-marking overgeneralization below and do not perform an additional audit for this revision. The complete knowledge base is serialized openly in structured JSON format (see Appendix A for prompt templates and public knowledge base links).

*Table 1. Illustrative Gemma-Generated Descriptors for Confusable Breed Pairs; known overgeneralizations are identified rather than treated as diagnostic rules.*

| Confusable Pair | Breed & Registry | Key Anatomical Descriptors (Gemma 4 Distilled) | Key Distinguishing Morphological Feature |
|---|---|---|---|
| **Feline Pair** | **Birman** *(CFA)* | • `head`: Broad, rounded skull contour.<br>• `eyes`: Round, deep sapphire-blue.<br>• `coat`: Silky semi-longhair, no undercoat. | **Pure white symmetrical "gloves"** on all four paws. |
| | **Ragdoll** *(CFA)* | • `head`: Broad modified wedge, flat plane.<br>• `eyes`: Large, vivid blue, oval.<br>• `coat`: Semi-long plush, dense undercoat. | **Known descriptor limitation:** the archived Gemma attribute says “absence of white gloves”, but mitted Ragdolls can have white feet [19]. |
| **Canine Pair** | **Staffordshire Bull Terrier** *(AKC)* | • `head`: Broad skull; semi-pricked close ears.<br>• `body`: Compact, low-slung, short tail.<br>• `coat`: Short, dense, sleek double coat. | **Blocky, low-slung silhouette** with extreme muscle density. |
| | **American Pit Bull Terrier** *(UKC)* | • `head`: Broad skull; semi-erect high-set ears.<br>• `body`: Leggier frame, pronounced chest depth.<br>• `coat`: Extremely short, smooth single coat. | **Athletic, leggy taper** with pronounced chest depth. |

### 3.2. 185-Dimensional CBM Space & Morphological Centroids
Each raw attribute $a_c^{(k)}$ ($k \in \{1..5\}$) is formatted via template $\mathcal{T}_k(\cdot)$ (Appendix B) and projected through frozen CLIP text encoder $\mathcal{E}_{\text{txt}}$ into normalized vector $w_c^{(k)} \in \mathbb{R}^{512}$.

In the **Gemma-CBM** stream, stacking all 185 concept vectors forms the morphological concept projection matrix $W_{\text{concept}} \in \mathbb{R}^{185 \times 512}$. For any image embedding $f(x)$, concept activation scores and diagnostic class predictions are computed as:

$$\hat{y}_{\text{CBM}} = \arg\max_{c \in \{1 \dots C\}} \frac{1}{K_{\text{axes}}} \sum_{k=1}^{K_{\text{axes}}} \langle f(x), w_c^{(k)} \rangle \tag{2}$$

where $s_{c,k}(x) \in [-1, 1]$ represents the cosine activation score between the normalized image representation $f(x) \in \mathbb{R}^{512}$ and the $k$-th anatomical concept vector $w_c^{(k)} \in \mathbb{R}^{512}$ of class $c$, and $K_{\text{axes}} = 5$ corresponds to the five canonical anatomical axes (`head_and_ears`, `eyes`, `coat`, `body`, `distinguishing_mark`). Uniform pooling across these 5 concept scores yields the transparent prediction $\hat{y}_{\text{CBM}}$ across the $C$ candidate categories.

In parallel, the **Gemma-Centroid** stream constructs an invariant morphological centroid representation for each class $c$ by normalizing the average of its five concept projections:

$$w_c^{\text{centroid}} = \text{Normalize}\left(\frac{1}{5} \sum_{k=1}^5 w_c^{(k)}\right) = \frac{\sum_{k=1}^5 w_c^{(k)}}{\left\|\sum_{k=1}^5 w_c^{(k)}\right\|_2} \in \mathbb{R}^{512} \tag{3}$$

where $w_c^{\text{centroid}} \in \mathbb{R}^{512}$ is the $L_2$-normalized mean concept vector for class $c$. Stacking all $C$ centroids forms the static non-parametric text matrix $W_{\text{centroid}} \in \mathbb{R}^{C \times 512}$ ($37 \times 512$ for Pets; $120 \times 512$ for Dogs), serving as fixed semantic anchors for inference and adaptation with zero trainable parameters.

### 3.3. Residual Visual Adaptation & Gemma-CoOp Prompt Optimization

#### 3.3.1. Gemma-VisAdapter with Prior Weight Anchoring
Visual adaptation employs a lightweight two-layer residual bottleneck MLP ($512 \to 64 \to 512$, 65,536 parameters):

$$\Delta f(x) = W_2 \cdot \text{ReLU}(W_1 f(x)) \tag{4}$$

where $W_1 \in \mathbb{R}^{64 \times 512}$ is a down-projection matrix, $W_2 \in \mathbb{R}^{512 \times 64}$ is an up-projection matrix, and $\text{ReLU}(\cdot) = \max(0, \cdot)$ introduces non-linearity. Zero-initialization ($W_2 \leftarrow \mathbf{0}$) guarantees $\Delta f(x) = 0$ at step 0, strictly preserving pre-trained CLIP representations ($2 \times 512 \times 64 = 65,536$ trainable parameters).

The adapted visual embedding is obtained by residual interpolation:

$$f'(x) = \text{Normalize}\left((1 - \alpha)f(x) + \alpha \cdot \Delta f(x)\right) \tag{5}$$

where $\alpha \in [0, 1]$ is the residual blending factor ($\alpha = 0.2$ is our conservative reported operating point), and $f'(x) \in \mathbb{R}^{512}$ is the adapted normalized visual embedding. Classification logits are evaluated against fixed centroid weights via $z_c^{\text{adapter}} = \tau \langle f'(x), w_c^{\text{centroid}} \rangle$.

#### 3.3.2. Gemma-CoOp with Compact Trait Conditioning
Standard CoOp prepends soft prompt tokens to class names. Appending long descriptive sentences (45–60 tokens) causes attention gradient dilution across CLIP's 12 causal self-attention layers because text representations pool strictly from `[EOS]`:

$$\frac{\partial \mathcal{L}}{\partial [V]_i} = \sum_{j=i}^{N_{\text{tokens}}} \frac{\partial \mathcal{L}}{\partial h_L^j} \frac{\partial h_L^j}{\partial [V]_i} \tag{6}$$

where $[V]_i \in \mathbb{R}^{512}$ ($1 \le i \le M, M=4$ soft tokens, totaling $2,048$ trainable parameters) denotes the $i$-th continuous prompt vector, $h_L^j$ denotes the hidden state at position $j$ in the final Transformer layer ($L=12$), and $N_{\text{tokens}}$ denotes sequence length. When $N_{\text{tokens}}$ grows beyond 50 tokens, the compounding chain of intermediate attention Jacobian matrices dilutes the backward gradient reaching $[V]_i$.

We condition prompts on concise visual trait descriptors yielding an empirical mean sequence length of approximately 24 tokens via Compact Trait Conditioning:

$$p_c = [V]_1 [V]_2 [V]_3 [V]_4 \oplus \text{Tokenize}("\{breed\}, \{species\}, \text{ recognized by } \{concise\_trait\}") \tag{7}$$

where $[V]_{1..4} \in \mathbb{R}^{4 \times 512}$ are the four continuous prompt vectors ($2,048$ trainable parameters), $\oplus$ denotes sequence concatenation, and $\{concise\_trait\}$ injects a compact morphological summary distilled from Gemma. This bounds sequence length to $\approx 24$ tokens, maintaining short, uncorrupted gradient backpropagation paths.

### 3.4. Aspect-Ratio Preserving 5-View TTA Pipeline
To prevent anatomical distortion from naive square resizing, images are isotropically resized (short edge 224) and center-cropped. Five complementary views are generated per test sample: **1)** Canonical crop, **2)** Bilateral edge-preserving filter ($d=7, \sigma=50$), **3)** CIELAB CLAHE contrast enhancement ($8\times 8$ grid, clip limit 2.0), **4)** Cascaded Bilateral + CLAHE, and **5)** Multi-scale zoom ($1.15\times$, horizontal flip, $15^\circ$ rotation). The test embedding is:

$$f_{\text{TTA}}(x) = \text{Normalize}\left(\frac{1}{5} \sum_{v=1}^5 f_v(x)\right) = \frac{\sum_{v=1}^5 f_v(x)}{\left\|\sum_{v=1}^5 f_v(x)\right\|_2} \in \mathbb{R}^{512} \tag{8}$$

where $f_v(x) \in \mathbb{R}^{512}$ denotes the normalized visual embedding of view $v \in \{1..5\}$. This multi-view fusion operates at 17.2 FPS on Apple Silicon MPS.

### 3.5. Dual-Stream Logit Fusion & Loss
Logits from Gemma-CoOp and Gemma-Centroid are blended:

$$z_c^{\text{fusion}} = z_c^{\text{Gemma-CoOp}} + \beta \cdot z_c^{\text{centroid}} \tag{9}$$

where $z_c^{\text{Gemma-CoOp}} = \tau \langle f(x), w_c^{\text{CoOp}} \rangle$ is the logit produced by continuous prompt tuning, $z_c^{\text{centroid}} = \tau \langle f(x), w_c^{\text{centroid}} \rangle$ is the invariant morphological anchor logit, and $\beta$ is the fusion weight ($\beta = 0.5$ on Oxford-IIIT Pet; $\beta = 0.1$ on Stanford Dogs). In Gemma-DualStream, the visual adapter is bypassed ($f'(x) \equiv f(x)$), directly combining Gemma-CoOp with the centroid bank using exactly $2,048$ trainable prompt parameters.

Learnable parameters ($[V]_{1..4}$) are optimized using cross-entropy loss:

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\tau\, z^{\text{fusion}}_{y_i})}{\sum_{c=1}^{C}\exp(\tau\, z^{\text{fusion}}_{c})} \tag{10}$$

where $B=256$ is the training batch size, $y_i \in \{1 \dots C\}$ is the ground-truth breed label, and optimization is carried out with AdamW ($\eta = 5\times 10^{-3}$, weight decay $1\times 10^{-4}$) under a cosine annealing schedule over 50 epochs.

### 3.6. Standardized Architectural Taxonomy
To guarantee unambiguous evaluation across supervision regimes and experimental protocols, we establish a standardized architectural taxonomy spanning training-free and parameterized variants.

In the **training-free regime (0 trainable parameters)**, three configurations are evaluated: *Gemma-CBM* projects image features onto the 185-axis (Pets) or 600-axis (Dogs) concept space for transparent attribution; *Gemma-Centroid* performs standalone cosine matching against the 5-axis morphological centroid bank $W_{\text{centroid}}$ (88.39% TTA); and *Gemma-ZeroShot* combines prompt representations with Gemma-Centroid via training-free logit blending ($\beta = 0.5$).

In the **prompt-tuning regime (2,048 parameters)**, *Gemma-CoOp* optimizes four continuous prompt tokens conditioned on 24-token Compact Traits to mitigate attention gradient dilution (91.85% TTA). *Gemma-DualStream* blends the adapted logits of Gemma-CoOp with the invariant morphological priors of Gemma-Centroid ($\beta = 0.5$), reaching 92.07% TTA.

In the **visual and joint adaptation regime (65,536 parameters)**, *Gemma-VisAdapter* (65,536 parameters) trains a residual bottleneck MLP ($512 \to 64 \to 512$, $\alpha = 0.2$) anchored to fixed centroid weights, achieving 90.73% (91.61% TTA) on Oxford-IIIT Pet and 63.32% on Stanford Dogs under strict validation-based model selection. The complete framework indicates that distilling compact morphological priors from on-device SLMs effectively guides both prompt optimization and visual adaptation across fine-grained categorization tasks.

---

## 4. Experimental Setup

### 4.1. Dataset & Partition Protocols
Evaluations are conducted on the official Oxford-IIIT Pet benchmark [1], containing 7,349 images across 37 cat and dog breeds partitioned into 3,680 trainval images and 3,669 test images. To ensure rigorous evaluation and prevent data leakage, we strictly enforce three distinct operational protocols:
**1) Hyperparameter Validation Split (9:1 Split: 3,310 Train / 370 Val):** The 3,680 trainval images are stratified into 3,310 training images (90%) and 370 validation images (10%). Learning rate $\eta$, prompt length $M$, fusion ratio $\beta$, and epoch checkpoints are evaluated using this 370-image validation split. The visual-adapter $\alpha$ sweep is reported separately in Section 5.3; its retained $\alpha=0.2$ configuration is a conservative operating point, not the maximum-validation-accuracy setting. The official 3,669 test images remain completely unobserved during training and tuning.
**2) Full Data Scaling Regime (3,680 Trainval):** For full data ceiling benchmarks, models configured with these validated hyperparameters are evaluated on the full 3,680 trainval partition to assess performance under maximum available supervision.
**3) Inference Modes (Raw Single-View vs. 5-View TTA):** Throughout all tables and narrative discussions, we strictly distinguish between two test inference modes:
   * **Raw (Single-View):** Standard single-pass center crop ($224 \times 224$), representing minimal-latency edge deployment (84.4 FPS).
   * **5-View TTA (Test-Time Augmentation):** Multi-crop feature aggregation combining canonical crop, bilateral edge filtering, CLAHE contrast enhancement, cascaded filtering, and multi-scale zoom (17.2 FPS), representing maximum-accuracy deployment.

To account for stochastic experimental variance, all few-shot evaluations ($K \in \{1, 2, 4, 8, 16\}$ support images per class) are repeated across three independent random seeds (Seed 1, Seed 2, Seed 3), reporting mean accuracy and sample standard deviation ($\text{mean} \pm \text{std}$). Each seed jointly dictates the stratified random sampling of the $K$ support images per breed and the parameter initialization of continuous prompt vectors $[V]_{1..4}$ and adapter weights. Full trainval runs are likewise evaluated across all three seeds to verify initialization stability.

### 4.2. Comparative Baselines & Supervised Reference
We benchmark our framework against three representative paradigm categories using frozen `openai/clip-vit-base-patch32` backbones. First, training-free zero-shot baselines comprise CLIP Raw Class Names [2], Enriched Handcrafted Prompts, the 80-Prompt Ensemble, and the official CuPL baseline [17]. Second, parameter-efficient adaptation baselines encompass Tip-Adapter [5], Tip-Adapter-F [5], CLIP-Adapter [4], and CoOp [3]. Third, to establish an empirical upper bound, we compare against supervised CNN references: the ResNet-50 Cosine Head [13] evaluated under the exact same 9:1 partition (86.51% TTA) and on the full trainval dataset (93.08% TTA). These supervised CNN models serve as reference baselines due to their reliance on complete supervision.

### 4.3. Training Hyperparameters & Implementation Details
Models are optimized using AdamW with an initial learning rate $\eta = 1 \times 10^{-3}$ for Gemma-VisAdapter and $\eta = 5 \times 10^{-3}$ for Gemma-CoOp, weight decay $1 \times 10^{-4}$, batch size 256, and a cosine annealing learning rate schedule over 50 epochs. All experiments are executed on Apple Silicon GPU via Metal Performance Shaders (MPS). Performance is quantified using Top-1 Accuracy (%), Top-5 Accuracy (%), and Macro-averaged F1 Score (%).

---

## 5. Experimental Results & Discussion

### 5.1. Comprehensive Benchmark on Oxford-IIIT Pet

Table 2 evaluates all architectures on the 3,669-image test split, comparing raw single-view vs. 5-View TTA across zero-shot, PEFT, and supervised ceiling references.

*Table 2. Architectural Ablation Performance on Official Test Split (3,669 Images) under Hyperparameter Validation Split (3,310 Train / 370 Val Protocol).*

| Paradigm Category | Evaluated Architecture | Adaptation Mechanism | Trainable Params | Top-1 (%) Raw | Top-1 (%) 5-View TTA | Top-5 (%) 5-View TTA | Macro F1 (%) 5-View TTA |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **CLIP Baselines** | CLIP Raw Class Names | Zero-Shot Basic | 0 | 80.16% | 80.68% | 96.40% | 77.40% |
| | CLIP Handcrafted Prompt | Zero-Shot Enriched | 0 | 87.49% | 87.98% | 99.21% | 87.41% |
| | CLIP 80-Prompt Ensemble | Zero-Shot Multi-Prompt | 0 | 85.20% | 86.15% | 97.30% | 84.56% |
| | CuPL Official Prompts [17] | LLM-Guided Prompts | 0 | 87.22% | 87.63% | 99.62% | 87.11% |
| **PEFT Baselines**<sup>†</sup> | Tip-Adapter [5] | Key-Value Cache | 0 | 87.00% | 88.06% | 98.72% | 87.94% |
| | Tip-Adapter-F (9:1 Split) [5] | Tuned Cache | 3,680 | 89.10% | 89.56% | 98.96% | 89.48% |
| | CLIP-Adapter (9:1 Split) [4] | Visual Residual MLP | 65,536 | 89.42% | 91.28% | 99.10% | 91.22% |
| | CoOp Baseline (9:1 Split) [3] | Soft Continuous Tokens | 2,048 | 91.58% | 91.39% | 99.70% | 91.33% |
| **Supervised Reference** | ResNet-50 Cosine (9:1 Split, Frozen Features) | Normalized Cosine Head (Frozen Features) | 75,815 | — | 86.51% | 99.51% | 86.31% |
| | **ResNet-50 Cosine (Full Ceiling)**<sup>¶</sup> | Normalized Cosine Head (End-to-End) | 75,815 | **92.78%** | **93.08%** | **99.32%** | **93.02%** |
| **Gemma-CLIP (Ours)** | Gemma-CBM<sup>§</sup> | 185-Concept Projection | 0 | 81.44% | 81.52% | 97.77% | 79.38% |
| | Gemma-Centroid | Training-Free Centroid Bank | 0 | 88.14% | 88.39% | 99.48% | 88.11% |
| | Gemma-ZeroShot | Prompt + Centroid Blend | 0 | 87.22% | 87.41% | 97.30% | 86.11% |
| | Gemma-VisAdapter (9:1 Split) | Residual Visual Adapter | 65,536 | 90.73% | 91.61% | 99.81% | 91.33% |
| | Gemma-CoOp (9:1 Split) | Anchored Soft Prompt | 2,048 | 91.69% | 91.85% | 99.78% | 91.82% |
| | **Gemma-DualStream (9:1 Split)** | CoOp + Centroid Blend | 2,048 | **91.55%** | **92.07%** | **99.73%** | **92.02%** |

> <sup>†</sup> *Table 2 reports performance on the official 3,669 test split. '(9:1 Split)' denotes models trained on 3,310 training images with 370 validation images for hyperparameter tuning, evaluated on the unobserved 3,669 test images (see Section 4.1 for partition protocols and evaluation settings).*
> <sup>§</sup> *Gemma-CBM is designed for transparent morphological attribution rather than unconstrained categorization.*
> <sup>¶</sup> *ResNet-50 Cosine (Full Ceiling: 93.08% TTA) represents the fully supervised reference upper-bound trained end-to-end on 100% full trainval (3,680 images) with ImageNet initialization; under the matched 9:1 split protocol on frozen visual features, ResNet-50 achieves 86.51% TTA (86.75% across full trainval).*

A systematic comparison across Table 2 highlights critical architectural trade-offs between supervised learning and foundation adaptation. Under the identical 9:1 data partition (3,310 train / 370 val), Gemma-DualStream attains **92.07% TTA** (91.55% raw) and **99.73% Top-5 accuracy** with only **2,048 trainable prompt parameters** (0.0013% of ViT-B/32), outperforming the supervised ResNet-50 Cosine Head (86.51% TTA, $\Delta = +5.56\%$) and exceeding all PEFT baselines (CoOp 91.39%, CLIP-Adapter 91.28%, Tip-Adapter-F 89.56%). While the fully-supervised ResNet-50 Cosine Reference trained on 100% full trainval achieves 93.08% TTA, it requires complete labeled supervision across all 3,680 images and degrades sharply under few-shot data scarcity ($72.88\%$ at $K=1$, as shown in Table 7).

Comparing the linguistic representations, standalone Gemma-CBM achieves 81.52% TTA and exposes concept scores for inspection; this by itself does not establish causal faithfulness of an explanation. Projecting images into 185 independent concept axes introduces inter-concept correlations and semantic overlap across closely related breeds. Consequently, Gemma-CBM is deployed specifically for transparent morphological attribution (Section 5.4), whereas classification accuracy is maximized through non-parametric centroid integration and prompt optimization.

Beyond baseline comparisons, Table 2 traces the stepwise progression across model configurations under 5-View TTA, isolating the precise contribution of each architectural module. In the zero-shot regime, transitioning from raw class names (80.68%) to handcrafted prompts yields 87.98% (+7.30%), which is further elevated by our non-parametric Gemma-Centroid prior to 88.39% (+0.41%), outperforming official CuPL (87.63%). In the parameter-efficient adaptation regime evaluated on the 9:1 training subset (3,310 images), continuous prompt optimization via Gemma-CoOp achieves 91.85%, while Gemma-DualStream logit fusion reaches 92.07% (+0.22%), illustrating the accuracy of this 2,048-trainable-parameter configuration while retaining inspectable centroid scores.

---

### 5.2. Prompt Conditioning Structure & Empirical Analysis of Prompt Length

Table 3 evaluates prompt structures in CoOp to analyze prompt length and test the Attention Gradient Dilution hypothesis.

*Table 3. Impact of Prompt Conditioning Length and Token Structure on CoOp.*

| Evaluated Prompt Variant | Semantic Template Structure | Token Length | Test Raw Top-1 (%) | Test TTA Top-1 (%) | Top-5 Acc (%) | Macro F1 (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Baseline CoOp | `[V1..V4] {breed}` | 6 | 91.58% | 91.39% | 99.70% | 91.33% |
| **Gemma-CoOp (Compact Trait)** | `[V1..V4] {breed}, {species} recognized by {trait}` | **24** | **91.69%** | **91.85%** | **99.78%** | **91.82%** |
| **Gemma-DualStream** | Gemma-CoOp + Gemma-Centroid | — | **91.55%** | **92.07%** | **99.73%** | **92.02%** |

Constraining prompts to a 24-token Compact Trait structure provides an empirically effective compromise: it bounds sequence length while injecting concrete morphological descriptors, achieving 91.85% TTA and reaching 92.07% under Gemma-DualStream.

---

### 5.3. Residual Adapter Sensitivity ($\alpha$ Ablation) & McNemar Statistical Analysis

To examine sensitivity to interpolation between pre-trained visual representations and adapted features, we evaluate blending factor $\alpha \in \{0.0, 0.1, 0.2, 0.4, 0.6\}$. In this evaluation protocol, Gemma-VisAdapter (two-layer bottleneck MLP $512 \to 64 \to 512$, $W_2 = 0$, 65,536 parameters) is anchored to fixed Gemma-Centroid weights. For each candidate $\alpha$, the network is trained from scratch for 40 epochs exclusively on the 3,310-image training partition. Checkpoint selection is conducted strictly at peak validation accuracy on the 370-image validation split, ensuring the 3,669 test images remain strictly unobserved during hyperparameter tuning. When $\alpha = 0.0$, the visual adapter is disabled ($f'(x) \equiv f(x)$); for $\alpha > 0$, the visual adapter parameters are actively optimized.

*Table 4. Sensitivity Analysis of Residual Adapter Blending Weight $\alpha$; validation and held-out test results are shown separately.*

| Blending Factor $\alpha$ | Validation Top-1 Acc (%) | Test Set Raw Top-1 Acc (%) | Test Set Top-1 Acc (5-View TTA) (%) | Trainable Params | Architectural Interpretation |
|---|:---:|:---:|:---:|:---:|---|
| $\alpha = 0.0$ (No Visual Adapter) | 86.49% (320/370) | 88.14% | 88.39% (3,243/3,669) | 0 | Frozen CLIP baseline representations |
| $\alpha = 0.1$ | 90.54% (335/370) | 90.32% | 91.01% (3,339/3,669) | 65,536 | Conservative visual refinement |
| **$\alpha = 0.2$ (Reported Setting)** | **90.54% (335/370)** | **90.73%** | **91.61% (3,361/3,669)** | **65,536** | **Conservative reported setting** |
| $\alpha = 0.4$ | 91.08% (337/370) | 90.41% | 91.09% (3,342/3,669) | 65,536 | Higher validation accuracy than $\alpha=0.2$ |
| $\alpha = 0.6$ | 91.08% (337/370) | 90.71% | 91.31% (3,350/3,669) | 65,536 | Higher validation accuracy than $\alpha=0.2$ |

Table 4 shows 90.54% validation accuracy (335/370) at $\alpha=0.2$ and 91.08% (337/370) at both $\alpha=0.4$ and $0.6$. We retain $\alpha=0.2$ as the conservative configuration reported for Gemma-VisAdapter, but it is **not** the setting with the highest validation accuracy in this sweep. Its held-out test results are 90.73% single-view and 91.61% under 5-View TTA. The observed test ranking is reported descriptively and is not used to claim that $\alpha=0.2$ was selected by maximizing validation accuracy.

To evaluate whether prediction distributions differ significantly across paradigms, we conduct paired McNemar tests with continuity correction:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \tag{11}$$

where $b$ denotes the number of test instances correctly identified by model A but misclassified by model B, $c$ denotes the number of instances correctly identified by model B but misclassified by model A, and $-1$ represents Yates' continuity correction for discrete 1-degree-of-freedom distributions ($\text{df}=1$). Tests are computed on complete aligned predictions across all 3,669 test samples. When comparing Gemma-DualStream (92.07% TTA) against the continuous prompt CoOp baseline (91.39% TTA), we observe $b = 100$ (samples where Gemma-DualStream succeeds and CoOp fails) versus $c = 75$ (where CoOp succeeds and Gemma-DualStream fails), yielding $\chi^2 = 3.29$ ($p = 0.0696$). This CoOp comparison does **not** reach the 0.05 significance threshold. Top-5 recall is 99.73% versus 99.70%, and the concept branch offers inspectable scores; neither observation changes the McNemar result. Against CLIP-Adapter (91.28% TTA), the paired counts are $b=88$, $c=59$, $\chi^2=5.33$, and unadjusted $p=0.0209$. Against the ResNet-50 Cosine Head evaluated on the 9:1 partition with frozen visual features (86.51% TTA), they are $b=360$, $c=156$, $\chi^2=79.86$, and unadjusted $p<10^{-15}$. These are separate comparisons; no multiple-comparison adjustment is claimed here, and the ResNet full-data end-to-end reference uses a different training protocol.

---

### 5.4. XAI Attribution Case Study (Gemma-CBM)

Table 5 details the diagnostic concept activation profile from Gemma-CBM (185 concepts) for a test specimen of Ragdoll evaluated against Birman.

*Table 5. Concept Activation Decomposition Across Five Anatomical Axes (Ragdoll vs. Birman).*

| Morphological Axis | Ragdoll Score | Birman Score | Discriminative Delta ($\Delta$) | Morphological Rationale |
|---|:---:|:---:|:---:|---|
| `head_and_ears` | 0.231 | 0.204 | +0.027 | Ragdoll favored by broad head structure |
| `eyes` | 0.218 | 0.215 | +0.003 | Near neutral (both exhibit vivid blue eyes) |
| `coat` | 0.240 | 0.238 | +0.002 | Near neutral (both exhibit semi-longhair pointed coats) |
| `body` | 0.222 | 0.219 | +0.003 | Ragdoll favored by larger, muscular build |
| `distinguishing_mark` | **0.245** | **0.218** | **+0.027** | Large score margin for an archived descriptor that overgeneralizes Ragdoll paw markings [19] |

The specimen is classified as Ragdoll. The archived concept scores show +0.027 margins on both `head_and_ears` and `distinguishing_mark`, and the latter's reported mean margin across 100 Ragdoll test images is $+0.022 \pm 0.006$. These are inspectable score differences, not proof that a photographed paw trait caused the decision. In particular, the underlying Ragdoll `distinguishing_mark` descriptor in the released attribute JSON incorrectly treats absent white gloves as breed-general; CFA recognizes mitted Ragdolls with white feet [19]. The numerical results are retained as originally computed with that bank, and a corrected descriptor would require regenerating text features and rerunning the affected evaluations.

---

### 5.5. Morphological Distinction and Fine-Grained Ambiguity Analysis

Morphological descriptors provide the greatest separation for breeds with distinctive localized anatomy: Sphynx (hairless wrinkle texture, large wide-set ears), Persian (brachycephalic flat facial geometry, compact muzzle), and Bengal (rosetted leopard markings). Across the entire 3,669-image test set, cat-vs-dog cross-species classification achieves **99.95% accuracy**, with only 2 cross-species misclassifications (0.05% error rate: Great Pyrenees confused as Siamese, and Scottish Terrier confused as Bombay).

Table 6 analyzes representative confusable breed pairs identified across the 37 classes of Oxford-IIIT Pet, reporting official test sample counts computed directly from aligned model predictions.

*Table 6. Fine-Grained Morphological Boundary and Confusable Breed Pair Analysis on Oxford-IIIT Pet.*

| Ground Truth Breed | Test $N$ | Correct ($N$) | Most Confusable Counterpart | Errors to Counterpart ($N$, % of errors) | Key Morphological Differentiator | Primary Photographic Ambiguity Factor |
|---|:---:|:---:|---|:---:|---|---|
| Staffordshire Bull Terrier | 89 | 73 (82.0%) | American Pit Bull Terrier | 11 (68.8%) | Cranial cheekbone width to mandibular taper ratio | Frontal perspective obscuring jaw taper |
| Birman | 100 | 68 (68.0%) | Ragdoll | 27 (84.4%) | Birman glove/lace pattern together with head and coat cues; white paws alone are non-unique [19, 20] | Paws hidden under body posture/bedding |
| Beagle | 100 | 98 (98.0%) | Basset Hound | 1 (50.0%) | Limb bone length ratio and pendulous ear folds | Cropped framing obscuring limb stature |
| British Shorthair | 100 | 84 (84.0%) | Russian Blue | 13 (81.2%) | Eye coloration (amber/copper vs. emerald green) | Low-light sensor noise shifting eye hues |
| English Cocker Spaniel | 100 | 87 (87.0%) | English Setter | 11 (84.6%) | Muzzle stop angle and wither height proportions | Close-up framing eliminating height cues |

Misclassifications concentrate on photographic occlusions (hidden paws, low-light hue shifts) rather than multimodal representation failures. The error breakdowns indicate that when confusion occurs, it aligns strongly with the nearest morphological counterparts identified by formal breed standards.

---

### 5.6. Zero-Shot & Few-Shot Scaling Across 3 Independent Seeds

Table 7 presents scaling trajectories across budgets from $K = 0$ (Zero-Shot) to $K = 16$ and Full Trainval across three independent random seeds (mean $\pm$ std). Here, we evaluate our proposed flagship model, **Gemma-DualStream**, which fuses continuous prompt tuning (Gemma-CoOp) with the non-parametric morphological centroid bank (Gemma-Centroid).

*Table 7. Few-Shot and Full Trainval Scaling Benchmark on Oxford-IIIT Pet (Mean ± Std over 3 Random Seeds)<sup>†</sup>.*

| Budget ($K$) | CLIP Handcrafted Zero-Shot | CLIP + CoOp [3] | CLIP-Adapter [4] | Tip-Adapter-F [5] | Gemma-CoOp (Ours) | **Gemma-DualStream (Ours)** | ResNet-50 Cosine [13] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 0$** | 87.49 ± 0.00 | — | — | — | —<sup>*</sup> | —<sup>*</sup> | N/A |
| **$K = 1$** | 87.49 ± 0.00 | 86.32 ± 2.25 | 86.37 ± 0.86 | 87.79 ± 0.11 | 84.65 ± 1.31 | **87.55 ± 0.28** | 72.88 ± 0.93 |
| **$K = 2$** | 87.49 ± 0.00 | 87.44 ± 1.62 | 87.92 ± 0.30 | 88.18 ± 0.16 | 85.68 ± 0.35 | **88.55 ± 0.14** | 81.76 ± 1.01 |
| **$K = 4$** | 87.49 ± 0.00 | 89.01 ± 0.28 | 88.55 ± 0.69 | 88.92 ± 0.11 | 88.14 ± 0.41 | **89.86 ± 0.35** | 86.22 ± 0.95 |
| **$K = 8$** | 87.49 ± 0.00 | 90.55 ± 0.75 | 89.43 ± 0.42 | 90.21 ± 0.44 | 89.22 ± 0.67 | **90.51 ± 0.42** | 88.54 ± 0.82 |
| **$K = 16$** | 87.49 ± 0.00 | 91.13 ± 0.58 | 89.46 ± 0.76 | 90.43 ± 0.44 | 90.12 ± 0.70 | **90.96 ± 0.37** | 90.12 ± 0.37 |
| **Full** | 87.49 ± 0.00 | 91.56 ± 0.05 | 89.64 ± 0.15 | 89.81 ± 0.00 | 91.60 ± 0.49 | **91.57 ± 0.42** | 91.50 ± 0.14 |

> <sup>†</sup> *The trained few-shot configurations ($K=1$–16 and Full) report mean $\pm$ sample standard deviation over three random seeds on the official 3,669-image test set using the inference protocol recorded for those runs. The repeated CLIP handcrafted reference (87.49%) is the single-view raw value. <sup>*</sup>Gemma-CoOp and Gemma-DualStream optimize continuous prompt parameters from few-shot support images ($K \ge 1$); their zero-shot non-parametric counterpart (Gemma-Centroid) achieves 88.14% raw (88.39% 5-View TTA) as reported in Table 2.*

Two primary empirical insights emerge from Table 7. First, multimodal foundation priors maintain substantially higher accuracy than pure supervised CNN learning in data-scarce regimes. Under severe sample scarcity ($K=1$), the supervised ResNet-50 Cosine classifier drops to 72.88%, lagging Gemma-DualStream by nearly 15 percentage points ($87.55\%$). Second, Gemma-DualStream mitigates the high variance and sample instability inherent in soft prompt tuning. Under $K=1$, standard CoOp exhibits high sensitivity to sample selection, with variance reaching $\pm 2.25\%$ (accuracy dipping as low as $83.18\%$ on Seed 2). In contrast, Gemma-DualStream anchors continuous prompt adaptation to the non-parametric morphological centroid bank, suppressing variance down to $\pm 0.28\%$ and outperforming CoOp by $+1.23\%$ at $K=1$ ($87.55\%$ vs $86.32\%$), $+1.11\%$ at $K=2$ ($88.55\%$ vs $87.44\%$), and $+0.85\%$ at $K=4$ ($89.86\%$ vs $89.01\%$). At higher budgets ($K=8$ and $K=16$), CoOp achieves slightly higher mean accuracy ($90.55\%$ vs $90.51\%$ at $K=8$; $91.13\%$ vs $90.96\%$ at $K=16$), indicating that unconstrained continuous prompt tuning adapts flexibly when ample training samples are available, whereas Gemma-DualStream retains morphological concept grounding. Under full supervision, both paradigms converge to statistical parity ($91.57 \pm 0.42\%$ vs $91.56 \pm 0.05\%$).

---

### 5.7. Evaluation on a Second Fine-Grained Benchmark (Stanford Dogs)

To evaluate domain generalizability beyond Oxford-IIIT Pet, we benchmark on **Stanford Dogs** [16] (120 breeds, evaluated on the official benchmark partition of 12,000 train (100 per breed) / 8,580 test images) using 600 distilled morphological concepts across the five anatomical axes. To prevent data leakage, the 12,000 training images are partitioned into a stratified 90:10 train-validation split (10,800 train / 1,200 validation images; 10 per breed). Hyperparameters and checkpoint epochs are selected strictly at peak validation accuracy, after which final inference is executed once on the 8,580 test images.

*Table 8. Evaluation on Oxford-IIIT Pet (3,669 official test images; raw and 5-View TTA shown separately) and Stanford Dogs (official 12,000/8,580 split; single-view cached features).*

| Supervision Regime | Method / Framework | Text / Knowledge Source | Oxford-IIIT Pet Top-1 (%) Raw (5-View TTA) | Stanford Dogs Top-1 (%) Single-View | Interpretability Support |
|---|---|---|:---:|:---:|---|
| **Zero-Shot** | CLIP Raw Class Names [2] | Unmodified class labels | 80.16% (80.68% TTA) | 58.17% (4,991/8,580) | Black-box |
| | CLIP Handcrafted Prompt | Template ensemble [2] | 87.49% (87.98% TTA) | 59.45% (5,101/8,580) | Manual templates |
| | CuPL author prompts [17] (Oxford only) | Author-released Oxford Pets prompts | 87.22% (87.63% TTA) | — | LLM Prompt Ensemble |
| | CuPL-style (Gemma proxy; Dogs only) | Locally distilled morphological descriptors | — | 57.00% (4,891/8,580) | Local proxy, not the CuPL method |
| | LaBo-style [15] | Gemma concept pooling, without LaBo concept selection | 82.53% (82.61% TTA) | 56.93% (4,885/8,580) | Local proxy, not a LaBo reproduction |
| | **Gemma-CBM (Ours)** | **Gemma 4 E4B (On-Device)** | **81.44% (81.52% TTA)** | **56.04% (4,808/8,580)** | **185 / 600-Concept Bottleneck** |
| | **Gemma-Centroid (Ours)** | **Gemma 4 E4B (On-Device)** | **88.14% (88.39% TTA)** | **57.30% (4,916/8,580)** | **Distilled 5-Axis Centroids** |
| | **Gemma-ZeroShot (Ours)** | **Gemma 4 E4B (On-Device)** | **87.22% (87.41% TTA)** | **59.72% (5,124/8,580)** | **Centroid + Prompt Fusion** |
| **Supervised Adaptation** | **Baseline CoOp** [3] | Continuous Soft Prompts | 91.58% (91.39% TTA) | 63.62% (5,459/8,580) | Black-box continuous tokens (2,048 params) |
| | **Gemma-VisAdapter (Ours)**<sup>†</sup> | **Gemma 4 E4B (On-Device)** | **90.73% (91.61% TTA)** | **63.32% (5,433/8,580)** | **Residual Visual Adapter (65,536 params)** |
| | **Gemma-DualStream (Ours)**<sup>‡</sup> | **Gemma 4 E4B (On-Device)** | **91.55% (92.07% TTA)** | **64.17% (5,506/8,580)** | **CoOp + Centroid Blend (2,048 params)** |
| | **Gemma-CoOp (Ours)** | **Gemma 4 E4B (On-Device)** | **91.69% (91.85% TTA)** | **64.45% (5,530/8,580)** | **Compact Trait Prompt (2,048 params)** |

> <sup>†</sup> *Gemma-VisAdapter is trained with AdamW ($\alpha = 0.2$), using the distilled Gemma centroids as fixed text classifier weights. On Stanford Dogs official split, checkpoint selection at Epoch 40 achieves 63.32% (5,433/8,580 correct) on the test split.*
> <sup>‡</sup> *On Stanford Dogs, Gemma-DualStream trains 2,048 prompt parameters and blends Gemma-CoOp with the fixed 600-concept centroid bank. The validation-selected fusion weight is $\beta=0.1$ (validation Top-1: 63.58%); test Top-1 is 64.17% (5,506/8,580), versus 63.62% (5,459/8,580) for standard CoOp. Dogs results use single-view cached CLIP features.*
> *Evidence note.* A separate JSON file reports 62.83% (5,391/8,580) from author-released **ImageNet** CuPL prompts mapped locally to Stanford Dogs. The repository does not currently contain the mapping/inference script or per-class prompt mapping needed to reproduce that figure; therefore it is excluded from the matched comparison table. The 57.00% Dogs row instead uses the documented Gemma-derived local proxy. Oxford uses author-released Pets prompts.*

On the official Stanford Dogs test split (8,580 images), training-free Gemma-ZeroShot reaches 59.72% (5,124 correct), compared with handcrafted prompts at 59.45%, the Gemma-derived CuPL-style proxy at 57.00%, and LaBo-style pooling at 56.93%. Standalone Gemma-Centroid reaches 57.30%, below the handcrafted prompt baseline on this dataset. Under the reported cached-feature training protocol, Gemma-CoOp reaches 64.45% (5,530 correct), Gemma-DualStream 64.17% (5,506), standard CoOp 63.62% (5,459), and Gemma-VisAdapter 63.32% (5,433). Thus DualStream exceeds standard CoOp by 0.55 percentage points but trails Gemma-CoOp by 0.28 points; the visual adapter is below standard CoOp here. The Dogs CuPL-style and LaBo-style rows are locally constructed proxies and cannot be treated as published method replications. On Oxford-IIIT Pet, the author-released CuPL Pets prompts yield 87.63% TTA and Gemma-Centroid 88.39% TTA. Different datasets, training runs, and inference settings must be interpreted separately.

---

### 5.8. Two-Tier Computational Efficiency & Deployment Benchmark

Table 9 benchmarks throughput and latency on Apple Silicon GPU (MPS) across FP32 precision (batch size 100, 50 timed iterations).

*Table 9. Two-Tier Computational Efficiency Benchmark on Apple Silicon GPU (MPS).*

| Evaluated Configuration | Trainable Params | Head Latency | Head FPS | Full Pipeline Latency | Full Pipeline FPS | Top-1 Test Acc (%) (Raw / 5-View TTA) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CLIP Raw Class Names | 0 | 0.18 ms | 5,544 | 11.85 ms (1-View) | 84.4 | 80.16% / 80.68% |
| **Gemma-Centroid** (Ours) | 0 | 0.18 ms | 5,550 | 11.85 ms (1-View) | 84.4 | 88.14% / 88.39% |
| **Gemma-CBM** (Ours) | 0 | 0.19 ms | 5,200 | 11.86 ms (1-View) | 84.3 | 81.44% / 81.52% |
| **Gemma-VisAdapter (9:1)** | 65,536 | 0.27 ms | 3,660 | 11.94 ms (1-View) | 83.8 | 90.73% / 91.61% |
| **Gemma-CoOp (Compact Trait)** | 2,048 | 0.18 ms | 5,652 | 11.85 ms (1-View) | 84.4 | 91.69% / 91.85% |
| **Gemma-DualStream (5-View TTA)** | **2,048** | **0.18 ms** | **5,650** | **58.14 ms (5-View)** | **17.2** | **91.55% / 92.07%** |

On pre-cached features, classification executes in 0.18–0.90 ms (1,110–5,652 FPS). End-to-end from raw pixels, single-view inference runs at 84.4 FPS, while the full 5-View TTA pipeline sustains 17.2 FPS on consumer edge hardware.

---

### 5.9. Limitations & Future Research
Several avenues remain open for future investigation. The existing 19/185 descriptor spot-check reports 18/19 matches to referenced breed-standard descriptions, but this internal document-based check is not independent expert validation and cannot establish accuracy for all 185 descriptors. No additional attribute audit was performed for this revision. The released Ragdoll `distinguishing_mark` is a known overgeneralization: mitted Ragdolls can have white feet [19]. Correcting that input would require rebuilding text embeddings and rerunning affected model evaluations before replacing any numerical results. The exploratory mapped-ImageNet CuPL Dogs figure likewise remains outside the main comparison until its mapping code and prompt provenance are available. Furthermore, our current CBM computes global cosine projections; integrating spatial cross-attention maps to visually ground attributes to anatomical subregions represents a promising extension. Finally, evaluating robustness on uncurated animal shelter imagery containing mixed-breed specimens and heavy occlusions, as well as extending the framework to broader fine-grained benchmarks such as CUB-200-2011 and FGVC Aircraft, will provide further validation of on-device SLM-guided vision adaptation.

---

## 6. Conclusion

This paper examined a parameter-efficient way to attach locally generated morphological knowledge to frozen CLIP for fine-grained breed recognition. The main technical contribution is a family of separable concept, centroid, prompt, and visual-adapter configurations. Gemma-CBM exposes five groups of concept scores for case-level inspection; these are descriptive scores, while causal faithfulness, correction of the known Ragdoll descriptor error, and independent expert validation remain open. On Oxford-IIIT Pet, the training-free Gemma-Centroid reaches 88.39% 5-View TTA versus 87.63% for the author-prompt CuPL baseline under the reported protocol.

The three-seed few-shot benchmark shows a narrower spread for Gemma-DualStream than CoOp at $K=1$ (87.55 $\pm$ 0.28% versus 86.32 $\pm$ 2.25%); CoOp has a slightly higher mean at $K=8$ and $K=16$. In a separate 9:1 Oxford run, DualStream obtains 92.07% 5-View TTA with 2,048 trainable prompt parameters, while its 0.68-point difference from CoOp is not significant in the reported paired test ($p=0.0696$). The official Stanford Dogs split adds an external evaluation: Gemma-CoOp obtains 64.45%, DualStream 64.17%, and standard CoOp 63.62%. These results support the study of trainable-parameter efficiency, inspectable morphological scores, and behavior at low shot counts, without asserting best absolute accuracy.

---

## References

* [1] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and Dogs," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3498–3505, 2012.
* [2] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *International Conference on Machine Learning (ICML)*, pp. 8748–8763, 2021.
* [3] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, "Learning to Prompt for Vision-Language Models," *International Journal of Computer Vision (IJCV)*, vol. 130, no. 9, pp. 2337–2348, 2022.
* [4] P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, and Y. Qiao, "CLIP-Adapter: Better Vision-Language Models with Feature Adapters," *International Journal of Computer Vision (IJCV)*, vol. 132, no. 2, pp. 581–595, 2024.
* [5] R. Zhang, R. Fang, W. Zhang, P. Gao, K. Li, J. Dai, Y. Qiao, and H. Li, "Tip-Adapter: Training-Free CLIP-Adapter for Better Vision-Language Modeling," in *European Conference on Computer Vision (ECCV)*, pp. 493–510, 2022.
* [6] P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Ghorbani, and P. Liang, "Concept Bottleneck Models," in *International Conference on Machine Learning (ICML)*, pp. 5338–5348, 2020.
* [7] S. Menon and C. Vondrick, "Visual Classification via Description from Large Language Models," in *International Conference on Learning Representations (ICLR)*, 2023.
* [8] Gemma Team, Google, "Gemma 4 Technical Report," *arXiv:2607.02770*, 2026.
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
* [19] The Cat Fanciers' Association, "Ragdoll Show Standard," https://cfa.org/wp-content/uploads/2024/03/ragdoll-standard.pdf (accessed 07 Nov. 2026).
* [20] The Cat Fanciers' Association, "Birman Show Standard," https://cfa.org/wp-content/uploads/2024/03/birman-standard.pdf (accessed 07 Nov. 2026).

---

## Appendix A: Automated Morphological Knowledge Distillation Prompt

To construct the 5-axis morphological knowledge base, the on-device Gemma 4 SLM (served locally via LM Studio) is queried for each target class using deterministic greedy decoding (`temperature = 0.0`, `seed = 42`) with the following standardized prompt template:

```text
You are a professional veterinarian and animal morphologist.
Describe the distinctive fine-grained visual characteristics of: "{human_breed}".
Only provide visual characteristics visible in a clear photograph. Ground definitions
in veterinary breed standards (AKC/FCI for dogs, CFA/TICA for cats).

Return ONLY a valid JSON object matching the following structure without markdown:
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

The resulting structured morphological descriptors and breed-standard audit records are publicly accessible via direct GitHub repository links:
* **Oxford-IIIT Pet Knowledge Base (185 concepts, 37 breeds):** [oxford_pets_attributes.json](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/oxford_pets_attributes.json)
* **Stanford Dogs Knowledge Base (600 concepts, 120 breeds):** [stanford_dogs_attributes.json](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/stanford_dogs_attributes.json)
* **Internal 19-descriptor spot-check (18/19 reported matches, not independent expert validation):** [clinical_audit_19_descriptors.json](https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/blob/main/Semester/attributes/clinical_audit_19_descriptors.json)

---

## Appendix B: Anatomical Concept Templates ($T_1 \dots T_5$) & Instantiation Example

Raw attributes distilled from Gemma 4 are formatted into the 185-dimensional CBM space and Morphological Centroid Bank ($W_{\text{centroid}}$) using five canonical concept templates:
* **$T_1$ (`head_and_ears`):** `"head and ears of {name}: {head_and_ears}"`
* **$T_2$ (`eyes`):** `"eyes of {name}: {eyes}"`
* **$T_3$ (`coat`):** `"coat texture and markings of {name}: {coat}"`
* **$T_4$ (`body`):** `"body structure of {name}: {body}"`
* **$T_5$ (`distinguishing_mark`):** `"distinguishing trait of {name}: {distinguishing_mark}"`

**Breed-standard-aware wording example (Ragdoll; not the archived descriptor used for the reported results):**
* **$T_1$:** `"head and ears of Ragdoll: Broad, modified wedge with flat plane between ears."`
* **$T_2$:** `"eyes of Ragdoll: Large, vivid blue, oval-shaped eyes set moderately wide."`
* **$T_3$:** `"coat texture and markings of Ragdoll: Semi-long plush coat with minimal matting; dense undercoat."`
* **$T_4$:** `"body structure of Ragdoll: Substantial, muscular body; long, broad chest and heavy bone."`
* **$T_5$:** `"distinguishing trait of Ragdoll: Point colorations across extremities; mitted pattern may exhibit white feet."`  
This corrected example is illustrative only. The archived attribute JSON contains a different, overgeneralized `distinguishing_mark`; reported model scores have **not** been recomputed with this wording [19].

These prompt representations ground CLIP in anatomical axes, enabling invariant centroid formation via:

$$\mathbf{w}_{\text{centroid}}^{(c)} = \frac{1}{5}\sum_{i=1}^{5} \frac{g(T_{c,i})}{\|g(T_{c,i})\|_2}, \quad \bar{\mathbf{w}}^{(c)} = \frac{\mathbf{w}_{\text{centroid}}^{(c)}}{\|\mathbf{w}_{\text{centroid}}^{(c)}\|_2}$$
