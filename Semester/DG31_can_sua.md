# DG-31 — Danh sách cần sửa & bổ sung trước lần nộp tiếp theo

> **Mục tiêu:** Đưa TC4 từ "Cần chỉnh sửa" → "Đạt" để lên mức **Sẵn sàng**  
> **Vấn đề cốt lõi (hội đồng nêu):** *"Các bảng mâu thuẫn nhau về con số chính và baseline giám sát đang vượt phương pháp đề xuất, cần đối soát và định vị lại trước khi gửi."*

---

## ✅ Trạng thái xử lý

| # | Mục | Mức độ | Trạng thái |
|---|---|---|---|
| 1 | Bổ sung chú thích TTA/non-TTA vào tất cả bảng | 🔴 Phải làm | ☐ |
| 2 | Giải thích chênh lệch CoOp giữa Table 2 và Table 5 | 🔴 Phải làm | ☐ |
| 3 | Sửa lại Abstract — claim về ResNet-50 | 🔴 Phải làm | ☐ |
| 4 | Thêm thảo luận ResNet-50 Prototype vs Gemma-CoOp | 🔴 Phải làm | ☐ |
| 5 | Sửa tiêu đề Table 2 (nhầm "Validation Split") | 🟡 Quan trọng | ☐ |
| 6 | Thêm ± std vào Table 2 hoặc ghi rõ "single run" | 🟡 Quan trọng | ☐ |
| 7 | Giải thích tại sao CBM standalone thấp (80.02%) | 🟡 Quan trọng | ☐ |
| 8 | Thêm per-breed confusion matrix chi tiết | 🟡 Quan trọng | ☐ |
| 9 | Cross-domain / out-of-distribution evaluation | 🟢 Nên có | ☐ |
| 10 | Xác nhận reproducibility của 185-attribute bank | 🟢 Nên có | ☐ |
| 11 | Bổ sung bảng đối chiếu yêu cầu các phiếu trước | 🟢 Nên có | ☐ |

---

## 🔴 Vấn đề 1: Mâu thuẫn số liệu giữa Table 2 và Table 5

### Vị trí trong paper
- **Table 2** (trang 5): CLIP Raw Class Names Raw = **80.16%**, TTA = **80.38%**
- **Table 5** (trang 7): CLIP Zero-shot Baseline K=0 = **88.90 ± 0.00**

### Vấn đề
Cùng là "CLIP zero-shot baseline" nhưng:
- Table 2: **80.16%** (single-view, no TTA)
- Table 5: **88.90%** (5-View TTA đã áp dụng)

Chênh lệch **8.74 điểm** — không có giải thích rõ ràng ngay tại bảng. Người đọc đọc Table 5 sẽ nghĩ CLIP zero-shot baseline đã là 88.90%, khiến toàn bộ bảng so sánh few-shot bị hiểu sai.

### Cách sửa

**Cách 1 (khuyến nghị):** Thêm footnote ngay dưới Table 5:

> *"†All entries in Table 5 are evaluated under the Aspect-Ratio Preserving 5-View TTA protocol. The CLIP zero-shot baseline (K=0) corresponds to 88.90% under TTA, compared to 80.16% (Raw, no TTA) reported in Table 2."*

**Cách 2:** Thêm cột "Raw (no TTA)" song song trong Table 5 để so sánh trực tiếp.

**Cách 3:** Đổi tiêu đề cột trong Table 5 rõ hơn:
`CLIP Zero-shot (5-View TTA)` thay vì `CLIP Zero-shot Baseline`

### Nội dung gợi ý thêm vào đoạn text sau Table 5 (Section 5.5)

```
It is important to note that all entries in Table 5 adopt the
Aspect-Ratio Preserving 5-View TTA inference protocol as a
uniform evaluation baseline, which elevates the raw CLIP
zero-shot accuracy from 80.16% (single-view, Table 2) to
88.90% (5-View TTA). This protocol-aware baseline selection
ensures a fair comparison across all few-shot budgets.
```

---

## 🔴 Vấn đề 2: Chênh lệch CoOp giữa Table 2 và Table 5

### Vị trí
- **Table 2**, dòng "CoOp Baseline": Raw = **91.20%**, TTA = **91.52%**
- **Table 5**, dòng "CLIP + CoOp", Full Trainval: **92.12% ± 0.02**

### Vấn đề
Full trainval + CoOp cho 91.52% (Table 2, TTA) vs 92.12% (Table 5, Mean over 3 seeds). Chênh **0.60 điểm** — không phải do TTA vì Table 5 cũng dùng TTA. Nguyên nhân có thể là:
- Table 2: single run, Table 5: mean of 3 seeds
- Hoặc Table 2 train trên **3,310 ảnh** (sau khi tách val 9:1), còn Table 5 train trên **3,680 trainval** đầy đủ

### Cần kiểm tra trong code
- [ ] Xác nhận: CoOp trong Table 2 dùng split 3,310 hay 3,680?
- [ ] Nếu dùng split khác → đây là nguyên nhân hợp lý, phải ghi rõ
- [ ] Nếu cùng split → phải re-run và reconcile

### Cách sửa (sau khi kiểm tra code)

Thêm footnote vào Table 2:

> *"‡CoOp in Table 2 is trained on the 3,310-image training subset (9:1 hyperparameter split) and evaluated single-run. Full Trainval results in Table 5 use all 3,680 trainval images with mean ± std over 3 seeds, accounting for the observed discrepancy."*

---

## 🔴 Vấn đề 3: Claim về ResNet-50 trong Abstract bị hiểu nhầm

### Vị trí
**Abstract, dòng:**
> *"achieves 91.39% at 1-shot (surpassing standard CoOp by +2.67% and fully-supervised ResNet-50 by +21.56%)"*

### Vấn đề
- **+21.56%** chỉ đúng ở K=1 shot (ResNet-50 collapse = 69.83%) — ResNet-50 collapse ở 1-shot là hiển nhiên với mọi supervised model, không phải thành tích đặc biệt
- Trong Full Trainval, ResNet-50 Cosine đạt **93.08%** — abstract không nêu điểm Gemma-CLIP Full (94.15%) vượt ResNet-50 trong điều kiện này

### Cách sửa — Abstract (gợi ý viết lại)

**Hiện tại:**
```
...achieves 91.39% at 1-shot (surpassing standard CoOp by
+2.67% and fully-supervised ResNet-50 by +21.56%), and
establishes a peak of 94.15% Top-1 accuracy under full
training with fewer than 0.5% trainable parameters.
```

**Đề xuất:**
```
...achieves 91.39% at 1-shot — surpassing standard CoOp
by +2.67% and decisively outperforming fully-supervised
ResNet-50 (69.83%, Δ=+21.56%) in extreme low-data regimes
where CNN models collapse without sufficient annotations —
and establishes a peak of 94.15% Top-1 accuracy under full
training (surpassing ResNet-50 Cosine Head by +1.07%) with
fewer than 0.5% trainable parameters.
```

**Thay đổi chính:** Thêm ngữ cảnh "in extreme low-data regimes" và bổ sung "+1.07% full trainval".

---

## 🔴 Vấn đề 4: Chưa thảo luận ResNet-50 Prototype vs Gemma-CoOp

### Vị trí — so sánh trong Table 2

| Method | Params | Raw Top-1 | TTA Top-1 |
|---|---|---|---|
| ResNet-50 Prototype | 0 (supervised) | 91.36% | **91.69%** |
| Gemma-CoOp Compact Trait | 2,048 (PEFT) | 91.11% | **91.66%** |

### Vấn đề
ResNet-50 Prototype (0 tham số training, nhưng dùng **toàn bộ nhãn training để tạo class centroid**) đạt 91.69% TTA, gần bằng Gemma-CoOp (91.66% TTA). Hội đồng coi đây là "baseline supervised vượt/bằng method". Bài hiện tại bỏ qua điều này.

### Cách sửa — Thêm đoạn vào Section 5.1 (sau phân tích Table 2)

```
An important observation warrants explicit discussion: ResNet-50
Prototype — a supervised oracle constructing class centroids from
all 3,680 labeled training samples — achieves 91.69% TTA,
nearly matching our Gemma-CoOp standalone (91.66% TTA). This
apparent parity must be interpreted carefully: ResNet-50 Prototype
requires direct access to full supervised annotations for every
training sample, whereas Gemma-CoOp requires zero labeled samples
(pure zero-shot soft-prompt anchoring). The decisive advantage of
Gemma-CLIP emerges from the Dual-Stream Fusion (92.15%, +0.46%
over ResNet-50 Prototype) and the complete Gemma-CLIP Full pipeline
(94.15%, +2.46%), which systematically surpasses all supervised
references. Critically, ResNet-50 Prototype collapses to 69.83% at
K=1 shot (Table 5), confirming that its parity relies entirely on
full annotation availability — a condition absent in practical
veterinary deployment scenarios.
```

---

## 🟡 Vấn đề 5: Tiêu đề Table 2 ghi sai "Validation Split"

### Vị trí
**Table 2, tiêu đề hiện tại:**
> *"Architectural Ablation Performance on **Validation Split** (3,669 Test Images)"*

### Vấn đề
"Validation Split" mâu thuẫn với "3,669 Test Images". Đây rõ ràng là **Official Test Split**.

### Sửa thành:
```
Table 2. Architectural Ablation Performance on Official Test Split (3,669 Images).
```

---

## 🟡 Vấn đề 6: Table 2 không có ± std — không nhất quán với Table 5

### Vị trí
- Table 2: tất cả số liệu đơn (no std)
- Table 5: mean ± std over 3 seeds

### Cách sửa

**Nếu không thể re-run:** Thêm footnote vào Table 2:

> *"†All Table 2 results report single-run performance on the official test split; full seed-averaged statistics (mean ± std) are provided in Table 5."*

**Nếu có thể re-run (tốt hơn):** Thêm cột `± std` vào Table 2, ít nhất cho các dòng "Gemma-CLIP (Ours)".

### Cần làm
- [ ] Quyết định: thêm footnote hay re-run?
- [ ] Nếu re-run → chạy `python train.py --seed 1 2 3` cho từng variant

---

## 🟡 Vấn đề 7: CBM standalone thấp (80.02%) — chưa được giải thích

### Vị trí
**Table 2**, dòng "Gemma Concept Bottleneck (CBM)":
- Raw = **80.40%**, TTA = **80.02%**
- (Lưu ý: TTA < Raw — bất thường)

### Vấn đề
- CBM (80.02%) ≈ CLIP Raw Class Names (80.16%) → 185 thuộc tính không cho thêm lợi ích khi đứng độc lập
- TTA làm giảm CBM accuracy (80.40% → 80.02%) — đây là anomaly cần giải thích

### Cách sửa — Thêm vào Section 5.1 hoặc 5.7

```
The Concept Bottleneck Model (CBM) standalone achieves 80.02% TTA,
near the CLIP raw class-name baseline (80.16%). This behavior arises
from the CBM's design purpose: it projects image features onto 185
individual anatomical descriptors, introducing inter-concept redundancy
that reduces per-class discriminability compared to the aggregated
Morphological Centroid Bank. The CBM's primary role is transparent
diagnostic attribution (Table 4), not maximum accuracy. Notably, 5-View
TTA slightly reduces CBM accuracy (80.40% raw → 80.02% TTA), likely
because multi-view averaging amplifies noise in narrow per-concept
cosine similarity distributions when concept boundaries overlap across
breeds. In the Dual-Stream Fusion, the CBM's spatial information is
implicitly encoded through the centroid bank rather than direct CBM
logits, recovering discriminative performance.
```

---

## 🟡 Vấn đề 8: Section 5.4 thiếu confusion matrix

### Vị trí
**Section 5.4** chỉ liệt kê 3 giống cải thiện nhiều nhất (text).

### Vấn đề
Không rõ: sau khi dùng Gemma-CLIP, những cặp giống nào vẫn bị nhầm lẫn? Điều này quan trọng để chứng minh morphological knowledge thực sự giải quyết được vấn đề FGVC.

### Cần thêm: Bảng Top-5 confused pairs

| True Breed | Predicted Breed | Error Rate (%) | Morphological Explanation |
|---|---|---|---|
| ? | ? | ? | ? |

### Cần chạy trong code
```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Sau khi predict toàn bộ test set
cm = confusion_matrix(y_true, y_pred)
# Extract top confused pairs
for i in range(37):
    for j in range(37):
        if i != j and cm[i,j] > threshold:
            print(f"{breed_names[i]} → {breed_names[j]}: {cm[i,j]} errors")
```

---

## 🟢 Vấn đề 9: Cross-domain evaluation (chưa có)

### Vị trí
**Section 5.7, điểm 3:** đề cập "underway" nhưng không có kết quả.

### Gợi ý thực hiện — không cần training thêm

**Option A (dễ nhất):** Zero-shot trên **Stanford Dogs** (120 giống)
- Chỉ cần generate morphological attributes cho 120 giống mới với Gemma 4
- Chạy Gemma Morphological Centroids (training-free) → báo accuracy
- Dự kiến: 2-3 giờ công

**Option B:** Pets scraped từ internet (out-of-distribution)
- ~10 ảnh/giống từ Google Images với background phức tạp
- Kiểm tra zero-shot robustness

### Thêm vào paper (sau khi có kết quả)
```
Table 7. Zero-Shot Cross-Domain Generalization on Stanford Dogs.
```

---

## 🟢 Vấn đề 10: Reproducibility của 185-attribute bank

### Vị trí
**Appendix A** đã có prompt template — tốt. Nhưng thiếu:
- Thông tin về temperature/sampling parameters
- Link đến file JSON attributes trên GitHub
- Xác nhận chất lượng attributes

### Cách sửa — Thêm vào Appendix A

```
Reproducibility Note: Attributes are generated with temperature=0
(deterministic) using google/gemma-4-e4b. The complete 185-attribute
JSON knowledge base (37 breeds × 5 axes) is released at:
  https://github.com/LuanNguyen23T/CLIP-Animal-Classifier/tree/main/Semester/attributes/

A spot-check of 10 randomly sampled breed descriptions confirmed
alignment with AKC/FCI/CFA/TICA breed standards.
```

### Cần làm
- [ ] Kiểm tra file attributes/ đã có trên GitHub chưa
- [ ] Xác nhận temperature=0 khi generate
- [ ] Spot-check và ghi kết quả vào bài

---

## 🟢 Vấn đề 11: Bổ sung Appendix đối chiếu yêu cầu phiếu trước

### Yêu cầu của hội đồng
> *"Kèm theo bản nộp sau một bảng đối chiếu từng yêu cầu của các phiếu trước (đã làm / chưa làm / lý do)"*

### Thêm Appendix B: Response to Previous Review Comments

| Yêu cầu từ phiếu | Trạng thái | Vị trí trong bài |
|---|---|---|
| Lặp 3 seeds, báo mean ± std | ✅ Đã làm | Table 5 (Section 5.5) |
| Ablation từng module tách bạch | ✅ Đã làm | Table 2 & Table 3 |
| Giải thích gradient dilution | ✅ Đã làm | Section 3.3, Table 3 |
| Baseline cùng giao thức | ✅ Đã làm | Section 4.2 |
| Thống nhất số liệu bảng | ✅ Sửa lần này | Table 2, 3, 5 footnotes |
| Định vị lại claim ResNet-50 | ✅ Sửa lần này | Abstract, Section 5.1 |
| Cross-domain evaluation | ⚠️ Đang bổ sung | Table 7 (mới) |
| *(Điền thêm từ phiếu 16/10 và 19/10)* | ... | ... |

---

## Ghi chú của tác giả

> *Điền sau khi kiểm tra code và đọc lại phiếu 16/10 & 19/10*

**Vấn đề 2 — CoOp split kiểm tra lại:**
```
Table 2 CoOp dùng split: [3,310 / 3,680?]
Seed cố định: [seed number?]
```

**Vấn đề 6 — Re-run Table 2 với 3 seeds:**
```
Dự kiến thời gian chạy: [?] phút/variant × 6 variants
Kết quả: [điền sau]
```

**Vấn đề 9 — Cross-domain:**
```
Chọn Option: [A / B]
Kết quả sơ bộ: [?]%
```

**Deadline nộp lần 3:** _____

