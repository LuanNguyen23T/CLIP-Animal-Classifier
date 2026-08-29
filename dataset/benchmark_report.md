# Báo Cáo Kết Quả Benchmark Tiền Xử Lý & Tăng Cường Dữ Liệu Với CLIP Zero-Shot

Báo cáo này tổng hợp kết quả thực nghiệm đánh giá hiệu năng phân loại vật nuôi (37 giống chó mèo của tập dữ liệu Oxford-IIIT Pet) sử dụng mô hình **CLIP Zero-Shot (`openai/clip-vit-base-patch32`)** trên các phân chia stratified split của tập `trainval` và tập `test` độc lập đầy đủ.

---

## 1. Bảng Kết Quả Accuracy Tuyệt Đối (%)

Độ chính xác phân loại (Accuracy) của các cấu hình xử lý ảnh khác nhau trên các phân chia validation của `trainval` (tỉ lệ 6:4, 7:3, 8:2, 9:1) và trên toàn bộ tập dữ liệu `test` độc lập đầy đủ (3,669 mẫu ảnh):

| Tập Dữ Liệu / Split | Baseline | Bilateral Denoise | CLAHE Only | Bilateral + CLAHE | Bilateral + CLAHE + Geom | Test-Time Augmentation (TTA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trainval 6:4 (Val Split)** | 75.12% | 74.17% | 75.39% | 73.08% | 72.34% | **76.41%** (+1.29%) |
| **Trainval 7:3 (Val Split)** | 75.29% | 73.94% | 75.48% | 73.03% | 72.94% | **76.47%** (+1.18%) |
| **Trainval 8:2 (Val Split)** | 75.64% | 73.61% | 75.51% | 73.61% | 73.48% | **76.73%** (+1.08%) |
| **Trainval 9:1 (Val Split)** | 75.68% | 72.97% | 75.14% | 72.43% | 72.16% | **76.49%** (+0.81%) |
| **Full Test Set (100%)** | **80.16%** | 78.71% | 78.74% | 78.14% | 75.74% | **80.65%** (+0.49%) |

---

## 2. Bảng Thay Đổi Accuracy So Với Baseline (Delta Accuracy)

Sự chênh lệch độ chính xác ($\Delta$ Accuracy) của từng phương pháp xử lý so với **Baseline** (ảnh gốc chạy qua bộ tiền xử lý chuẩn của CLIP với nhãn thô):

| Tập Dữ Liệu / Split | Baseline | Bilateral Denoise | CLAHE Only | Bilateral + CLAHE | Bilateral + CLAHE + Geom | Test-Time Augmentation (TTA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trainval 6:4 (Val Split)** | +0.00% | -0.95% | **+0.27%** | -2.03% | -2.78% | **+1.29%** |
| **Trainval 7:3 (Val Split)** | +0.00% | -1.36% | **+0.18%** | -2.26% | -2.35% | **+1.18%** |
| **Trainval 8:2 (Val Split)** | +0.00% | -2.03% | -0.14% | -2.03% | -2.17% | **+1.08%** |
| **Trainval 9:1 (Val Split)** | +0.00% | -2.70% | -0.54% | -3.24% | -3.51% | **+0.81%** |
| **Full Test Set (100%)** | +0.00% | -1.44% | -1.42% | -2.02% | -4.42% | **+0.49%** |

---

## 3. Nhận Xét & Phân Tích Chuyên Sâu

### A. Hiệu quả vượt trội và ổn định của Test-Time Augmentation (TTA)
*   **Cải thiện đồng nhất:** TTA (tích hợp đặc trưng trung bình của 5 views ảnh) đem lại sự cải thiện độ chính xác rất ổn định trên cả tập `test` độc lập (+0.49%) lẫn các split validation của `trainval` (dao động từ **+0.81%** đến **+1.29%**). Điều này chứng minh sức mạnh của cơ chế biểu quyết trung bình đặc trưng trong việc tăng cường tính vững chãi cho biểu diễn đặc trưng của mô hình CLIP.

### B. Ảnh hưởng của bộ lọc CLAHE trên tập Trainval vs Test
*   **CLAHE đơn lẻ (CLAHE Only):** CLAHE đem lại cải thiện nhẹ trên các split validation nhỏ như **6:4** (+0.27%) và **7:3** (+0.18%). Điều này cho thấy trong các tập con nhỏ, việc cân bằng lại histogram có thể giúp làm rõ các chi tiết đặc trưng (như kết cấu lông thú cưng), giúp cải thiện xác suất khớp nhãn. 
*   Tuy nhiên, khi đánh giá trên tập Test lớn đầy đủ hoặc trên các split có kích thước lớn hơn (8:2, 9:1), hiệu quả này giảm nhẹ hoặc chuyển sang âm (-1.42% trên Full Test) do hiện tượng **Domain Shift** (phép biến đổi làm thay đổi phân phối ảnh so với những gì CLIP được pre-train).

### C. Ảnh hưởng tiêu cực từ các phép biến đổi đơn lẻ và hình học
*   **Bilateral Denoise & Bilateral + CLAHE:** Đều làm giảm hiệu năng phân loại (giảm từ -0.95% đến -3.24%). Bộ lọc Bilateral làm mịn ảnh có thể làm mất đi các chi tiết tần số cao cực kỳ quan trọng đối với CLIP (như ria mép hoặc các chi tiết lông mảnh).
*   **Bilateral + CLAHE + Geom:** Làm giảm hiệu năng nhiều nhất (lên tới -4.42% trên tập Test). Điều này dễ hiểu vì việc xoay nghiêng ảnh và lật ảnh làm xáo trộn cấu trúc không gian mà CLIP Vision Encoder mong đợi ở chế độ zero-shot thuần túy.

---

## 4. Đề Xuất Quy Trình Cho AI Học

1.  **Sử dụng Test-Time Augmentation (TTA):** TTA luôn là sự lựa chọn tối ưu và an toàn nhất để tăng độ chính xác mà không lo ngại rủi ro giảm hiệu năng.
2.  **Tránh áp dụng trực tiếp các bộ lọc OpenCV cổ điển đơn lẻ vào mô hình CLIP Zero-Shot:** Các bộ lọc như song phương hay CLAHE nên tránh dùng trực tiếp làm đầu vào duy nhất cho CLIP do gây ra Domain Shift.
3.  **Tập trung vào Prompt Engineering:** Cân nhắc cải tiến các câu Prompt mô tả lớp (như `"Picture of a pet with breed: {breed}, which is a type of {cat/dog}"`) để định vị vector văn bản tốt hơn, kết hợp cùng TTA để đạt hiệu quả tối đa.
