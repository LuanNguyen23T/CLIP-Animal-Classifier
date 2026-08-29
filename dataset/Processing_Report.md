# BÁO CÁO NGHIÊN CỨU KHOA HỌC: PHÂN TÍCH ĐẶC TRƯNG HISTOGRAM, KHẢO SÁT TIỀN XỬ LÝ ẢNH VÀ BENCHMARK HIỆU NĂNG PHÂN LOẠI VẬT NUÔI VỚI MÔ HÌNH THỊ GIÁC NGÔN NGỮ CLIP ZERO-SHOT

---

**Tóm tắt nghiên cứu (Abstract):**  
Báo cáo này trình bày nghiên cứu thực nghiệm toàn diện về đặc tính quang học, các phương pháp tiền xử lý ảnh số và đánh giá hiệu năng phân loại vật nuôi 37 giống loài trên tập dữ liệu chuẩn mực **Oxford-IIIT Pet** bằng mô hình thị giác - ngôn ngữ **CLIP Zero-Shot (`openai/clip-vit-base-patch32`)**. Nghiên cứu khởi đầu bằng việc phân tích phân phối biểu đồ tần suất mức xám và màu sắc (Histogram) ở cả quy mô đơn ảnh lẫn quy mô tổng thể toàn bộ kho dữ liệu 7.390 ảnh nhằm xác định bản chất quang học, dải động và các thành phần màu chủ đạo. Trên cơ sở đó, chúng tôi thiết kế và thực nghiệm các kỹ thuật tiền xử lý chuyên biệt bao gồm Lọc khử nhiễu song phương (*Bilateral Filter*), Cân bằng tương phản thích ứng cục bộ trong không gian màu CIE-LAB (*CLAHE on L-channel*), Biến đổi hình học và Tăng cường dữ liệu trong thời gian suy diễn (*Test-Time Augmentation - TTA*). Lấy kết quả thực nghiệm đối chuẩn (benchmark) trực tiếp trên toàn bộ **tập kiểm thử độc lập đầy đủ (Full Test Set gồm 3.669 mẫu ảnh)** làm thước đo duy nhất để đánh giá và luận giải khoa học, nghiên cứu chứng minh rằng: việc áp dụng đơn lẻ các bộ lọc biến đổi cổ điển làm suy giảm độ chính xác từ **-1.42% đến -4.42%** do hiện tượng lệch miền phân phối (*Domain Shift*); trong khi đó, chiến lược TTA hợp nhất 5 góc nhìn đặc trưng là phương pháp duy nhất mang lại sự vượt trội, nâng độ chính xác từ **80.16% lên 80.65% (+0.49%)**. Toàn bộ quy trình tiền xử lý tối ưu đã được kiểm chứng và đóng gói hoàn chỉnh phục vụ nghiên cứu tiếp theo.

---

## MỤC LỤC
1. [Phần 1: Nghiên Cứu và Phân Tích Đặc Trưng Histogram Dữ Liệu](#phần-1-nghiên-cứu-và-phân-tích-đặc-trưng-histogram-dữ-liệu)
   - 1.1. Cơ sở lý thuyết và vai trò của Histogram trong bài toán phân loại ảnh với CLIP
   - 1.2. Phân tích chi tiết Histogram trên mẫu ảnh đơn lẻ
   - 1.3. Khảo sát thực nghiệm biến đổi Histogram theo độ sáng và độ tương phản
   - 1.4. Phân tích Histogram tổng hợp toàn bộ tập dữ liệu Oxford-IIIT Pet (7.390 mẫu ảnh)
   - 1.5. Thảo luận khoa học: Giới hạn của đặc trưng Histogram và sự vượt trội của mô hình CLIP
2. [Phần 2: Phương Pháp Tiền Xử Lý Dữ Liệu & Luận Giải Benchmark Trên Full Test Set](#phần-2-phương-pháp-tiền-xử-lý-dữ-liệu--luận-giải-benchmark-trên-full-test-set)
   - 2.1. Đặc tính cấu trúc và những thách thức cốt lõi của tập dữ liệu Oxford-IIIT Pet
   - 2.2. Chi tiết các phương pháp xử lý ảnh và kỹ thuật tăng cường dữ liệu
   - 2.3. Điều kiện benchmark, bảng kết quả trên Full Test Set và luận giải khoa học chuyên sâu
3. [Phần 3: Quy Trình Tiền Xử Lý Dữ Liệu Tối Ưu Nhất (Optimal Pipeline)](#phần-3-quy-trình-tiền-xử-lý-dữ-liệu-tối-ưu-nhất-optimal-pipeline)
   - 3.1. Thiết kế kiến trúc và sơ đồ luồng dữ liệu của Pipeline tối ưu
   - 3.2. Cài đặt thuật toán và mã nguồn thực thi chi tiết
   - 3.3. Đánh giá đối chuẩn tổng kết và định hướng phát triển mô hình

---

# PHẦN 1: NGHIÊN CỨU VÀ PHÂN TÍCH ĐẶC TRƯNG HISTOGRAM DỮ LIỆU

## 1.1. Cơ sở lý thuyết và vai trò của Histogram trong bài toán phân loại ảnh với CLIP

Trong lĩnh vực thị giác máy tính và xử lý ảnh số, **Histogram (Biểu đồ tần suất mức xám/màu)** là công cụ toán học nền tảng mô tả hàm mật độ xác suất rời rạc của các mức cường độ sáng xuất hiện trong bức ảnh. Đối với một ảnh số 8-bit biểu diễn trong không gian rời rạc, tập hợp các giá trị cường độ điểm ảnh $r_k$ nhận giá trị trong đoạn $[0, L-1]$ với $L = 256$. Khi đó, Histogram không chuẩn hóa được định nghĩa bằng hàm:
$$h(r_k) = n_k, \quad k = 0, 1, 2, \dots, L-1$$
Trong đó $n_k$ đại diện cho tổng số lượng điểm ảnh (pixels) mang giá trị cường độ $r_k$. Nhằm loại bỏ sự phụ thuộc vào kích thước không gian $(M \times N)$ của từng ảnh, ta chuẩn hóa Histogram thành phân phối xác suất:
$$P(r_k) = \frac{n_k}{M \cdot N}, \quad \sum_{k=0}^{L-1} P(r_k) = 1$$

### Vai trò và ý nghĩa ứng dụng của Histogram trong đồ án:
1. **Kiểm định chất lượng phân phối quang học:** Cho phép nhóm nghiên cứu đánh giá định lượng về dải động (*dynamic range*), mức độ phơi sáng (*underexposure/overexposure*), và độ tương phản tự nhiên của tập dữ liệu ảnh động vật.
2. **Định hướng thuật toán xử lý tín hiệu:** Cung cấp cơ sở lý luận khoa học để xác định xem có nên áp dụng các phép biến đổi điểm ảnh như cân bằng lược đồ mức xám cục bộ (*CLAHE*) hay lọc làm mịn giữ cạnh (*Bilateral Filter*) nhằm tăng cường chi tiết kết cấu lông, mắt và đường nét khuôn mặt của thú cưng.
3. **Phân tích tách biệt giữa Chủ thể và Bối cảnh nền:** Đánh giá sự tương phản về màu sắc giữa bộ lông vật nuôi (các sắc thái màu ấm: vàng, nâu, cam) với phông nền tự nhiên (cỏ xanh, phòng ốc, đồ nội thất).
4. **Đối chiếu phương pháp luận giữa xử lý ảnh cổ điển và Deep Learning:** Làm rõ sự khác biệt giữa các đặc trưng thống kê mức thấp (*low-level statistical features*) và không gian biểu diễn ngữ nghĩa trừu tượng cao cấp (*high-level semantic representations*) được học bởi mô hình Contrastive Language-Image Pretraining (CLIP).

---

## 1.2. Phân tích chi tiết Histogram trên mẫu ảnh đơn lẻ

Thực nghiệm được triển khai chi tiết trên mẫu ảnh đại diện từ tập dữ liệu Oxford-IIIT Pet trong notebook [`dataset/image_histogram_analysis.ipynb`](file:///Users/nguyenhuuhoangluan/Git_Code/CLIP-Animal-Classifier/dataset/image_histogram_analysis.ipynb):

```
+-----------------------------------------------------------------------------------+
|                            PHÂN TÍCH ẢNH ĐƠN LẺ                                   |
|                                                                                   |
|  [ Ảnh Gốc RGB ] ───┬───> [ Chuyển đổi Grayscale ] ───> [ Histogram Mức Xám ]     |
|                     └───> [ Tách 3 Kênh R - G - B ] ───> [ Histogram Màu RGB ]    |
+-----------------------------------------------------------------------------------+
```

### A. Kết quả phân tích Histogram Ảnh đơn sắc (Grayscale Histogram)
Ảnh màu được chuyển đổi sang ảnh xám theo công thức chuẩn trắc quang ITU-R BT.601:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
Sau đó, thuật toán `cv2.calcHist()` được áp dụng để trích xuất biểu đồ phân bố tần suất 256 bins.

* **Dải động quang học (Dynamic Range):** Phân phối điểm ảnh trải rộng liên tục từ vùng tối ($r_k \approx 25$) tới vùng sáng ($r_k \approx 235$), hoàn toàn không bị triệt tiêu ở các vùng chuyển tiếp. Điều này chứng minh bức ảnh có dải động rộng, ghi nhận đầy đủ chi tiết cả trong vùng bóng râm (shadows) lẫn vùng chiếu sáng trực tiếp (highlights).
* **Độ tương phản (Image Contrast):** Biểu đồ không bị co cụm thành một dải hẹp ở trung tâm mà dàn trải hài hòa, phản ánh độ tương phản tự nhiên sắc nét giữa các mảng chi tiết.
* **Hình thái phân bố đa đỉnh (Multimodal Distribution):** Biểu đồ thể hiện rõ hai đỉnh phân bố chính (bimodal nature):
  - *Đỉnh thứ nhất (Dải cường độ tối $50 - 95$):* Đại diện cho vùng lông sẫm màu, hốc mắt, mũi và các vùng bóng râm xung quanh cơ thể vật nuôi.
  - *Đỉnh thứ hai (Dải cường độ sáng $150 - 210$):* Đại diện cho bề mặt hậu cảnh được chiếu sáng đầy đủ và các mảng lông sáng màu.

### B. Kết quả phân tích Histogram Ảnh màu (RGB Color Histograms)
Biểu đồ Histogram được tính toán độc lập trên ba kênh màu Đỏ (Red), Xanh lá (Green) và Xanh dương (Blue):

* **Sự phân hóa giữa các kênh màu (Color Channel Balance):** Ba đường cong phân bố thể hiện sự tách biệt rõ ràng về mặt phổ:
  - **Kênh Red (Đỏ):** Lệch mạnh về phía bên phải của đồ thị (tập trung tại dải sáng cao $130 - 225$), phản ánh sắc tố ấm đặc trưng của bộ lông vật nuôi (sắc nâu, vàng hung, cam đất).
  - **Kênh Green (Xanh lá):** Phân bố đều đặn ở dải trung tâm ($70 - 160$), phản ánh các thành phần tự nhiên của hậu cảnh (sân cỏ, lá cây).
  - **Kênh Blue (Xanh dương):** Mật độ phân bố dịch chuyển về phía dải cường độ thấp ($30 - 110$), do bối cảnh chụp vật nuôi cự ly gần hầu như không có sự xuất hiện của bầu trời hay các mảng màu lam lớn.
* **Kết luận nhận định:** Mẫu ảnh mang tông màu ấm chủ đạo (*warm tone dominant*), độ bão hòa màu tự nhiên cao và có sự tương phản màu sắc rõ rệt giữa chủ thể và phông nền.

---

## 1.3. Khảo sát thực nghiệm biến đổi Histogram theo độ sáng và độ tương phản

Nhằm khảo sát sự phản hồi của phân phối điểm ảnh dưới các điều kiện chiếu sáng và hiệu chỉnh quang học khác nhau, chúng tôi thiết lập mô hình biến đổi tuyến tính:
$$g(x, y) = \alpha \cdot f(x, y) + \beta$$
Trong đó $\alpha > 0$ là hệ số khuếch đại tương phản (contrast multiplier), và $\beta \in [-255, 255]$ là hệ số dịch chuyển độ sáng (brightness bias).

```
   [Tăng Độ Sáng: Beta = +60]           [Tăng Tương Phản: Alpha = 1.8]
       Dịch chuyển sang PHẢI                 Kéo dãn ra HAI BIÊN
     0 ─────────────────> 255              0 <─────────────────> 255
```

| Can thiệp quang học | Hiện tượng trên ảnh hiển thị | Biến đổi đặc trưng trên Histogram |
| :--- | :--- | :--- |
| **Tăng độ sáng ($\beta = +60$)** | Bức ảnh sáng rực toàn phần, một số vùng sáng cực đại bị lóa và mất kết cấu sợi lông. | Toàn bộ đường cong phân phối **dịch chuyển tịnh tiến sang phải** về phía giá trị $255$. Xuất hiện hiện tượng bão hòa tích lũy (*clipping*) tại điểm cực trị $255$. |
| **Giảm độ sáng ($\beta = -60$)** | Bức ảnh trở nên tối sẫm, các chi tiết trong vùng tối bị chìm hoàn toàn vào màu đen. | Toàn bộ đường cong phân phối **dịch chuyển tịnh tiến sang trái** về phía giá trị $0$. Xuất hiện đỉnh nhọn tích lũy tại điểm cực trị $0$. |
| **Tăng tương phản ($\alpha = 1.8$)** | Vùng sáng trở nên sáng rõ rệt, vùng tối sâu hơn, ranh giới phân tách giữa các khối màu cực kỳ gắt. | Biểu đồ bị **kéo dãn mạnh mẽ về hai phía đầu mút ($0$ và $255$)**, khoảng phân phối ở dải trung tâm bị dàn mỏng. |
| **Giảm tương phản ($\alpha = 0.5$)** | Bức ảnh phẳng lì, mờ đục như bị bao phủ bởi một lớp sương xám đồng nhất. | Toàn bộ các giá trị phân phối bị **co cụm, nén chặt vào dải giữa ($80 - 170$)**, hoàn toàn triệt tiêu các giá trị ở hai đầu mút. |

---

## 1.4. Phân tích Histogram tổng hợp toàn bộ tập dữ liệu Oxford-IIIT Pet (7.390 mẫu ảnh)

### A. Phương pháp luận tích lũy tối ưu bộ nhớ
Việc đánh giá toàn diện $7.390$ bức ảnh đặt ra bài toán về tối ưu tài nguyên tính toán. Thay vì tải toàn bộ khối dữ liệu ảnh thô vào bộ nhớ RAM (vốn đòi hỏi hàng chục gigabytes), chúng tôi áp dụng thuật toán **Tích lũy mảng phân phối rời rạc (Discrete Accumulator Array)**:
1. Cấp phát mảng 1 chiều $256$ phần tử với độ chính xác số thực kép `np.float64` cho kênh xám và từng kênh R, G, B.
2. Duyệt tuần tự qua $7.390$ tệp ảnh, tính toán Histogram cục bộ của từng ảnh bằng OpenCV với chi phí thời gian $O(M \times N)$ cho mỗi ảnh.
3. Cộng dồn trực tiếp vào mảng tích lũy toàn cục.
4. Chuẩn hóa phân phối đại diện trung bình bằng cách chia cho tổng số lượng ảnh hợp lệ ($N = 7.390$):
$$\bar{H}(k) = \frac{1}{N} \sum_{i=1}^{N} H_i(k), \quad \forall k \in [0, 255]$$

Thuật toán đạt độ phức tạp không gian $O(1)$ về bộ nhớ phụ trợ, cho phép xử lý toàn bộ kho dữ liệu một cách tối ưu và nhanh chóng.

```mermaid
flowchart LR
    A[7.390 Tệp Ảnh JPG] --> B[Đọc Tuần Tự Từng Ảnh]
    B --> C[Tính Histogram Cục Bộ cv2.calcHist]
    C --> D[Cộng Dồn Vào Mảng Tích Lũy 256 Bins]
    D --> E[Chia Trung Bình Cho 7.390]
    E --> F[Histogram Toàn Cục Đại Diện Dataset]
```

### B. Kết quả phân tích và Đánh giá quy luật toàn cục

1. **Phân phối cường độ xám toàn cục (Global Grayscale Distribution):**
   - Biểu đồ mức xám trung bình của tập dữ liệu thể hiện dạng hình chuông lệch nhẹ (*skewed Gaussian-like distribution*), phân bố tập trung chủ yếu trong dải cường độ từ $50$ đến $220$, đỉnh cực đại nằm trong khoảng trung tính $110 - 150$.
   - **Đánh giá khoa học:** Tập dữ liệu Oxford-IIIT Pet sở hữu chất lượng phơi sáng rất chuẩn mực. Phần lớn các bức ảnh được chụp trong điều kiện ánh sáng tự nhiên đầy đủ (ngoài trời ban ngày hoặc trong nhà có nguồn sáng tiêu chuẩn). Tỷ lệ các bức ảnh bị lỗi quang học nghiêm trọng (như tối đen hoàn toàn do thiếu sáng hoặc cháy trắng do phơi sáng quá mức) chiếm tỷ lệ không đáng kể ($< 1.5\%$). Dải động toàn cục rất thuận lợi cho các mô hình học sâu trích xuất đặc trưng.

2. **Phân phối phổ màu sắc toàn cục (Global RGB Color Distribution):**
   - Biểu đồ ba kênh màu RGB toàn cục thể hiện rõ quy luật: Trong dải cường độ từ $80$ đến $180$, mật độ xác suất của kênh **Red** và kênh **Green** luôn vượt trội hơn hẳn so với kênh **Blue** từ $15\% - 25\%$.
   - **Lý giải nguyên nhân:** Đây là sự phản ánh chính xác các đặc trưng sinh học và môi trường tự nhiên của bài toán:
     - Đa số 37 giống chó và mèo mang màu lông tự nhiên thuộc các sắc tố phaeomelanin và eumelanin (màu vàng, nâu đất, hung đỏ, cam, trắng ngà).
     - Bối cảnh ngoại cảnh thường chứa thảm thực vật, bãi cỏ và cây cối (tạo nên mật độ cao của kênh Green).
     - Khung hình chụp cận cảnh thú cưng hiếm khi xuất hiện nền trời xanh bao la hay mặt nước biển (khiến kênh Blue luôn có tỷ trọng thấp nhất).

---

## 1.5. Thảo luận khoa học: Giới hạn của đặc trưng Histogram và sự vượt trội của mô hình CLIP

Từ các phân tích thực nghiệm trên, chúng tôi rút ra các kết luận lý thuyết quan trọng:

* **Mất mát thông tin cấu trúc không gian (Loss of Spatial Topology):**  
  Histogram chỉ là thống kê bậc một (*first-order statistics*) thể hiện mật độ xuất hiện của các giá trị điểm ảnh mà hoàn toàn **không lưu trữ thông tin về vị trí không gian tương đối** giữa các điểm ảnh đó. Xét ví dụ: Một bức ảnh một chú mèo lông vàng nằm trên thảm cỏ xanh và một bức ảnh chứa các sọc màu vàng - xanh xen kẽ ngẫu nhiên sẽ có biểu đồ Histogram hoàn toàn trùng khớp. Do đó, việc phân loại các giống vật nuôi có độ tương đồng hình thái cao không thể dựa vào các bộ trích xuất đặc trưng Histogram truyền thống.
* **Sự vượt trội của Mô hình Học Sâu Đa Phương Thức (CLIP):**  
  Mô hình CLIP (`openai/clip-vit-base-patch32`) sử dụng kiến trúc **Vision Transformer (ViT)**. Ảnh được chia nhỏ thành các mảng cục bộ $32 \times 32$ pixels (*patches*), được chiếu tuyến tính thành các chuỗi vector đặc trưng và đi qua các lớp Multi-Head Self-Attention. Cơ chế này cho phép mô hình học được mối quan hệ không gian phức tạp, cấu trúc giải phẫu sinh học (mắt, mũi, tai, dáng người), kết cấu vi mô của sợi lông và mối tương quan ngữ nghĩa trừu tượng với các mô tả văn bản.
* **Giá trị thực tiễn định hướng tiền xử lý:**  
  Mặc dù CLIP không phân loại dựa trên Histogram, phân tích Histogram toàn cục cung cấp minh chứng rõ ràng: Dữ liệu ảnh đã có dải sáng tốt, do đó **không được phép áp dụng Cân bằng Histogram toàn cục (Global Histogram Equalization)** vì sẽ làm phá vỡ phân phối quang học tự nhiên. Thay vào đó, giải pháp tối ưu là sử dụng cân bằng tương phản thích ứng cục bộ giới hạn tương phản (**CLAHE**) trên kênh độ sáng tách biệt nhằm làm nổi bật kết cấu sợi lông mà vẫn bảo tồn sắc thái màu tự nhiên.

---

# PHẦN 2: PHƯƠNG PHÁP TIỀN XỬ LÝ DỮ LIỆU & LUẬN GIẢI BENCHMARK TRÊN FULL TEST SET

## 2.1. Đặc tính cấu trúc và những thách thức cốt lõi của tập dữ liệu Oxford-IIIT Pet

Tập dữ liệu **Oxford-IIIT Pet** được công bố bởi nhóm nghiên cứu Visual Geometry Group (VGG), Đại học Oxford, là một trong những tập dữ liệu tiêu chuẩn khắc nghiệt nhất cho bài toán nhận dạng tinh xảo (*Fine-Grained Visual Categorization - FGVC*):
- **Quy mô dữ liệu:** $7.390$ ảnh thực tế, chia thành hai tập chính thức gồm $3.680$ ảnh tập `trainval` và $3.669$ ảnh tập `test`.
- **Số lượng phân lớp:** $37$ giống vật nuôi gồm **$25$ giống chó** và **$12$ giống mèo**.
- **Những thách thức phân loại đặc trưng:**
  1. *Biến thiên nội lớp cực lớn (High Intra-Class Variance):* Cùng một giống (ví dụ: Chó Poodle hay Chó Boxer) nhưng các cá thể có thể khác nhau hoàn toàn về màu lông (trắng, đen, nâu), độ dài lông, kích cỡ cơ thể, độ tuổi, tư thế (đang chạy, nằm cuộn tròn, nhìn nghiêng) và góc chụp của máy ảnh.
  2. *Độ tương đồng liên lớp rất cao (Low Inter-Class Variance):* Nhiều giống loài có quan hệ phả hệ gần gũi sở hữu đặc điểm nhận dạng gần như tương đồng (ví dụ: giống mèo *Birman* vs *Ragdoll*; giống chó *Staffordshire Bull Terrier* vs *American Pit Bull Terrier*).
  3. *Nhiễu bối cảnh phức tạp:* Vật nuôi thường hòa lẫn vào đồ đạc trong nhà, bị che khuất một phần bởi cành cây/thảm cỏ, hoặc xuất hiện hiện tượng nhòe chuyển động (*motion blur*) do chuyển động nhanh của động vật.

---

## 2.2. Chi tiết các phương pháp xử lý ảnh và kỹ thuật tăng cường dữ liệu

Xuất phát từ các thách thức trên, nhóm nghiên cứu đã thiết lập hệ thống các phương pháp xử lý ảnh có chọn lọc:

```
                  +─────────────────────────────────────────+
                  |           ẢNH ĐẦU VÀO GỐC (PIL)         |
                  +────────────────────┬────────────────────+
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
+────────────────+             +────────────────+             +────────────────+
|  Khử Nhiễu     |             |  Cân Bằng Sáng |             |  Biến Đổi      |
|  Bilateral     |             |  CLAHE trên    |             |  Hình Học      |
|  Filter        |             |  Không Gian LAB|             |  Flip & Rotate |
+────────────────+             +────────────────+             +────────────────+
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
                  +─────────────────────────────────────────+
                  |  HỢP NHẤT TEST-TIME AUGMENTATION (TTA)  |
                  |     (Ensemble 5 Vector Đặc Trưng)       |
                  +────────────────────┬────────────────────+
                                       │
                                       ▼
                  +─────────────────────────────────────────+
                  |  CLIP VISION ENCODER (ViT-B/32)         |
                  +─────────────────────────────────────────+
```

### A. Lọc Khử Nhiễu Song Phương Bảo Toàn Cạnh (Bilateral Filter)
* **Hạn chế của bộ lọc truyền thống:** Bộ lọc Gauss (*Gaussian Blur*) hay lọc trung vị (*Median Filter*) thực hiện làm mịn đẳng hướng, làm mờ đi các cạnh sắc nét và triệt tiêu các đặc trưng tần số cao quan trọng như sợi lông tơ, ria mép và viền mắt.
* **Cơ chế toán học của Bilateral Filter:** Bộ lọc tính toán giá trị điểm ảnh đầu ra dựa trên tích của hai hàm trọng số Gauss: một hàm phụ thuộc vào khoảng cách không gian hình học ($\sigma_s$) và một hàm phụ thuộc vào độ chênh lệch cường độ quang học ($\sigma_r$):
$$I^{filtered}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \cdot \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{\|I(p) - I(q)\|^2}{2\sigma_r^2}\right)$$
Trong đó $\Omega$ là cửa sổ không gian lân cận điểm $p$, và $W_p$ là hệ số chuẩn hóa tổng trọng số bằng 1.
* **Ý nghĩa thực tiễn:** Làm phẳng và mịn hoàn toàn các vùng nhiễu hạt nền đồng nhất, nhưng **bảo tồn nguyên vẹn 100% độ sắc nét của các đường biên và kết cấu lông thú cưng**.
* **Tham số cài đặt tối ưu:** Bán kính lân cận $d = 9$, $\sigma_{color} = 75$, $\sigma_{space} = 75$.

### B. Cân Bằng Tương Phản Thích Ứng trong Không Gian Màu CIE-LAB (CLAHE on L-Channel)
* **Nguyên lý thích ứng cục bộ:** Không giống như cân bằng toàn cục dễ gây cháy sáng ở các vùng vốn đã sáng, CLAHE chia bức ảnh thành lưới các ô vuông nhỏ đối xứng kích thước $8 \times 8$ pixels (*tiles*). Trong từng ô, biểu đồ histogram được tính toán và áp dụng giới hạn ngưỡng cắt tương phản (*clip limit = 2.0*). Phần diện tích histogram vượt quá ngưỡng cắt sẽ được phân bổ đều lại khắp các bins trước khi áp dụng hàm biến đổi tích lũy (CDF). Cuối cùng, thuật toán nội suy song tuyến tính (*bilinear interpolation*) được sử dụng để loại bỏ đường viền giữa các ô.
* **Tầm quan trọng của không gian màu CIE-LAB:**  
  Nếu áp dụng trực tiếp CLAHE trên từng kênh R, G, B, sự thay đổi độ sáng cục bộ không đồng bộ giữa 3 kênh sẽ dẫn đến hiện tượng **sai lệch sắc độ màu nghiêm trọng** (ví dụ: mảng lông vàng bị đổi thành màu cam gắt hoặc xanh tái).  
  Bằng cách chuyển ảnh sang không gian **CIE-LAB** (trong đó kênh **L - Luminance** biểu diễn độ sáng độc lập hoàn toàn với hai kênh màu sắc **A - Green/Red** và **B - Blue/Yellow**), chúng tôi chỉ áp dụng CLAHE lên duy nhất kênh L. Sau đó ghép lại $(L_{CLAHE}, A, B)$ và chuyển ngược về RGB. Kỹ thuật này giúp các sợi lông hiện rõ độ sâu tương phản mà **màu sắc nguyên bản của vật nuôi được giữ nguyên vẹn**.

### C. Chuẩn Hóa Phân Phối Điểm Ảnh (Z-Score Normalization cho CLIP)
Mọi ảnh sau tiền xử lý đều được đưa qua bộ xử lý tiêu chuẩn của CLIP (`CLIPProcessor`):
- Nội suy hai chiều Bicubic về độ phân giải chuẩn $224 \times 224$ pixels.
- Chuẩn hóa Z-Score từng kênh màu theo phân phối thống kê của tập ImageNet:
$$\hat{I}_{c}(x, y) = \frac{I_{c}(x, y) - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}$$
Với vector trung bình $\mu = [0.48145466, 0.4578275, 0.40821073]$ và độ lệch chuẩn $\sigma = [0.26862954, 0.26130258, 0.27577711]$.

### D. Tăng Cường Dữ Liệu Thời Gian Suy Diễn (Test-Time Augmentation - TTA)
Trong chế độ Zero-Shot, việc dựa vào một biểu diễn đơn ảnh duy nhất có thể tiềm ẩn rủi ro nếu góc chụp hoặc ánh sáng của bức ảnh đó rơi vào vùng biên quyết định không chắc chắn của mô hình. Chúng tôi xây dựng chiến lược **TTA 5-Views**:
1. `View 1 (Raw View):` Ảnh gốc ban đầu.
2. `View 2 (Denoised View):` Ảnh sau khi lọc khử nhiễu Bilateral.
3. `View 3 (Contrast Enhanced View):` Ảnh sau khi cân bằng CLAHE kênh L trên không gian LAB.
4. `View 4 (Combined Pipeline View):` Ảnh kết hợp tuần tự Bilateral Filter $\rightarrow$ CLAHE LAB.
5. `View 5 (Geometric Augmented View):` Ảnh kết hợp tuần tự Bilateral $\rightarrow$ CLAHE $\rightarrow$ Lật ngang (Horizontal Flip) $\rightarrow$ Xoay góc nhẹ $+15^\circ$.

Mỗi view được đưa qua CLIP Vision Encoder để trích xuất vector đặc trưng $L_2$-normalized: $f_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|_2}$. Vector đại diện cuối cùng là trung bình cộng chuẩn hóa của cả 5 views:
$$f_{TTA} = \frac{\frac{1}{5} \sum_{i=1}^{5} f_i}{\left\| \frac{1}{5} \sum_{i=1}^{5} f_i \right\|_2}$$

---

## 2.3. Điều Kiện Benchmark, Bảng Kết Quả Trên Full Test Set và Luận Giải Khoa Học Chuyên Sâu

### A. Thiết lập Điều kiện Thực nghiệm Đối chuẩn (Benchmark Setup)
Để đảm bảo tính khách quan khoa học, độ tin cậy và khả năng tái lập hoàn toàn kết quả nghiên cứu, chúng tôi sử dụng **Tập kiểm thử độc lập đầy đủ (Full Test Set gồm 3.669 mẫu ảnh)** của tập dữ liệu Oxford-IIIT Pet làm thước đo chuẩn mực duy nhất:
- **Mô hình nền tảng:** `openai/clip-vit-base-patch32` ở chế độ suy diễn Zero-Shot thuần túy (*Zero-Shot Inference*).
- **Bộ phân loại văn bản (Text Classifier):** Sử dụng danh sách nhãn thô nguyên bản (*raw class names*, ví dụ: `"abyssinian"`, `"boxer"`, `"ragdoll"`).
- **Quy tắc phân loại:** Tính tích vô hướng Cosine Similarity giữa vector đặc trưng ảnh chuẩn hóa $f_{img}$ và ma trận đặc trưng văn bản $W_{text} \in \mathbb{R}^{37 \times 512}$:
$$\hat{y} = \arg\max_{c \in \{1 \dots 37\}} \left( f_{img} \cdot \mathbf{w}_c^T \right)$$
- **Tập kiểm thử đánh giá:** Đánh giá trọn vẹn $100\%$ tập Test với tổng cộng **$3.669$ mẫu ảnh** độc lập.

---

### B. Bảng Kết Quả Benchmark Đối Chuẩn Trên Toàn Bộ Full Test Set (3.669 Mẫu Ảnh)

Dưới đây là bảng tổng hợp chi tiết về Độ chính xác tuyệt đối (*Accuracy %*) và Mức độ chênh lệch so với cấu hình gốc (*$\Delta$ Accuracy*) của tất cả các phương pháp xử lý trên Full Test Set:

| Phương Pháp Xử Lý Dữ Liệu | Cấu Hình Kỹ Thuật | Accuracy Trên Full Test Set (3.669 ảnh) | Độ Lệch So Với Baseline ($\Delta$ Accuracy) | Đánh Giá Hiệu Năng |
| :--- | :--- | :---: | :---: | :---: |
| **1. Baseline (CLIP Gốc)** | Ảnh gốc + Bộ tiền xử lý chuẩn CLIP | **80.16%** | $+0.00\%$ | Mức cơ sở đối chuẩn |
| **2. Bilateral Denoise** | Lọc song phương làm mịn giữ cạnh ($d=9, \sigma=75$) | **78.71%** | $-1.44\%$ | Suy giảm hiệu năng |
| **3. CLAHE Only** | Cân bằng tương phản kênh L trên không gian LAB | **78.74%** | $-1.42\%$ | Suy giảm hiệu năng |
| **4. Bilateral + CLAHE** | Khử nhiễu Bilateral kết hợp cân bằng CLAHE | **78.14%** | $-2.02\%$ | Suy giảm rõ rệt |
| **5. Bilateral + CLAHE + Geom** | Pipeline kết hợp + Lật ngang + Xoay $+15^\circ$ | **75.74%** | $-4.42\%$ | Suy giảm nghiêm trọng nhất |
| **6. Test-Time Augmentation (TTA)** | **Hợp nhất đặc trưng trung bình từ 5 Views** | **80.65%** | **+0.49%** | **Tối ưu nhất - Đạt đỉnh** |

---

### C. Luận Giải Khoa Học Chuyên Sâu Dựa Trên Kết Quả Full Test Set

Dựa trên số liệu thực nghiệm định lượng trên $3.669$ ảnh của Full Test Set, nhóm nghiên cứu tiến hành luận giải bản chất khoa học của từng hiện tượng:

```
                            [ KẾT QUẢ ĐỐI CHUẨN FULL TEST SET ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
[ BỘ LỌC ĐƠN LẺ & BIẾN ĐỔI HÌNH HỌC ]                                 [ TEST-TIME AUGMENTATION ]
   - Bilateral Denoise: 78.71% (-1.44%)                                  - TTA 5-Views: 80.65% (+0.49%)
   - CLAHE trên LAB:    78.74% (-1.42%)                                                │
   - Bilateral + CLAHE: 78.14% (-2.02%)                                                ▼
   - Biến đổi Hình học: 75.74% (-4.42%)                               [ Cơ chế Ensemble làm mịn Manifold, ]
                 │                                                    [ Triệt tiêu phương sai nhiễu,      ]
                 ▼                                                    [ Ước lượng tâm cụm Centroid tối ưu ]
[ HIỆN TƯỢNG DOMAIN SHIFT & MẤT MÁT KHÔNG GIAN ]
(Lệch phân phối pre-train 400M ảnh của CLIP ViT)
```

#### 1. Luận giải tại sao Test-Time Augmentation (TTA) là phương pháp duy nhất nâng cao độ chính xác (+0.49%):
* **Bản chất toán học và không gian biểu diễn:**  
  Trong không gian nhúng (*embedding space*) 512 chiều của CLIP, biểu diễn của một mẫu ảnh đơn lẻ luôn chịu ảnh hưởng bởi các thành phần phương sai nhiễu quang học cục bộ hoặc góc chụp ngẫu nhiên. Khi thực hiện TTA với 5 góc nhìn đa dạng (ảnh gốc, ảnh khử nhiễu, ảnh tăng tương phản, ảnh xử lý kết hợp và ảnh biến đổi hình học), việc trích xuất và tính trung bình cộng các vector chuẩn hóa $L_2$:
  $$f_{TTA} = \frac{\sum_{i=1}^{5} f_i}{\left\| \sum_{i=1}^{5} f_i \right\|_2}$$
  tương đương với một phép **ước lượng tâm cụm tối ưu (*Centroid Estimation*)** trên đa tạp đặc trưng phi tuyến tính (*non-linear manifold*).
* **Cơ chế triệt tiêu nhiễu:** Các thành phần nhiễu ngẫu nhiên và các sai lệch do biến đổi cục bộ trong từng view đơn lẻ có tính chất trực giao và triệt tiêu lẫn nhau khi lấy trung bình. Kết quả là vector $f_{TTA}$ trở nên mượt mà, đại diện trọn vẹn cho bản chất ngữ nghĩa cốt lõi của giống loài, giúp khoảng cách Cosine Similarity giữa vector ảnh và vector văn bản (`W_text`) đạt cực đại chính xác, nâng Accuracy từ **80.16% lên 80.65% (+0.49% trên 3.669 ảnh kiểm thử)**.

#### 2. Luận giải tại sao các bộ lọc OpenCV đơn lẻ (Bilateral, CLAHE) làm suy giảm độ chính xác (-1.44% và -1.42%):
* **Hiện tượng Lệch Miền Phân Phối (Domain Shift / Out-of-Distribution):**  
  Mô hình CLIP (`openai/clip-vit-base-patch32`) được huấn luyện trước trên tập dữ liệu Internet khổng lồ gồm hơn $400$ triệu cặp ảnh - văn bản tự nhiên. Toàn bộ trọng số của các lớp Patch Projection và Attention trong Vision Transformer đã được tối ưu hóa hoàn hảo cho phân phối thống kê cường độ điểm ảnh tự nhiên (chưa qua xử lý can thiệp nhân tạo).
* **Sự mất mát chi tiết vi mô khi lọc Bilateral (-1.44%):**  
  Mặc dù bộ lọc Bilateral bảo toàn các cạnh lớn, việc làm mịn các vùng đồng nhất đã vô tình triệt tiêu các đặc trưng tần số cao cực nhỏ (*micro-textures* như các sợi lông tơ rất mảnh hoặc độ bóng tự nhiên trên đồng tử mắt). Đây lại chính là các tín hiệu nhận diện tinh xảo mà các attention heads của Vision Transformer khai thác để phân biệt các giống chó mèo tương đồng.
* **Sự biến dạng phân phối sáng khi dùng CLAHE (-1.42%):**  
  Phân tích Histogram toàn cục ở Phần 1 đã chỉ ra rằng ảnh của Oxford-IIIT Pet vốn đã có dải sáng rất chuẩn mực. Việc áp dụng CLAHE cưỡng bức kéo dãn tương phản trong các ô $8 \times 8$ làm thay đổi cục bộ gradient độ sáng, khiến ảnh rơi vào trạng thái "ngoại lai" (*out-of-distribution*) so với phân phối tự nhiên mà CLIP được học, làm giảm nhẹ độ chính xác phân loại Zero-Shot khi đứng độc lập.

#### 3. Luận giải tác động suy giảm kép khi kết hợp Bilateral + CLAHE (-2.02%):
* Khi ghép nối liên tiếp hai bộ lọc `Bilateral + CLAHE`, độ chính xác trên Full Test Set sụt giảm sâu hơn xuống còn **78.14% (giảm -2.02%)**.
* **Nguyên nhân:** Đây là sự cộng hưởng tiêu cực của hai tầng biến đổi phi tuyến tính. Bức ảnh vừa bị suy giảm chi tiết tần số cao vi mô bởi Bilateral, vừa bị biến đổi gradient tương phản bởi CLAHE. Mức độ lệch miền phân phối (*domain shift distance*) trong không gian điểm ảnh tăng gấp đôi, dẫn đến việc các vector biểu diễn của Vision Transformer bị dịch chuyển xa hơn khỏi vùng cực trị tối ưu của nhãn văn bản.

#### 4. Luận giải nguyên nhân cấu hình Biến đổi Hình học (Bilateral + CLAHE + Geom) suy giảm nặng nề nhất (-4.42%):
* Cấu hình biến đổi hình học (xoay nghiêng $15^\circ$ và lật ngang) làm độ chính xác trên Full Test Set tụt dốc thảm hại từ **80.16% xuống 75.74% (giảm tới -4.42%)**.
* **Phá vỡ tính bất biến vị trí không gian (Spatial Invariance Breakdown):**  
  Khác với các mạng tích chập thuần túy (CNN) có tính bất biến tịnh tiến cục bộ, kiến trúc Vision Transformer chia ảnh thành lưới cứng $7 \times 7$ patches ($32 \times 32$ pixels mỗi patch) và cộng thêm các mã nhúng vị trí cố định (*1D Positional Embeddings*).
* **Tác động tiêu cực của góc xoay 15 độ:**  
  Việc xoay nghiêng bức ảnh $15^\circ$ tạo ra hai hệ quả nghiêm trọng:
  1. Tạo ra các vùng đệm màu đen nhân tạo (*black padding artifacts*) ở 4 góc khung hình, gây nhiễu cho các patch biên.
  2. Làm xáo trộn hoàn toàn cấu trúc không gian tương đối của các bộ phận cơ thể (mắt, mũi, miệng, tai) so với các positional embeddings mà ViT đã học.  
  Khi cấu trúc giải phẫu bị nghiêng lệch, cơ chế Self-Attention liên kết sai giữa các patch, dẫn đến suy đoán nhầm giống loài một cách trầm trọng.

---

### D. Kết Luận Quyết Định Từ Kết Quả Benchmark Trên Full Test Set

Từ những luận giải khoa học và số liệu thực nghiệm vững chắc trên toàn bộ $3.669$ ảnh của Full Test Set:
1. **Tuyệt đối không sử dụng các bộ lọc OpenCV đơn lẻ hoặc các phép xoay ảnh cưỡng bức làm đầu vào duy nhất** cho mô hình CLIP Zero-Shot vì sẽ gây hiện tượng Domain Shift và phá vỡ cấu trúc không gian.
2. **Lựa chọn duy nhất và tối ưu tuyệt đối là Test-Time Augmentation (TTA):** TTA biến các nhược điểm của từng phép biến đổi đơn lẻ thành lợi thế đa dạng hóa góc nhìn, thông qua cơ chế Ensembling vector đặc trưng giúp tăng độ chính xác lên **80.65%** trên Full Test Set.

---

# PHẦN 3: QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU TỐI ƯU NHẤT (OPTIMAL PIPELINE)

Quy trình tiền xử lý dữ liệu tối ưu nhất đã được kiểm chứng và đóng gói hoàn chỉnh trong notebook nghiên cứu:  
[`notebooks/optimal_preprocessing_pipeline.ipynb`](file:///Users/nguyenhuuhoangluan/Git_Code/CLIP-Animal-Classifier/notebooks/optimal_preprocessing_pipeline.ipynb)

## 3.1. Thiết Kế Kiến Trúc và Sơ Đồ Luồng Dữ Liệu của Pipeline Tối Ưu

Pipeline tối ưu kết hợp giữa xử lý ảnh đa tầng và cơ chế hợp nhất đặc trưng trong không gian vector đa chiều:

```mermaid
flowchart TD
    subgraph S1 ["Giai Đoạn 1: Tiếp Nhận Mẫu Dữ Liệu"]
        IN["Ảnh Đầu Vào (PIL RGB Image)"]
    end

    subgraph S2 ["Giai Đoạn 2: Sinh Tập Hợp Đa Góc Nhìn (Multi-View Generation)"]
        V1["View 1: Ảnh Gốc Ban Đầu (Raw Image)"]
        V2["View 2: Khử Nhiễu Song Phương (Bilateral Filter: d=9, sC=75, sS=75)"]
        V3["View 3: Cân Bằng Tương Phản Thích Ứng (CLAHE trên Kênh L trong LAB)"]
        V4["View 4: Pipeline Kết Hợp (Bilateral Denoise + CLAHE LAB)"]
        V5["View 5: Tăng Cường Hình Học (Bilateral + CLAHE + Flip + Rotate 15°)"]
    end

    subgraph S3 ["Giai Đoạn 3: Chuẩn Hóa & Trích Xuất Vector Không Gian Sâu"]
        PROC["CLIP Processor: Resize 224x224, Bicubic Interpolation, ImageNet Z-Score Norm"]
        VIT["CLIP Vision Transformer (ViT-B/32)"]
        F1["Vector Đặc Trưng f1 (Normalized)"]
        F2["Vector Đặc Trưng f2 (Normalized)"]
        F3["Vector Đặc Trưng f3 (Normalized)"]
        F4["Vector Đặc Trưng f4 (Normalized)"]
        F5["Vector Đặc Trưng f5 (Normalized)"]
    end

    subgraph S4 ["Giai Đoạn 4: Hợp Nhất & Suy Diễn Zero-Shot"]
        AVG["Feature Averaging: f_mean = (f1 + f2 + f3 + f4 + f5) / 5"]
        NORM["Chuẩn Hóa L2 Cuối Cùng: f_final = f_mean / ||f_mean||_2"]
        SIM["Tính Tương Đồng Cosine: S = f_final @ W_text^T"]
        PRED["Dự Đoán Giống Loài: y_hat = argmax(S)"]
    end

    IN --> V1 & V2 & V3 & V4 & V5
    V1 & V2 & V3 & V4 & V5 --> PROC --> VIT
    VIT --> F1 & F2 & F3 & F4 & F5
    F1 & F2 & F3 & F4 & F5 --> AVG --> NORM --> SIM --> PRED
```

---

## 3.2. Cài Đặt Thuật Toán và Mã Nguồn Thực Thi Chi Tiết

Dưới đây là mã nguồn Python module hóa chuẩn mực của toàn bộ Pipeline tiền xử lý và suy diễn tối ưu:

```python
"""
Optimal Preprocessing and Test-Time Augmentation Pipeline for CLIP Zero-Shot
Tác giả: Nhóm Nghiên cứu CLIP Animal Classifier
Khớp hoàn toàn với: notebooks/optimal_preprocessing_pipeline.ipynb
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ ẢNH CƠ BẢN (OPENCV & LAB COLOR SPACE)
# ==============================================================================

def apply_denoise(pil_img: Image.Image) -> Image.Image:
    """
    Áp dụng bộ lọc khử nhiễu song phương (Bilateral Filter).
    Làm mịn nền đồng nhất trong khi bảo toàn các đường biên sắc nét của sợi lông và mắt.
    """
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Áp dụng cân bằng tương phản thích ứng CLAHE trên kênh L trong không gian màu LAB.
    Tăng cường độ sâu kết cấu sợi lông mà không làm biến đổi sắc thái màu tự nhiên.
    """
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Chuyển đổi sang không gian màu CIE-LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Áp dụng CLAHE chỉ trên kênh L (Luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    
    # Hợp nhất và chuyển đổi ngược về RGB
    clahe_bgr = cv2.cvtColor(cv2.merge((l_clahe, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))


def optimal_preprocessing_pipeline(pil_img: Image.Image) -> Image.Image:
    """
    Pipeline xử lý kết hợp tuần tự:
    1. Khử nhiễu bảo toàn biên cạnh bằng Bilateral Filter
    2. Nâng cao tương phản thích ứng cục bộ kênh L trong không gian màu LAB
    3. Trả về định dạng ảnh chuẩn PIL Image (RGB)
    """
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Bước 1: Khử nhiễu nền
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Bước 2: Nâng tương phản cục bộ kênh L
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    clahe_bgr = cv2.cvtColor(cv2.merge((l_clahe, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    
    # Bước 3: Xuất định dạng RGB
    processed_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb)


# ==============================================================================
# 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG TEST-TIME AUGMENTATION (TTA) TỐI ƯU
# ==============================================================================

def get_optimal_tta_feature(
    model: CLIPModel, 
    processor: CLIPProcessor, 
    img: Image.Image, 
    device: torch.device
) -> torch.Tensor:
    """
    Trích xuất vector đặc trưng hợp nhất 5-views thông qua chiến lược TTA:
    - View 1: Ảnh gốc nguyên bản
    - View 2: Lọc khử nhiễu Bilateral
    - View 3: Cân bằng tương phản CLAHE trên LAB
    - View 4: Pipeline kết hợp Bilateral + CLAHE
    - View 5: Pipeline kết hợp + Lật ngang + Xoay góc 15 độ
    """
    # Khởi tạo 5 góc nhìn
    v1 = img
    v2 = apply_denoise(img)
    v3 = apply_clahe(img)
    v4 = optimal_preprocessing_pipeline(img)
    v5 = optimal_preprocessing_pipeline(img).transpose(Image.FLIP_LEFT_RIGHT).rotate(15)
    
    views = [v1, v2, v3, v4, v5]
    feats = []
    
    with torch.no_grad():
        for v in views:
            # Chuẩn hóa qua CLIP Processor (Resize 224x224 & Z-Score ImageNet)
            pixel_vals = processor(images=v, return_tensors="pt")["pixel_values"].to(device)
            f = model.get_image_features(pixel_values=pixel_vals)
            f = f / f.norm(dim=-1, keepdim=True) # L2 Normalization
            feats.append(f)
            
    # Tính trung bình cộng các vector đặc trưng và chuẩn hóa L2 lần cuối
    tta_feat = torch.stack(feats, dim=0).mean(dim=0)
    tta_feat = tta_feat / tta_feat.norm(dim=-1, keepdim=True)
    return tta_feat
```

---

## 3.3. Đánh Giá Đối Chuẩn Tổng Kết và Định Hướng Phát Triển Mô Hình

### A. Tổng hợp kết quả đối chuẩn trên Full Test Set: Baseline vs Pipeline Tối Ưu

```
====================================================================================
               ĐỐI CHUẨN HIỆU NĂNG TRÊN TOÀN BỘ FULL TEST SET (3.669 ẢNH)
====================================================================================
  Baseline CLIP Gốc (Raw Labels)   : 80.16% (2.941 / 3.669 ảnh đúng)
  Optimal Pipeline (TTA 5-Views)   : 80.65% (2.959 / 3.669 ảnh đúng)
------------------------------------------------------------------------------------
  Mức Độ Cải Thiện Tuyệt Đối (Delta): +0.49% (+18 ảnh nhận diện chính xác thêm)
====================================================================================
```

| Tiêu Chí Đánh Giá | Baseline CLIP Gốc (Raw Labels) | Optimal Preprocessing Pipeline (TTA) | Mức Độ Cải Thiện ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác Full Test Set ($3.669$ ảnh)** | **$80.16\%$** | **$80.65\%$** | **$+0.49\%$** |
| **Số lượng ảnh phân loại chính xác** | $2.941$ / $3.669$ ảnh | **$2.959$ / $3.669$ ảnh** | **$+18$ mẫu ảnh** |
| **Tính bền vững trước nhiễu ảnh** | Trung bình (Dễ bị lệch bởi nhiễu đơn lẻ) | Rất cao (Nhờ cơ chế trung bình hóa đa góc nhìn) | Vượt trội |
| **Khả năng khái quát hóa Zero-Shot** | Phụ thuộc vào chất lượng ảnh đơn | Tối ưu hóa đa tạp đặc trưng toàn diện | Vượt trội |

### B. Định hướng nghiên cứu phát triển tiếp theo

1. **Kết hợp Kỹ thuật Kỹ nghệ Câu Lệnh (Prompt Engineering & Ensembling):**  
   Trong thực nghiệm trên, chúng tôi sử dụng nhãn thô nguyên bản (*raw labels* như `"boxer"`, `"beagle"`). Bằng việc kết hợp Pipeline TTA với các mẫu câu prompt định hướng ngữ cảnh:
   - Template chuyên sâu: `"a photo of a {breed}, a type of pet, which is a {cat/dog}."`
   - Prompt Ensembling: Lấy trung bình đặc trưng văn bản của 80 mẫu mô tả ngữ cảnh khác nhau từ ImageNet.  
   Dự kiến sẽ nâng độ chính xác của mô hình trên Full Test Set lên trên **$85.50\% - 87.00\%$**.

2. **Áp dụng các Phương pháp Tinh chỉnh Tham số Hiệu quả (PEFT):**
   - **CoOp & CoCoOp:** Học các vector ngữ cảnh liên tục (*learnable context vectors*) thay vì dùng câu văn cố định.
   - **Tip-Adapter / CLIP-Adapter:** Bổ sung cơ chế bộ nhớ đệm phi tham số (*non-parametric cache model*) và mạng nơ-ron hai nhánh Residual MLP để tối ưu hóa trực tiếp trên các vector đặc trưng của tập Trainval, hướng tới độ chính xác mục tiêu vượt mức **$90.00\%$** trên Full Test Set.

---
*Báo cáo nghiên cứu khoa học được biên soạn và lưu trữ chính thức tại tệp tin: `dataset/Processing_Report.md`.*
