# Đánh giá bản thảo Gemma-CLIP v5 theo phiếu phản biện Lần 3

**Ngày rà soát:** 07/11/2026  
**Bản được đánh giá:** `paper_CLIP_Gemma4_revised_v5.md`  
**Căn cứ:** Phiếu `DG31_NhanXet_BanNopLan3_Gemma_CLIP_Oxford_Pet_2026-11-06.pdf`, paper v4 vừa pull, các JSON kết quả và mã trong repository. Bản v5 là bản sửa biên tập từ v4; **không có thí nghiệm huấn luyện hoặc suy luận mới trong lần này**.

## 1. Kết luận

V5 đã sửa ba điểm trình bày quan trọng của v4: (1) Bảng 8 ghi đúng Oxford **raw (5-View TTA)**, (2) không gán số CuPL tác giả của Oxford cho hàng **Gemma proxy** trên Oxford, và (3) không đưa kết quả CuPL ánh xạ ImageNet sang Stanford Dogs vào bảng so sánh chính khi repo chưa có mã ánh xạ và suy luận tái lập kết quả đó. Bản này cũng giảm các khẳng định nhân quả về độ ổn định few-shot và tính giải thích, đồng thời chỉ ra một mô tả Ragdoll chưa đúng trong ngân hàng thuộc tính.

**Chưa thể gọi là đáp ứng 100% mọi yêu cầu của thầy.** Phiếu Lần 3 đánh giá TC2–TC5 đạt, TC1 cần chỉnh sửa. V5 đã định vị TC1 tốt hơn, nhưng thầy chưa đánh giá lại bản này. Mục ưu tiên 2 về kiểm tra thuộc tính bổ sung được nhóm quyết định không thực hiện. Một thuộc tính Ragdoll có vấn đề vẫn nằm trong JSON đã dùng để tính các kết quả cũ; sửa dữ liệu ấy về sau sẽ đòi hỏi chạy lại các phép đánh giá liên quan.

## 2. Đối chiếu tiêu chí của phiếu Lần 3

| Tiêu chí | Bằng chứng hiện có trong v5 | Đánh giá công tâm |
|---|---|---|
| **TC1 – Tính mới và hàm lượng khoa học** | Abstract, Introduction và Conclusion tập trung vào 2.048 tham số **được huấn luyện** của DualStream, điểm khái niệm có thể xem xét, và few-shot ba seed. Bài nêu rõ khác biệt với CoOp trên Oxford chưa đạt ý nghĩa thống kê (`p=0,0696`). | **Đã phản hồi đúng hướng; chưa thể tự kết luận “Đạt” thay thầy.** Tính giải thích hiện có bằng chứng mô tả điểm, chưa có thử nghiệm faithfulness nhân quả. Kết quả ít mẫu mới có ba seed. |
| **TC2 – Loại đóng góp** | Họ cấu hình CBM/Centroid/CoOp/VisAdapter/DualStream, ablation Bảng 2–4; ngân hàng thuộc tính công khai. | **Phù hợp nhận xét Đạt**, với giới hạn: 2.048 là tham số prompt được tối ưu, không tính CLIP đóng băng, việc sinh thuộc tính offline và chi phí TTA. |
| **TC3 – Dữ liệu** | Oxford dùng official test 3.669 ảnh; Stanford Dogs dùng split 12.000/8.580, nội bộ tách 10.800/1.200 để chọn mô hình. | **Phù hợp nhận xét Đạt** trên cấu trúc và artifact đang có. Không gọi đây là chứng nhận độc lập rằng mọi bước trích đặc trưng đã được chạy lại trong lần rà soát này. |
| **TC4 – Phương pháp đánh giá** | Có baseline, few-shot mean ± std ba seed, McNemar ghép cặp, khảo sát $\alpha$, benchmark thứ hai và đo độ trễ. | **Phù hợp nhận xét Đạt.** Cần giữ chú thích các run khác nhau, nhất là raw/TTA và 9:1/Full. Các p của nhiều so sánh đang báo **chưa hiệu chỉnh**, đúng như paper v5 nói. |
| **TC5 – Phân loại bài** | Module prompt/adapter trên CLIP đóng băng, neo tri thức SLM; TTA là thành phần bổ trợ. | **Phù hợp Nhóm B/Ứng dụng AI** như phiếu nhận xét. |

## 3. Những điểm v5 đã sửa so với v4

| Vị trí | Vấn đề ở v4 | Cách xử lý trong v5 |
|---|---|---|
| Abstract | “Suppresses sample selection variance” và “human-auditable interpretability” có thể bị đọc như kết luận nhân quả rộng. | Ghi **mức biến thiên quan sát được tại K=1 trên ba seed**, phân biệt điểm khái niệm có thể xem xét với faithfulness chưa được chứng minh. |
| Bảng 8 | Tiêu đề cột Oxford ghi chỉ TTA nhưng ô số có cả raw và TTA. | Đổi thành **Raw (5-View TTA)**, Stanford Dogs ghi **Single-View**. |
| Bảng 8, CuPL-style (Gemma proxy) | Ô Oxford lặp 87,22% (87,63% TTA) của CuPL tác giả. | Đặt `—` ở Oxford vì chưa có kết quả riêng của proxy này trên Oxford; Oxford author CuPL vẫn ở hàng riêng. |
| Bảng 8, CuPL tác giả ánh xạ sang Dogs | Có JSON báo 62,83% nhưng chưa thấy mã/ánh xạ prompt theo từng giống để tái lập. | Loại số khỏi bảng so sánh chính; ghi chú đây là **kết quả thăm dò trong JSON**, chưa đủ provenance. Giữ CuPL tác giả ở Oxford; Dogs 57,00% là proxy Gemma riêng. |
| Bảng 1, 5, 6 và Appendix B | Paper coi “Ragdoll không có găng trắng” là dấu hiệu phân biệt chung, dù chuẩn CFA cho phép Ragdoll dạng mitted có chân trắng. | Ghi rõ đây là **lỗi khái quát trong descriptor đã lưu**; không đổi số điểm khái niệm đã tính, không nói điểm cao chứng minh dấu hiệu đúng. Appendix B được nhãn là ví dụ câu chữ đã hiệu chỉnh, **chưa được dùng để chạy lại**. |

## 4. Những việc còn cần làm nếu nhóm muốn hoàn thiện sâu hơn

| Ưu tiên | Vấn đề chưa giải quyết | Hành động có thể kiểm chứng |
|---|---|---|
| **Cao** | Descriptor `ragdoll.distinguishing_mark` trong `Semester/attributes/oxford_pets_attributes.json` vẫn ghi “Absence of white gloves”. Bản v5 chỉ **cảnh báo**, không âm thầm sửa dữ liệu rồi giữ số cũ. | Nếu nhóm quyết định sửa descriptor, lưu phiên bản thuộc tính mới; tạo lại text embeddings, chạy lại các cấu hình bị ảnh hưởng và cập nhật Bảng 1, 2, 5, 6 cùng kết quả liên quan. Không thay số nếu chưa chạy. [Chuẩn CFA Ragdoll](https://cfa.org/wp-content/uploads/2024/03/ragdoll-standard.pdf) có mitted pattern với chân trắng; [chuẩn Birman](https://cfa.org/wp-content/uploads/2024/03/birman-standard.pdf) mô tả glove/lace cụ thể. |
| **Cao** | `verified_cupl_official_stanford_dogs.json` báo 62,83% (5.391/8.580), nhưng chưa có mã ánh xạ synset → 120 lớp, prompt list/hash và mã suy luận tạo ra số đó. Kiểm tra v4 trước đây gán cứng con số vào script. | Nếu muốn đưa lại hàng CuPL ImageNet-mapped vào bảng chính: nộp prompt nguồn, quy tắc ánh xạ và mã chạy; kiểm tra class order, số prompt/lớp, tiền xử lý và dự đoán trên 8.580 ảnh. Khi đó gọi rõ **đánh giá tại chỗ bằng prompt ImageNet tác giả được ánh xạ**, không gọi là kết quả Stanford Dogs do tác giả công bố. [Repo CuPL](https://github.com/sarahpratt/CuPL) công bố công cụ cho ImageNet. |
| **Trung bình** | Phiếu thầy yêu cầu khoảng 10% thuộc tính; nhóm đã thống nhất **không làm thêm**. JSON audit 19 mục là kiểm tra nội bộ theo tài liệu, không có chứng thực chuyên gia. Lỗi Ragdoll nằm ngoài khả năng được con số 18/19 bảo đảm. | Giữ phát biểu và giới hạn như v5; nếu bị hỏi, trình bày trung thực phần chưa làm. Không tạo thêm số liệu hay tự nhận đã qua thẩm định chuyên gia. |
| **Trung bình** | Bảng 4: $\alpha=0,2$ có 335/370 validation, thấp hơn $\alpha=0,4$ và $0,6$ (337/370). V5 đã ghi đây là cấu hình bảo thủ, không phải điểm cao nhất validation. | Nếu nhóm muốn tuyên bố “chọn tối ưu bằng validation”, cần có quy tắc chọn được định trước hoặc một lần chọn lại và chạy test đúng giao thức; không dùng test để biện minh ngược cho $\alpha=0,2$. |
| **Trung bình** | Bảng 7 ghi rõ CLIP handcrafted 87,49% là raw và các run ít mẫu khác được tổng hợp riêng. Độ lệch chuẩn ba seed chưa đủ để suy luận ưu thế ổn định phổ quát. | Giữ nhận xét ở mức **quan sát tại các seed/K đã thử**; báo chính xác CoOp nhỉnh hơn DualStream tại $K=8,16$. |
| **Trước khi gửi thầy** | Bản paper Word/PDF hiện có được tạo trước v5. | Xuất lại Word/PDF từ v5, rà soát bằng mắt Bảng 1, 2, 5, 7, 8, chú thích và ký hiệu toán; không gửi PDF cũ với tên bản mới. |

## 5. Giới hạn của lần rà soát này

Tôi đã đọc paper v4, phiếu Lần 3, các mã benchmark và JSON trong repo; kiểm tra phép chia `correct/8.580` của Bảng 8 và sự khác nhau giữa source prompt CuPL Oxford với proxy Dogs. Tôi **không** chạy lại Gemma, CLIP, quá trình huấn luyện, kiểm tra ảnh đầu vào hoặc phép thẩm định thuộc tính mới. Các số giữ nguyên trong v5 là số đã có ở v4/JSON; file đánh giá này không tuyên bố chúng được tái lập độc lập.

**Không có tuyên bố SOTA trong v5.** Kết quả trên hai dataset thuộc các giao thức và những lần train khác nhau; paper không gộp chúng thành một checkpoint hay một phép so sánh hơn kém tuyệt đối.
