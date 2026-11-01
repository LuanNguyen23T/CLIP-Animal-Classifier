# Đánh giá sau khi chạy notebook và push kết quả ngày 01/11/2026

**Đề tài:** DG31 — Gemma-CLIP.  
**Bản kiểm:** paper `paper/paper_CLIP_Gemma4_revised_v2.md` và commit kết quả `d001f15`.  
**Phạm vi:** Đọc source notebook, JSON, prompt bank, prediction đã push; tính lại count, split và kiểm định từ tensor. Không huấn luyện hoặc chạy CLIP lại trong lượt đánh giá này. Các đường dẫn trong tài liệu là đường dẫn tương đối theo repository.

## 1. Kết luận công tâm

**Có tiến bộ thực nghiệm thật:** lần này nhóm đã lưu notebook chạy, output, prompt CuPL Oxford do tác giả công bố, JSON và prediction theo từng ảnh. Hai notebook được chạy với source code giống bản trước commit; không phải chỉ điền số vào bảng. Kết quả Stanford sau sửa checkpoint và CuPL Oxford có thể kiểm lại từ tensor.

**Paper chưa phản ánh kết quả mới:** paper vẫn ghi Stanford DualStream **59,62%**, CoOp **57,55%**, β=0,7, CuPL Oxford **86,02% TTA**. Artifact mới lần lượt cho **59,57%**, **57,64%**, **β=1,0** và **CuPL Oxford 87,63% TTA**. Không tự thay số paper bằng một run khác khi chưa chốt run chính; cần ghi rõ run ID/protocol và chọn bộ số nhất quán.

Các mô hình **được phép train/chạy ở những lần khác nhau** vì giới hạn hiệu năng. Không cần gộp mọi mô hình vào một script. So sánh độ chính xác cần nêu cùng dataset/test IDs, backbone, tiền xử lý/view và mức giám sát hoặc ghi rõ chỗ khác giao thức. McNemar cần hai vector prediction ghép đúng từng ảnh; ngày train hay notebook train có thể khác. Tôi không xếp mô hình vào SOTA.

## 2. Đối soát artifact mới

| Thí nghiệm | Số được push | Kiểm lại độc lập | Nhận xét |
|---|---|---|---|
| Stanford CoOp baseline | 3.559/6.174 | **57,64496%**; JSON và prediction khớp | Checkpoint epoch 8; validation 799/1.382. |
| Stanford Gemma-CoOp | 3.564/6.174 | **57,72595%**; JSON và prediction khớp | Checkpoint epoch 11; validation 792/1.382. |
| Stanford Gemma-DualStream | 3.678/6.174 | **59,57240%**; JSON và prediction khớp | β=**1,0** được chọn bởi 812/1.382 validation = 58,76%; paper còn ghi run cũ 59,62%, β=0,7. |
| CuPL Oxford, prompt tác giả, raw | 3.200/3.669 | **87,21723%**; JSON và prediction khớp | 37 lớp, mỗi lớp 20 prompt. |
| CuPL Oxford, prompt tác giả, TTA cache | 3.215/3.669 | **87,62606%**; JSON và prediction khớp | Khác **86,02%** của Table 8, vốn là script điền Gemma descriptor vào template tự viết. |

Tensor Stanford lưu train indices 13.024, validation indices 1.382, hai tập không giao nhau và phủ đủ 14.406 ảnh cache; test labels và class order khớp cache. Tensor CuPL Oxford cũng có test labels/class order khớp cache. Source code của hai notebook so với commit trước **không đổi**; commit mới thêm output và artifact. Đây là bằng chứng tốt cho *hai run cụ thể*, chưa chứng minh toàn bộ các bảng Oxford/Stanford trong paper.

Với prediction Stanford mới, so DualStream và CoOp trên cùng 6.174 hàng cache cho **b=534, c=415**, chênh 119 ảnh = **+1,93 điểm phần trăm**. McNemar có hiệu chỉnh liên tục: χ²≈14,67, p≈0,000128. Đây là kiểm định **bổ sung do tôi tính** từ tensor mới, chưa được paper báo và không thay phép McNemar Oxford mà phiếu thầy yêu cầu. Cần manifest ID ảnh nếu muốn chứng minh alignment từ ảnh gốc ngoài cache.

### Prompt CuPL và hash

`notebooks/paper_test/external/pets_prompts_full.json` có 37 lớp × 20 prompt, nguồn [CuPL của tác giả](https://github.com/sarahpratt/CuPL/tree/main/all_prompts/full_prompts). Notebook không dùng `Semester/attributes` để tạo prompt CuPL. Hash SHA-256 ghi trong JSON **khớp nội dung file Git với xuống dòng LF**. Sau checkout trên Windows, file làm việc đổi sang CRLF nên hash byte trực tiếp khác; khi chuẩn hóa CRLF→LF lại khớp. Đây là vấn đề chuẩn hóa xuống dòng, không phải bằng chứng prompt bị thay nội dung. Nên lưu thêm Git blob/hash chuẩn hóa hoặc thiết lập `.gitattributes` cho file JSON để việc kiểm hash không gây hiểu nhầm.

## 3. Bảng nhận xét cần sửa sau run mới

| Mã | Mức | Vị trí | Đánh giá | Việc cần làm |
|---|---|---|---|---|
| **S01** | **P0** | Table 8; Abstract; §5.7 | Paper ghi Stanford DualStream 59,62%/CoOp 57,55%/β=0,7; notebook sau sửa checkpoint ra **59,57%/57,64%/β=1,0**. Hai bộ số là **hai run khác nhau**, không phải lỗi số học. Nhưng nếu gọi run mới là đã xác minh thì paper phải theo đúng run mới. | Chốt một run chính. Nếu chọn run mới, cập nhật đồng bộ count, accuracy, β, validation, epoch và chú thích. Nếu giữ run cũ, trỏ rõ JSON/script cũ và giải thích checkpoint-selection bug đã tìm thấy; không trộn hai run. |
| **S02** | **P0** | Table 8 hàng CuPL Oxford; §5.7 | Run mới dùng prompt CuPL tác giả cho **87,22% raw / 87,63% TTA**, có prediction và prompt bank. Paper còn ghi **85,88% / 86,02%** từ script dùng Gemma descriptors. Run mới đúng bản chất CuPL hơn; **không** được tự coi hai run là một. | Gắn hàng CuPL Oxford với run mới và ghi “CuPL full prompts released by authors, locally re-evaluated with CLIP ViT-B/32 and project TTA cache”; cập nhật số nếu chốt run này. Không gọi 86,02% là CuPL nguyên bản. |
| **S03** | **P0** | Table 8 CuPL Stanford và LaBo hai dataset | Commit mới **chưa chạy CuPL Stanford với prompt tác giả**; số 56,85% vẫn từ Gemma descriptors. Hàng LaBo vẫn là trung bình 5 concept Gemma, chưa có bước chọn concept và classifier của [LaBo gốc](https://github.com/YueYANG1996/LaBo). Phiếu thầy yêu cầu so CuPL/LaBo nên **giữ mục so sánh trong kế hoạch**, không xóa vì thiếu run. | Chạy hai đối chứng theo phương pháp được ghi tên, có bank/config/prediction riêng; hoặc tạm ghi chính xác “CuPL-style/LaBo-style dùng Gemma descriptors”, không tuyên bố vượt CuPL/LaBo gốc cho những ô chưa có run. Có thể chạy từng baseline riêng. |
| **S04** | **P0** | §5.7, caption Table 8 | Stanford 14.406/6.174 là **custom cached 70:30 split**, không phải official 12.000/8.580 theo [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/stanford_dogs). Notebook mới vẫn dùng custom split. | Sửa “official split” thành “custom cached split” cho mọi hàng Dogs; nếu muốn số official, cần chạy lại trên 12.000/8.580. |
| **S05** | **P0** | §5.6 Table 7, Abstract/Conclusion | Kết quả Table 7 không đổi: hàng “Gemma-CLIP Full” **thấp hơn CoOp ở K=1,2,4,8,16 và Full** (−0,78 đến −1,35 điểm). Notebook mới Stanford/CuPL không khắc phục luận điểm Oxford few-shot. | Viết lại kết luận “comprehensive gains across all supervision regimes”; nêu cả điểm yếu so CoOp và tách run Table 2 khỏi Table 7. |
| **S06** | **P0** | §5.3 McNemar Oxford | Không có prediction mới cho Oxford 92,15%. Paper vẫn ghép b/c=(100,88) của run DualStream **91,71%** với accuracy **92,15%** từ run khác. Cặp ResNet (352,161) nằm ở tệp prediction khác với tệp paper dẫn. Phiếu DG31 còn yêu cầu Full vs CoOp/CLIP-Adapter; paper hiện so DualStream vs CoOp/ResNet. | Gắn từng phép kiểm với đúng vector/count/run, bổ sung CLIP-Adapter nếu giữ yêu cầu thầy. Run được train riêng vẫn dùng McNemar được nếu cùng test IDs. |
| **S07** | **P1** | §3.3.2; Table 3 | Notebook Stanford đo token không padding của Gemma-CoOp **min 27, median 37, max 52**, nên không hỗ trợ phát biểu prompt “24 token” cho run này. Đây là số đo mới rõ hơn nhận xét trước. | Nêu token distribution thực theo dataset/run hoặc đưa pipeline rút gọn 24 token thật rồi chạy lại. Không quy gain cho “gradient dilution” nếu chưa có ablation tách độ dài khỏi nội dung. |
| **S08** | **P1** | §5.3 Table 4; Table 1/5/6; Table 9 | JSON α vẫn không chứa TTA và raw test khác Table 4; descriptor Birman/Ragdoll còn khác bank; Table 6 Beagle ghi “most confusable Basset” dù run lưu cho 0 lỗi sang Basset; định nghĩa zoom 1,15× và benchmark 17,2 FPS chưa được artifact mới kiểm. | Xem chi tiết và tiêu chí hoàn thành trong bản đánh giá DG31 ngày 01/11. Các notebook mới không tự giải quyết các mục này. |

## 4. Kết luận hành động

Ưu tiên cập nhật **Table 8 theo run đã chốt**, ghi đúng custom split, và phân biệt CuPL Oxford vừa chạy với các ô CuPL/LaBo còn là proxy. Giữ các notebook độc lập trong `notebooks/paper_test`; lưu prompt/concept bank, config, test count và prediction theo từng run. Sau đó sửa Abstract/Conclusion theo Table 7 hiện hành và đối soát các bảng Oxford còn thiếu artifact. Không cần gộp tất cả model vào một code hoặc một lần train.

**Giới hạn:** Tôi chỉ tính lại từ artifact và source đã push, chưa chạy lại huấn luyện/CLIP inference. Không có manifest ảnh gốc cho cache Stanford; không xác nhận tuyên bố tuyệt đối “zero test leakage” trên toàn bộ quá trình nghiên cứu. Không kết luận các số chưa truy được là bịa.
