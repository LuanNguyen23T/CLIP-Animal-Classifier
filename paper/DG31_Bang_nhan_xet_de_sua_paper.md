# Bảng nhận xét để sửa bản thảo Gemma-CLIP

**Người nhận:** Luân và công cụ AI hỗ trợ biên tập  
**Bản được xem xét:** `paper_CLIP_Gemma4_revised_v2.md`  
**Mục tiêu:** Sửa bài báo và kiểm chứng kết quả trước khi chuyển sang báo cáo Word.

> Đây là bản nhận xét độc lập. Những mục ghi **chưa kiểm chứng được** có nghĩa là chưa tìm thấy đủ dữ liệu đầu ra hoặc mã tái tính trong các tài liệu được cung cấp; không khẳng định thí nghiệm chưa từng được chạy. Không được tự tạo số liệu, log, dự đoán hay kết quả kiểm định để lấp chỗ trống.

## Kết luận chung

Bản thảo đã bổ sung phương pháp, bảng ablation, phản hồi DG31 và một thí nghiệm trên Stanford Dogs. Tuy nhiên, hiện **chưa nên tuyên bố đã giải quyết hoàn toàn mọi ý kiến phản biện**. Có lỗi số học/dữ liệu rõ ràng ở bảng nhầm lẫn Oxford Pets, rò rỉ thông tin test khi chọn epoch Stanford Dogs, gọi sai cấu hình mô hình, và so sánh CuPL/LaBo không có nguồn phù hợp. Các kết quả Oxford quan trọng còn thiếu dấu vết đủ để người khác tái tính.

## Bảng nhận xét và yêu cầu sửa

| Ưu tiên | Vị trí trong bài | Nhận xét có căn cứ | Yêu cầu sửa cụ thể | Chỉ xem là hoàn thành khi |
|---|---|---|---|---|
| **P0** | Mục 5.5, Table 6 | Bảng ghi `18/373` ảnh Staffordshire Bull Terrier, `12/370` Birman, `11/386` Beagle, v.v. Nhưng cache nhãn test Oxford Pets có **3.669 ảnh/37 lớp**; số ảnh test thực tế lần lượt là **89, 100, 100** cho ba lớp này. Không thể dùng các mẫu số 373/370/386 như số ảnh test của từng giống. | Tạo lại ma trận nhầm lẫn từ `ground truth` và dự đoán của đúng checkpoint/cấu hình đang báo cáo. Ghi số ca lỗi, mẫu số theo từng lớp và tỷ lệ tính lại. Nếu không có dự đoán từng ảnh, bỏ Table 6 và những câu suy luận từ bảng. | Mọi hàng của Table 6 khớp với nhãn và dự đoán có thể kiểm tra lại; tổng số ảnh và số ca lỗi hợp lệ. |
| **P0** | Mục 5.7, Table 8; thí nghiệm Stanford Dogs | Script đánh giá test mỗi 5 epoch rồi lấy **độ chính xác test cao nhất** làm kết quả `72,64%`. Như vậy tập test tham gia chọn checkpoint. | Tách validation từ 14.406 ảnh train, chọn epoch và siêu tham số trên validation; chỉ dùng 6.174 ảnh test một lần sau khi cố định mô hình. Báo cáo lại Top-1 và seed/split. | Có mã, seed, split, log validation, checkpoint được chọn và phép đánh giá test cuối cùng. |
| **P0** | Mục 5.7, Table 8, Abstract/Conclusion | Kết quả Stanford Dogs `72,64%` đến từ **visual residual adapter với centroid làm text weights**. Script không huấn luyện Gemma-CoOp và không tính dual-stream fusion cho cấu hình này. Tên “Gemma-CLIP Full” khiến người đọc hiểu sai mô hình đã chạy. | Hoặc triển khai và đánh giá đúng cấu hình Full như phương pháp mô tả, hoặc đổi tên hàng thành `Gemma centroid + residual visual adapter`; sửa mọi câu kết luận tương ứng. | Tên cấu hình, sơ đồ phương pháp, mã thực thi và bảng kết quả mô tả cùng một hệ thống. |
| **P0** | Mục 5.7, Table 8 | Các số CuPL `68,30%` và LaBo `68,75%` trên Stanford Dogs được nhập **cố định** vào JSON/script, không do script chạy. Bài CuPL gốc báo Oxford Pets nhưng không liệt kê Stanford Dogs; bài LaBo gốc không đánh giá Oxford Pets hay Stanford Dogs. Số CuPL `88,90%` trên Oxford Pets trong bảng cũng trùng với baseline CLIP 80-prompt của chính bản thảo, không khớp bảng Oxford Pets trong CuPL gốc. | Xác định đúng nguồn, backbone, split, prompt, chế độ zero-shot/few-shot/full-data và TTA của từng số. Chỉ so sánh trực tiếp khi cùng giao thức; tốt nhất chạy lại baseline trên cùng split. Nếu không làm được, bỏ các số và câu “vượt CuPL/LaBo”. | Mỗi số CuPL/LaBo có nguồn bảng/trang chính xác hoặc mã chạy lại; không đặt zero-shot cạnh full-data rồi kết luận phương pháp vượt trội. |
| **P1** | Mục 5.3, McNemar | Công thức và phép tính từ các cặp `b,c` trong bài **đúng số học**: `(104,29) → χ²≈41,17`, `(92,31) → χ²≈29,27`, `(76,37) → χ²≈12,78`. Nhưng chưa có dự đoán từng ảnh của các mô hình để xác nhận các cặp bất đồng này. | Lưu prediction theo cùng thứ tự 3.669 ảnh cho từng mô hình; viết mã tạo `b,c`, χ², p-value và Holm correction. Nếu thiếu prediction, chuyển kết luận thống kê sang “chưa kiểm chứng” và bỏ p-value. | Người khác chạy script từ prediction và nhận đúng các cặp `b,c` cùng p-value được công bố. |
| **P1** | Mục 4.1, Table 7 | Bài tuyên bố mean ± std qua 3 seed cho mọi mức K, nhưng chưa kèm log từng seed, danh sách support sample theo seed và checkpoint tương ứng. Các giá trị trung bình/độ lệch chuẩn không thể đối soát từ một bảng tổng hợp. | Công bố kết quả riêng từng seed, cách lấy K mẫu/lớp, seed khởi tạo, validation và cách tính std; tạo bảng tổng hợp bằng mã. | Tái tính được từng ô mean ± std của Table 7 từ kết quả 3 lần chạy. |
| **P1** | Mục 5.3, Table 4 | Bài nêu khảo sát `α = 0; 0,1; 0,2; 0,4; 0,6` trên 370 ảnh validation, song chưa có log hoặc script sinh các số `92,43; 93,24; 93,78; 92,70; 90,81%`. | Bổ sung mã và log khảo sát; nêu rõ mỗi α có huấn luyện lại hay chỉ thay hệ số khi suy luận, chọn checkpoint theo tiêu chí nào, và thời điểm xem tập test. Nếu thiếu bằng chứng, bỏ số và câu “α tối ưu”. | Mọi số trong Table 4 được tái tính trên đúng validation split, không chọn α bằng test. |
| **P1** | Mục 3.1, 5.4; Table 1/5 | Bài tuyên bố kiểm tra chuyên gia độc lập `18/19`, Cohen’s `κ=0,88`, và đưa điểm kích hoạt khái niệm cho một ảnh Ragdoll. Chưa có phiếu chấm của hai người, tiêu chí chấm, mã tính κ hoặc ID ảnh và vector điểm để kiểm tra. | Đính kèm bảng 19 mục được lấy mẫu, đánh giá riêng của từng chuyên gia và mã tính κ; lưu ID ảnh cùng 185 concept scores của ca minh họa. Nếu không có, trình bày là ví dụ minh họa và bỏ tuyên bố audit định lượng. | Tính lại được 18/19, κ và toàn bộ số Table 5 từ dữ liệu gốc. |
| **P1** | Abstract, Mục 5.1, 5.6, 5.8 | Các kết quả chính Oxford Pets `90,85%`, `91,39%`, `94,15%`, latency/FPS và các bảng Top-5/Macro-F1 chưa đi kèm cùng một bộ mã, prediction, log và cấu hình có thể tái tạo. Có cache đặc trưng và notebook, nhưng cache chỉ chứng minh đã lưu đặc trưng; notebook vẽ biểu đồ không tự chạy toàn bộ benchmark. | Lập bảng nguồn gốc số liệu: mỗi metric → dataset/split → seed → cấu hình → checkpoint → prediction/log → script tổng hợp. Sửa hoặc hạ mức tuyên bố nào không truy được nguồn. | Mỗi số chính trong Abstract và Conclusion đối chiếu được với artifact thực nghiệm cụ thể. |
| **P2** | Tiêu đề Mục 5.7 và Conclusion | Stanford Dogs được **huấn luyện adapter mới trên chính train split Stanford Dogs**. Đây là đánh giá phương pháp trên dataset thứ hai, không phải kiểm tra khả năng chuyển trực tiếp checkpoint từ Oxford Pets sang Dogs. | Đổi “cross-dataset generalization” thành “evaluation on a second fine-grained dataset”, trừ khi có thí nghiệm chuyển checkpoint không huấn luyện lại. | Cách diễn đạt đúng với giao thức đã chạy. |
| **P2** | Mục 5.2 và Conclusion | Bài viết nói compact prompt đã “giải quyết attention gradient dilution”, nhưng chỉ đưa chênh lệch accuracy rất nhỏ; chưa đo gradient norm, độ dài token thực tế trên toàn bộ lớp hoặc kiểm định seed cho ablation này. | Đổi thành giả thuyết/cách diễn giải phù hợp với kết quả quan sát, hoặc bổ sung đo đạc trực tiếp gradient và độ ổn định nhiều seed. | Kết luận không mạnh hơn bằng chứng. |

## Những điểm cần sửa ngay trong Table 8 Stanford Dogs

Kết quả JSON phân biệt **ba** phép đo sau:

| Cấu hình thực tế | Top-1 ghi trong JSON |
|---|---:|
| CBM 5 trục | 57,14% |
| Centroid Bank đứng riêng | **58,44%** |
| Handcrafted prompt + `0,35 ×` Centroid Bank | **59,62%** |

Do đó **59,62% không phải accuracy của Centroid Bank đứng riêng**. Bài và bảng phản hồi hiện gọi lẫn hai cấu hình. Phải sửa tên hàng, mô tả công thức và số liệu nhất quán.

## Thứ tự làm việc đề nghị

1. Tái tạo prediction và sửa Table 6; nếu không có dữ liệu, bỏ bảng.
2. Sửa giao thức Stanford Dogs: validation để chọn checkpoint, chạy test cuối cùng, đổi đúng tên cấu hình.
3. Xác minh hoặc bỏ toàn bộ so sánh CuPL/LaBo chưa cùng giao thức.
4. Thu thập chứng cứ cho Table 4, Table 7, McNemar, audit chuyên gia và các số chính Oxford Pets.
5. Viết lại Abstract, Results, Conclusion và bảng phản hồi DG31 dựa trên những kết quả còn đứng vững.
6. **Sau cùng** mới dàn trang báo cáo Word.

## Yêu cầu dành cho AI hỗ trợ sửa bài

Hãy xử lý bảng nhận xét này như một phản biện khoa học khó tính. Với từng hàng:

1. Phân loại `đã xác thực`, `cần chạy lại`, hoặc `không có chứng cứ`.
2. Chỉ dùng dữ liệu, mã và nguồn thực sự được cung cấp; **không tự sinh số đẹp, prediction, p-value, log hay kết quả chuyên gia**.
3. Nếu có chứng cứ, tái tính số và ghi rõ công thức/giao thức. Nếu không có, viết lại câu theo mức độ chắc chắn thấp hơn hoặc đề xuất bỏ.
4. Giữ riêng các cấu hình zero-shot, few-shot và full-data; không dùng chúng để tuyên bố vượt trội khi giao thức không giống nhau.
5. Trả lại: (a) bảng lỗi đã xử lý, (b) bản thảo Markdown đã sửa, (c) danh sách thí nghiệm/dữ liệu còn thiếu. Chỉ đánh dấu “hoàn thành” khi số liệu tái tạo được.

Đừng trả lời chung chung rằng bài “đã tốt” hoặc “đã giải quyết hết”. Hãy chỉ ra chính xác phần nào chưa thể bảo vệ trước phản biện và sửa phần có bằng chứng trước.
