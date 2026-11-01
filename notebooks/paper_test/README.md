# Notebook kiểm chứng paper

Đặt mọi notebook mới dùng để kiểm chứng số liệu, chạy đối chứng hoặc thử phương án sửa paper trong thư mục này (`notebooks/paper_test/`).

Mỗi notebook nên ghi rõ: câu hỏi cần kiểm, phiên bản paper và mã nguồn, dataset/split, seed, mô hình và checkpoint, nguồn descriptor/prompt, cách tạo feature/TTA, quy tắc chọn siêu tham số trên validation, và tệp kết quả đầu ra. Lưu số ảnh đúng cùng mẫu số và prediction theo ảnh khi báo accuracy hoặc kiểm định ghép cặp. Phân biệt các run khác giao thức; không dùng kết quả test để chọn mô hình rồi gọi đó là đánh giá cuối.

**Không cần gộp các mô hình vào một notebook hay một lần train.** Có thể chạy riêng từng mô hình do giới hạn hiệu năng. Khi so kết quả, ghi rõ test split/ID, backbone, prompt bank, view, mức giám sát và cách chọn model. McNemar cần prediction căn theo đúng cùng ảnh test; hai model vẫn có thể được train riêng.

Notebook hiện có:

- `audit_paper_results.ipynb`: chỉ đọc JSON/prediction để tính lại các kết luận chính; đã chạy kiểm được trong workspace.
- `stanford_dualstream_recheck.ipynb`: chạy lại CoOp/Gemma-CoOp/DualStream trên Stanford Dogs custom cached split, sửa bước chọn checkpoint; **chưa chạy toàn bộ huấn luyện**, đầu ra riêng trong `notebooks/paper_test/results/`.
- `cupl_official_prompts_oxford.ipynb`: đánh giá lại Oxford bằng prompt CuPL Pets do tác giả công bố, **không dùng Gemma descriptors**. Cần tải prompt từ GitHub và chạy CLIP; notebook mới được kiểm cú pháp, chưa có kết quả inference. Đây là run độc lập. Với Stanford Dogs, chưa có prompt tác giả tương ứng trong nguồn đã kiểm nên không tự gán số CuPL chính thức.

Các notebook đã có ở thư mục khác không cần di chuyển chỉ để đáp ứng quy ước này. Khi sửa hoặc bổ sung **thí nghiệm cho paper từ thời điểm này**, tạo notebook tương ứng ở đây và trỏ tới notebook đó trong phần mô tả kết quả.
