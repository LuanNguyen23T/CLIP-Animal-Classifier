# Notebook kiểm chứng paper

Đặt mọi notebook mới dùng để kiểm chứng số liệu, chạy đối chứng hoặc thử phương án sửa paper trong thư mục này (`notebooks/paper_test/`).

Mỗi notebook nên ghi rõ: câu hỏi cần kiểm, phiên bản paper và mã nguồn, dataset/split, seed, mô hình và checkpoint, nguồn descriptor/prompt, cách tạo feature/TTA, quy tắc chọn siêu tham số trên validation, và tệp kết quả đầu ra. Lưu số ảnh đúng cùng mẫu số và prediction theo ảnh khi báo accuracy hoặc kiểm định ghép cặp. Phân biệt các run khác giao thức; không dùng kết quả test để chọn mô hình rồi gọi đó là đánh giá cuối.

**Không cần gộp các mô hình vào một notebook hay một lần train.** Có thể chạy riêng từng mô hình do giới hạn hiệu năng. Khi so kết quả, ghi rõ test split/ID, backbone, prompt bank, view, mức giám sát và cách chọn model. McNemar cần prediction căn theo đúng cùng ảnh test; hai model vẫn có thể được train riêng.

Notebook hiện có:

- `audit_paper_results.ipynb`: chỉ đọc JSON/prediction để tính lại các kết luận chính; đã chạy kiểm được trong workspace.
- `stanford_dualstream_recheck.ipynb`: đã chạy trên custom cached split Stanford; kết quả mới lưu tại `results/stanford_dualstream_recheck.json` và prediction `.pt` (DualStream 59,57%, β=1,0). Xem đánh giá sau run trước khi thay số trong paper.
- `cupl_official_prompts_oxford.ipynb`: đã chạy với 20 prompt/lớp do tác giả CuPL công bố; kết quả Oxford 87,63% TTA lưu tại `results/cupl_official_prompts_oxford.json` và prediction `.pt`. Không dùng Gemma descriptors. Stanford Dogs CuPL và LaBo gốc vẫn chưa có run tương ứng.

Các notebook đã có ở thư mục khác không cần di chuyển chỉ để đáp ứng quy ước này. Khi sửa hoặc bổ sung **thí nghiệm cho paper từ thời điểm này**, tạo notebook tương ứng ở đây và trỏ tới notebook đó trong phần mô tả kết quả.

## Bộ kiểm tra theo từng bảng và cách chạy trên Colab

Các notebook `verification/verify_table_01.ipynb` đến `verification/verify_table_09.ipynb` kiểm từng bảng; verification/verify_all_paper_claims.ipynb tổng hợp và kiểm tra tính nhất quán giữa bảng, prediction và run. verification/verify_paper.py là mã kiểm tra dùng chung. Trạng thái:

- `match`: số khớp đúng artifact được nêu, chỉ xác nhận phép tính hoặc nội dung artifact đó.
- `conflict`: paper và artifact không khớp.
- `protocol_mismatch`: hai số thuộc run hoặc phương pháp khác nhau; cần gắn nhãn đúng.
- `missing_evidence`: thiếu đầu vào để tái tính.
- `manual_review`: cần xem ảnh, descriptor hay tài liệu gốc bằng người.

**Chạy trên Colab:** trong VS Code, mở `00_colab_runtime_check.ipynb`, chọn **Select Kernel → Colab → Auto Connect** rồi chọn runtime có GPU nếu cần huấn luyện/suy luận. Kernel Colab chạy trên máy chủ Google và không tự thấy file trên ổ Windows. Sau khi kết nối, dùng **Upload to Colab** cho thư mục repository hoặc đặt repository ở `/content/CLIP-Animal-Classifier` trên runtime; chạy notebook kiểm tra môi trường trước. Chỉ chạy các notebook khác khi tất cả đường dẫn kiểm tra đều hiện `OK`. Tài liệu của extension: https://github.com/googlecolab/colab-vscode/wiki/User-Guide.

Các notebook mới hiện chỉ **đối soát artifact đã lưu**, không tự huấn luyện lại CLIP và không biến JSON thành bằng chứng của chính nó. Chúng cũng không tự khẳng định một run đúng chỉ vì bảng số khớp. Để xác nhận một thí nghiệm đầy đủ, lưu thêm: commit mã, dataset và test image IDs, split train/val/test, seed, checkpoint, prompt/descriptor bank, cấu hình TTA, prediction theo image ID, logits nếu tính Top-5/F1, và thời gian đo từng lượt nếu báo latency. Có thể train các model trong notebook/run khác nhau; khi làm McNemar phải ghép prediction trên đúng cùng test IDs.

**Thứ tự ưu tiên chạy lại trên GPU:** Table 2 (các hàng Oxford chưa có prediction của đúng run), Table 3 (prompt/token và prediction), Table 4 (alpha TTA và validation tie break), Table 5 (XAI scores theo ảnh), Table 7 (per-seed image IDs/checkpoint), Table 8 (CuPL Stanford và LaBo đúng phương pháp), Table 9 (latency thực đo). Table 1 cần xác minh thủ công với registry; Table 6 cần ảnh và ID để kiểm giải thích lỗi. Không dùng nhãn SOTA hoặc kết luận hơn toàn diện khi bằng chứng còn mâu thuẫn.


## Mac MPS workflow and folder layout

- 00_colab_runtime_check.ipynb checks the runtime, device, and repository files. It works on Mac even though its historical name mentions Colab.
- 01_mps_clip_smoke.ipynb tests a small MPS forward/backward pass and the CLIP text encoder. It does not train a model.
- verification/ contains the shared verifier, full audit notebook, and one notebook per paper table.
- stanford_dualstream_recheck.ipynb and cupl_official_prompts_oxford.ipynb are independent experiments. Both prefer MPS, then CUDA, then CPU.
- results/ holds per-run JSON and predictions; external/ holds the CuPL prompt bank.
- audit_paper_results.ipynb is the earlier audit. Its path remains stable for older review documents.

On Mac, run 00 and 01 before the two experiment notebooks. The verification notebooks only inspect saved artifacts and do not require a GPU. Save execution outputs and record the device, package versions, run ID, split, seed, and checkpoint. Do not update the paper from a notebook that merely prints a hardcoded table.


### Existing training notebook caveat

The notebook notebooks/training/run_full_table7_benchmark.ipynb contains a hardcoded older Table 7 (including 94.15 for one Full row), which is not the current paper's Table 7 and is not an independent rerun. The notebook notebooks/training/few_shot_multiseed_scaling_benchmark.ipynb also prints the older table and a SimpleAdapter result. Do not use either printed table as proof for the current three-seed row without run IDs, checkpoints, and per-image predictions. Table 2's DualStream and Table 7's Gemma-CLIP Full must be identified as separate configurations until the original run manifest resolves the latter.
