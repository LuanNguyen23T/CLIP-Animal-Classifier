# DG31 – Kế hoạch xác minh toàn bộ thông số paper (01/11/2026)

**Bản đối chiếu:** paper/paper_CLIP_Gemma4_revised_v2.md và artifact hiện có trong repository.  
**Tình trạng thực thi:** Các notebook kiểm tra mới được chuẩn bị cho Colab, **chưa chạy trên Colab**. Không có thí nghiệm CLIP/Gemma mới trong lần cập nhật này. Các nhận xét dưới đây dựa trên việc đọc paper, notebook và artifact đã push trước đó; số liệu chưa có artifact được ghi là chưa xác nhận.

## Nguyên tắc xác nhận

Một số khớp JSON chỉ xác nhận phép tính trên JSON đó. Để xác nhận accuracy của một run cần prediction và label theo cùng image ID, checkpoint, model config, split, seed, prompt/descriptor bank và TTA. Các model được train riêng ở các thời điểm khác nhau hoàn toàn có thể so sánh; phải mô tả đúng giao thức và căn test ID khi dùng McNemar. Không dùng số test để chọn hyperparameter. Không xếp mô hình là SOTA.

## Bảng công việc theo từng phần

| Phần | Bằng chứng hiện có | Nhận xét thực chất | Notebook kiểm tra | Điều kiện hoàn thành |
|---|---|---|---|---|
| Table 1 – thuộc tính giống | Descriptor bank và mô tả trong paper | Nhận định Birman/Ragdoll về “không có găng trắng” cần đối chiếu registry; con số spot check 18/19 thiếu danh sách item và phiếu chấm. | notebooks/paper_test/verification/verify_table_01.ipynb | Lưu ID từng descriptor được chấm, bản gốc, người chấm/rubric và quyết định; sửa mô tả không đúng. |
| Table 2 – Oxford benchmark | Semester/results/verified_oxford_pets_benchmark.json, hai file prediction Oxford, notebook train | Nhiều số TTA trong paper khác artifact này. Điều đó có thể do run khác, nhưng mỗi hàng phải trỏ tới đúng run; DualStream 92,15% chưa có prediction tương ứng. | notebooks/paper_test/verification/verify_table_02.ipynb | Với từng hàng, lưu run ID, config, checkpoint, test IDs, prediction/logits và tính lại Raw/TTA/Top-5/F1. |
| Table 3 – prompt length | Các notebook CoOp/Gemma-CoOp | Chưa có log tokenizer và prediction đúng từng hàng. Số 24 token không được suy ra cho Stanford: notebook Stanford đã ghi prompt dài 27–52 token. | notebooks/paper_test/verification/verify_table_03.ipynb | Xuất prompt/token length theo lớp và prediction của từng biến thể; ghi rõ dataset/run. |
| Table 4 – alpha | Semester/results/table5_alpha_verification.json, verified_alpha_ablation.json | Validation 0,2 và 0,4 cùng 92,43%; raw test của một số alpha khác paper, artifact đầu tiên không có TTA. | notebooks/paper_test/verification/verify_table_04.ipynb | Chốt run chính, tính lại TTA trên đúng checkpoint, quy định tie break chỉ từ validation. |
| Table 5 – XAI | Bảng score trong paper | Thiếu image ID, vector concept, code gộp trục và quy tắc chuẩn hoá; chưa thể xác minh các số 0,231, 0,245… | notebooks/paper_test/verification/verify_table_05.ipynb | Lưu ảnh/case ID, score vector hai giống và code tái tạo bảng. |
| Table 6 – lỗi nhầm | Semester/results/verified_confusable_pairs.json | Các count chính khớp file tổng hợp; Beagle có 0 lỗi sang Basset nên gọi Basset là “most confusable” là sai cho run này. Giải thích nguyên nhân từ ảnh cần người xem. | notebooks/paper_test/verification/verify_table_06.ipynb | Lưu prediction theo image ID, confusion matrix và case ảnh cho các diễn giải định tính. |
| Table 7 – few-shot | Semester/results/table7_full_benchmark_results.json | Có ba số run để tính mean/std, nhưng thiếu image IDs/checkpoint theo seed. Gemma-CLIP Full thấp hơn CoOp ở K=1,2,4,8,16 và Full; kết luận tăng toàn diện không đúng. K=0 paper chưa gắn được với run phù hợp. | notebooks/paper_test/verification/verify_table_07.ipynb | Lưu sampling ID, seed, checkpoint, prediction từng run; sửa Abstract/Conclusion và K=0 theo bằng chứng. |
| Table 8 – Stanford, CuPL, LaBo | Stanford và CuPL Oxford có JSON/prediction run mới; proxy cũ nằm trong verified_cupl_labo_baselines.json | Stanford mới: DualStream 3.678/6.174 = 59,57%, beta=1,0; paper còn 59,62%/beta=0,7 của run cũ. CuPL Oxford dùng prompt tác giả: 3.215/3.669 = 87,63% TTA; paper còn 86,02% của proxy. CuPL Stanford và LaBo đúng phương pháp chưa có. Stanford dùng custom cached split, không được gọi official split. | notebooks/paper_test/verification/verify_table_08.ipynb; notebook CuPL/Stanford đã có | Chọn một run chính cho mỗi ô, ghi protocol riêng; chạy CuPL Stanford và LaBo đúng định nghĩa hoặc ghi rõ proxy. |
| Table 9 – latency/params | Bảng trong paper và vài tham số ở JSON | Phép đổi ms→FPS kiểm được về số học, nhưng thiếu log timing, cấu hình phần cứng và phép đo thật. Param count ResNet cosine ở paper và JSON không giống nhau. | notebooks/paper_test/verification/verify_table_09.ipynb | Chạy đo trên runtime được mô tả, lưu từng lượt, warm-up, sync, batch size, cache status; đếm tham số từ checkpoint. |
| McNemar Oxford | Hai file prediction và verified_mcnemar_stats.json | b/c = 100/88 thuộc prediction DualStream 91,71%, không chứng minh hàng 92,15%; b/c ResNet 352/161 thuộc file khác file paper dẫn. | notebooks/paper_test/verification/verify_all_paper_claims.ipynb | Căn đúng test IDs, tính lại b/c và p từ cùng cặp run mà paper báo; bổ sung CLIP-Adapter nếu giữ yêu cầu của phiếu DG31. |

**Table 2 versus Table 7 model identity:** Table 2 names Gemma-DualStream; Table 7 names Gemma-CLIP Full. The few-shot notebook contains a SimpleAdapter run, consistent with the hypothesis that Table 7 used a lighter adapter. However, the three-seed JSON does not record the exact architecture and checkpoints for that row. Do not interpret Table 7 as DualStream results. Confirm the run manifest before renaming that row; then update the abstract and conclusion to distinguish the two configurations.

## Chạy bằng Colab extension

1. Mở notebooks/paper_test/00_colab_runtime_check.ipynb trong VS Code, chọn **Select Kernel → Colab → Auto Connect** và chọn máy chủ GPU khi chạy thí nghiệm.
2. Tải repository vào runtime, giữ cấu trúc thư mục, rồi chạy notebook kiểm tra môi trường. Runtime Colab không tự đọc được ổ đĩa Windows. [Hướng dẫn chính thức của extension](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#uploading-files-to-colab-servers).
3. Chạy verify_all_paper_claims.ipynb và các notebook theo bảng. Trạng thái match chỉ xác nhận artifact được nêu; missing_evidence phải được giải quyết bằng artifact mới, không điền số thủ công.
4. Chỉ khi run có đầy đủ bằng chứng, cập nhật paper đồng bộ số liệu, caption, Abstract và Conclusion. Giữ bảng CuPL/LaBo trong kế hoạch vì phiếu nhận xét yêu cầu; nếu chưa chạy đúng phương pháp thì ghi proxy chính xác.

**Kết luận hiện tại:** Paper **chưa đủ cơ sở để xác nhận toàn bộ thông số**. Một số khác biệt là do run khác nhau và có thể giải thích được bằng metadata; một số là lỗi ghép run hoặc suy luận quá mức. Chưa có căn cứ để nói các số thiếu artifact là bịa, cũng chưa có căn cứ để công nhận chúng đã được kiểm chứng.

