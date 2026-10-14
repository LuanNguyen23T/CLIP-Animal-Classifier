# Huấn luyện NewDataset: CLIP và ResNet-50 trên Kaggle / MacBook

Hai notebook chỉ sử dụng các lớp trong `dataset/NewDataset`, không gộp Oxford-IIIT Pet. Kiến trúc CNN là **ResNet-50 giống benchmark cũ**, không phải ResNeSt.

Kiểm tra dữ liệu hiện tại: **16 lớp, 2.842 file ảnh**, không có ảnh lỗi; loại **5 ảnh trùng pixel**, còn **2.837 ảnh duy nhất**. Với cấu hình mặc định: **train 2.035**, **validation 235**, **test 567**. Thứ tự lớp, số lượng và fingerprint được tính lại nếu bạn thay đổi ảnh.

## 1. Đưa gì vào Kaggle Input?

**Chỉ cần ảnh của NewDataset.** Có thể tải lên file `artifacts/kaggle/NewDataset.zip` tạo bằng lệnh dưới từ thư mục gốc dự án:

```powershell
python scripts/package_new_dataset_kaggle.py
```

ZIP chứa cấu trúc:

```text
NewDataset/
  Airedale/*.jpg
  basenji/*.jpg
  Bernese_mountain_dog/*.jpg
  ...
  West_Highland_white_terrier/*.jpg
```

1. Tạo Kaggle Dataset mới, ví dụ đặt tên `animal-newdataset`.
2. Upload `NewDataset.zip`, tạo phiên bản dataset.
3. Trong mỗi notebook, **Add Input** → chọn dataset vừa tạo.
4. Kiểm tra cây thư mục Input. `DATA_ROOT` phải trỏ tới thư mục chứa trực tiếp 16 thư mục giống, ví dụ `/kaggle/input/animal-newdataset/NewDataset`. Tên đường dẫn thực tế phụ thuộc slug và cách Kaggle giải nén; dùng đúng đường dẫn hiển thị.

**Không đưa vào Input:** `trainval_features.pt`, `test_features.pt`, các file nhãn `.pt`, `dataset/cached`, `dataset/cached_resnet50`, `backend/weights`, `oxford-iiit-pet`, backend/frontend hay toàn bộ repository. Cache cũ không khớp bộ dữ liệu mới.

Hai `.ipynb` dưới đây được **Import Notebook**, không phải upload cùng ảnh vào Input. Chúng đã nhúng các hàm cần thiết nên không cần thêm file `.py` vào Kaggle:

- `benchmark_clip_newdataset.ipynb`
- `benchmark_resnet50_newdataset.ipynb`

## 2. Chạy notebook CLIP

1. Tạo notebook Kaggle → Import Notebook → chọn `benchmark_clip_newdataset.ipynb`.
2. Add Input bộ ảnh như bước 1.
3. Trong Settings chọn **GPU**, bật **Internet** để cài dependencies và tải pretrained weights. Notebook tự dùng một GPU, không có huấn luyện phân tán hai GPU.
4. Chạy cell cài dependencies. Nếu Kaggle yêu cầu restart kernel sau cài đặt, restart rồi chạy từ đầu.
5. Trong cell cấu hình, đặt `DATA_ROOT` theo cây Input nếu tự tìm không được; mặc định `BATCH_SIZE=128` như benchmark cũ, seed 42, test 20%. Giảm batch trích xuất xuống 32/16/8 nếu thiếu bộ nhớ; batch huấn luyện head giữ nguyên 256 (CLIP) hoặc 128 (ResNet).
6. **Run All**, theo thứ tự: nạp ảnh và chia dữ liệu → tiền xử lý/TTA → train/val 9:1 → zero-shot → Tip-Adapter → Tip-Adapter-F → CLIP-Adapter → CoOp → latency → biểu đồ và bảng kết quả.
7. **Save Version / Save & Run All**, tải artifacts từ Output sau khi phiên bản hoàn tất.

Pretrained CLIP: `openai/clip-vit-base-patch32`. Mô hình backbone đóng băng; học cache keys, MLP adapter hoặc 4 context vectors tùy phương pháp.

## 3. Chạy notebook ResNet-50

Import `benchmark_resnet50_newdataset.ipynb`, thêm cùng Input, bật GPU/Internet và chạy các bước tương tự.

Thứ tự: nạp ảnh/chia dữ liệu → pretrained ImageNet ResNet-50 → tiền xử lý/TTA → train/val 9:1 → Prototype → Linear Probe → Residual Adapter → Cosine Head → latency → biểu đồ và bảng kết quả.

Đây là benchmark **frozen backbone và huấn luyện head/adapter như bản cũ**, không phải fine-tune toàn bộ ResNet-50. CLIP huấn luyện trên raw features; ResNet sử dụng raw hoặc TTA train features đúng cấu trúc benchmark gốc.

| Mô hình | Phương pháp | Số epoch mặc định |
| --- | --- | ---: |
| CLIP | Zero-shot / Tip-Adapter Free | 0; Free tìm alpha/beta trên validation |
| CLIP | Tip-Adapter-F | 20 |
| CLIP | CLIP-Adapter | 50 |
| CLIP | CoOp | 50 |
| ResNet-50 | Prototype | 0 |
| ResNet-50 | Linear / Residual Adapter / Cosine | 30 |

Mỗi phương pháp có ba kịch bản giống benchmark cũ: **Raw Full Trainval**, **TTA 5-view 9:1**, **TTA 5-view Full Trainval**. Vì ảnh mới chưa có test split chính thức, trước hết giữ riêng khoảng 20% mỗi lớp làm test. 80% còn lại là trainval; chia tiếp 9:1. “Full” luôn chỉ toàn bộ trainval, không gồm test. Validation chọn checkpoint tốt nhất trong số epoch cố định; không triển khai dừng sớm theo patience.

TTA giữ đúng năm view trong mã benchmark cũ: ảnh gốc, bilateral denoise, CLAHE, bilateral + CLAHE, bilateral + CLAHE + flip + rotate 15°. Mô tả center crop/scale/shift trong markdown cũ không khớp mã thực thi; bản mới theo mã thực thi.

## 4. Kết quả nằm ở đâu?

```text
/kaggle/working/newdataset_benchmark/
  clip/
  resnet50/
```

Trong mỗi thư mục có:

- `split_manifest.json`, `train_val_indices.json`, `classes.json`: ảnh từng split, seed, fingerprint và thứ tự lớp.
- `cache/<fingerprint>/...`: đặc trưng mới, không sử dụng cache Oxford cũ.
- `model_*.pt`: checkpoint adapter/head; CoOp lưu context đã học, cần tải lại pretrained CLIP khi inference.
- `tip_free_cache.pt` (CLIP), `prototypes.pt` (ResNet): dữ liệu cho phương pháp training-free.
- `benchmark_results.csv`, `benchmark_results.json`, `benchmark_report.md`.
- `training_history.json`, `environment.json`, `classification_report.json` và 7 biểu đồ PNG.

Latency đo từng head đúng kịch bản Raw Full, TTA 9:1 và TTA Full trên cached features. Cột `Head_Latency_ms` và throughput trong các biểu đồ là **head-only**, chưa gồm encoder. Encoder raw/5-view được đo riêng trên tensor đã nằm trên thiết bị. Không gọi các số này là latency end-to-end vì chưa tính đọc ảnh, OpenCV và processor. Không so trực tiếp latency giữa phần cứng khác nhau.

## Tính tương đồng với benchmark cũ và các sửa lỗi

Công thức, kiến trúc, optimizer, learning rate, weight decay, scheduler, số epoch, batch huấn luyện, năm view TTA và ba kịch bản giữ nguyên. Batch trích xuất mặc định trở lại 128. Split train/validation dùng global RNG seed 42 như mã cũ. ResNet giữ điều kiện `>` khi chọn checkpoint (giữ checkpoint đầu khi hòa); CLIP giữ `>=` như mã cũ.

Giữ mã vẽ gốc: CLIP có heatmap accuracy → grouped bars → loss/validation hai panel → throughput/tham số → confusion CoOp 9:1 → delta từng giống so với zero-shot raw → accuracy/latency Pareto. ResNet có grouped bars so với CoOp → loss/validation hai panel → trade-off → throughput/tham số → confusion Cosine 9:1 → delta TTA cùng Cosine head → đối chiếu Raw/TTA với CoOp. Màu sắc, bố cục và thứ tự kế thừa mã cũ; số lớp và giới hạn trục thích ứng dữ liệu mới.

Các sửa lỗi: CoOp dùng context riêng cho mỗi classifier và đóng băng pretrained CLIP; tên lớp văn bản của zero-shot/CoOp cùng chuyển `_` thành khoảng trắng nhưng mapping checkpoint vẫn giữ tên thư mục. ResNet báo best validation accuracy của checkpoint đã phục hồi; khởi tạo best score -1 để vẫn lưu checkpoint nếu tất cả epoch có accuracy 0%. Bỏ số accuracy/latency ghi sẵn, sửa mask CPU/GPU khi vẽ delta CLIP, tính Pareto theo dominance thực tế và hiển thị dấu tăng/giảm đúng.

Biểu đồ ResNet chỉ có số liệu CoOp khi output CLIP cùng fingerprint đã tồn tại. Nếu chưa chạy CLIP, đánh dấu thiếu kết quả thay vì dùng số Oxford cũ; sau khi chạy CLIP, chạy lại cell báo cáo ResNet để bổ sung.

Phần thay dữ liệu bắt buộc khác: folder ảnh NewDataset, số lớp động, loại ảnh trùng và giữ riêng test 20% vì không có split test chính thức Oxford. Không tuyên bố tái lập accuracy Oxford hay sao chép nguyên các lỗi thực thi cũ.

Để có bảng CLIP vs ResNet trong một lần chạy, dùng cùng Kaggle notebook: sau khi CLIP hoàn tất, chạy các cell ResNet bên dưới với cùng `OUTPUT_ROOT`. Các artifact CLIP đã được lưu trước đó. Hoặc tải output CLIP và đặt thư mục `clip/` bên cạnh `resnet50/` trong Working trước cell báo cáo ResNet. Notebook chỉ ghép khi fingerprint split giống nhau. Add Input output notebook trước không tự chép sang Working.

Để chạy lại từ cache sau một phiên Kaggle mới, Add Input output phiên trước và **copy** thư mục cache tương ứng vào đường dẫn Working phía trên trước cell trích xuất. Input chỉ đọc; notebook luôn ghi kết quả vào Working. Cache giúp tránh trích xuất lại, chưa phải resume optimizer giữa epoch.

## 5. Chạy trên MacBook Apple Silicon (MPS)

Từ thư mục gốc dự án:

```bash
python3 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install torch torchvision 'transformers>=4.45,<5' 'opencv-python-headless>=4.8,<5' scikit-learn matplotlib seaborn pandas tabulate tqdm jupyter ipykernel
python -m ipykernel install --user --name newdataset-benchmark --display-name 'NewDataset Benchmark'
python -c "import torch; print(torch.backends.mps.is_available())"
jupyter notebook
```

Chọn kernel `NewDataset Benchmark`, mở notebook trong `notebooks/training/new_dataset/`. `DATA_ROOT=None` tự tìm `dataset/NewDataset` qua các thư mục cha; hoặc nhập đường dẫn tuyệt đối. Tự chọn CUDA → MPS → CPU. MacBook nên bắt đầu batch 16 hoặc 32, giảm xuống 8 nếu hết bộ nhớ. Nếu gặp một toán tử không hỗ trợ MPS, thử khởi động Jupyter với `PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter notebook`; fallback có thể chậm hơn.

## 6. Kiểm tra trước khi báo cáo

Notebook kiểm tra ảnh có thể mở, loại ảnh trùng pixel trước khi chia, và dừng nếu cùng ảnh mang hai nhãn khác nhau. Xem `invalid_images` và `duplicates_removed` trong manifest. Hash chỉ phát hiện trùng chính xác sau giải mã; không đảm bảo các ảnh gần trùng hoặc cùng cá thể không rơi vào nhiều split.

Kết quả chưa được tính sẵn: cần hoàn tất Run All trên GPU/MPS. Không đưa các accuracy Oxford cũ vào báo cáo mới. Checkpoint 16 lớp không thể thay trực tiếp vào backend 37 lớp mà chưa cập nhật class mapping và logic inference.

Nguồn về môi trường Kaggle: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks). Model CLIP: [Hugging Face CLIP](https://huggingface.co/openai/clip-vit-base-patch32).
