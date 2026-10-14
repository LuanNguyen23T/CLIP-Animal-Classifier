"""Build standalone Kaggle/MPS notebooks from the existing benchmark methods."""
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebooks/training/new_dataset'


def source(cell):
    return ''.join(cell['source'])


def put(cell, value):
    cell['source'] = value.splitlines(keepends=True)


CONFIG = '''
from pathlib import Path
def as_feature_tensor(value):
    # Transformers 4 returns a tensor; newer APIs may return a pooled model output.
    return value if isinstance(value, torch.Tensor) else value.pooler_output

DATA_ROOT = None  # e.g. "/kaggle/input/animal-newdataset/NewDataset"
SEED = 42
TEST_RATIO = 0.20
BATCH_SIZE = 32  # reduce to 16/8 if GPU/MPS memory is limited
TIP_EPOCHS = 20
ADAPTER_EPOCHS = 50
COOP_EPOCHS = 50
RESNET_EPOCHS = 30
OUTPUT_ROOT = Path('/kaggle/working/newdataset_benchmark') if Path('/kaggle').exists() else Path('results/newdataset_benchmark')
OUTPUT_ROOT = OUTPUT_ROOT.resolve()
output_dir = OUTPUT_ROOT / RUN_NAME
output_dir.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
'''

SPLIT = '''
# Identical raw/TTA and CLIP/ResNet split indices; at least one validation sample/class.
train_indices, val_indices = [], []
generator = torch.Generator().manual_seed(SEED)
for c in range(num_classes):
    indices = (trainval_labels == c).nonzero(as_tuple=True)[0]
    indices = indices[torch.randperm(len(indices), generator=generator)]
    n_train = min(len(indices) - 1, max(1, int(len(indices) * 0.9)))
    train_indices.append(indices[:n_train]); val_indices.append(indices[n_train:])
train_idx, val_idx = torch.cat(train_indices), torch.cat(val_indices)
(output_dir / 'train_val_indices.json').write_text(json.dumps({
    'train': train_idx.tolist(), 'val': val_idx.tolist(), 'fingerprint': manifest['fingerprint']}, indent=2), encoding='utf-8')
print(f'Train={len(train_idx)} | Validation={len(val_idx)} | Test={len(test_labels)}')
'''

COOP_CLASS = '''class CoOpClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=None, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
        self.num_ctx = num_ctx
        self.ctx = nn.Parameter(torch.randn(num_ctx, clip_model.config.text_config.hidden_size, device=device) * 0.02)

    def get_text_features(self, input_ids):
        # Scoped hook: each classifier uses its own context, including after subsequent fits.
        def replace_context(module, args, output):
            return torch.cat([output[:, :1], self.ctx.to(output.dtype).unsqueeze(0).expand(output.shape[0], -1, -1), output[:, 1 + self.num_ctx:]], dim=1)
        self.clip_model.eval()
        handle = self.clip_model.text_model.embeddings.token_embedding.register_forward_hook(replace_context)
        try:
            features = self.clip_model.get_text_features(input_ids=input_ids)
        finally:
            handle.remove()
        return F.normalize(as_feature_tensor(features), dim=-1)

    def forward(self, image_features, input_ids):
        return (image_features @ self.get_text_features(input_ids).t()) * self.clip_model.logit_scale.exp()


'''


def build(kind):
    original = 'benchmark_clip_animal_classifier.ipynb' if kind == 'clip' else 'benchmark_resnet50.ipynb'
    book = copy.deepcopy(json.loads((DEST.parent / original).read_text(encoding='utf-8')))
    cells = book['cells']
    for cell in cells:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
        cell['metadata'] = {}
        text = source(cell).replace('Oxford-IIIT Pet Dataset (37 Classes)', 'NewDataset (số lớp tự động)')
        text = text.replace('OXFORD PETS', 'NEWDATASET').replace('37 giống', 'các giống').replace('37 lớp', 'các lớp')
        text = text.replace('num_classes=37', 'num_classes=None')
        # Defaults must be dynamic for constructors invoked without arguments.
        text = text.replace('self.num_classes = num_classes', 'self.num_classes = num_classes')
        text = text.replace('Image.FLIP_LEFT_RIGHT', 'Image.Transpose.FLIP_LEFT_RIGHT')
        text = text.replace('sample_idx = 10', 'sample_idx = min(10, len(pets_test) - 1)')
        if cell['cell_type'] == 'code':
            text = text.replace('epochs=20', 'epochs=TIP_EPOCHS').replace('epochs=50', 'epochs=ADAPTER_EPOCHS').replace('epochs=30', 'epochs=RESNET_EPOCHS')
            text = text.replace('if val_acc > best_val_acc:', 'if val_acc >= best_val_acc:')
            text = text.replace('return val_acc if val_feats is not None else None', 'return best_val_acc if val_feats is not None else None')
            for name in ('f', 'text_feats_enr', 'text_feats_raw'):
                text = re.sub(rf'({name} = )(clip_model\.get_(?:image|text)_features\([^\n]+\))', r'\1as_feature_tensor(\2)', text)
        put(cell, text)
    put(cells[0], f'''# BENCHMARK TOÀN DIỆN {kind.upper()} — NEWDATASET

Chỉ huấn luyện/đánh giá các thư mục lớp trong `NewDataset`; không gộp Oxford-IIIT Pet.
Notebook độc lập, giữ thứ tự và các phương pháp từ `{original}`.

**Kaggle:** Import notebook này; Add Input dataset chứa `NewDataset/<giống>/<ảnh>`; bật GPU và Internet.
**MacBook:** cài dependencies ở cell dưới, chọn kernel Python của môi trường; tự chọn CUDA → MPS → CPU.
Đường dẫn `DATA_ROOT` phải trỏ tới thư mục chứa trực tiếp các thư mục giống.

**Thiết kế:** loại ảnh trùng pixel trước khi chia, giữ 20% mỗi lớp làm test; 80% còn lại là trainval.
Trainval chia 9:1 theo lớp. Full trainval dùng toàn bộ 80%, không dùng test để chọn tham số.
Ảnh gần trùng hoặc cùng một cá thể không được phát hiện bởi hash; cần kiểm tra nguồn dữ liệu nếu muốn split theo cá thể.
Cache tách theo fingerprint dữ liệu và backbone; mọi output ghi vào thư mục có quyền ghi.
Các số liệu và biểu đồ sẽ được tính khi chạy, không sao chép kết quả Oxford cũ.
''')
    put(cells[1], source(cells[1]) + f"\nRUN_NAME = '{kind}'\n" + CONFIG + (DEST / 'evaluate.py').read_text(encoding='utf-8'))
    loader = (DEST / 'data_utils.py').read_text(encoding='utf-8') + '''
dataset_root = discover_dataset(DATA_ROOT)
pets_trainval, pets_test, manifest = prepare_dataset(dataset_root, output_dir, SEED, TEST_RATIO)
class_names = pets_trainval.classes
num_classes = len(class_names)
trainval_labels = torch.tensor([r['label'] for r in pets_trainval.records], dtype=torch.long)
test_labels = torch.tensor([r['label'] for r in pets_test.records], dtype=torch.long)
'''
    if kind == 'clip':
        loader += '''
MODEL_ID = 'openai/clip-vit-base-patch32'
clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
clip_model.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
logit_scale_val = clip_model.logit_scale.exp().item()
prompts_raw = [c.replace('_', ' ') for c in class_names]
prompts_enriched = [f'Picture of a pet with breed: {c.replace("_", " ")}, which is a type of animal.' for c in class_names]
'''
    else:
        loader += "\nMODEL_ID = 'torchvision/resnet50/IMAGENET1K_V2'\n"
    put(cells[2], loader)
    feature_code = source(cells[4])
    if kind == 'clip':
        feature_code = feature_code[:feature_code.index('cached_dir =')]
        feature_code += '''
cache_dir = output_dir / 'cache' / manifest['fingerprint'] / 'clip_vit_b32_preprocess_v1'
cache_dir.mkdir(parents=True, exist_ok=True)
def get_features(name, dataset, transform=None):
    path = cache_dir / f'{name}.pt'
    if path.exists():
        features = torch.load(path, map_location='cpu', weights_only=True)
        assert features.shape == (len(dataset), clip_model.config.projection_dim)
        return features
    features = extract_features(TransformedDataset(dataset, transform, clip_processor), batch_size=BATCH_SIZE, desc=name)
    torch.save(features, path)
    return features
trainval_features_raw = get_features('trainval_raw', pets_trainval)
test_features_raw = get_features('test_raw', pets_test)
test_features_opt = get_features('test_optimal', pets_test, optimal_preprocessing_pipeline)
test_features_tta = F.normalize(torch.stack([
    test_features_raw, get_features('test_denoise', pets_test, apply_denoise),
    get_features('test_clahe', pets_test, apply_clahe), test_features_opt,
    get_features('test_geom', pets_test, transform_geom)]).mean(0), dim=-1)
print(trainval_features_raw.shape, test_features_tta.shape)
'''
        split = SPLIT + '''
train_feats_90, train_lbls_90 = trainval_features_raw[train_idx], trainval_labels[train_idx]
val_feats_10, val_lbls_10 = trainval_features_raw[val_idx], trainval_labels[val_idx]
train_feats_full, train_lbls_full = trainval_features_raw, trainval_labels
benchmark_12_results = {}
'''
        coop = source(cells[15]).replace('epochs=ADAPTER_EPOCHS', 'epochs=COOP_EPOCHS')
        start = coop.index('class CoOpClassifier')
        stop = coop.index('def train_coop', start)
        coop = coop[:start] + COOP_CLASS + coop[stop:]
        coop = coop.replace('best_weights = {k: v.clone() for k, v in model_coop.state_dict().items()}', 'best_weights = model_coop.ctx.detach().clone()')
        coop = coop.replace('model_coop.load_state_dict(best_weights)', 'model_coop.ctx.data.copy_(best_weights)')
        put(cells[15], coop)
        for index, names in [(11, ['model_tip_90', 'model_tip_full']), (13, ['model_ca_90', 'model_ca_full']), (15, ['model_coop_90', 'model_coop_full'])]:
            put(cells[index], source(cells[index]) + '\n' + '\n'.join(f"save_checkpoint('{name}', {name})" for name in names))
        # Text/cache metadata needed to reconstruct training-free inference too.
        put(cells[11], source(cells[11]) + "\ntorch.save({'text_features': text_feats_enr.cpu(), 'support_features': train_feats_full, 'support_labels': train_lbls_full, 'classes': class_names, 'fingerprint': manifest['fingerprint'], 'alpha': best_alpha_90, 'beta': best_beta_90, 'logit_scale': logit_scale_val}, output_dir / 'tip_free_cache.pt')\n")
        predict = '''
predictors = {
    'CLIP (Baseline)': lambda x: x @ text_feats_enr.t() * logit_scale_val,
    'CLIP + Tip-Adapter (Free)': lambda x: run_tip_adapter_free(x, text_feats_enr, train_feats_90.to(device), train_lbls_90.to(device), best_alpha_90, best_beta_90),
    'CLIP + Tip-Adapter-F (Tuned)': model_tip_90.eval(),
    'CLIP + CLIP-Adapter': model_ca_90.eval(),
    'CLIP + Prompt Tuning': lambda x: model_coop_90(x, coop_input_ids),
}
histories = {'Tip-Adapter-F': hist_tip_90, 'CLIP-Adapter': hist_ca_90, 'CoOp': hist_coop_90}
results = benchmark_12_results
encoder = lambda x: clip_model.get_image_features(pixel_values=x)
'''
    else:
        feature_code = feature_code[:feature_code.index('cached_res_dir =')]
        feature_code = feature_code.replace('backbone.eval()', 'backbone.eval()\nbackbone.requires_grad_(False)')
        feature_code += '''
cache_dir = output_dir / 'cache' / manifest['fingerprint'] / 'resnet50_imagenet_v2_preprocess_v1'
cache_dir.mkdir(parents=True, exist_ok=True)
def get_features(name, dataset, transform=None):
    path = cache_dir / f'{name}.pt'
    if path.exists():
        features = torch.load(path, map_location='cpu', weights_only=True)
        assert features.shape == (len(dataset), 2048)
        return features
    features, _ = extract_resnet_features(dataset, transform_fn=transform, batch_size=BATCH_SIZE, desc=name)
    torch.save(features, path)
    return features
def tta_features(prefix, dataset, raw):
    return torch.stack([raw, get_features(prefix + '_denoise', dataset, apply_denoise),
        get_features(prefix + '_clahe', dataset, apply_clahe),
        get_features(prefix + '_optimal', dataset, optimal_preprocessing_pipeline),
        get_features(prefix + '_geom', dataset, transform_geom)]).mean(0)
trainval_features_raw = get_features('trainval_raw', pets_trainval)
test_features_raw = get_features('test_raw', pets_test)
trainval_features_tta = tta_features('trainval', pets_trainval, trainval_features_raw)
test_features_tta = tta_features('test', pets_test, test_features_raw)
print(trainval_features_raw.shape, test_features_tta.shape)
'''
        split = SPLIT + '''
X_tr_tta_90, y_tr_90 = trainval_features_tta[train_idx], trainval_labels[train_idx]
X_val_tta_90, y_val_90 = trainval_features_tta[val_idx], trainval_labels[val_idx]
X_tr_raw_90, X_val_raw_90 = trainval_features_raw[train_idx], trainval_features_raw[val_idx]
benchmark_resnet_results = {}
'''
        for index, short in [(11, 'lp'), (13, 'adp'), (15, 'cos')]:
            text = source(cells[index]).replace('num_classes=None', 'num_classes=num_classes')
            # Retain every fitted head and its history for checkpointing and audit.
            text = text.replace(f't_{short}_raw, _, _', f't_{short}_raw, hist_{short}_raw, model_{short}_raw')
            text = text.replace(f't_{short}_full, _, _', f't_{short}_full, hist_{short}_full, model_{short}_full')
            text += '\n' + '\n'.join(f"save_checkpoint('model_{short}_{suffix}', model_{short}_{suffix})" for suffix in ['raw', '90', 'full'])
            put(cells[index], text)
        predict = '''
prototypes = F.normalize(torch.stack([F.normalize(X_tr_tta_90[y_tr_90 == c], dim=-1).mean(0) for c in range(num_classes)]), dim=-1).to(device)
torch.save({'prototypes': prototypes.cpu(), 'classes': class_names, 'fingerprint': manifest['fingerprint']}, output_dir / 'prototypes.pt')
predictors = {'ResNet-50 (Prototype)': lambda x: F.normalize(x, dim=-1) @ prototypes.t(),
    'ResNet-50 (Linear Probe)': model_lp_90.eval(),
    'ResNet-50 (Adapter)': model_adp_90.eval(),
    'ResNet-50 (Cosine Head)': model_cos_90.eval()}
histories = {'Linear Probe': hist_lp_90, 'Residual Adapter': hist_adp_90, 'Cosine Head': hist_cos_90}
results = benchmark_resnet_results
encoder = lambda x: backbone(x).flatten(1)
'''
    put(cells[4], feature_code)
    put(cells[6], '''## 2. CHIẾN LƯỢC PHÂN CHIA DỮ LIỆU: TRAIN:VAL (9:1) & FULL TRAINVAL

20% ảnh duy nhất mỗi lớp được giữ làm test từ đầu. Chia 80% trainval còn lại theo lớp:
90% để train/support, 10% validation chọn tham số và checkpoint có validation accuracy cao nhất.
Full trainval không có validation độc lập; chạy đúng số epochs đã cấu hình.
Đây là chọn checkpoint tốt nhất, không phải dừng sớm. CLIP train trên raw features;
ResNet có cả raw và TTA train features như benchmark gốc.
''')
    put(cells[7], split)
    put(cells[16], '## 7. ĐO ĐẠC ĐỘ TRỄ SUY DIỄN\n\nĐo trên thiết bị thực tế, đồng bộ CUDA/MPS. Tách head-only và encoder-only; không tính đọc ảnh/tiền xử lý.\n')
    put(cells[17], predict)
    put(cells[18], '## 8. TỔNG HỢP KẾT QUẢ & TRỰC QUAN HÓA\n\nBảy hình: accuracy, loss, validation, latency, tham số, confusion matrix, TTA gain từng lớp. Confusion matrix chọn phương pháp theo validation.\n')
    put(cells[19], 'build_report(results, predictors, histories, encoder)\n')
    put(cells[20], '## 9. BẢNG TỔNG KẾT & XUẤT ARTIFACT\n\nXem bảng được tạo phía trên và `benchmark_report.md`, CSV/JSON, checkpoints, split manifest, class mapping và PNG trong output. So sánh CLIP/ResNet chỉ khi fingerprint trùng khớp. Không dùng trực tiếp checkpoint mới với backend 37 lớp khi chưa cập nhật ánh xạ lớp và cấu hình inference.\n')
    install = {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
               'source': ['%pip install "transformers>=4.45,<5" "opencv-python-headless>=4.8,<5" scikit-learn matplotlib seaborn pandas tabulate tqdm\n',
                          '# macOS only, if torch/torchvision are missing: %pip install torch torchvision\n']}
    cells.insert(1, install)
    book['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                        'language_info': {'name': 'python'}}
    book['nbformat'], book['nbformat_minor'] = 4, 5
    for i, cell in enumerate(cells):
        cell['id'] = f'{kind}-{i:02d}'
    target = DEST / f'benchmark_{kind}_newdataset.ipynb'
    target.write_text(json.dumps(book, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    build('clip')
    build('resnet50')
