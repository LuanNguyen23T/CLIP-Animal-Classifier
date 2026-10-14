"""Offline integration smoke test with synthetic features and a tiny CLIP text encoder.

Checks all methods and reports; does not measure real dataset accuracy.
"""
import ast
import json
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def verify(kind, temporary):
    path = ROOT / f'notebooks/training/new_dataset/benchmark_{kind}_newdataset.ipynb'
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == 'code' and not cell.source.startswith('%'):
            ast.parse(cell.source)
    context = {}
    exec(book.cells[2].source, context)
    exec('''
import matplotlib
matplotlib.use('Agg')
plt.show = lambda: plt.close('all')
torch.set_num_threads(2)
device = torch.device('cpu')
TIP_EPOCHS = ADAPTER_EPOCHS = COOP_EPOCHS = RESNET_EPOCHS = 1
SEED = 42
num_classes = 3
class_names = ['breed_a', 'breed_b', 'breed_c']
manifest = {'fingerprint': 'synthetic_smoke'}
trainval_labels = torch.arange(num_classes).repeat_interleave(4)
test_labels = torch.arange(num_classes)
''', context)
    context['output_dir'] = Path(temporary) / kind
    context['output_dir'].mkdir()
    context['MODEL_ID'] = f'synthetic-{kind}'
    dimension = 512 if kind == 'clip' else 2048
    exec(f'''
trainval_features_raw = F.normalize(torch.randn(12, {dimension}), dim=-1)
test_features_raw = F.normalize(torch.randn(3, {dimension}), dim=-1)
test_features_tta = F.normalize(test_features_raw + torch.randn_like(test_features_raw) * .01, dim=-1)
trainval_features_tta = trainval_features_raw.clone()
''', context)
    if kind == 'clip':
        exec('''
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from types import SimpleNamespace
text_config = CLIPTextConfig(vocab_size=49408, hidden_size=512, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=8)
vision_config = CLIPVisionConfig(hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4)
clip_model = CLIPModel(CLIPConfig(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), projection_dim=512))
clip_model.requires_grad_(False)
clip_model.eval()
clip_model.get_image_features = lambda pixel_values: torch.ones(pixel_values.shape[0], 512)
clip_processor = SimpleNamespace(tokenizer=lambda name, **kw: {'input_ids': torch.tensor([[49406, 3, 49407]])})
logit_scale_val = clip_model.logit_scale.exp().item()
text_feats_enr = F.normalize(torch.randn(num_classes, 512), dim=-1)
text_feats_raw = text_feats_enr.clone()
''', context)
    else:
        exec('backbone = lambda x: torch.ones(x.shape[0], 2048, 1, 1)', context)
    exec(book.cells[8].source, context)  # shared train/validation split
    if kind == 'clip':
        # Baseline text features are synthetic, preserving the result/evaluation part.
        baseline = book.cells[10].source
        exec(baseline[baseline.index('# 1.1. Raw Test'):], context)
    else:
        exec(book.cells[10].source, context)
    for index in [12, 14, 16, 18, 20]:
        exec(book.cells[index].source, context)
    results = json.loads((context['output_dir'] / 'benchmark_results.json').read_text(encoding='utf-8'))
    assert len(results['results']) == (15 if kind == 'clip' else 12)
    assert len(list(context['output_dir'].glob('0*.png'))) == 7
    if kind == 'clip':
        # Regression: fitting full CoOp must not change the 9:1 model's context hook.
        first = context['model_coop_90']
        second = context['model_coop_full']
        ids = context['coop_input_ids']
        before = first.get_text_features(ids).detach().clone()
        second.ctx.data.add_(2)
        after = first.get_text_features(ids).detach()
        assert context['torch'].allclose(before, after)
        assert not context['clip_model'].text_model.embeddings.token_embedding._forward_hooks
        assert context['clip_model'].text_model.embeddings.token_embedding.weight.grad is None
    print(f'PASS: {kind}, all methods, 7 figures, reports, checkpoints')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='newdataset-smoke-') as temporary:
        verify('clip', temporary)
        verify('resnet50', temporary)
