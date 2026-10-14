"""Image-folder dataset, content deduplication and reproducible benchmark splits."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def discover_dataset(explicit=None):
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root.resolve()
    candidates = []
    for parent in [Path.cwd(), *Path.cwd().parents]:
        for name in ('NewDataset', 'NewDatasett'):
            candidate = parent / 'dataset' / name
            if candidate.is_dir():
                candidates.append(candidate.resolve())
    kaggle = Path('/kaggle/input')
    if kaggle.exists():
        candidates.extend(p.resolve() for p in kaggle.rglob('*')
                          if p.is_dir() and p.name in ('NewDataset', 'NewDatasett'))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f'Set DATA_ROOT to the folder containing breed folders; found {candidates}')
    return candidates[0]


class BreedDataset(Dataset):
    def __init__(self, root, records, classes):
        self.root, self.records, self.classes = Path(root), records, classes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        with Image.open(self.root / item['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
        return image, item['label']


def prepare_dataset(root, output_dir, seed=42, test_ratio=0.2):
    """Split unique decoded images per class; reject conflicting duplicate labels."""
    root, output_dir = Path(root), Path(output_dir)
    if not 0 < test_ratio < 1:
        raise ValueError('test_ratio must be between 0 and 1')
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if len(classes) < 2:
        raise ValueError('Expected at least two breed folders')
    groups, seen, duplicates, invalid = defaultdict(list), {}, [], []
    for label, name in enumerate(classes):
        for path in sorted((root / name).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            except (OSError, ValueError) as error:
                invalid.append({'path': relative, 'error': str(error)})
                continue
            if digest in seen:
                previous = seen[digest]
                if previous['label'] != label:
                    raise ValueError(f'Identical image has conflicting labels: {previous["path"]}, {relative}')
                duplicates.append({'path': relative, 'same_as': previous['path']})
                continue
            record = {'path': relative, 'label': label, 'sha256': digest}
            seen[digest] = record
            groups[label].append(record)
    trainval, test = [], []
    for label, name in enumerate(classes):
        items = groups[label][:]
        if len(items) < 4:
            raise ValueError(f'{name}: need >= 4 valid unique images for train/val/test')
        random.Random(seed + label).shuffle(items)
        count = min(len(items) - 2, max(1, round(len(items) * test_ratio)))
        test.extend(items[:count])
        trainval.extend(items[count:])
    manifest = {'seed': seed, 'test_ratio': test_ratio, 'classes': classes,
                'trainval': trainval, 'test': test, 'duplicates_removed': duplicates,
                'invalid_images': invalid, 'version': 1}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    manifest['fingerprint'] = fingerprint
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'classes.json').write_text(json.dumps(classes, indent=2), encoding='utf-8')
    print(f'{len(classes)} classes | trainval={len(trainval)} | test={len(test)}')
    print(f'Removed duplicates={len(duplicates)}; invalid images={len(invalid)}; fingerprint={fingerprint}')
    print({classes[k]: v for k, v in sorted(Counter(r['label'] for r in trainval + test).items())})
    return BreedDataset(root, trainval, classes), BreedDataset(root, test, classes), manifest
