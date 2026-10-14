"""Package image folders only; excludes legacy features, code and model weights."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset/NewDataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/kaggle/NewDataset.zip')
    args = parser.parse_args()
    root = args.dataset.resolve()
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f'No images in {root}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, Path('NewDataset') / path.relative_to(root))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    print(f'{len(files)} images; {args.output.stat().st_size / 1024**2:.1f} MiB; {args.output.resolve()}')


if __name__ == '__main__':
    main()
