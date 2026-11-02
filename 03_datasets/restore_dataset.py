"""Reconstruct the original ZIP from verified GitHub Release binary parts."""
import argparse
import hashlib
import json
from pathlib import Path


def restore(parts_dir: Path, output: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    archive = manifest['archive']
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.incomplete')
    if temporary.exists():
        raise FileExistsError(f'Partial output already exists: {temporary}')
    whole = hashlib.sha256()
    total = 0
    with temporary.open('xb') as dest:
        for part in manifest['parts']:
            name = part['name']
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Invalid part name')
            source = parts_dir / name
            if source.stat().st_size != part['bytes']:
                raise ValueError(f'Part size mismatch: {name}')
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                while block := stream.read(8 * 1024 ** 2):
                    dest.write(block)
                    digest.update(block)
                    whole.update(block)
                    total += len(block)
            if digest.hexdigest() != part['sha256']:
                raise ValueError(f'Part checksum mismatch: {name}; incomplete output retained')
            print(f'Verified {name}', flush=True)
    if total != archive['bytes'] or whole.hexdigest() != archive['sha256']:
        raise ValueError('Archive checksum mismatch; incomplete output retained')
    temporary.replace(output)
    print(f'Restored and verified: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, default=Path('E:/比赛数据集/release_downloads'))
    parser.add_argument('--output', type=Path, default=Path('E:/比赛数据集/downloads/cxr_first.zip'))
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    restore(args.parts_dir, args.output, args.manifest or args.parts_dir / 'DATASET_MANIFEST.json')
