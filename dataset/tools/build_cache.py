"""Create JPEG quality-92 caches with short-edge resizing and square padding."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image, ImageOps

if __package__:
    from .build_manifest import data_root_path, read_manifest, write_csv
else:
    from build_manifest import data_root_path, read_manifest, write_csv


SHORT_EDGE = 512
JPEG_QUALITY = 92


def cache_image_path(cache_root, image_path):
    path = cache_root / image_path
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        path = path.with_suffix(path.suffix + ".jpg")
    return path


def cache_image(row, root, cache_root):
    relative = row["image_path"]
    destination = cache_image_path(cache_root, relative)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        if destination.is_file():
            try:
                with Image.open(destination) as image:
                    if image.format == "JPEG" and image.width == image.height >= SHORT_EDGE:
                        image.load()
                        return "reused", relative, ""
            except (OSError, ValueError, Image.DecompressionBombError):
                pass
        with Image.open(root / relative) as image:
            scale = SHORT_EDGE / min(image.size)
            size = tuple(round(edge * scale) for edge in image.size)
            resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        edge = max(size)
        square = ImageOps.pad(resized, (edge, edge), color=0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        square.save(temporary, format="JPEG", quality=JPEG_QUALITY)
        temporary.replace(destination)
        return "written", relative, ""
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        temporary.unlink(missing_ok=True)
        return "failed", relative, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Official dataset directory (defaults to DATA_ROOT)")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("cache_512"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = data_root_path(args.data_root)
    output = args.output.resolve()
    if output == root or any(output.is_relative_to(root / split) for split in ("train", "valid", "test")):
        parser.error("Use a separate cache directory, outside the source splits.")
    rows = read_manifest(args.manifest)
    failures, counts = [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(partial(cache_image, root=root, cache_root=output), rows)
        for index, (state, path, reason) in enumerate(results, 1):
            counts[state] += 1
            if state == "failed":
                failures.append({"image_path": path, "error": reason})
            if index % 1000 == 0 or index == len(rows):
                print(f"{index:,}/{len(rows):,} images. {dict(counts)}", flush=True)
    write_csv(output.parent / "cache_failures.csv", ("image_path", "error"), failures)
    if failures:
        raise SystemExit(f"Cache failed for {len(failures)} images. See cache_failures.csv.")


if __name__ == "__main__":
    main()
