"""Audit the official split counts, labels and decoded image integrity."""

import argparse
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path
from statistics import median

from PIL import Image

if __package__:
    from .build_manifest import OFFICIAL_SPLITS, build_rows, data_root_path, parse_label_ids, write_csv
else:
    from build_manifest import OFFICIAL_SPLITS, build_rows, data_root_path, parse_label_ids, write_csv


def decode(row, root):
    try:
        with Image.open(root / row["image_path"]) as image:
            image.load()
            return image.size, ""
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return None, str(error)


def dimension_summary(values):
    if not values:
        return "—"
    values = sorted(values)
    return f"{values[0]} / {median(values):g} / {values[math.ceil(0.95 * len(values)) - 1]} / {values[-1]}"


def audit(root, report, bad_images, workers):
    rows = build_rows(root)
    stats = {
        split: {"subjects": set(), "studies": {}, "images": 0, "positives": Counter(),
                "widths": [], "heights": [], "bad": 0, "dimension_mismatch": 0}
        for split in OFFICIAL_SPLITS
    }
    bad = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(partial(decode, root=root), rows)
        for index, (row, (size, error)) in enumerate(zip(rows, results), 1):
            split = row["split"]
            stat = stats[split]
            labels = parse_label_ids(row["label_ids"])
            stat["subjects"].add(row["subject_id"])
            stat["studies"][row["study_id"]] = labels
            stat["images"] += 1
            stat["positives"].update(labels)
            if size is None:
                stat["bad"] += 1
                bad.append({"split": split, "image_path": row["image_path"], "error": error})
            else:
                width, height = size
                stat["widths"].append(width)
                stat["heights"].append(height)
                stat["dimension_mismatch"] += size != (row["width"], row["height"])
            if index % 2000 == 0 or index == len(rows):
                print(f"Decoded {index:,}/{len(rows):,} images, {len(bad)} unreadable", flush=True)
    write_csv(bad_images, ("split", "image_path", "error"), bad)

    lines = ["# 官方胸片数据审计", "", "按官方 CSV 的标识关联磁盘图片，逐张完整解码。验证集目录 `valid` 在 manifest 中记为 `val`。", "",
             "| 划分 | Subject | Study | 图片 | 坏图 | 图片尺寸与 CSV 不一致 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, s in stats.items():
        lines.append(f"| {split} | {len(s['subjects']):,} | {len(s['studies']):,} | {s['images']:,} | {s['bad']:,} | {s['dimension_mismatch']:,} |")
    lines += ["", "## 解码尺寸分布", "", "以下依次为最小值、中位数、P95、最大值，单位为像素。",
              "", "| 划分 | 宽 | 高 |", "| --- | --- | --- |"]
    for split, s in stats.items():
        lines.append(f"| {split} | {dimension_summary(s['widths'])} | {dimension_summary(s['heights'])} |")
    lines += ["", "## 10 类正样本", "", "分别按图片和唯一 Study 计数。测试集的标签留空。",
              "", "| 类别 ID | 训练图片 | 训练 Study | 验证图片 | 验证 Study |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    study_positives = {
        split: Counter(label for labels in stats[split]["studies"].values() for label in labels)
        for split in ("train", "val")
    }
    for label in range(1, 11):
        lines.append(f"| {label} | {stats['train']['positives'][label]:,} | {study_positives['train'][label]:,} | {stats['val']['positives'][label]:,} | {study_positives['val'][label]:,} |")
    lines += ["", "## 跨划分 Subject 重叠", "", "| 划分 | 重叠 Subject 数 |", "| --- | ---: |"]
    for left, right in combinations(stats, 2):
        overlap = len(stats[left]["subjects"] & stats[right]["subjects"])
        lines.append(f"| {left} 与 {right} | {overlap:,} |")
    lines += ["", f"坏图明细写入本地 `{Path(bad_images).name}`，共 {len(bad):,} 条。",
              "测试集仅用于文件完整性和划分审计。训练、阈值选择和模型评估使用有标签的划分。", ""]
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report}. Unreadable images: {len(bad):,}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Official dataset directory (defaults to DATA_ROOT)")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "data_audit.md")
    parser.add_argument("--bad-images", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    audit(data_root_path(args.data_root), args.report, args.bad_images or args.report.with_name("bad_images.csv"), args.workers)


if __name__ == "__main__":
    main()
