"""Compare image-level and mean-pooled study-level multilabel metrics."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import CXRDataset, build_transform, resolve_manifest
from tools.build_manifest import NUM_CLASSES, data_root_path, relative_image_path, split_name


def score(labels, probabilities, threshold):
    auc = np.full(NUM_CLASSES, np.nan)
    for index in range(NUM_CLASSES):
        if len(np.unique(labels[:, index])) == 2:
            auc[index] = roc_auc_score(labels[:, index], probabilities[:, index])
    f1 = f1_score(labels, probabilities >= threshold, average=None, zero_division=0)
    defined = np.isfinite(auc)
    return {"auc": auc, "f1": f1, "macro_auc": float(auc[defined].mean()) if defined.any() else np.nan,
            "macro_f1": float(f1.mean()), "auc_classes": int(defined.sum()), "count": len(labels)}


def evaluate_predictions(labels, probabilities, study_ids, threshold=0.5):
    """Use class-wise mean probabilities before thresholding each study."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    expected = (len(study_ids), NUM_CLASSES)
    if not len(study_ids) or labels.shape != expected or probabilities.shape != expected:
        raise ValueError(f"Labels and probabilities must both have shape {expected} with at least one image")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    grouped = defaultdict(list)
    for index, study in enumerate(study_ids):
        grouped[study].append(index)
    study_labels, study_probs = [], []
    for study, indices in grouped.items():
        target = labels[indices[0]]
        if not (labels[indices] == target).all():
            raise ValueError(f"Inconsistent labels within study {study}")
        study_labels.append(target)
        study_probs.append(probabilities[indices].mean(axis=0))
    return {"image": score(labels, probabilities, threshold),
            "study": score(np.stack(study_labels), np.stack(study_probs), threshold)}


def read_predictions(path, dataset):
    """Read the predict.py long-table format for every image in the chosen split."""
    index_by_image = {row["image_path"]: i for i, row in enumerate(dataset.rows)}
    probabilities = np.full((len(dataset), NUM_CLASSES), np.nan)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {"study_id", "image_path", "predict_class", "probability"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("Prediction CSV needs study_id,image_path,predict_class,probability")
        for line, row in enumerate(reader, 2):
            image = Path(row["image_path"].strip())
            if image.is_absolute():
                image = image.resolve().relative_to(dataset.data_root)
            key = relative_image_path(image.as_posix())
            index = index_by_image.get(key)
            if index is None or row["study_id"].strip() != dataset.rows[index]["study_id"]:
                raise ValueError(f"Prediction row {line} does not match the selected split and study")
            label, value = int(row["predict_class"]) - 1, float(row["probability"])
            if not 0 <= label < NUM_CLASSES or not 0 <= value <= 1:
                raise ValueError(f"Prediction row {line} has an invalid class or probability")
            if not np.isnan(probabilities[index, label]):
                raise ValueError(f"Prediction row {line} repeats an image/class pair")
            probabilities[index, label] = value
    if np.isnan(probabilities).any():
        raise ValueError("Prediction CSV does not cover every image and all ten classes")
    return probabilities


def predict_checkpoint(dataset, checkpoint, cfg, device_name):
    # The teammate-owned model module is needed only for checkpoint inference.
    from model import build_model, load_checkpoint

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(num_classes=NUM_CLASSES, pretrained=False, drop_path=0.0)
    load_checkpoint(model, str(checkpoint))
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 8)),
                        num_workers=int(cfg.get("num_workers", 0)), shuffle=False)
    with torch.inference_mode():
        return np.concatenate([model(batch["image"].to(device)).sigmoid().float().cpu().numpy() for batch in loader])


def show_metrics(metrics, threshold):
    def fmt(value):
        return f"{value:.4f}" if np.isfinite(value) else "N/A"

    image, study = metrics["image"], metrics["study"]
    print(f"Images: {image['count']:,}. Studies: {study['count']:,}. Fixed threshold: {threshold:g}.")
    print("| Class | Image AUC | Study AUC | Image F1 | Study F1 |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for index in range(NUM_CLASSES):
        values = (image["auc"][index], study["auc"][index], image["f1"][index], study["f1"][index])
        print(f"| {index + 1} | {' | '.join(map(fmt, values))} |")
    values = (image["macro_auc"], study["macro_auc"], image["macro_f1"], study["macro_f1"])
    print(f"| Macro | {' | '.join(map(fmt, values))} |")
    if image["auc_classes"] != NUM_CLASSES or study["auc_classes"] != NUM_CLASSES:
        print("AUC is undefined for classes without both labels and omitted from macro AUC. Macro F1 covers all ten classes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", type=Path)
    source.add_argument("--preds", type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "valid", "validation"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", help="Inference device, defaults to CUDA when available")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.ckpt is not None and not args.ckpt.is_file():
        parser.error(f"Checkpoint not found: {args.ckpt}")
    cfg = {}
    config = args.config or Path("configs/base.yaml")
    if args.config is not None or config.is_file():
        from utils import load_config

        cfg = load_config(str(config))
    root = data_root_path(args.data_root or cfg.get("data_root") or os.environ.get("DATA_ROOT"))
    manifest = args.manifest or resolve_manifest(cfg, root)
    dataset = CXRDataset(manifest, root, split_name(args.split), build_transform(False, int(cfg.get("img_size", 384))))
    if args.preds is not None:
        probabilities = read_predictions(args.preds, dataset)
    else:
        probabilities = predict_checkpoint(dataset, args.ckpt, cfg, args.device)
    targets = np.zeros((len(dataset), NUM_CLASSES), dtype=np.uint8)
    for index, labels in enumerate(dataset.labels):
        targets[index, list(labels)] = 1
    study_ids = [row["study_id"] for row in dataset.rows]
    show_metrics(evaluate_predictions(targets, probabilities, study_ids, args.threshold), args.threshold)


if __name__ == "__main__":
    main()
