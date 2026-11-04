"""Local-only MambaOut training, evaluation, example submission and timing."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import timm
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from dataset import CXRDataset, build_transform
from dataset.evaluate import evaluate_predictions, show_metrics
from dataset.tools.build_manifest import write_csv


ROOT = Path(__file__).resolve().parent
CLASSES = 10
ARCHITECTURE = "mambaout_tiny"
SUBMISSION_FIELDS = ("Subject_id", "Study_id", "Predict_class", "Probability", "Gate")


def loader(dataset, args, shuffle=False):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )


def load_pretrained():
    weights = ROOT / "pretrain_model/model.safetensors"
    metadata = json.loads((weights.parent / "weight_metadata.json").read_text(encoding="utf-8"))
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata["weight_sha256"]:
        raise ValueError("ImageNet weight SHA-256 does not match the recorded source")
    model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(weights), strict=True)
    model.reset_classifier(CLASSES)
    print(f"Loaded verified ImageNet-1K weights: {digest}", flush=True)
    return model.to(device="cuda", memory_format=torch.channels_last)


def train(cfg, args, checkpoint):
    data = CXRDataset(cfg["manifest"], cfg["data_root"], "train", build_transform(True, args.img_size), cfg["cache_dir"])
    if not 0 < args.subset <= len(data):
        raise ValueError("Training subset must fit the official train split")
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed))[:args.subset].tolist()
    batches = loader(Subset(data, indices), args, shuffle=True)
    model = load_pretrained()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        shape = tuple(model(torch.zeros(2, 3, args.img_size, args.img_size, device="cuda")).shape)
    if shape != (2, CLASSES):
        raise ValueError(f"Unexpected output shape: {shape}")
    print(f"Model forward: {shape}. Training {len(indices):,} images for {args.epochs} epochs.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    loss_function = torch.nn.BCEWithLogitsLoss()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, images = 0.0, 0
        for step, batch in enumerate(batches, 1):
            x = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
            target = batch["target"].to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_function(model(x), target)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise ValueError(f"Non-finite training loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss_value * len(x)
            images += len(x)
            if step % 25 == 0:
                print(f"Epoch {epoch}/{args.epochs}, {images:,}/{len(indices):,} images, loss={running/images:.5f}", flush=True)
        if images != len(indices):
            raise ValueError("Training epoch did not cover the full sample")
        losses.append(running / images)
        print(f"Epoch {epoch} mean BCE = {losses[-1]:.6f}", flush=True)
    torch.cuda.synchronize()
    training = {
        "architecture": ARCHITECTURE, "subset": len(indices), "epochs": args.epochs, "seed": args.seed,
        "img_size": args.img_size, "batch_size": args.batch_size, "learning_rate": 1e-4,
        "losses": losses, "seconds": perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "training": training}, temporary)
    temporary.replace(checkpoint)
    print(f"Checkpoint saved: {checkpoint} ({checkpoint.stat().st_size:,} bytes)", flush=True)


def load_checkpoint(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info = state["training"]
    if info["architecture"] != ARCHITECTURE:
        raise ValueError("Unexpected checkpoint architecture")
    model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=CLASSES)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device="cuda", memory_format=torch.channels_last).eval(), info


@torch.inference_mode()
def evaluate(model, cfg, args, img_size):
    data = CXRDataset(cfg["manifest"], cfg["data_root"], "val", build_transform(False, img_size))
    targets, probabilities, studies = [], [], []
    for step, batch in enumerate(loader(data, args), 1):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last))
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        targets.append(batch["target"].numpy())
        studies.extend(batch["study_id"])
        if step % 30 == 0:
            print(f"Validation: {len(studies):,}/{len(data):,} images", flush=True)
    metrics = evaluate_predictions(np.concatenate(targets), np.concatenate(probabilities), studies, threshold=0.5)
    show_metrics(metrics, 0.5)
    return {level: {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in scores.items()} for level, scores in metrics.items()}


def test_sample(cfg):
    data = CXRDataset(cfg["manifest"], cfg["data_root"], "test")
    grouped = defaultdict(list)
    for row in data.rows:
        grouped[(row["subject_id"], row["study_id"])].append(row)
    selected = []
    for study in sorted(grouped):
        views = grouped[study]
        if len(selected) + len(views) <= 100:
            selected.extend(views)
        if len(selected) == 100:
            break
    chosen = {r["image_path"] for r in selected}
    warmup = [r for r in data.rows if r["image_path"] not in chosen][:5]
    if len(selected) != 100 or len(warmup) != 5:
        raise ValueError("Could not assemble a 100-image set of complete studies and five warmup images")
    return data.data_root, selected, warmup


@torch.inference_mode()
def infer_images(model, root, rows, transform):
    probabilities, decode, forward = [], 0.0, 0.0
    for row in rows:
        begin = perf_counter()
        with Image.open(root / row["image_path"]) as image:
            x = transform(image).unsqueeze(0)
        prepared = perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(x.to("cuda", memory_format=torch.channels_last))
        probabilities.append(prediction.float().sigmoid().cpu().numpy()[0])
        end = perf_counter()
        decode += prepared - begin
        forward += end - prepared
    return np.stack(probabilities), decode, forward


def make_submission(rows, probabilities):
    grouped = defaultdict(list)
    if probabilities.shape != (len(rows), CLASSES):
        raise ValueError("Expected ten predictions per image")
    for row, values in zip(rows, probabilities):
        grouped[(row["subject_id"], row["study_id"])].append(values)
    return [
        {"Subject_id": subject, "Study_id": study, "Predict_class": label,
         "Probability": f"{value:.8f}", "Gate": ""}
        for (subject, study), predictions in sorted(grouped.items())
        for label, value in enumerate(np.mean(predictions, axis=0), 1)
    ]


def validate_submission(path, images):
    expected = {(row["subject_id"], row["study_id"], label)
                for row in images for label in range(1, CLASSES + 1)}
    values = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or []) != SUBMISSION_FIELDS:
            raise ValueError("Unexpected submission columns")
        for row in reader:
            key = row["Subject_id"], row["Study_id"], int(row["Predict_class"])
            value = float(row["Probability"])
            if key in values or key not in expected or not math.isfinite(value) or not 0 <= value <= 1 or row["Gate"]:
                raise ValueError("Invalid submission row")
            values[key] = value
    if values.keys() != expected:
        raise ValueError("Missing study/class predictions")
    return values


def benchmark(model, cfg, train_size, output):
    root, rows, warmup = test_sample(cfg)
    report = {"images": len(rows), "warmup_images": len(warmup), "rounds": 3}
    reference = None
    for size in dict.fromkeys((train_size, 224, 384)):
        transform = build_transform(False, size)
        infer_images(model, root, warmup, transform)
        trials = []
        for trial in range(3):
            begin = perf_counter()
            predictions, decode, forward = infer_images(model, root, rows, transform)
            destination = output / ("submission.csv" if size == train_size and trial == 0 else "timing_output.csv")
            result = make_submission(rows, predictions)
            write_csv(destination, SUBMISSION_FIELDS, result)
            elapsed = perf_counter() - begin
            saved = validate_submission(destination, rows)
            if size == train_size:
                if reference is None:
                    reference, reference_images = saved, predictions
                else:
                    if any(abs(saved[key] - reference[key]) > 1e-6 for key in reference):
                        raise ValueError("Repeated inference differs from submission.csv")
                    mismatched = int(np.count_nonzero(np.any(np.abs(predictions - reference_images) > 1e-6, axis=1)))
                    report["mismatched_images"] = max(report.get("mismatched_images", 0), mismatched)
            trials.append([decode * 1000 / len(rows), forward * 1000 / len(rows), elapsed * 1000 / len(rows)])
        averages = np.asarray(trials).mean(axis=0)
        report[str(size)] = dict(zip(("decode_and_transform_ms", "forward_and_transfer_ms", "end_to_end_ms"), averages.tolist()))
        print(f"{size}: {averages[0]:.2f} ms decode/preprocess, {averages[1]:.2f} ms forward, {averages[2]:.2f} ms end-to-end", flush=True)
    (output / "timing_output.csv").unlink(missing_ok=True)
    if report.get("mismatched_images") != 0:
        raise ValueError("Image predictions were not consistent across repeated inference")
    report["studies"] = len(reference) // CLASSES
    report["submission_rows"] = len(reference)
    print(f"Local submission.csv: {report['studies']} complete studies, {len(reference)} rows, 0 inconsistent images.", flush=True)
    return report


def write_report(path, result):
    train, val, timing = result["training"], result["validation"], result["timing"]
    lines = [
        "# 胸片分类本地跑通结果", "",
        f"设备为 {result['gpu']}。使用 ImageNet-1K 预训练 MambaOut-Tiny，在 {train['subset']:,} 张训练图上跑 {train['epochs']} 个 epoch。",
        f"输入 {train['img_size']}，batch {train['batch_size']}，混合精度。训练平均 BCE 为 " + " → ".join(f"{v:.5f}" for v in train["losses"]) + "。",
        f"训练耗时 {train['seconds']:.1f} 秒。Checkpoint 大小 {result['checkpoint_bytes'] / 1e6:.2f} MB。",
        "", "| 验证口径 | 样本数 | Macro AUC | Macro F1（阈值 0.5） |",
        "| --- | ---: | ---: | ---: |",
        f"| 逐图 | {val['image']['count']} | {val['image']['macro_auc']:.4f} | {val['image']['macro_f1']:.4f} |",
        f"| Study | {val['study']['count']} | {val['study']['macro_auc']:.4f} | {val['study']['macro_f1']:.4f} |", "",
        "| 输入 | 读图和预处理 ms/张 | 前向和传输 ms/张 | 端到端 ms/张 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for size in (224, 384):
        stats = timing[str(size)]
        lines.append(f"| {size} | {stats['decode_and_transform_ms']:.2f} | {stats['forward_and_transfer_ms']:.2f} | {stats['end_to_end_ms']:.2f} |")
    lines += ["", f"计时使用本机显卡，5 张预热后对 100 张测试图重复 3 轮。端到端包含按 Study 聚合并写出 CSV。",
              f"本地样例 submission.csv 包含 {timing['studies']} 次完整检查、{timing['submission_rows']} 行，重复推理不一致 {timing['mismatched_images']} 张。",
              "该 CSV 用于本地格式和一致性验收。正式比赛包需覆盖全部测试检查，延迟以赛方 T4 实测为准。", "",
              "| 类别 | 逐图 AUC | Study AUC |", "| --- | ---: | ---: |"]
    with (ROOT / "class_mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["class_id"]) - 1
            lines.append(f"| {row['chinese_name']} | {val['image']['auc'][index]:.4f} | {val['study']['auc'][index]:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or ROOT.drive.upper() != "E:":
        raise RuntimeError("The local smoke test expects the E-drive workspace and CUDA runtime")
    cfg = yaml.safe_load((ROOT / "data_local.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    ckpt = ROOT / "checkpoints/smoke.pth"
    print(f"CUDA device: {torch.cuda.get_device_name()}. PyTorch {torch.__version__}. timm {timm.__version__}.", flush=True)
    if not args.reuse_checkpoint:
        train(cfg, args, ckpt)
    model, training = load_checkpoint(ckpt)
    print("Checkpoint reloaded with strict state-dict validation.", flush=True)
    validation = evaluate(model, cfg, args, training["img_size"])
    timing = benchmark(model, cfg, training["img_size"], out)
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "timm": timm.__version__,
              "training": training, "validation": validation, "timing": timing, "checkpoint_bytes": ckpt.stat().st_size}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(out / "结果.md", result)
    print(f"Results ready: {out}", flush=True)


if __name__ == "__main__":
    main()
