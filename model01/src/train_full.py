"""Train and compare the full official training split against the local smoke baseline."""

import csv
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

OUT = Path(__file__).resolve().parent
SMOKE = OUT.parent / "本地跑通"
sys.path.insert(0, str(SMOKE))
from run_local import ARCHITECTURE, CXRDataset, build_transform, evaluate, load_checkpoint, load_pretrained


SETTINGS = {"architecture": ARCHITECTURE, "epochs": 5, "seed": 42,
            "img_size": 224, "batch_size": 12, "learning_rate": 1e-4,
            "weight_decay": 0.01, "train_images": 30344}


def save_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def epoch_train(model, dataset, optimizer, scaler, epoch):
    seed(SETTINGS["seed"] + epoch)
    batches = DataLoader(dataset, batch_size=SETTINGS["batch_size"], shuffle=True, num_workers=0,
                         pin_memory=True, generator=torch.Generator().manual_seed(SETTINGS["seed"] + epoch))
    model.train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    running, count = 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for step, batch in enumerate(batches, 1):
        image = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
        target = batch["target"].to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_function(model(image), target)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Non-finite loss at epoch {epoch}, step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss_value * len(image)
        count += len(image)
        if step % 250 == 0:
            print(f"Epoch {epoch}/{SETTINGS['epochs']}: {count:,}/{len(dataset):,} images. BCE {running/count:.5f}.", flush=True)
    torch.cuda.synchronize()
    if count != len(dataset):
        raise ValueError("Training did not cover the entire official split")
    return {"epoch": epoch, "images": count, "train_bce": running/count,
            "train_seconds": perf_counter()-started, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2}


def write_report(result):
    smoke, best, history = result["smoke_baseline"], result["best_validation"], result["history"]
    lines = [
        "# MambaOut-Tiny 全量训练结果", "",
        f"本机 GPU 为 {result['gpu']}。从 ImageNet-1K 公开权重初始化。每轮用全部 30,344 张官方训练图。",
        "输入 224，batch 12，FP16 混合精度，AdamW 1e-4，BCE。每轮用相同的 1,422 张图和 859 次独立验证检查。",
        "", "| 轮次 | 训练 BCE | Study Macro AUC | Study Macro F1（阈值 0.5） | 训练分钟 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in history:
        scores = entry["validation"]["study"]
        lines.append(f"| {entry['epoch']} | {entry['train_bce']:.5f} | {scores['macro_auc']:.4f} | {scores['macro_f1']:.4f} | {entry['train_seconds']/60:.2f} |")
    improvement = best["study"]["macro_auc"] - smoke["study"]["macro_auc"]
    lines += [
        "", f"第 {result['best_epoch']} 轮的 Study Macro AUC 最高，已保存并重新加载 checkpoint。",
        f"Study Macro AUC 从试跑的 {smoke['study']['macro_auc']:.4f} 到 {best['study']['macro_auc']:.4f}，变化 {improvement*100:+.2f} 个百分点。",
        f"Study Macro F1 从 {smoke['study']['macro_f1']:.4f} 到 {best['study']['macro_f1']:.4f}。",
        "两轮全量结果可与两轮小样本结果对照。最佳轮次还包含增加训练轮数的影响。",
        "", "| 类别 | 试跑 Study AUC | 全量最佳 Study AUC |", "| --- | ---: | ---: |",
    ]
    with (SMOKE / "class_mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["class_id"]) - 1
            lines.append(f"| {row['chinese_name']} | {smoke['study']['auc'][index]:.4f} | {best['study']['auc'][index]:.4f} |")
    (OUT / "结果.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    if not torch.cuda.is_available() or OUT.drive.upper() != "E:":
        raise RuntimeError("Expected the CUDA environment and the E-drive output folder")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed(SETTINGS["seed"])
    cfg = yaml.safe_load((SMOKE / "data_local.yaml").read_text(encoding="utf-8"))
    smoke = json.loads((SMOKE / "outputs/metrics.json").read_text(encoding="utf-8"))["validation"]
    train_data = CXRDataset(cfg["manifest"], cfg["data_root"], "train",
                            build_transform(True, SETTINGS["img_size"]), cfg["cache_dir"])
    if len(train_data) != SETTINGS["train_images"]:
        raise ValueError("Official train split size does not match expected count")
    checkpoint = OUT / "checkpoints"
    last, best = checkpoint / "last_training.pth", checkpoint / "best.pth"
    model = load_pretrained()
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    history, best_epoch = [], 0
    best_score = (-math.inf, -math.inf)
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["settings"] != SETTINGS or not best.is_file():
            raise ValueError("Saved training state does not match the current run")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_epoch = state["history"], state["best_epoch"]
        best_score = tuple(state["best_score"])
        del state
    args = SimpleNamespace(batch_size=SETTINGS["batch_size"], workers=0)
    started = perf_counter()
    print(f"Full training on {torch.cuda.get_device_name()}. {len(train_data):,} images per epoch.", flush=True)
    for epoch in range(len(history)+1, SETTINGS["epochs"]+1):
        record = epoch_train(model, train_data, optimizer, scaler, epoch)
        model.eval()
        val_start = perf_counter()
        record["validation"] = evaluate(model, cfg, args, SETTINGS["img_size"])
        record["validation_seconds"] = perf_counter() - val_start
        val = record["validation"]["study"]
        if val["count"] != 859 or record["validation"]["image"]["count"] != 1422:
            raise ValueError("Validation set changed since the smoke baseline")
        history.append(record)
        score = val["macro_auc"], val["macro_f1"]
        weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(best, {"state_dict": weights, "training": {**SETTINGS, "best_epoch": best_epoch}})
        save_checkpoint(last, {"state_dict": weights, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                               "settings": SETTINGS, "history": history, "best_score": list(best_score), "best_epoch": best_epoch})
        save_json(OUT / "history.json", history)
        print(f"EPOCH_DONE {epoch}/{SETTINGS['epochs']} BCE={record['train_bce']:.5f} "
              f"Study_AUC={val['macro_auc']:.5f} Study_F1={val['macro_f1']:.5f} "
              f"train_s={record['train_seconds']:.1f} best_epoch={best_epoch}", flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    model, info = load_checkpoint(best)
    if info["best_epoch"] != best_epoch:
        raise ValueError("Reloaded best checkpoint epoch differs from training history")
    print(f"Reloaded best checkpoint from epoch {best_epoch}. Running final verification.", flush=True)
    verified = evaluate(model, cfg, args, SETTINGS["img_size"])
    if abs(verified["study"]["macro_auc"] - best_score[0]) > 1e-6:
        raise ValueError("Reloaded checkpoint did not reproduce validation AUC")
    result = {"settings": SETTINGS, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "smoke_baseline": smoke, "best_epoch": best_epoch, "best_validation": verified, "history": history,
              "checkpoint_bytes": best.stat().st_size, "current_process_seconds": perf_counter()-started}
    save_json(OUT / "results.json", result)
    write_report(result)
    print(f"Full baseline complete. Best Study AUC {verified['study']['macro_auc']:.5f}.", flush=True)


if __name__ == "__main__":
    main()
