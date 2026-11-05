"""Train MambaOut on the full official CXR training split and verify the best epoch."""

import ctypes
import hashlib
import json
import math
import msvcrt
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "全量基线"
sys.path.insert(0, str(PREVIOUS))
from train_full import (ARCHITECTURE, CXRDataset, SMOKE, build_transform,
                        load_checkpoint, load_pretrained, save_checkpoint, save_json, seed)
from dataset.evaluate import evaluate_predictions

ARGS = SimpleNamespace(batch_size=12, workers=0)
IMAGE_SIZE = 224
TRAIN_IMAGES, TRAIN_STUDIES = 30344, 18513
VAL_IMAGES, VAL_STUDIES = 1422, 859
SEED = 42
PROGRESS = ROOT / "status.json"


def status(**fields):
    save_json(PROGRESS, {"pid": os.getpid(), "time": datetime.now().astimezone().isoformat(), **fields})


def cosine_lr(epoch, epochs, first, last, hold):
    fraction = max(0, epoch - hold) / (epochs - hold)
    return last + (first - last) * (1 + math.cos(math.pi * fraction)) / 2


def scores_pair(metrics):
    study = metrics["study"]
    return study["macro_auc"], study["macro_f1"]


@torch.inference_mode()
def validate(model, data, keep_predictions=False):
    model.eval()
    loader = DataLoader(data, batch_size=ARGS.batch_size, pin_memory=True, shuffle=False)
    probabilities, targets, studies, subjects = [], [], [], []
    for batch in loader:
        images = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        targets.append(batch["target"].numpy())
        studies.extend(batch["study_id"])
        subjects.extend(batch["subject_id"])
    targets, probabilities = np.concatenate(targets), np.concatenate(probabilities)
    metrics = evaluate_predictions(targets, probabilities, studies)
    if metrics["study"]["count"] != VAL_STUDIES or len(studies) != VAL_IMAGES:
        raise ValueError("Unexpected number of validation images or studies")
    if metrics["study"]["auc_classes"] != 10:
        raise ValueError("Validation must contain both outcomes for every class")
    metrics = {
        level: {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in values.items()}
        for level, values in metrics.items()
    }
    if keep_predictions:
        groups = defaultdict(list)
        for index, study in enumerate(studies):
            groups[study].append(index)
        study_targets, study_probabilities, study_subjects = [], [], []
        for indices in groups.values():
            subject = subjects[indices[0]]
            if any(subjects[index] != subject for index in indices):
                raise ValueError("Study ID shared by different patients")
            study_targets.append(targets[indices[0]])
            study_probabilities.append(probabilities[indices].mean(axis=0))
            study_subjects.append(subject)
        return metrics, (np.stack(study_targets), np.stack(study_probabilities), np.array(study_subjects))
    return metrics


def train_epoch(model, data, optimizer, scaler, settings, epoch, completed, best_epoch, best_score):
    seed(SEED + epoch)
    batches = DataLoader(data, batch_size=ARGS.batch_size, shuffle=True, pin_memory=True,
                         generator=torch.Generator().manual_seed(SEED + epoch))
    model.train()
    count, running_loss = 0, 0.0
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for step, batch in enumerate(batches, 1):
        images = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
        targets = batch["target"].to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"{settings['name']} epoch {epoch}, step {step}: {value}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count += len(images)
        running_loss += value * len(images)
        if step == 1 or step % 250 == 0:
            status(state="training", stage=settings["name"], completed=completed, epoch=epoch,
                   images=count, total_images=TRAIN_IMAGES, best_epoch=best_epoch, best_score=best_score)
            print(f"{settings['name']} epoch {epoch}/{settings['epochs']} "
                  f"images={count}/{len(data)} loss={running_loss/count:.5f}", flush=True)
    torch.cuda.synchronize()
    if count != TRAIN_IMAGES:
        raise ValueError(f"Expected {TRAIN_IMAGES} train images, processed {count}")
    return {"epoch": epoch, "images": count, "train_loss": running_loss / count,
            "train_seconds": perf_counter() - start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}


def save_best(path, weights, settings, epoch):
    save_checkpoint(path, {"state_dict": weights, "training": {**settings, "best_epoch": epoch}})


def run_stage(settings, train, val, completed):
    stage = ROOT / settings["name"]
    stage.mkdir(exist_ok=True)
    last, best = stage / "last_training.pth", stage / "best.pth"
    history, best_epoch, best_score = [], None, None
    seed(SEED)
    if last.is_file():
        model, _ = load_checkpoint(last)
    else:
        model = load_pretrained()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["first_lr"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["training"].get("settings") != settings:
            raise ValueError(f"Training settings changed for {stage}")
        history, best_epoch, best_score = state["history"], state["best_epoch"], tuple(state["best_score"])
        saved_epoch = None
        if best.is_file():
            saved_best = torch.load(best, map_location="cpu", weights_only=True, mmap=True)
            saved_epoch = saved_best["training"]["best_epoch"]
            del saved_best
        if saved_epoch != best_epoch:
            if best_epoch != len(history):
                raise ValueError(f"Best checkpoint does not match the saved training state for {stage}")
            # The last checkpoint is committed first so a crash between writes is recoverable.
            save_best(best, state["state_dict"], settings, best_epoch)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        print(f"RESUME {stage.name} from epoch {len(history)}", flush=True)
        del state
    completed[stage.name] = len(history)
    for epoch in range(len(history) + 1, settings["epochs"] + 1):
        lr = cosine_lr(epoch, settings["epochs"], settings["first_lr"], settings["last_lr"], settings["hold"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        record = train_epoch(model, train, optimizer, scaler, settings, epoch, completed, best_epoch, best_score)
        status(state="validating", stage=stage.name, completed=completed, epoch=epoch)
        start = perf_counter()
        metrics = validate(model, val)
        score = scores_pair(metrics)
        record.update(learning_rate=lr, validation=metrics, validation_seconds=perf_counter() - start)
        history.append(record)
        improved = best_score is None or score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        weights = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        save_checkpoint(last, {"state_dict": weights, "optimizer": optimizer.state_dict(),
                               "scaler": scaler.state_dict(), "training": {"architecture": ARCHITECTURE,
                               "settings": settings}, "history": history, "best_epoch": best_epoch,
                               "best_score": list(best_score)})
        if improved:
            save_best(best, weights, settings, best_epoch)
        completed[stage.name] = epoch
        save_json(stage / "history.json", history)
        status(state="checkpointed", stage=stage.name, completed=completed, epoch=epoch,
               best_epoch=best_epoch, best_score=best_score, latest_score=score)
        print(f"EPOCH_DONE {stage.name} {epoch}/{settings['epochs']} AUC={score[0]:.8f} "
              f"F1={score[1]:.8f} best_epoch={best_epoch} "
              f"train_s={record['train_seconds']:.1f} val_s={record['validation_seconds']:.1f}", flush=True)
    del model, optimizer, scaler
    model, info = load_checkpoint(best)
    verified = validate(model, val)
    if info["best_epoch"] != best_epoch or not np.allclose(scores_pair(verified), best_score, atol=1e-6, rtol=0):
        raise ValueError(f"Reloaded best checkpoint does not reproduce results for {stage}")
    result = {"settings": settings, "best_epoch": best_epoch, "validation": verified,
              "best_checkpoint": str(best), "checkpoint_bytes": best.stat().st_size,
              "training_seconds": sum(record["train_seconds"] for record in history),
              "validation_seconds": sum(record["validation_seconds"] for record in history),
              "last_epoch": history[-1]}
    save_json(stage / "results.json", result)
    return result


def fit_gates(targets, probabilities):
    candidates = np.arange(1, 40) / 40
    by_preference = sorted(candidates, key=lambda gate: (abs(gate - 0.5), gate))
    f1s = [f1_score(targets, probabilities >= gate, average=None, zero_division=0) for gate in by_preference]
    return np.array(by_preference)[np.argmax(f1s, axis=0)]


def evaluate_gates(predictions):
    targets, probabilities, patients = predictions
    decisions = np.zeros_like(targets)
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit, held_out in folds.split(probabilities, targets, patients):
        decisions[held_out] = probabilities[held_out] >= fit_gates(targets[fit], probabilities[fit])
    f1 = f1_score(targets, decisions, average=None, zero_division=0)
    return {"oof_macro_f1": float(f1.mean()), "oof_f1_by_class_id": f1.tolist(),
            "final_gates_by_class_id": fit_gates(targets, probabilities).tolist(),
            "patients": int(len(np.unique(patients)))}


def write_report(result):
    gates = result["gates"]
    lines = [
        "# MambaOut-Tiny 100 轮全量训练", "",
        f"训练设备为 {result['gpu']}。主实验从 ImageNet-1K 权重开始，全量训练 100 轮。每轮处理 30,344 张训练图。",
        "用 1,422 张验证图对应的 859 次检查选模型。同次检查的图像概率取平均，先按 Study Macro AUC、再按固定阈值 0.5 的 Macro F1 选权重。",
        "训练集用于更新参数，官方验证集用于选轮次和校准阈值，测试集留作官方评测。",
        "", "| 训练方案 | 训练轮数 | 最佳轮次 | Study Macro AUC | Macro F1（0.5） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in result["candidates"].items():
        auc, f1 = scores_pair(entry["validation"])
        lines.append(f"| {name} | {entry['settings']['epochs']} | {entry['best_epoch']} | {auc:.6f} | {f1:.6f} |")
    lines += [
        "", f"最终选择 **{result['winner']}**，其权重已重新加载验证。",
        f"100 轮实验最佳 AUC 相对历史最佳变化 {result['auc_change_vs_previous'] * 100:+.3f} 个百分点。",
        f"最终模型的逐类阈值在按患者分组的 5 折验证中得到 Macro F1 {gates['oof_macro_f1']:.6f}。",
        "逐类阈值只用于生成分类决策，概率和 AUC 保持相同。最优轮次在验证集上筛选，官方测试成绩待评测。",
        "", "| 类别 ID | Study AUC | 默认 F1 | 分折阈值 F1 | 全验证集拟合 Gate |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    final = result["candidates"][result["winner"]]["validation"]["study"]
    for i, (auc, f1, oof, gate) in enumerate(zip(final["auc"], final["f1"], gates["oof_f1_by_class_id"],
                                               gates["final_gates_by_class_id"]), 1):
        lines.append(f"| {i} | {auc:.6f} | {f1:.6f} | {oof:.6f} | {gate:.3f} |")
    (ROOT / "结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(completed):
    if ROOT.drive.upper() != "E:" or not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA and an E-drive output directory")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = yaml.safe_load((SMOKE / "data_local.yaml").read_text(encoding="utf-8"))
    train = CXRDataset(cfg["manifest"], cfg["data_root"], "train", build_transform(True, IMAGE_SIZE), cfg["cache_dir"])
    val = CXRDataset(cfg["manifest"], cfg["data_root"], "val", build_transform(False, IMAGE_SIZE))
    if (len(train), len({row["study_id"] for row in train.rows})) != (TRAIN_IMAGES, TRAIN_STUDIES):
        raise ValueError("Official training split size changed")
    if {row["subject_id"] for row in train.rows} & {row["subject_id"] for row in val.rows}:
        raise ValueError("A patient appears in both train and validation")
    with Path(cfg["manifest"]).open("rb") as handle:
        manifest_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    common = {"architecture": ARCHITECTURE, "img_size": IMAGE_SIZE, "batch_size": ARGS.batch_size,
              "seed": SEED, "train_images": TRAIN_IMAGES, "weight_decay": 0.01, "manifest_sha256": manifest_hash}
    long_run = run_stage({**common, "name": "baseline_100", "loss": "BCE", "epochs": 100,
                          "first_lr": 1e-4, "last_lr": 1e-6, "hold": 3}, train, val, completed)

    previous = json.loads((PREVIOUS / "results.json").read_text(encoding="utf-8"))
    candidates = {
        "历史 5 轮 BCE": {"settings": previous["settings"], "validation": previous["best_validation"],
                         "best_epoch": previous["best_epoch"],
                         "best_checkpoint": str(PREVIOUS / "checkpoints/best.pth")},
        "100 轮 BCE": long_run,
    }
    winner = max(candidates, key=lambda name: scores_pair(candidates[name]["validation"]))
    best_path = Path(candidates[winner]["best_checkpoint"])
    model, _ = load_checkpoint(best_path)
    status(state="calibrating", stage="final", completed=completed, winner=winner)
    final_metrics, predictions = validate(model, val, keep_predictions=True)
    if not np.allclose(scores_pair(final_metrics), scores_pair(candidates[winner]["validation"]), atol=1e-6, rtol=0):
        raise ValueError("Final checkpoint failed score verification")
    gates = evaluate_gates(predictions)
    temporary = ROOT / "best.tmp"
    shutil.copyfile(best_path, temporary)
    temporary.replace(ROOT / "best.pth")
    save_json(ROOT / "gates.json", gates)
    result = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "candidates": candidates, "winner": winner, "gates": gates,
              "auc_change_vs_previous": scores_pair(long_run["validation"])[0] -
                                       scores_pair(previous["best_validation"])[0]}
    save_json(ROOT / "results.json", result)
    write_report(result)
    status(state="complete", stage="final", completed=completed, winner=winner,
           best_score=scores_pair(final_metrics), oof_macro_f1=gates["oof_macro_f1"])
    print(f"COMPLETE {winner} AUC={scores_pair(final_metrics)[0]:.8f}", flush=True)


if __name__ == "__main__":
    completed = {}
    with (ROOT / "training.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        # Keep the system awake while training. This resets when the process exits.
        execution_state = ctypes.windll.kernel32.SetThreadExecutionState
        awake = execution_state(0x80000001)
        print(f"Training process {os.getpid()}; idle sleep prevention active={bool(awake)}", flush=True)
        try:
            status(state="starting", completed=completed)
            main(completed)
        except Exception as error:
            status(state="failed", completed=completed, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if awake:
                execution_state(0x80000000)