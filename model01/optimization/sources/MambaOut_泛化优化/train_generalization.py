"""Local, patient-disjoint MambaOut comparison with early stopping."""
import csv
import ctypes
import gc
import hashlib
import json
import math
import msvcrt
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT.parent / "本地跑通"
sys.path.insert(0, str(SMOKE))
from dataset import CXRDataset, build_transform
from patient_split import make_patient_split
from experiment_metrics import evaluate_model, paired_auc_interval

SEED, SPLIT_SEED = 42, 20261104
MAX_EPOCHS, PATIENCE, MIN_DELTA = 12, 2, 0.001
BATCH_SIZE, IMAGE_SIZE = 12, 224
ARCHITECTURE = "mambaout_tiny"
RECIPES = {
    "baseline": {"backbone_lr": 1e-4, "classifier_lr": 1e-4,
                 "weight_decay": 0.01, "drop_rate": 0.0, "drop_path_rate": 0.0},
    "regularized": {"backbone_lr": 3e-5, "classifier_lr": 3e-4,
                    "weight_decay": 0.05, "drop_rate": 0.2, "drop_path_rate": 0.1},
}
PROGRESS = ROOT / "status.json"


def replace_file(temporary, destination):
    for attempt in range(10):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_file(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    replace_file(temporary, path)


def status(**fields):
    save_json(PROGRESS, {"pid": os.getpid(), "time": datetime.now().astimezone().isoformat(), **fields})


def seed(value):
    import random
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def score(metrics):
    return metrics["study"]["macro_auc"], metrics["study"]["macro_f1"]


def build_model(settings, state_dict=None):
    seed(SEED)
    model = timm.create_model(
        ARCHITECTURE, pretrained=False, num_classes=10 if state_dict is not None else 1000,
        drop_rate=settings["drop_rate"], drop_path_rate=settings["drop_path_rate"],
    )
    if state_dict is None:
        model.load_state_dict(load_file(SMOKE / "pretrain_model/model.safetensors"), strict=True)
        model.reset_classifier(10)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model.to(device="cuda", memory_format=torch.channels_last)


def make_optimizer(model, settings):
    classifier = list(model.head.fc.parameters())
    classifier_ids = {id(parameter) for parameter in classifier}
    pretrained = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
    return torch.optim.AdamW(
        [{"params": pretrained, "lr": settings["backbone_lr"]},
         {"params": classifier, "lr": settings["classifier_lr"]}],
        weight_decay=settings["weight_decay"],
    )


def save_best(path, weights, settings, epoch):
    save_checkpoint(path, {"state_dict": weights,
                          "training": {**settings, "architecture": ARCHITECTURE, "img_size": IMAGE_SIZE,
                                       "best_epoch": epoch}})


def train_epoch(model, dataset, optimizer, scaler, name, epoch, completed_epochs):
    seed(SEED + epoch)
    batches = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
                         generator=torch.Generator().manual_seed(SEED + epoch))
    model.train()
    running, processed = 0.0, 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(batches, 1):
        images = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
        targets = batch["target"].to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite loss in {name} epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        processed += len(images)
        running += value * len(images)
        if step == 1 or step % 250 == 0:
            status(state="training", stage=name, epoch=epoch, max_epochs=MAX_EPOCHS,
                   completed_epochs=completed_epochs, images=processed, total_images=len(dataset))
            print(f"{name} {epoch}/{MAX_EPOCHS} images={processed}/{len(dataset)} BCE={running/processed:.6f}",
                  flush=True)
    torch.cuda.synchronize()
    if processed != len(dataset):
        raise ValueError("Training did not cover every selected image")
    return {"epoch": epoch, "train_images": processed, "train_bce": running / processed,
            "train_seconds": time.perf_counter() - start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}


def run_stage(name, recipe, train_data, early_data, split_hash, manifest_hash, completed):
    stage = ROOT / name
    stage.mkdir(exist_ok=True)
    settings = {**recipe, "name": name, "architecture": ARCHITECTURE,
                "img_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "data_workers": 0,
                "seed": SEED, "split_seed": SPLIT_SEED, "split_hash": split_hash,
                "manifest_hash": manifest_hash, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "min_delta": MIN_DELTA, "final_lr_ratio": 0.1, "train_images": len(train_data)}
    last, best = stage / "last_training.pth", stage / "best.pth"
    history, best_epoch, best_score = [], 0, None
    stale, stop_reference = 0, None
    state = None
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError(f"Resume settings changed for {name}")
        model = build_model(recipe, state["state_dict"])
    else:
        model = build_model(recipe)
    optimizer = make_optimizer(model, recipe)
    scaler = torch.amp.GradScaler("cuda")
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_epoch = state["history"], state["best_epoch"]
        best_score = tuple(state["best_score"])
        stale, stop_reference = state["stale"], state["stop_reference"]
        recorded_best = None
        if best.is_file():
            best_state = torch.load(best, map_location="cpu", weights_only=True, mmap=True)
            recorded_best = best_state["training"]["best_epoch"]
            del best_state
        if recorded_best != best_epoch:
            if best_epoch != len(history):
                raise ValueError(f"Missing or inconsistent best checkpoint for {name}")
            save_best(best, state["state_dict"], settings, best_epoch)
        print(f"RESUME {name} from epoch {len(history)}", flush=True)
        del state
    completed[name] = len(history)
    for epoch in range(len(history) + 1, MAX_EPOCHS + 1):
        if stale >= PATIENCE:
            break
        factor = 0.1 + 0.9 * (1 + math.cos(math.pi * (epoch - 1) / (MAX_EPOCHS - 1))) / 2
        optimizer.param_groups[0]["lr"] = recipe["backbone_lr"] * factor
        optimizer.param_groups[1]["lr"] = recipe["classifier_lr"] * factor
        record = train_epoch(model, train_data, optimizer, scaler, name, epoch, completed)
        status(state="early_stop_validation", stage=name, epoch=epoch, completed_epochs=completed)
        start = time.perf_counter()
        validation = evaluate_model(model, early_data, BATCH_SIZE)
        record.update(early_validation=validation["metrics"], early_bce=validation["bce"],
                      validation_seconds=time.perf_counter() - start,
                      learning_rates=[group["lr"] for group in optimizer.param_groups])
        if validation["metrics"]["image"]["count"] != 4571 or validation["metrics"]["study"]["count"] != 2831:
            raise ValueError("Early-stopping cohort size changed")
        current_score = score(validation["metrics"])
        improved = best_score is None or current_score > best_score
        if improved:
            best_score, best_epoch = current_score, epoch
        if stop_reference is None or current_score[0] > stop_reference + MIN_DELTA:
            stop_reference, stale = current_score[0], 0
        else:
            stale += 1
        history.append(record)
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        save_checkpoint(last, {"state_dict": weights, "optimizer": optimizer.state_dict(),
                               "scaler": scaler.state_dict(), "settings": settings, "history": history,
                               "best_score": list(best_score), "best_epoch": best_epoch,
                               "stale": stale, "stop_reference": stop_reference})
        if improved:
            save_best(best, weights, settings, best_epoch)
        save_json(stage / "history.json", history)
        completed[name] = epoch
        status(state="checkpointed", stage=name, epoch=epoch, completed_epochs=completed,
               early_auc=current_score[0], best_early_auc=best_score[0], best_epoch=best_epoch,
               stale=stale, patience=PATIENCE)
        print(f"EPOCH_DONE {name} {epoch}/{MAX_EPOCHS} train_BCE={record['train_bce']:.6f} "
              f"early_BCE={record['early_bce']:.6f} early_AUC={current_score[0]:.7f} "
              f"F1={current_score[1]:.7f} best_epoch={best_epoch} stale={stale}/{PATIENCE}", flush=True)
        if (ROOT / "pause_after_epoch.flag").is_file():
            status(state="paused_after_epoch", stage=name, epoch=epoch, completed_epochs=completed)
            raise SystemExit(0)
        del validation, weights
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    result = {"settings": settings, "epochs_completed": len(history), "best_epoch": best_epoch,
              "best_early_validation": history[best_epoch - 1]["early_validation"],
              "best_early_bce": history[best_epoch - 1]["early_bce"],
              "stop_reason": "early_stopping" if stale >= PATIENCE else "epoch_limit",
              "best_checkpoint": str(best)}
    save_json(stage / "results.json", result)
    return result


def evaluate_checkpoint(stage, dataset, cohort):
    state = torch.load(Path(stage["best_checkpoint"]), map_location="cpu", weights_only=True)
    if state["training"]["best_epoch"] != stage["best_epoch"]:
        raise ValueError("Best checkpoint metadata mismatch")
    model = build_model(stage["settings"], state["state_dict"]).eval()
    del state
    measured = evaluate_model(model, dataset, BATCH_SIZE)
    expected = {"audit": (4454, 2692), "official_validation": (1422, 859)}[cohort]
    if (measured["metrics"]["image"]["count"], measured["metrics"]["study"]["count"]) != expected:
        raise ValueError(f"Unexpected cohort size for {cohort}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return measured


def write_report(result):
    names = {"baseline": "同划分 BCE 基线", "regularized": "小学习率与正则化组合"}
    lines = [
        "# MambaOut 泛化与早停实验", "",
        "两组均从校验过的 ImageNet 权重重新初始化。按患者固定划分，训练 21,319 图，早停验证 4,571 图，最终复核 4,454 图。",
        "训练患者、早停患者、最终复核患者互不重叠。选轮次与选方案只用早停集合，选定后才执行最终复核。",
        "官方验证集在本轮训练结束后评估，其在此前实验中已用于模型开发。", "",
        "| 方案 | 实际轮数 | 最佳轮次 | 早停集 AUC | 最终复核 AUC | 官方验证 AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stage in result["stages"].items():
        lines.append(f"| {names[name]} | {stage['epochs_completed']} | {stage['best_epoch']} | "
                     f"{score(stage['best_early_validation'])[0]:.5f} | "
                     f"{score(stage['audit']['metrics'])[0]:.5f} | "
                     f"{score(stage['official_validation']['metrics'])[0]:.5f} |")
    difference = result["paired_audit_auc"]
    lines += [
        "", f"按早停集预先选定的方案为 {names[result['selected_stage']]}。",
        f"正则化组合相对基线的最终复核 AUC 差为 {difference['delta']*100:+.3f} 个百分点。",
        f"患者配对 bootstrap 95% 区间为 [{difference['ci95'][0]*100:+.3f}, {difference['ci95'][1]*100:+.3f}] 个百分点。",
        "该区间只反映当前留出患者与固定权重下的不确定性。一次随机种子的实验用于筛选训练策略。",
        "旧全量模型见过本实验的留出患者，原第 3 轮权重继续保留为历史参照。",
        "本次训练样本量为官方训练图的约 70%，与旧全量训练的样本量不同。", "",
        "| 类别 ID | 最终复核阳性 Study | 基线 AUC | 正则化 AUC |",
        "| ---: | ---: | ---: | ---: |",
    ]
    positive = result["split_summary"]["partitions"]["audit"]["study_positive_counts"]
    for index in range(10):
        base = result["stages"]["baseline"]["audit"]["metrics"]["study"]["auc"][index]
        candidate = result["stages"]["regularized"]["audit"]["metrics"]["study"]["auc"][index]
        lines.append(f"| {index+1} | {positive[str(index+1)]} | {base:.5f} | {candidate:.5f} |")
    (ROOT / "结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if ROOT.drive.upper() != "E:" or not torch.cuda.is_available():
        raise RuntimeError("Expected the E-drive CUDA environment")
    if (ROOT / "results.json").is_file():
        existing = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
        write_report(existing)
        status(state="complete", stage="final", selected_stage=existing["selected_stage"],
               completed_epochs={name: value["epochs_completed"] for name, value in existing["stages"].items()})
        print("Experiment already complete. Read results.json and 结果.md.", flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = yaml.safe_load((SMOKE / "data_local.yaml").read_text(encoding="utf-8"))
    with Path(cfg["manifest"]).open("rb") as handle:
        manifest_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = json.loads((SMOKE / "pretrain_model/weight_metadata.json").read_text(encoding="utf-8"))
    with (SMOKE / "pretrain_model/model.safetensors").open("rb") as handle:
        weight_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if weight_hash != metadata["weight_sha256"]:
        raise ValueError("ImageNet checkpoint hash mismatch")
    training = CXRDataset(cfg["manifest"], cfg["data_root"], "train", build_transform(True, IMAGE_SIZE), cfg["cache_dir"])
    cached_evaluation = CXRDataset(cfg["manifest"], cfg["data_root"], "train", build_transform(False, IMAGE_SIZE), cfg["cache_dir"])
    raw_evaluation = CXRDataset(cfg["manifest"], cfg["data_root"], "train", build_transform(False, IMAGE_SIZE))
    official_validation = CXRDataset(cfg["manifest"], cfg["data_root"], "val", build_transform(False, IMAGE_SIZE))
    indices, summary = make_patient_split(training.rows, SPLIT_SEED)
    if summary["assignment_sha256"] != "564a4a846c17474d2fbff17f899018113394029752045e7c837f6589b09949b5":
        raise ValueError("Fixed patient split changed")
    if {row["subject_id"] for row in training.rows} & {row["subject_id"] for row in official_validation.rows}:
        raise ValueError("Training and official-validation patient overlap")
    save_json(ROOT / "patient_split.json", summary)
    save_json(ROOT / "experiment_config.json", {"recipes": RECIPES, "max_epochs": MAX_EPOCHS,
                                               "patience": PATIENCE, "min_delta": MIN_DELTA,
                                               "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
                                               "manifest_sha256": manifest_hash,
                                               "pretrain_sha256": weight_hash,
                                               "early_validation_images": "cached_512",
                                               "final_audit_images": "original_raw"})
    train_data = Subset(training, indices["train"])
    early_data = Subset(cached_evaluation, indices["early_stop"])
    audit_data = Subset(raw_evaluation, indices["audit"])
    completed, stages = {}, {}
    for name, recipe in RECIPES.items():
        stages[name] = run_stage(name, recipe, train_data, early_data,
                                 summary["assignment_sha256"], manifest_hash, completed)
    selected = max(stages, key=lambda name: score(stages[name]["best_early_validation"]))
    selection = {"selected_stage": selected, "selection_cohort": "early_stop",
                 "selected_before_final_audit": True,
                 "early_scores": {name: score(stage["best_early_validation"]) for name, stage in stages.items()}}
    save_json(ROOT / "selection_before_audit.json", selection)
    predictions = {}
    for name, stage in stages.items():
        status(state="final_audit", stage=name, completed_epochs=completed, selected_stage=selected)
        audit = evaluate_checkpoint(stage, audit_data, "audit")
        status(state="official_validation", stage=name, completed_epochs=completed, selected_stage=selected)
        official = evaluate_checkpoint(stage, official_validation, "official_validation")
        stage["audit"] = {"metrics": audit["metrics"], "bce": audit["bce"]}
        stage["official_validation"] = {"metrics": official["metrics"], "bce": official["bce"]}
        predictions[name] = audit
        save_json(ROOT / name / "results.json", stage)
    baseline, regularized = predictions["baseline"], predictions["regularized"]
    if baseline["study_ids"] != regularized["study_ids"] or not np.array_equal(
            baseline["study_targets"], regularized["study_targets"]):
        raise ValueError("Paired audit predictions are not aligned")
    interval = paired_auc_interval(baseline["study_targets"], baseline["study_probabilities"],
                                   regularized["study_probabilities"], baseline["study_subjects"], SPLIT_SEED, 400)
    if interval["valid_repeats"] < 300:
        raise ValueError("Insufficient valid patient bootstrap replicates")
    selected_checkpoint = Path(stages[selected]["best_checkpoint"])
    temporary = ROOT / "selected_model.tmp"
    shutil.copyfile(selected_checkpoint, temporary)
    replace_file(temporary, ROOT / "selected_model.pth")
    public_summary = {key: value for key, value in summary.items() if key != "assignments"}
    result = {"stages": stages, "selected_stage": selected, "selection": selection,
              "split_summary": public_summary, "paired_audit_auc": interval,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    save_json(ROOT / "results.json", result)
    write_report(result)
    status(state="complete", stage="final", completed_epochs=completed, selected_stage=selected,
           audit_auc_delta=interval["delta"], audit_auc_ci95=interval["ci95"])
    print(f"COMPLETE selected={selected} independent_audit_AUC_delta={interval['delta']:+.7f}", flush=True)


if __name__ == "__main__":
    with (ROOT / "training.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            status(state="starting", stage="initialization")
            main()
        except Exception as error:
            status(state="failed", stage="exception", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)