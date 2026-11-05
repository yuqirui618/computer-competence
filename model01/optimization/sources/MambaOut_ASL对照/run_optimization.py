"""Full-data ASL comparison, one evidence-based fallback, and best-model retention."""
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
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
GENERALIZATION = ROOT.parent / "MambaOut_泛化优化"
BCE = ROOT.parent / "MambaOut_100轮"
sys.path.insert(0, str(GENERALIZATION))
import train_generalization as core
sys.path.insert(0, str(BCE))
import train_and_optimize as reference
from asl_loss import asymmetric_loss

MAX_ASL_EPOCHS, PATIENCE, MIN_DELTA = 12, 2, 0.001
BATCH_SIZE, IMAGE_SIZE, SEED = 12, 224, 42


def status(**fields):
    core.save_json(ROOT / "status.json", {"pid": os.getpid(), "time": datetime.now().astimezone().isoformat(), **fields})


def hash_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def score(metrics):
    return metrics["study"]["macro_auc"], metrics["study"]["macro_f1"]


def initial_state(recipe):
    if recipe["initial_checkpoint"]:
        return torch.load(Path(recipe["initial_checkpoint"]), map_location="cpu", weights_only=True)["state_dict"]
    return None


def train_epoch(model, dataset, optimizer, scaler, recipe, epoch, completed):
    core.seed(SEED + epoch)
    batches = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
                         generator=torch.Generator().manual_seed(SEED + epoch))
    model.train()
    if recipe["loss"] == "ASL":
        parameters = recipe["asl"]
        loss_function = partial(asymmetric_loss, gamma_neg=parameters["gamma_neg"],
                                gamma_pos=parameters["gamma_pos"], clip=parameters["clip"], eps=parameters["eps"])
    else:
        loss_function = torch.nn.functional.binary_cross_entropy_with_logits
    total, count = 0.0, 0
    started = time.perf_counter()
    for step, batch in enumerate(batches, 1):
        images = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
        targets = batch["target"].to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        loss = loss_function(logits.float(), targets)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count += len(images)
        total += value * len(images)
        if step == 1 or step % 250 == 0:
            status(state="training", stage=recipe["name"], epoch=epoch, completed_epochs=completed,
                   images=count, total_images=len(dataset), max_epochs=recipe["max_epochs"])
            print(f"{recipe['name']} {epoch}/{recipe['max_epochs']} images={count}/{len(dataset)} "
                  f"{recipe['loss']}_mean={total/count:.6f}", flush=True)
    torch.cuda.synchronize()
    if count != 30344:
        raise ValueError("A full-data epoch must process all 30344 training images")
    return {"epoch": epoch, "images": count, "train_loss": total/count,
            "train_seconds": time.perf_counter()-started}


def run_trial(recipe, train_data, val_data, manifest_hash, completed):
    stage = ROOT / recipe["name"]
    stage.mkdir(exist_ok=True)
    settings = {**recipe, "architecture": "mambaout_tiny", "img_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE, "seed": SEED, "train_images": 30344,
                "manifest_sha256": manifest_hash, "patience": PATIENCE, "min_delta": MIN_DELTA}
    last, best = stage / "last_training.pth", stage / "best.pth"
    history, best_epoch, best_validation = [], None, None
    stale, stop_reference = 0, None
    state = None
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Resume settings changed")
        model = core.build_model(recipe, state["state_dict"])
    else:
        model = core.build_model(recipe, initial_state(recipe))
    if recipe["name"] == "asl":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    else:
        optimizer = core.make_optimizer(model, recipe)
    scaler = torch.amp.GradScaler("cuda")
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_epoch, best_validation = state["history"], state["best_epoch"], state["best_validation"]
        stale, stop_reference = state["stale"], state["stop_reference"]
        saved_epoch = None
        if best.is_file():
            saved = torch.load(best, map_location="cpu", weights_only=True, mmap=True)
            saved_epoch = saved["training"]["best_epoch"]
            del saved
        if saved_epoch != best_epoch:
            if best_epoch != len(history):
                raise ValueError("Missing best checkpoint after interrupted save")
            core.save_best(best, state["state_dict"], settings, best_epoch)
        print(f"RESUME {recipe['name']} from epoch {len(history)}", flush=True)
        del state
    elif recipe["initial_checkpoint"]:
        status(state="initial_checkpoint_validation", stage=recipe["name"])
        best_validation = reference.validate(model, val_data)
        best_epoch, stop_reference = 0, score(best_validation)[0]
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        core.save_best(best, weights, settings, 0)
        del weights
    completed[recipe["name"]] = len(history)
    for epoch in range(len(history) + 1, recipe["max_epochs"] + 1):
        if stale >= PATIENCE:
            break
        if recipe["lr_mode"] == "baseline_100_horizon":
            lr = reference.cosine_lr(epoch, 100, 1e-4, 1e-6, 3)
            optimizer.param_groups[0]["lr"] = lr
        else:
            minimum_ratio = recipe.get("final_lr_ratio", 0.1)
            ratio = minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * (epoch - 1) / (recipe["max_epochs"] - 1))) / 2
            optimizer.param_groups[0]["lr"] = recipe["backbone_lr"] * ratio
            optimizer.param_groups[1]["lr"] = recipe["classifier_lr"] * ratio
        record = train_epoch(model, train_data, optimizer, scaler, recipe, epoch, completed)
        status(state="validation", stage=recipe["name"], epoch=epoch, completed_epochs=completed)
        started = time.perf_counter()
        validation = reference.validate(model, val_data)
        current = score(validation)
        record.update(validation=validation, validation_seconds=time.perf_counter()-started,
                      learning_rates=[group["lr"] for group in optimizer.param_groups])
        improved = best_validation is None or current > score(best_validation)
        if improved:
            best_epoch, best_validation = epoch, validation
        if stop_reference is None or current[0] > stop_reference + MIN_DELTA:
            stop_reference, stale = current[0], 0
        else:
            stale += 1
        history.append(record)
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        core.save_checkpoint(last, {"state_dict": weights, "optimizer": optimizer.state_dict(),
                                     "scaler": scaler.state_dict(), "settings": settings, "history": history,
                                     "best_epoch": best_epoch, "best_validation": best_validation,
                                     "stale": stale, "stop_reference": stop_reference})
        if improved:
            core.save_best(best, weights, settings, best_epoch)
        core.save_json(stage / "history.json", history)
        completed[recipe["name"]] = epoch
        status(state="checkpointed", stage=recipe["name"], epoch=epoch, completed_epochs=completed,
               auc=current[0], f1=current[1], best_epoch=best_epoch, best_auc=score(best_validation)[0],
               stale=stale, patience=PATIENCE)
        print(f"EPOCH_DONE {recipe['name']} {epoch} AUC={current[0]:.8f} F1={current[1]:.8f} "
              f"best_epoch={best_epoch} stale={stale}/{PATIENCE}", flush=True)
        del weights
        if (ROOT / "pause_after_epoch.flag").is_file():
            status(state="paused_after_epoch", stage=recipe["name"], epoch=epoch, completed_epochs=completed)
            raise SystemExit(0)
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    model = core.build_model(recipe, checkpoint["state_dict"]).eval()
    del checkpoint
    verified = reference.validate(model, val_data)
    if not np.allclose(score(verified), score(best_validation), atol=1e-6, rtol=0):
        raise ValueError("Best checkpoint failed score reproduction")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result = {"settings": settings, "epochs_completed": len(history), "best_epoch": best_epoch,
              "best_validation": verified, "best_checkpoint": str(best),
              "stop_reason": "early_stopping" if stale >= PATIENCE else "epoch_limit"}
    core.save_json(stage / "results.json", result)
    return result


def write_report(result):
    lines = ["# 全量 MambaOut 优化结果", "",
             "各全量试验每轮处理全部 30,344 张官方训练图，输入 224，验证 1,422 张图并汇总为 859 次检查。",
             "ASL 与原 BCE 保持相同 ImageNet 初始化、seed、数据顺序、增强、AdamW、权重衰减与原 100 轮学习率轨迹。",
             "新增试验采用早停和预算上限。ASL 的损失数值使用全元素 mean，其数值尺度与 BCE 不同。",
             "", "| 方案 | 已训练轮数 | 最佳轮次 | Study AUC | F1@0.5 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, entry in result["candidates"].items():
        auc, f1 = score(entry["best_validation"])
        lines.append(f"| {name} | {entry['epochs_completed']} | {entry['best_epoch']} | {auc:.6f} | {f1:.6f} |")
    lines += ["", f"最终保留的方案为 {result['winner']}，权重已重新加载复核。",
              f"最终模型的患者分组阈值内部估计 F1 为 {result['threshold_calibration']['oof_macro_f1']:.6f}。",
              "阈值只改变分类判定，概率与 AUC 保持相同。模型和轮次使用整个官方验证集筛选，此处 F1 属于内部开发估计。",
              "原独立患者实验的留出集属于官方训练集；全量再训练会使用这些患者，因此其得分用于此前的训练策略筛选。",
              "微调试验中的第 0 轮表示原第 3 轮权重，尚未执行额外参数更新。",
              "最终泛化能力仍需官方隐藏测试或后续独立新图像验证。"]
    (ROOT / "结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if ROOT.drive.upper() != "E:" or not torch.cuda.is_available():
        raise RuntimeError("Expected E-drive CUDA execution")
    if (ROOT / "results.json").is_file():
        result = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
        write_report(result)
        status(state="complete", stage="final", winner=result["winner"])
        return
    if json.loads((GENERALIZATION / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise RuntimeError("Complete the patient-disjoint experiment first")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = yaml.safe_load((reference.SMOKE / "data_local.yaml").read_text(encoding="utf-8"))
    manifest_hash = hash_file(Path(cfg["manifest"]))
    original_history = json.loads((BCE / "baseline_100/history.json").read_text(encoding="utf-8"))
    original_best = max(original_history, key=lambda item: score(item["validation"]))
    original_state = torch.load(BCE / "baseline_100/best.pth", map_location="cpu", weights_only=True, mmap=True)
    original_header = original_state["training"]
    if (original_header["best_epoch"] != original_best["epoch"] or
            original_header["manifest_sha256"] != manifest_hash or
            original_header["img_size"] != IMAGE_SIZE or original_header["batch_size"] != BATCH_SIZE or
            original_header["seed"] != SEED):
        raise ValueError("Historical BCE checkpoint/data/settings do not match the planned loss comparison")
    del original_state
    original = {"epochs_completed": len(original_history), "best_epoch": original_best["epoch"],
                "best_validation": original_best["validation"],
                "best_checkpoint": str(BCE / "baseline_100/best.pth"),
                "checkpoint_sha256": hash_file(BCE / "baseline_100/best.pth")}
    train_data = reference.CXRDataset(cfg["manifest"], cfg["data_root"], "train",
                                      reference.build_transform(True, IMAGE_SIZE), cfg["cache_dir"])
    val_data = reference.CXRDataset(cfg["manifest"], cfg["data_root"], "val",
                                    reference.build_transform(False, IMAGE_SIZE))
    if len(train_data) != 30344 or len(val_data) != 1422:
        raise ValueError("Official data split sizes changed")
    if {row["subject_id"] for row in train_data.rows} & {row["subject_id"] for row in val_data.rows}:
        raise ValueError("Patient leakage between train and official validation")
    pretrain_hash = hash_file(reference.SMOKE / "pretrain_model/model.safetensors")
    expected_hash = json.loads((reference.SMOKE / "pretrain_model/weight_metadata.json").read_text(encoding="utf-8"))["weight_sha256"]
    if pretrain_hash != expected_hash:
        raise ValueError("ImageNet weight hash mismatch")
    asl_recipe = {"name": "asl", "loss": "ASL", "initial_checkpoint": None,
                  "backbone_lr": 1e-4, "classifier_lr": 1e-4, "weight_decay": 0.01,
                  "drop_rate": 0.0, "drop_path_rate": 0.0,
                  "lr_mode": "baseline_100_horizon", "max_epochs": MAX_ASL_EPOCHS,
                  "asl": {"gamma_neg": 4, "gamma_pos": 1, "clip": 0.05, "eps": 1e-8, "focal_gradient": "detached",
                          "reduction": "mean"}, "pretrain_sha256": pretrain_hash}
    core.save_json(ROOT / "experiment_plan.json", {"asl": asl_recipe, "primary_metric": "study_macro_auc",
                                                   "fallback_only_if_asl_does_not_win": True,
                                                   "historical_best_epoch": original["best_epoch"],
                                                   "historical_best_score": score(original["best_validation"])})
    completed = {}
    asl = run_trial(asl_recipe, train_data, val_data, manifest_hash, completed)
    matched_best = max(original_history[:asl["epochs_completed"]], key=lambda item: score(item["validation"]))
    candidates = {"历史全量 BCE": original, "全量 ASL": asl}
    fallback_reason = None
    if score(asl["best_validation"]) <= score(original["best_validation"]):
        general = json.loads((GENERALIZATION / "results.json").read_text(encoding="utf-8"))
        interval = general["paired_audit_auc"]
        if interval["ci95"][0] > 0:
            fallback_recipe = {**core.RECIPES["regularized"], "name": "full_regularized_bce", "loss": "BCE",
                               "initial_checkpoint": None, "lr_mode": "short_cosine", "max_epochs": 12, "final_lr_ratio": 0.1,
                               "pretrain_sha256": pretrain_hash}
            fallback_reason = "Patient-disjoint paired audit supports the regularized recipe"
        else:
            fallback_recipe = {"name": "protected_finetune", "loss": "BCE",
                               "initial_checkpoint": original["best_checkpoint"],
                               "initial_checkpoint_sha256": hash_file(Path(original["best_checkpoint"])),
                               "backbone_lr": 1e-6, "classifier_lr": 1e-5, "weight_decay": 0.05,
                               "drop_rate": 0.2, "drop_path_rate": 0.1,
                               "lr_mode": "short_cosine", "max_epochs": 6, "final_lr_ratio": 0.2}
            fallback_reason = "Audit gain is uncertain; preserve the verified best3 features with conservative fine-tuning"
        core.save_json(ROOT / "fallback_plan.json", {"recipe": fallback_recipe, "reason": fallback_reason,
                                                    "patient_audit_delta": interval["delta"],
                                                    "patient_audit_ci95": interval["ci95"]})
        fallback = run_trial(fallback_recipe, train_data, val_data, manifest_hash, completed)
        fallback_label = "全量正则化 BCE" if fallback_recipe["name"] == "full_regularized_bce" else "最佳第3轮低学习率微调"
        candidates[fallback_label] = fallback
    winner = max(candidates, key=lambda name: score(candidates[name]["best_validation"]))
    chosen = candidates[winner]
    state = torch.load(Path(chosen["best_checkpoint"]), map_location="cpu", weights_only=True)
    settings = state["training"]
    model = core.build_model({"drop_rate": settings.get("drop_rate", 0.0),
                              "drop_path_rate": settings.get("drop_path_rate", 0.0)}, state["state_dict"]).eval()
    del state
    status(state="final_verification", stage="selection", winner=winner, completed_epochs=completed)
    verified, predictions = reference.validate(model, val_data, keep_predictions=True)
    if not np.allclose(score(verified), score(chosen["best_validation"]), atol=1e-6, rtol=0):
        raise ValueError("Final winner did not reproduce validation scores")
    gates = reference.evaluate_gates(predictions)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    temporary = ROOT / "best_model.tmp"
    shutil.copyfile(Path(chosen["best_checkpoint"]), temporary)
    core.replace_file(temporary, ROOT / "best_model.pth")
    if hash_file(ROOT / "best_model.pth") != hash_file(Path(chosen["best_checkpoint"])):
        raise ValueError("Selected checkpoint copy failed SHA-256 verification")
    if hash_file(Path(original["best_checkpoint"])) != original["checkpoint_sha256"]:
        raise ValueError("The preserved original checkpoint changed during optimization")
    core.save_json(ROOT / "gates.json", gates)
    result = {"candidates": candidates, "winner": winner, "winner_validation": verified,
              "matched_budget_bce": {"epochs": asl["epochs_completed"], "best_epoch": matched_best["epoch"],
                                      "best_validation": matched_best["validation"]},
              "fallback_reason": fallback_reason, "threshold_calibration": gates,
              "preserved_original_checkpoint": original["best_checkpoint"]}
    core.save_json(ROOT / "results.json", result)
    write_report(result)
    status(state="complete", stage="final", winner=winner, completed_epochs=completed,
           best_auc=score(verified)[0], best_f1=score(verified)[1], oof_macro_f1=gates["oof_macro_f1"])
    print(f"COMPLETE winner={winner} AUC={score(verified)[0]:.8f}", flush=True)


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