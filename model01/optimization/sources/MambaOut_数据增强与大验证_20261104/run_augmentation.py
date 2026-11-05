"""Sequential, patient-disjoint augmentation ablations. Outputs stay beside this file.

Train/development use the existing JPEG512 cache; final audit and official val
read raw images. Audit is never used to choose recipes, checkpoints or gates.
Run --check-only for CPU-only integrity, split, transform and MixUp checks.
"""
import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import msvcrt
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA = Path(r"E:\比赛数据集")
SMOKE = DATA / "本地跑通"
GENERAL = DATA / "MambaOut_泛化优化"
sys.path.insert(0, str(SMOKE))
sys.path.insert(0, str(GENERAL))
from dataset import CXRDataset, build_transform
from dataset.tools.build_cache import cache_image_path
from patient_split import make_patient_split
import train_generalization as core
from experiment_metrics import evaluate_model, paired_auc_interval

MANIFEST = DATA / "喻数据产物/manifest.csv"
CACHE = DATA / "喻数据产物/cache_512"
RAW = DATA / "raw/cxr_first"
PRETRAIN = SMOKE / "pretrain_model/model.safetensors"
CHAMPION = DATA / "MambaOut_ASL对照/best_model.pth"
SUBMISSION = DATA / "提交结果/MambaOut_S7微调第1轮_20261104/submission.csv"
EXPECTED_HASHES = {
    str(MANIFEST): "b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142",
    str(PRETRAIN): "ac9535c5b9fe1d325c5bf58dd4ffc3201c1fc04992b37e393977a7526cccafb4",
    str(CHAMPION): "9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585",
    str(SUBMISSION): "d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8",
}
SPLIT_HASH = "1096c6caa265067c0a4c7422dc7c0ce017919fd18174be7848368028bd39bb5d"
SEED, SPLIT_SEED, EPOCHS, BATCH, SIZE = 42, 20261105, 3, 12, 224
COMMON = dict(backbone_lr=3e-5, classifier_lr=3e-4, weight_decay=0.05,
              drop_rate=0.2, drop_path_rate=0.1)
RECIPES = {
    "A_current_affine": {"photometric": False, "mixup_alpha": 0.0, "mixup_probability": 0.0},
    "B_mild_photometric": {"photometric": True, "mixup_alpha": 0.0, "mixup_probability": 0.0},
    "C_photometric_mixup": {"photometric": True, "mixup_alpha": 0.1, "mixup_probability": 0.25},
}
EXPECTED_COUNTS = {
    "train": (21291, 13025, 7739), "early_stop": (4404, 2646, 1658), "audit": (4649, 2842, 1658)
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    core.save_json(Path(path), value)


def status(**fields):
    save_json(ROOT / "status.json", {"pid": os.getpid(), "time": datetime.now().astimezone().isoformat(), **fields})


def freeze_json(path, value):
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen record changed: {path.name}")
    else:
        save_json(path, value)


def verify_hashes():
    actual = {path: sha(path) for path in EXPECTED_HASHES}
    for path, expected in EXPECTED_HASHES.items():
        if actual[path] != expected:
            raise ValueError(f"Protected input SHA mismatch: {Path(path).name}")
    return actual


def transform_for(recipe):
    transform = build_transform(True, SIZE)
    if recipe["photometric"]:
        position = next(i for i, item in enumerate(transform.transforms) if isinstance(item, transforms.ToTensor))
        transform.transforms.insert(position, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    return transform


def load_data():
    base = CXRDataset(MANIFEST, RAW, "train", build_transform(False, SIZE), CACHE)
    indices, summary = make_patient_split(base.rows, SPLIT_SEED)
    if summary["assignment_sha256"] != SPLIT_HASH:
        raise ValueError("Patient split SHA changed")
    for part, counts in EXPECTED_COUNTS.items():
        actual = tuple(summary["partitions"][part][key] for key in ("images", "studies", "patients"))
        if actual != counts:
            raise ValueError(f"Unexpected cohort counts: {part}")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = {part: {row["subject_id"] for row in rows if row["split"] == part} for part in ("train", "val", "test")}
    if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Patient overlap in official splits")
    for i in indices["train"] + indices["early_stop"]:
        if not cache_image_path(CACHE, base.rows[i]["image_path"]).is_file():
            raise FileNotFoundError("Required training/development cache image is absent")
    return base, indices, summary


def fit_gates(targets, probabilities):
    # Identical finite grid and tie preference to the existing project fitter.
    ordered = sorted(np.arange(1, 40) / 40, key=lambda gate: (abs(gate - 0.5), gate))
    values = [f1_score(targets, probabilities >= gate, average=None, zero_division=0) for gate in ordered]
    return np.array(ordered)[np.argmax(values, axis=0)]


def save_predictions(path, evaluation):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, targets=evaluation["study_targets"],
                            probabilities=evaluation["study_probabilities"],
                            subjects=np.asarray(evaluation["study_subjects"], dtype=str),
                            study_ids=np.asarray(evaluation["study_ids"], dtype=str))
    core.replace_file(temporary, path)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as array:
        return {key: array[key].copy() for key in ("targets", "probabilities", "subjects", "study_ids")}


def array_metrics(predictions, gates):
    y, p = predictions["targets"], predictions["probabilities"]
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 10:
        raise ValueError("Incorrect prediction shape")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Invalid labels or probabilities")
    if not ((y == 0).any(axis=0) & (y == 1).any(axis=0)).all():
        raise ValueError("Undefined class AUC")
    auc = roc_auc_score(y, p, average=None)
    ap = average_precision_score(y, p, average=None)
    decisions = {"fixed_0_5": p >= 0.5, "frozen_development_gates": p >= np.asarray(gates)}
    result = {"macro_auc": float(auc.mean()), "auc_by_class": auc.tolist(),
              "macro_ap": float(ap.mean()), "ap_by_class": ap.tolist(),
              "study_count": len(y), "patient_count": int(len(np.unique(predictions["subjects"]))),
              "positive_by_class": y.sum(axis=0).astype(int).tolist(), "gates": np.asarray(gates).tolist()}
    for label, decision in decisions.items():
        f1 = f1_score(y, decision, average=None, zero_division=0)
        result[label] = {"macro_f1": float(f1.mean()), "f1_by_class": f1.tolist(),
                         "macro_precision": float(precision_score(y, decision, average="macro", zero_division=0)),
                         "macro_recall": float(recall_score(y, decision, average="macro", zero_division=0)),
                         "tp_by_class": ((y == 1) & decision).sum(axis=0).astype(int).tolist(),
                         "fp_by_class": ((y == 0) & decision).sum(axis=0).astype(int).tolist(),
                         "fn_by_class": ((y == 1) & ~decision).sum(axis=0).astype(int).tolist()}
    return result


def score(evaluation):
    return evaluation["study"]["macro_auc"], evaluation["study"]["macro_f1"]


def cpu_check(base, indices, summary, hashes):
    core.seed(SEED)
    checks = {}
    for name, recipe in RECIPES.items():
        data = CXRDataset(MANIFEST, RAW, "train", transform_for(recipe), CACHE)
        sample = data[indices["train"][0]]
        if tuple(sample["image"].shape) != (3, SIZE, SIZE) or not torch.isfinite(sample["image"]).all():
            raise ValueError("Augmentation shape or finiteness failure")
        expected = base[indices["train"][0]]["target"]
        if not torch.equal(sample["target"], expected):
            raise ValueError("Augmentation changed labels")
        checks[name] = str(data.transform)
    z = torch.linspace(-2, 2, 40).reshape(4, 10)
    y = (torch.arange(40).reshape(4, 10) % 3 == 0).float()
    permutation = torch.tensor([2, 0, 3, 1])
    lam = 0.37
    mixed = lam * y + (1 - lam) * y[permutation]
    soft = torch.nn.functional.binary_cross_entropy_with_logits(z, mixed)
    reference = lam * torch.nn.functional.binary_cross_entropy_with_logits(z, y) + (1 - lam) * torch.nn.functional.binary_cross_entropy_with_logits(z, y[permutation])
    if not torch.allclose(soft, reference, atol=1e-7, rtol=1e-6):
        raise ValueError("MixUp target loss identity failed")
    result = {"mode": "CPU_only", "hashes_verified": hashes, "split_sha256": summary["assignment_sha256"],
              "partitions": summary["partitions"], "transforms": checks, "mixup_loss_identity": True,
              "cached_training_and_development_images_verified": len(indices["train"]) + len(indices["early_stop"]),
              "protected_files_unmodified": verify_hashes() == hashes}
    save_json(ROOT / "preflight.json", result)
    print(json.dumps({"CPU_preflight": "passed", "cohorts": summary["partitions"], "mixup_loss_identity": True}, ensure_ascii=False), flush=True)


def train_one_epoch(model, data, optimizer, scaler, recipe, name, epoch, completed):
    core.seed(SEED + epoch)
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(SEED + epoch), drop_last=False)
    mix_rng = np.random.default_rng(SEED + 10000 * epoch)
    permutation_rng = torch.Generator().manual_seed(SEED + 20000 * epoch)
    model.train()
    loss_sum, processed, mixed_batches, mixed_images = 0.0, 0, 0, 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(loader, 1):
        images = batch["image"].to("cuda", non_blocking=True, memory_format=torch.channels_last)
        targets = batch["target"].to("cuda", non_blocking=True)
        if recipe["mixup_probability"] and mix_rng.random() < recipe["mixup_probability"]:
            lam = float(mix_rng.beta(recipe["mixup_alpha"], recipe["mixup_alpha"]))
            permutation = torch.randperm(len(images), generator=permutation_rng).to("cuda")
            images = (lam * images + (1 - lam) * images[permutation]).contiguous(memory_format=torch.channels_last)
            targets = lam * targets + (1 - lam) * targets[permutation]
            mixed_batches += 1
            mixed_images += len(images)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        processed += len(images)
        loss_sum += value * len(images)
        if step == 1 or step % 100 == 0 or step == len(loader):
            status(state="training", phase="training", stage=name, epoch=epoch, max_epochs=EPOCHS,
                   images=processed, total_images=len(data), batch=step, batches=len(loader), completed_epochs=completed)
            print(f"TRAIN {name} epoch={epoch}/{EPOCHS} images={processed}/{len(data)} BCE={loss_sum/processed:.7f}", flush=True)
    torch.cuda.synchronize()
    if processed != len(data):
        raise ValueError("Epoch did not cover every selected source image exactly once")
    return {"epoch": epoch, "train_images": processed, "train_bce": loss_sum / processed,
            "train_seconds": time.perf_counter() - start, "mixup_batches": mixed_batches,
            "mixup_source_images": mixed_images, "batches": len(loader),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}


def run_stage(name, recipe, indices, early_data, completed):
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    data = Subset(CXRDataset(MANIFEST, RAW, "train", transform_for(recipe), CACHE), indices["train"])
    settings = {**COMMON, **recipe, "stage": name, "architecture": "mambaout_tiny", "img_size": SIZE,
                "batch_size": BATCH, "workers": 0, "seed": SEED, "split_seed": SPLIT_SEED,
                "epochs": EPOCHS, "lr_factors": [1.0, 0.55, 0.1], "split_sha256": SPLIT_HASH,
                "train_images": len(data), "early_images": len(early_data), "initialization": "ImageNet_only",
                "pretrained_sha256": EXPECTED_HASHES[str(PRETRAIN)], "manifest_sha256": EXPECTED_HASHES[str(MANIFEST)],
                "train_and_early_source": "existing_cache512_JPEG92", "audit_and_official_source": "raw"}
    freeze_json(folder / "settings.json", settings)
    last_path = folder / "last_training.pth"
    state = torch.load(last_path, map_location="cpu", weights_only=True) if last_path.is_file() else None
    if state is not None and state["settings"] != settings:
        raise ValueError("Resume settings mismatch")
    model = core.build_model(COMMON, state["state_dict"] if state is not None else None)
    optimizer = core.make_optimizer(model, COMMON)
    scaler = torch.amp.GradScaler("cuda")
    history, best_epoch, best_score = [], 0, None
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_epoch, best_score = state["history"], state["best_epoch"], tuple(state["best_score"])
        if len(history) > EPOCHS or [item["epoch"] for item in history] != list(range(1, len(history) + 1)):
            raise ValueError("Resume history is not contiguous")
        expected_best = max(history, key=lambda entry: score(entry["early_metrics"]))
        if expected_best["epoch"] != best_epoch or score(expected_best["early_metrics"]) != best_score:
            raise ValueError("Resume best epoch disagrees with history")
        for entry in history:
            if not (folder / f"epoch_{entry['epoch']}.pth").is_file() or not (folder / f"early_epoch_{entry['epoch']}.npz").is_file():
                raise ValueError("Resume epoch artifacts missing")
        # A crash may have occurred between a newer best.pth and the atomic last checkpoint.
        # Always restore the committed best from its committed, epoch-specific checkpoint.
        committed = torch.load(folder / f"epoch_{best_epoch}.pth", map_location="cpu", weights_only=True)
        core.save_checkpoint(folder / "best.pth", committed)
        del committed, state
    completed[name] = len(history)
    for epoch in range(len(history) + 1, EPOCHS + 1):
        factor = settings["lr_factors"][epoch - 1]
        optimizer.param_groups[0]["lr"] = COMMON["backbone_lr"] * factor
        optimizer.param_groups[1]["lr"] = COMMON["classifier_lr"] * factor
        record = train_one_epoch(model, data, optimizer, scaler, recipe, name, epoch, completed)
        status(state="development_validation", phase="development_validation", stage=name, epoch=epoch,
               images=0, total_images=len(early_data), completed_epochs=completed)
        validation_start = time.perf_counter()
        evaluated = evaluate_model(model, early_data, BATCH)
        if evaluated["metrics"]["image"]["count"] != 4404 or evaluated["metrics"]["study"]["count"] != 2646:
            raise ValueError("Development evaluation cohort changed")
        record.update(early_metrics=evaluated["metrics"], early_bce=evaluated["bce"],
                      validation_seconds=time.perf_counter() - validation_start,
                      learning_rates=[item["lr"] for item in optimizer.param_groups])
        current = score(evaluated["metrics"])
        improved = best_score is None or current > best_score
        if improved:
            best_score, best_epoch = current, epoch
        history.append(record)
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        checkpoint = {"state_dict": weights, "training": {**settings, "epoch": epoch, "best_epoch": epoch}}
        core.save_checkpoint(folder / f"epoch_{epoch}.pth", checkpoint)
        save_predictions(folder / f"early_epoch_{epoch}.npz", evaluated)
        if improved:
            core.save_checkpoint(folder / "best.pth", checkpoint)
        core.save_checkpoint(last_path, {"state_dict": weights, "settings": settings, "history": history,
                                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                        "best_epoch": best_epoch, "best_score": list(best_score)})
        save_json(folder / "history.json", history)
        completed[name] = epoch
        status(state="checkpointed", phase="checkpointed", stage=name, epoch=epoch, completed_epochs=completed,
               early_auc=current[0], early_f1=current[1], best_epoch=best_epoch, best_early_auc=best_score[0])
        print(f"EPOCH_DONE {name} {epoch}/{EPOCHS} AUC={current[0]:.8f} F1={current[1]:.8f} best_epoch={best_epoch}", flush=True)
        del evaluated, weights, checkpoint
        if (ROOT / "pause_after_epoch.flag").is_file():
            status(state="paused_after_epoch", phase="paused", stage=name, epoch=epoch, completed_epochs=completed)
            raise SystemExit(0)
    best_path = folder / "best.pth"
    early_predictions = load_predictions(folder / f"early_epoch_{best_epoch}.npz")
    gates = fit_gates(early_predictions["targets"], early_predictions["probabilities"])
    result = {"stage": name, "settings": settings, "completed_epochs": len(history), "best_epoch": best_epoch,
              "best_score": list(best_score), "checkpoint": str(best_path), "checkpoint_sha256": sha(best_path),
              "gates_fitted_on": "early_stop_only", "gates": gates.tolist(),
              "early_metrics": array_metrics(early_predictions, gates), "history": history}
    save_json(folder / "results.json", result)
    save_json(folder / "history.json", history)
    del model, optimizer, scaler, data, early_predictions
    gc.collect()
    torch.cuda.empty_cache()
    return result


def final_evaluation(name, stage_result, dataset, cohort, expected_counts, completed):
    folder = ROOT / name
    record_path, pred_path = folder / f"{cohort}_evaluation.json", folder / f"{cohort}_predictions.npz"
    gate_list = stage_result["gates"]
    fingerprint = {"checkpoint_sha256": stage_result["checkpoint_sha256"], "cohort": cohort,
                   "image_source": "raw", "input_size": SIZE, "gates": gate_list,
                   "split_sha256": SPLIT_HASH, "manifest_sha256": EXPECTED_HASHES[str(MANIFEST)]}
    if record_path.is_file():
        cached = json.loads(record_path.read_text(encoding="utf-8"))
        if cached["fingerprint"] != fingerprint or not pred_path.is_file() or sha(pred_path) != cached["predictions_sha256"]:
            raise ValueError("Completed final evaluation artifacts changed")
        return cached, load_predictions(pred_path)
    if sha(stage_result["checkpoint"]) != stage_result["checkpoint_sha256"]:
        raise ValueError("Frozen candidate checkpoint changed")
    status(state="final_evaluation", phase=cohort, stage=name, completed_epochs=completed,
           images=0, total_images=len(dataset))
    state = torch.load(stage_result["checkpoint"], map_location="cpu", weights_only=True)
    model = core.build_model(COMMON, state["state_dict"])
    del state
    start = time.perf_counter()
    evaluated = evaluate_model(model, dataset, BATCH)
    if evaluated["metrics"]["image"]["count"] != expected_counts[0] or evaluated["metrics"]["study"]["count"] != expected_counts[1]:
        raise ValueError("Final evaluation did not cover the full intended cohort")
    if len(set(evaluated["study_subjects"])) != expected_counts[2]:
        raise ValueError("Final patient count mismatch")
    save_predictions(pred_path, evaluated)
    prediction = load_predictions(pred_path)
    record = {"fingerprint": fingerprint, "predictions_sha256": sha(pred_path),
              "seconds": time.perf_counter() - start, "metrics": evaluated["metrics"],
              "bce": evaluated["bce"], "study_details": array_metrics(prediction, gate_list)}
    save_json(record_path, record)
    del model, evaluated
    gc.collect()
    torch.cuda.empty_cache()
    return record, prediction


def compare_pair(baseline, candidate):
    for key in ("targets", "subjects", "study_ids"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError("Paired evaluation records are not aligned")
    interval = paired_auc_interval(baseline["targets"], baseline["probabilities"], candidate["probabilities"],
                                   baseline["subjects"].tolist(), seed=SPLIT_SEED, repeats=400)
    if interval["valid_repeats"] < 300:
        raise ValueError("Too few valid patient-bootstrap repeats")
    return interval


def run(check_only=False):
    if ROOT.drive.upper() != "E:":
        raise ValueError("Experiment outputs must stay on E:")
    hashes = verify_hashes()
    base, indices, summary = load_data()
    if check_only:
        cpu_check(base, indices, summary, hashes)
        return
    if (ROOT / "results.json").is_file():
        existing = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
        if existing.get("state") != "completed":
            raise ValueError("Existing root results are not complete")
        print("ALREADY_COMPLETE: existing results retained; protected hashes verified", flush=True)
        return
    if (ROOT / "pause_after_epoch.flag").is_file():
        status(state="paused_before_start", phase="paused")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing accidental CPU training")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = False
    plan = {"schema": 1, "epochs_per_recipe": EPOCHS, "recipes": RECIPES, "common_training": COMMON,
            "seed": SEED, "split_seed": SPLIT_SEED, "split_sha256": SPLIT_HASH, "image_size": SIZE,
            "cohorts": summary["partitions"], "batch_size": BATCH, "workers": 0, "cpu_threads": 4,
            "initialization": "ImageNet_only_for_each_recipe", "lr_factors": [1.0, 0.55, 0.1],
            "train_and_development_input": "cache512_JPEG92_then_BILINEAR_pad224",
            "audit_and_official_input": "raw_BILINEAR_pad224", "hashes": hashes,
            "selection": "epoch: max development study AUC then F1@0.5; augmented recipe: same among B/C",
            "thresholds": "fit only development, then freeze before audit",
            "final_audit": "baseline and development-selected augmentation once; 400 paired patient bootstrap",
            "limitations": ["Internal holdout from the same competition source, not external validation",
                            "Single training seed and only three epochs per recipe",
                            "Historical experiments have used this source; audit is frozen for this comparison",
                            "Cache/raw input mismatch is recorded and shared across recipes",
                            "Official validation has already been repeatedly inspected; scores are exploratory"],
            "protected_original_champion_and_submission": True}
    freeze_json(ROOT / "plan.json", plan)
    freeze_json(ROOT / "patient_split.json", summary)
    freeze_json(ROOT / "protected_inputs.json", hashes)
    early_data = Subset(base, indices["early_stop"])
    completed, stages = {}, {}
    for name, recipe in RECIPES.items():
        stages[name] = run_stage(name, recipe, indices, early_data, completed)
    selected = max((name for name in RECIPES if name != "A_current_affine"), key=lambda name: tuple(stages[name]["best_score"]))
    selection = {"criterion": "development study AUC, then fixed-threshold F1; audit and official val unused",
                 "selected_augmentation": selected, "baseline": "A_current_affine",
                 "before_audit": True, "candidates": {name: {"epoch": result["best_epoch"],
                    "score": result["best_score"], "checkpoint_sha256": result["checkpoint_sha256"],
                    "gates": result["gates"]} for name, result in stages.items()}}
    freeze_json(ROOT / "selection.json", selection)
    raw_train = CXRDataset(MANIFEST, RAW, "train", build_transform(False, SIZE), None)
    audit = Subset(raw_train, indices["audit"])
    official = CXRDataset(MANIFEST, RAW, "val", build_transform(False, SIZE), None)
    evaluation, intervals = {}, {}
    for cohort, dataset, counts in (("audit", audit, EXPECTED_COUNTS["audit"]), ("official_validation", official, (1422, 859, 228))):
        predictions = {}
        evaluation[cohort] = {}
        for name in ("A_current_affine", selected):
            record, prediction = final_evaluation(name, stages[name], dataset, cohort, counts, completed)
            evaluation[cohort][name], predictions[name] = record, prediction
        status(state="bootstrap", phase=cohort, completed_epochs=completed, repeats=400)
        intervals[cohort] = compare_pair(predictions["A_current_affine"], predictions[selected])
    protected = verify_hashes()
    result = {"state": "completed", "completed_epochs": completed, "total_new_epochs": sum(completed.values()),
              "selected_augmentation": selected, "stages": stages, "selection": selection,
              "evaluation": evaluation, "paired_auc_intervals": intervals, "plan": plan,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "protected_hashes_after": protected, "original_champion_replaced": False,
              "test_data_evaluated": False, "github_uploaded": False}
    save_json(ROOT / "results.json", result)
    status(state="completed", phase="complete", completed_epochs=completed, total_new_epochs=sum(completed.values()),
           selected_augmentation=selected, paired_audit_auc=intervals["audit"])
    print(f"COMPLETE selected={selected} audit_AUC_delta={intervals['audit']['delta']:+.8f}", flush=True)


@contextmanager
def exclusive_lock():
    with (ROOT / "runner.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    with exclusive_lock():
        awake = 0 if args.check_only else ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            if not args.check_only:
                status(state="starting", phase="input_checks")
            run(args.check_only)
        except Exception as error:
            status(state="failed", phase="exception", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
