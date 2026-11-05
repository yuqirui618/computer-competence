"""CPU-only verified Chinese report for the completed 3x3 augmentation ablation.

Uses saved predictions and stdlib/numpy/sklearn only. Never imports torch or
training code. Refuses incomplete results; writes reports only after checks.
"""
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME"):
    os.environ[key] = str(ROOT / "temp")
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

STAGES = ("A_current_affine", "B_mild_photometric", "C_photometric_mixup")
LABELS = {"A_current_affine": "A：当前仿射增强", "B_mild_photometric": "B：轻度亮度/对比度",
          "C_photometric_mixup": "C：亮度/对比度＋MixUp"}
CLASS_NAMES = {1: "纵隔心影增大", 3: "肺部实变", 4: "肺炎"}
CHAMPION = DATA / "MambaOut_ASL对照" / "best_model.pth"
SUBMISSION = DATA / "提交结果" / "MambaOut_S7微调第1轮_20261104" / "submission.csv"
EXPECTED_CHAMPION = "9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585"
EXPECTED_SUBMISSION = "d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8"
CHECKS, FINGERPRINTS = {}, {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message, atol=1e-10):
    require(np.shape(actual) == np.shape(expected), message + " [shape]")
    require(np.allclose(actual, expected, rtol=0, atol=atol), message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(relative):
    path = ROOT / relative
    require(path.is_file(), "Missing input: " + str(relative))
    FINGERPRINTS[str(relative)] = sha(path)
    return json.loads(path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON: " + value)))


def load_predictions(relative):
    path = ROOT / relative
    require(path.is_file(), "Missing predictions: " + str(relative))
    FINGERPRINTS[str(relative)] = sha(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in ("targets", "probabilities", "subjects", "study_ids")}
    y, p = result["targets"], result["probabilities"]
    require(y.ndim == 2 and y.shape == p.shape and y.shape[1] == 10, "Invalid prediction shape")
    require(np.isin(y, (0, 1)).all(), "Nonbinary target")
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "Invalid probability")
    require(((y == 0).any(axis=0) & (y == 1).any(axis=0)).all(), "Undefined class AUC")
    require(result["subjects"].shape == result["study_ids"].shape == (len(y),), "Invalid identifiers")
    require(len(np.unique(result["study_ids"])) == len(y), "Repeated Study ID")
    return result


def metrics(prediction, gates):
    y, p, gates = prediction["targets"], prediction["probabilities"], np.asarray(gates)
    require(gates.shape == (10,) and np.isfinite(gates).all(), "Invalid thresholds")
    auc, ap = roc_auc_score(y, p, average=None), average_precision_score(y, p, average=None)
    out = dict(macro_auc=float(auc.mean()), auc_by_class=auc.tolist(), macro_ap=float(ap.mean()),
        ap_by_class=ap.tolist(), study_count=len(y), patient_count=len(np.unique(prediction["subjects"])),
        positive_by_class=y.sum(axis=0).astype(int).tolist(), gates=gates.tolist())
    for label, d in (("fixed_0_5", p >= .5), ("frozen_development_gates", p >= gates)):
        f1 = f1_score(y, d, average=None, zero_division=0)
        out[label] = dict(macro_f1=float(f1.mean()), f1_by_class=f1.tolist(),
            macro_precision=float(precision_score(y, d, average="macro", zero_division=0)),
            macro_recall=float(recall_score(y, d, average="macro", zero_division=0)),
            tp_by_class=((y == 1) & d).sum(axis=0).astype(int).tolist(),
            fp_by_class=((y == 0) & d).sum(axis=0).astype(int).tolist(),
            fn_by_class=((y == 1) & ~d).sum(axis=0).astype(int).tolist())
    return out


def check_metrics(recomputed, saved, context):
    for key in ("macro_auc", "auc_by_class", "macro_ap", "ap_by_class", "study_count",
                "patient_count", "positive_by_class", "gates"):
        same(recomputed[key], saved[key], context + "/" + key)
    for threshold in ("fixed_0_5", "frozen_development_gates"):
        for key, value in recomputed[threshold].items():
            same(value, saved[threshold][key], context + "/" + threshold + "/" + key)


def fit_gates(y, p):
    candidates = sorted(np.arange(1, 40) / 40, key=lambda gate: (abs(gate - .5), gate))
    values = [f1_score(y, p >= gate, average=None, zero_division=0) for gate in candidates]
    return np.array(candidates)[np.argmax(values, axis=0)]


def interval_for(a, b):
    for key in ("targets", "subjects", "study_ids"):
        require(np.array_equal(a[key], b[key]), "Paired predictions not aligned: " + key)
    y, groups = a["targets"], defaultdict(list)
    for index, patient in enumerate(a["subjects"]):
        groups[str(patient)].append(index)
    groups = [np.asarray(indices, dtype=np.intp) for indices in groups.values()]
    rng, differences = np.random.default_rng(20261105), []
    delta = float(roc_auc_score(y, b["probabilities"], average="macro") -
                  roc_auc_score(y, a["probabilities"], average="macro"))
    for _ in range(400):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        subset = y[indices]
        if not ((subset == 0).any(axis=0) & (subset == 1).any(axis=0)).all():
            continue
        differences.append(float(roc_auc_score(subset, b["probabilities"][indices], average="macro") -
                                 roc_auc_score(subset, a["probabilities"][indices], average="macro")))
    require(len(differences) >= 300, "Insufficient valid bootstrap replicates")
    return dict(delta=delta, ci95=np.quantile(differences, [.025, .975]).tolist(),
                valid_repeats=len(differences), repeats=400)


def fmt(value):
    return f"{value:.6f}"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows]


def atomic_write(path, content, encoding):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding=encoding)
    temporary.replace(path)


def main():
    require(ROOT.drive.upper() == "E:", "Report output must stay on E:")
    result = read_json("results.json")
    require(result["state"] == "completed", "Experiment incomplete: no report will be written")
    require(result["total_new_epochs"] == 9 and result["completed_epochs"] == {name: 3 for name in STAGES},
            "Expected exactly 3 recipes x 3 completed epochs")
    for key in ("original_champion_replaced", "test_data_evaluated", "github_uploaded"):
        require(result[key] is False, "Unexpected experiment action: " + key)
    counts, probe = read_json("cohort_audit/counts.json"), read_json("resolution_probe/metrics.json")
    plan, selection = read_json("plan.json"), read_json("selection.json")
    require(plan == result["plan"] and selection == result["selection"], "Frozen records disagree")
    require(selection["before_audit"] is True, "Selection was not frozen before audit")
    require(plan["epochs_per_recipe"] == 3 and plan["seed"] == 42 and plan["split_seed"] == 20261105,
            "Unexpected training plan")
    require(plan["initialization"] == "ImageNet_only_for_each_recipe", "Invalid holdout initialization")
    require(plan["split_sha256"] == counts["assignment_sha256"], "Cohort split hash mismatch")
    require(all(value == 0 for item in counts["new_partition_pairwise_overlap_counts"].values()
                for value in item.values()), "New split contains leakage")
    require(all(value == 0 for item in counts["new_partition_official_val_test_patient_overlap_counts"].values()
                for value in item.values()), "New split overlaps official validation or test patients")
    protected = {path: sha(path) for path in plan["hashes"]}
    require(protected == plan["hashes"] == result["protected_hashes_after"], "Protected input hash changed")
    require(protected.get(str(CHAMPION)) == EXPECTED_CHAMPION and
            protected.get(str(SUBMISSION)) == EXPECTED_SUBMISSION, "Unexpected champion/submission")
    require(probe["checkpoint_sha256"] == EXPECTED_CHAMPION, "Probe used another champion")
    same(probe["baseline224"]["study"]["macro_auc"], .8019769550500871, "Champion AUC reference")
    same(probe["baseline224"]["calibration"]["oof_macro_f1"], .451054963682343, "Champion OOF F1 reference")
    require(probe["alignment"] == {"targets_identical": True, "subjects_identical": True}, "Probe misalignment")
    same(probe["candidate320"]["study"]["macro_auc"] - probe["baseline224"]["study"]["macro_auc"],
         probe["paired_patient_macro_auc_bootstrap"]["delta"], "Probe delta")
    CHECKS.update(completed_epochs=9, initialization="ImageNet_only_for_each_recipe",
        protected_hashes_verified=protected, split_disjoint=True, gpu_used_by_report=False,
        champion_replaced=False, test_predictions_read=False)
    stages, epoch_rows, best_predictions = result["stages"], [], {}
    expected_counts = {"new_train": (21291, 13025, 7739), "new_early_stop": (4404, 2646, 1658),
                       "new_audit": (4649, 2842, 1658), "official_val": (1422, 859, 228)}
    for key, expected in expected_counts.items():
        require(tuple(counts["cohorts"][key][field] for field in ("images", "studies", "patients")) == expected,
                "Unexpected cohort counts: " + key)
    for name in STAGES:
        stage = stages[name]
        require(stage == read_json(f"{name}/results.json"), "Stage disagrees with root")
        require(stage["history"] == read_json(f"{name}/history.json"), "History disagrees with root")
        history = stage["history"]
        require([row["epoch"] for row in history] == [1, 2, 3], "Noncontiguous epochs")
        chosen = max(history, key=lambda row: (row["early_metrics"]["study"]["macro_auc"],
                                               row["early_metrics"]["study"]["macro_f1"]))
        require(chosen["epoch"] == stage["best_epoch"], "Best epoch mismatch")
        same(stage["best_score"], [chosen["early_metrics"]["study"]["macro_auc"],
                                  chosen["early_metrics"]["study"]["macro_f1"]], "Best score")
        require(sha(stage["checkpoint"]) == stage["checkpoint_sha256"], "Candidate checkpoint changed")
        require(stage["gates_fitted_on"] == "early_stop_only", "Threshold fit source changed")
        require(stage["settings"]["drop_rate"] == .2 and stage["settings"]["drop_path_rate"] == .1,
                "Regularization differed between recipes")
        for row in history:
            epoch, sc = row["epoch"], row["early_metrics"]["study"]
            pred = load_predictions(f"{name}/early_epoch_{epoch}.npz")
            measured = metrics(pred, stage["gates"])
            require(measured["study_count"] == 2646 and measured["patient_count"] == 1658, "Early cohort changed")
            same(measured["macro_auc"], sc["macro_auc"], "Early AUC")
            same(measured["auc_by_class"], sc["auc"], "Early per-class AUC")
            same(measured["fixed_0_5"]["macro_f1"], sc["macro_f1"], "Early F1")
            same(measured["fixed_0_5"]["f1_by_class"], sc["f1"], "Early per-class F1")
            require(row["train_images"] == 21291 and row["early_metrics"]["image"]["count"] == 4404,
                    "Epoch image coverage mismatch")
            require(math.isfinite(row["train_bce"]) and math.isfinite(row["early_bce"]) and
                    row["train_bce"] >= 0 and row["early_bce"] >= 0, "Invalid BCE loss")
            if epoch == stage["best_epoch"]:
                best_predictions[name] = pred
                same(fit_gates(pred["targets"], pred["probabilities"]), stage["gates"], "Frozen gates")
                check_metrics(measured, stage["early_metrics"], "Best early metrics")
            epoch_rows.append({
                "组别": name, "方案": LABELS[name], "轮次": epoch, "本组最优": epoch == stage["best_epoch"],
                "训练图像数": row["train_images"], "开发图像数": 4404, "开发Study数": 2646,
                "开发患者数": 1658, "训练BCE": row["train_bce"], "开发BCE": row["early_bce"],
                "开发StudyMacroAUC": sc["macro_auc"], "开发MacroF1_阈值0.5": sc["macro_f1"],
                "骨干学习率": row["learning_rates"][0], "分类头学习率": row["learning_rates"][1],
                "训练秒数": row["train_seconds"], "开发验证秒数": row["validation_seconds"],
                "MixUp批次数": row["mixup_batches"], "MixUp源图像数": row["mixup_source_images"],
                "训练批次数": row["batches"], "峰值分配显存MiB": row["peak_allocated_mib"]})
        record = selection["candidates"][name]
        require(record["epoch"] == stage["best_epoch"] and
                record["checkpoint_sha256"] == stage["checkpoint_sha256"], "Frozen selection changed")
        same(record["score"], stage["best_score"], "Selection score")
        same(record["gates"], stage["gates"], "Selection gates")
    selected = max(STAGES[1:], key=lambda name: tuple(stages[name]["best_score"]))
    require(selected == result["selected_augmentation"] == selection["selected_augmentation"],
            "B/C recipe selection mismatch")
    CHECKS["best_epochs"] = {name: stages[name]["best_epoch"] for name in STAGES}
    CHECKS["selected_augmentation"] = selected
    CHECKS["all_9_early_metrics_recomputed"] = True
    CHECKS["bce_note"] = "Recorded image-level BCE; cannot reconstruct it from Study mean probabilities."
    comparison, finals = {}, {}
    for cohort, count_key in (("audit", "new_audit"), ("official_validation", "official_val")):
        predictions, finals[cohort] = {}, {}
        for name in (STAGES[0], selected):
            saved = result["evaluation"][cohort][name]
            require(saved == read_json(f"{name}/{cohort}_evaluation.json"), "Final JSON disagrees")
            pred = load_predictions(f"{name}/{cohort}_predictions.npz")
            predictions[name] = pred
            require(FINGERPRINTS[f"{name}/{cohort}_predictions.npz"] == saved["predictions_sha256"],
                    "Final prediction hash changed")
            require(saved["fingerprint"]["checkpoint_sha256"] == stages[name]["checkpoint_sha256"] and
                    saved["fingerprint"]["cohort"] == cohort and saved["fingerprint"]["image_source"] == "raw",
                    "Final fingerprint mismatch")
            same(saved["fingerprint"]["gates"], stages[name]["gates"], "Final gates changed")
            measured = metrics(pred, stages[name]["gates"])
            require((saved["metrics"]["image"]["count"], measured["study_count"], measured["patient_count"])
                    == expected_counts[count_key], "Final counts mismatch")
            check_metrics(measured, saved["study_details"], "Final metrics")
            same(measured["macro_auc"], saved["metrics"]["study"]["macro_auc"], "Final AUC implementations differ")
            same(measured["fixed_0_5"]["macro_f1"], saved["metrics"]["study"]["macro_f1"], "Final F1 differs")
            require(not set(pred["subjects"]).intersection(best_predictions[name]["subjects"]),
                    "Final evaluation overlaps development patients")
            finals[cohort][name] = measured
        recomputed = interval_for(predictions[STAGES[0]], predictions[selected])
        for key, value in recomputed.items():
            same(value, result["paired_auc_intervals"][cohort][key], "Bootstrap does not reproduce")
        comparison[cohort] = recomputed
    CHECKS["final_metrics_and_patient_bootstrap_recomputed"] = comparison
    CHECKS["all_thresholds_refitted_on_development_and_frozen"] = True
    lo, hi = comparison["audit"]["ci95"]
    if lo > 0:
        finding = "预选增强方案在本轮内部核验集上有正向证据，但尚不能替换全量冠军。"
        next_step = "可将选定增强迁移到全量训练，在相同训练预算和多个种子下与现方案对照；确认全量收益后再决定是否替换冠军。"
    elif hi < 0:
        finding = "预选增强方案在本轮内部核验集上弱于对照，不建议据此替换现方案。"
        next_step = "保留冠军；先审查增强是否影响细小阴影/边界、开发缓存与原图的输入差异及稀有类标签质量，再设计单因素对照，避免盲目加大增强。"
    else:
        finding = "本轮核验的 AUC 差值区间跨过或接触 0，没有足够证据声称增强能稳定提升。"
        next_step = "保留冠军；若点估计和稀有类表现值得复核，先在预设训练预算下补充多个种子，再决定是否做全量迁移。"
    CHECKS["audit_interpretation"] = finding
    CHECKS["audit_interval_excludes_zero_positive"] = bool(lo > 0)
    lines = [
        "# MambaOut：数据增强与扩大患者核验实验", "", finding, "",
        "**目前仍保留全量训练的原冠军：官方验证 Study Macro AUC = 0.801976955，"
        "患者五折阈值 OOF Macro F1 = 0.451054964。原权重和原 submission.csv 的 SHA256 与实验前一致。**",
        "本轮共完成 3 组 × 3 轮 = 9 轮，全部从同一 ImageNet 权重独立开始；"
        "子集模型只见过 21,291 张训练图，与全量冠军的训练量及核验人群不同，"
        "不能把内部核验 AUC 与冠军 0.80198 直接比较。官方验证分数也是开发观察，不是官方盲测成绩。", "",
        "## 本轮具体改了什么", "",
        "三组均为 MambaOut-Tiny、224 输入、BCE、dropout=0.2、DropPath=0.1、batch=12、训练 seed=42；"
        "使用相同患者划分、起始权重、学习率和 3 轮预算，逐轮训练。"
        "基础增强为 ±5° 旋转、最多 2% 平移和 0.95–1.05 倍缩放。",
        "A 使用基础增强；B 额外独立随机调整亮度和对比度，因子均在 0.9–1.1 内，"
        "模拟轻度曝光/显示变化，标签不变；是否有益由对照决定。"
        "参数语义见 [torchvision ColorJitter 官方说明](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html)。",
        "C 在 B 上以每批 25% 概率使用 MixUp：影像按 λ 混合，标签也按 λ 混合为软标签；"
        "λ 来自 Beta(0.1,0.1)，不是固定各取一半。soft-label BCE 的线性等价关系已在运行前检查。"
        "原论文通过样本及标签的凸组合正则化模型，但不保证本医学任务必然提分。"
        "参考 [mixup: Beyond Empirical Risk Minimization，ICLR 2018](https://arxiv.org/abs/1710.09412)。",
        "骨干来源：[MambaOut，CVPR 2025 原论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html)。", "",
        "## 验证集是否太少：按患者核实", ""]
    rows = []
    for key, label in (("official_train", "原官方训练集"), ("official_val", "原官方验证集"),
                       ("new_train", "本轮训练子集"), ("new_early_stop", "本轮开发：选轮次/方案/阈值"),
                       ("new_audit", "本轮核验：方案冻结后检查")):
        item = counts["cohorts"][key]
        rows.append([label, item["images"], item["studies"], item["patients"]])
    lines += table(["用途", "影像数", "Study 数", "患者数"], rows)
    original, audit_counts = counts["cohorts"]["official_val"], counts["cohorts"]["new_audit"]
    lines += ["", f"原验证有 859 次检查，实际只有 228 位患者；检查最多的 10 位患者占 "
        f"{original['top_patients_study_percentage']['10']:.2f}% 的 Study。"
        "同患者多次检查有关联，不能视作独立患者。",
        "本轮核验有 1,658 位患者。扩大患者数是为了提高比较可信度、减小少数患者主导的波动，不会自动提分。"]
    lines += table(["稀有类别", "原验证阳性 Study/患者", "本轮核验阳性 Study/患者"], [
        [CLASS_NAMES[c], f"{original['positive_studies_class1to10'][c-1]}/{original['positive_patients_class1to10'][c-1]}",
         f"{audit_counts['positive_studies_class1to10'][c-1]}/{audit_counts['positive_patients_class1to10'][c-1]}"]
        for c in (1, 3, 4)])
    overlap = counts["new_audit_patient_overlap_with_previous_partition_counts"]
    lines += ["", "本轮训练、开发、核验患者两两不重叠，也不与官方验证/测试患者重叠。"
        "但核验仍来自相同官方训练来源，不是新增医院或外部数据；"
        f"与以往划分存在患者重叠（旧训练 {overlap['train']}、旧开发 {overlap['early_stop']}、"
        f"旧核验 {overlap['audit']} 人），因此只称本轮预先冻结的内部核验。",
        "", "## 本轮 9 轮完整结果", "",
        "星号标记本组最优：先比开发 Study Macro AUC，再比 F1@0.5。"
        "训练 BCE 含随机增强，MixUp 还改变训练目标，所以不同组训练 BCE 不能单独判优。"]
    lines += table(["组别", "轮次", "训练 BCE", "开发 BCE", "开发 AUC", "开发 F1@0.5"], [
        [LABELS[row["组别"]], str(row["轮次"]) + (" ★" if row["本组最优"] else ""),
         fmt(row["训练BCE"]), fmt(row["开发BCE"]), fmt(row["开发StudyMacroAUC"]),
         fmt(row["开发MacroF1_阈值0.5"])] for row in epoch_rows])
    lines += ["", "每轮学习率、覆盖量、耗时、显存和 MixUp 实际批数详见 9轮明细.csv。", "",
              "## 先选方案，再打开核验集", ""]
    lines += table(["组别", "最佳轮次", "最佳开发 AUC", "最佳开发 F1@0.5", "核验角色"], [
        [LABELS[name], stages[name]["best_epoch"], fmt(stages[name]["best_score"][0]),
         fmt(stages[name]["best_score"][1]), "对照" if name == STAGES[0] else
         ("B/C 中预选增强" if name == selected else "未入选，不用核验反选")] for name in STAGES])
    lines += ["", f"B/C 中预选 **{LABELS[selected]}**，不代表它一定优于 A；即使 B/C 都弱于 A 也会产生这一预选记录。"
        "随后只对 A 和该方案做核验。最佳轮来自 history/results 的开发指标，而非单个 epoch 文件的轮次元数据。",
        "每组 10 个阈值只在各自最佳轮的开发集拟合，再冻结到内部核验和官方验证，不在核验/测试上重选阈值。"
        "开发集上用同一批标签拟合后的 F1 是拟合内结果，不当作泛化提升证据。",
        "下表“开发阈值 F1”是冻结阈值迁移后的评分，与冠军在官方验证内部五折得到的 OOF F1 "
        "不是同一估计方法，不能直接宣称提升。AUC 与 AP 不使用二分类阈值。", ""]
    for cohort, label in (("audit", "内部核验：2,842 Study / 1,658 患者"),
                          ("official_validation", "官方验证：859 Study / 228 患者")):
        lines += ["### " + label, ""]
        rows = []
        for name in (STAGES[0], selected):
            m = finals[cohort][name]
            rows.append([LABELS[name], fmt(m["macro_auc"]), fmt(m["macro_ap"]),
                         fmt(m["fixed_0_5"]["macro_f1"]), fmt(m["frozen_development_gates"]["macro_f1"])])
        lines += table(["方案", "Study Macro AUC", "Macro AP", "F1@0.5", "开发阈值 F1"], rows)
        ci = comparison[cohort]
        lines += ["", f"增强减 A 的 AUC 差值 **{ci['delta']:+.6f}**（{100*ci['delta']:+.3f} 个百分点）；"
                  f"患者配对 bootstrap 95% 区间 **[{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}]**，"
                  f"请求 400 次、有效 {ci['valid_repeats']} 次。每次整位患者重采样，保留其所有检查。",
                  "区间仅反映固定模型与当前人群的抽样变化，不涵盖重新训练的种子不确定性。", ""]
        rare_rows = []
        for name in (STAGES[0], selected):
            m = finals[cohort][name]
            for cls in (1, 3, 4):
                k = cls - 1
                for key, gate in (("fixed_0_5", "0.5"), ("frozen_development_gates", f"开发冻结 {m['gates'][k]:.3f}")):
                    d = m[key]
                    rare_rows.append([LABELS[name], CLASS_NAMES[cls], fmt(m["auc_by_class"][k]), fmt(m["ap_by_class"][k]),
                        gate, d["tp_by_class"][k], d["fp_by_class"][k], d["fn_by_class"][k], fmt(d["f1_by_class"][k])])
        lines += table(["方案", "类别", "AUC", "AP", "阈值", "TP", "FP", "FN", "F1"], rare_rows) + [""]
    lines += ["TP 是检出的阳性检查、FP 是误报、FN 是漏检。F1=0 表示当前阈值没有真阳性命中，"
              "不代表 AUC 必然随机。降低阈值可能减少漏检，也可能增加误报，所以同时列出 TP/FP/FN。",
              "", "## 其他方向：固定权重直接改输入到 320", ""]
    lines += table(["同一冠军输入", "Study Macro AUC", "Macro AP", "F1@0.5", "患者五折 OOF F1"], [
        [size, fmt(probe[key]["study"]["macro_auc"]), fmt(probe[key]["study"]["macro_ap"]),
         fmt(probe[key]["study"]["macro_f1"]), fmt(probe[key]["calibration"]["oof_macro_f1"])]
        for size, key in ((224, "baseline224"), (320, "candidate320"))])
    p_ci = probe["paired_patient_macro_auc_bootstrap"]
    lines += ["", f"320 相比 224，AUC 差值 {p_ci['delta']:+.6f}，"
        f"患者配对 95% 区间 [{p_ci['ci95'][0]:+.6f}, {p_ci['ci95'][1]:+.6f}]。"
        "本轮直接改推理分辨率无收益，保留 224；不等于“320 重新训练也无用”，后者未验证。",
        "", "## 局限与下一步", "", next_step,
        '下一项更值得单独验证的是“按检查等权训练”：官方训练集18,513次检查中，10,006次含多图，最多7张。当前逐图等权BCE会使多图检查获得更多训练权重，而评分按检查汇总。可单独比较每图损失乘以该检查图像数的倒数（并做均值归一），或直接优化检查级损失；这是由数据结构提出的待验证假设，本轮尚未训练，不能声称有效。其次应固定其他条件量化缓存与原图预处理差异；没有可靠增强信号时，不宜直接加大增强强度或延长到数十轮。',
        "每组仅 3 轮、单训练种子，学习率按 1.0/0.55/0.1 倍快速下降，"
        "只能筛查值得追踪的信号，不能宣布某种增强或 MambaOut 已达到性能上限。",
        "训练和开发用已有 512 JPEG92 缓存再缩放到 224；最终核验和官方验证用原图缩放到 224。"
        "三组规则一致，但缓存压缩/原图差异可能影响开发集选出的轮次和阈值。"
        "可在固定方案下统一输入路径确认，不能把不同集合的损失变化都解释为过拟合。",
        "核验是同来源数据，历史上研究过该来源；官方验证也已多次用于选模。"
        "这些结果不证明对新医院、新设备分布有同等表现，也不是临床效能结论。",
        "若考虑全量迁移，先冻结方案、预算和比较规则。"
        "不要反复利用已看过的核验结果调参后仍称其为未见数据。"
        "原冠军见过全量训练患者，不能拿它在本轮训练来源核验集上的成绩当作无泄漏对照。",
        "", "## 保存位置与复核", "",
        "环境沿用 E:\\比赛数据集\\envs\\cxr-smoke-cu128，没有新安装；结果和缓存都在 E 盘。"
        "没有上传 GitHub、没有改写原冠军或原提交文件。",
        f"原冠军 SHA256：{EXPECTED_CHAMPION}。",
        f"原提交 SHA256：{EXPECTED_SUBMISSION}。",
        "本报告在 CPU 重算全部 9 轮保存的检查级 AUC/F1，核对最佳轮及方案选择，重拟合开发阈值并核对冻结值；"
        "重算最终 AUC/AP/F1/混淆计数和 400 次患者 bootstrap，核对保护文件哈希。"
        "图像级 BCE 沿用训练/验证记录，不能由多图均值后的检查级预测反推。",
        "results.json 是训练总结果；cohort_audit/counts.json 是匿名人群统计；"
        "resolution_probe/metrics.json 是分辨率对照；report_checks.json 记录报告核对结果与输入文件哈希。"]
    require("torch" not in sys.modules, "Report accidentally imported torch")
    CHECKS.update(cpu_only_no_torch_import=True, input_sha256=FINGERPRINTS,
                  generated_at=datetime.now().astimezone().isoformat(), python_executable=sys.executable, status="passed")
    # No report is written until every validation above has passed.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(epoch_rows[0]))
    writer.writeheader()
    writer.writerows(epoch_rows)
    atomic_write(ROOT / "中文报告.md", "\n".join(lines) + "\n", "utf-8")
    atomic_write(ROOT / "9轮明细.csv", buffer.getvalue(), "utf-8-sig")
    atomic_write(ROOT / "report_checks.json", json.dumps(CHECKS, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(dict(state="report_completed", epochs=9, selected=selected,
                         audit=comparison["audit"], champion_replaced=False), ensure_ascii=False))


if __name__ == "__main__":
    main()

