"""Compare fixed checkpoints on the already-used official development validation set."""
import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TITLE = "## 固定权重患者配对审查"
SCOPE = ("官方开发验证集已用于模型和轮次筛选，患者配对区间描述固定模型在该队列上的"
         "条件不确定性。独立测试表现和训练随机种子变异仍待额外评估。")


def aggregate(metrics):
    return {level: {key: metrics[level][key] for key in ("macro_auc", "macro_f1", "count")}
            for level in ("image", "study")}


def main():
    if not (ROOT / "results.json").is_file():
        raise RuntimeError("Complete the main optimization before comparing checkpoints")
    status = json.loads((ROOT / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError("The main optimization status must be complete")
    result = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    import run_optimization as exp

    old = result["candidates"]["历史全量 BCE"]
    chosen = result["candidates"][result["winner"]]
    old_path, new_path = Path(old["best_checkpoint"]), ROOT / "best_model.pth"
    old_hash, new_hash = exp.hash_file(old_path), exp.hash_file(new_path)
    source_hash = exp.hash_file(Path(chosen["best_checkpoint"]))
    if old_hash != old["checkpoint_sha256"] or new_hash != source_hash:
        raise ValueError("The preserved baseline or exported winner failed SHA-256 verification")

    def check_metrics(actual, expected):
        for level in ("image", "study"):
            observed = [actual[level][key] for key in ("macro_auc", "macro_f1")]
            recorded = [expected[level][key] for key in ("macro_auc", "macro_f1")]
            if (actual[level]["count"] != expected[level]["count"] or
                    not exp.np.allclose(observed, recorded, atol=1e-6, rtol=0)):
                raise ValueError(f"Recomputed {level} macro metrics do not match the main results")

    check_metrics(result["winner_validation"], chosen["best_validation"])
    cfg = exp.yaml.safe_load((exp.reference.SMOKE / "data_local.yaml").read_text(encoding="utf-8"))
    data = exp.reference.CXRDataset(cfg["manifest"], cfg["data_root"], "val",
                                   exp.reference.build_transform(False, 224))
    counts = (len(data), len({row["study_id"] for row in data.rows}),
              len({row["subject_id"] for row in data.rows}))
    if counts != (1422, 859, 228):
        raise ValueError(f"Official development validation cohort changed: {counts}")
    identical = old_hash == new_hash
    if identical:
        check_metrics(result["winner_validation"], old["best_validation"])
        old_metrics, new_metrics = old["best_validation"], result["winner_validation"]
        interval = {"delta": 0.0, "ci95": [0.0, 0.0], "repeats": 0, "valid_repeats": 0}
    else:
        exp.torch.set_num_threads(4)
        exp.torch.backends.cudnn.benchmark = False

        def evaluate(path):
            state = exp.torch.load(path, map_location="cpu", weights_only=True)
            header = state["training"]
            model = exp.core.build_model(
                {"drop_rate": header.get("drop_rate", 0.0),
                 "drop_path_rate": header.get("drop_path_rate", 0.0)}, state["state_dict"]).eval()
            del state
            try:
                return exp.reference.validate(model, data, keep_predictions=True)
            finally:
                del model
                gc.collect()
                exp.torch.cuda.empty_cache()

        old_metrics, old_predictions = evaluate(old_path)
        new_metrics, new_predictions = evaluate(new_path)
        check_metrics(old_metrics, old["best_validation"])
        check_metrics(new_metrics, chosen["best_validation"])
        check_metrics(new_metrics, result["winner_validation"])
        targets, old_probs, subjects = old_predictions
        new_targets, new_probs, new_subjects = new_predictions
        if (not exp.np.array_equal(targets, new_targets) or
                not exp.np.array_equal(subjects, new_subjects) or
                targets.shape != (859, 10) or len(set(subjects.tolist())) != 228):
            raise ValueError("The two fixed-model study targets/patients are not aligned")
        interval = exp.core.paired_auc_interval(
            targets, old_probs, new_probs, subjects, seed=20261104, repeats=400)
        if interval["valid_repeats"] < 300:
            raise ValueError("Insufficient valid paired patient bootstrap replicates")
        expected_delta = new_metrics["study"]["macro_auc"] - old_metrics["study"]["macro_auc"]
        if not exp.np.isclose(interval["delta"], expected_delta, atol=1e-6, rtol=0):
            raise ValueError("Paired AUC delta disagrees with the verified macro metrics")
    if exp.hash_file(old_path) != old_hash or exp.hash_file(new_path) != new_hash:
        raise ValueError("A compared checkpoint changed during the fixed-model audit")
    comparison = {"winner": result["winner"], "baseline_metrics": aggregate(old_metrics),
                  "winner_metrics": aggregate(new_metrics),
                  "checkpoint_sha256": {"baseline": old_hash, "exported_winner": new_hash,
                                        "winner_source": source_hash},
                  "identical_checkpoint_bytes": identical, **interval, "bootstrap_seed": 20261104,
                  "images": 1422, "studies": 859, "patients": 228,
                  "training_updates": 0, "scope": SCOPE}
    exp.core.save_json(ROOT / "champion_comparison.json", comparison)
    report_path = ROOT / "结果.md"
    report = report_path.read_text(encoding="utf-8")
    if TITLE not in report:
        low, high = comparison["ci95"]
        paragraph = (f"\n\n{TITLE}\n\n固定权重的 Study Macro AUC 差值（最终模型减原 BCE）为 "
                     f"{comparison['delta']:+.8f}，患者配对 95% 区间为 [{low:+.8f}, {high:+.8f}]。"
                     f"队列含 859 次检查、228 名患者；本次新增训练更新为 0。{SCOPE}\n")
        if identical:
            paragraph += "两份 checkpoint 字节完全一致，差值和区间均为 0，无需重新运行模型。\n"
        report_path.write_text(report.rstrip() + paragraph, encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
