"""Fit predeclared thresholds on early_stop; report audit F1 separately from AUC."""
import contextlib
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    status_path, results_path = ROOT / "status.json", ROOT / "results.json"
    if not status_path.is_file() or not results_path.is_file():
        raise RuntimeError("The main experiment must finish and produce results.json first")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError("Calibration requires status.state == complete")
    result = json.loads(results_path.read_text(encoding="utf-8"))

    import train_generalization as exp
    from sklearn.metrics import f1_score, precision_score, recall_score
    sys.path.insert(0, str(ROOT.parent / "MambaOut_100轮"))
    from train_and_optimize import fit_gates

    selected = result["selected_stage"]
    if selected not in exp.RECIPES or status.get("selected_stage") != selected:
        raise ValueError("The recorded selected stage is inconsistent")
    stage = result["stages"][selected]
    exp.torch.set_num_threads(4)
    exp.torch.backends.cudnn.benchmark = False
    cfg = exp.yaml.safe_load((exp.SMOKE / "data_local.yaml").read_text(encoding="utf-8"))
    cached = exp.CXRDataset(cfg["manifest"], cfg["data_root"], "train",
                            exp.build_transform(False, 224), cfg["cache_dir"])
    indices, split = exp.make_patient_split(cached.rows, exp.SPLIT_SEED)
    if split["assignment_sha256"] != result["split_summary"]["assignment_sha256"]:
        raise ValueError("Patient assignments differ from the completed experiment")
    raw = exp.CXRDataset(cfg["manifest"], cfg["data_root"], "train",
                         exp.build_transform(False, 224))
    if raw.rows != cached.rows:
        raise ValueError("Raw and cached dataset row orders differ")
    early_data = exp.Subset(cached, indices["early_stop"])
    audit_data = exp.Subset(raw, indices["audit"])
    if split["partitions"]["audit"] != result["split_summary"]["partitions"]["audit"]:
        raise ValueError("The audit cohort differs from the main experiment")

    state = exp.torch.load(ROOT / selected / "best.pth", map_location="cpu", weights_only=True)
    if state["training"]["best_epoch"] != stage["best_epoch"]:
        raise ValueError("Selected best-checkpoint epoch differs from results.json")
    model = None
    try:
        model = exp.build_model(stage["settings"], state["state_dict"]).eval()
        del state
        with contextlib.redirect_stdout(sys.stderr):
            early = exp.evaluate_model(model, early_data, exp.BATCH_SIZE)
        if (early["metrics"]["image"]["count"], early["metrics"]["study"]["count"]) != (4571, 2831):
            raise ValueError("Unexpected early_stop evaluation size")
        gates = fit_gates(early["study_targets"], early["study_probabilities"])
        exp.save_json(ROOT / "selected_gates.json", {
            "selected_stage": selected, "fit_cohort": "early_stop",
            "method": "Per-class Study F1; grid 0.025:0.025:0.975; ties nearest 0.5, then lower",
            "gates": gates.tolist(),
        })
        del early
        # Gates are fixed on disk before evaluating any final-audit predictions.
        with contextlib.redirect_stdout(sys.stderr):
            audit = exp.evaluate_model(model, audit_data, exp.BATCH_SIZE)
        if (audit["metrics"]["image"]["count"], audit["metrics"]["study"]["count"],
                len(set(audit["study_subjects"]))) != (4454, 2692, 1658):
            raise ValueError("Unexpected audit images, Studies, or patient count")
        recorded = stage["audit"]["metrics"]
        for level in ("image", "study"):
            actual, expected = audit["metrics"][level], recorded[level]
            for key in ("macro_auc", "macro_f1"):
                if abs(actual[key] - expected[key]) > 1e-6:
                    raise ValueError("Audit AUC/F1 macro metrics failed checkpoint reproduction")
            for key in ("auc", "f1"):
                if len(actual[key]) != len(expected[key]) or any(
                        abs(a - b) > 1e-6 for a, b in zip(actual[key], expected[key])):
                    raise ValueError("Audit AUC/F1 class metrics failed checkpoint reproduction")

        targets, probabilities = audit["study_targets"], audit["study_probabilities"]
        classifications = {}
        for name, thresholds in (("default", 0.5), ("calibrated", gates)):
            decisions = probabilities >= thresholds
            values = {}
            for metric in (f1_score, precision_score, recall_score):
                key = metric.__name__.removesuffix("_score")
                by_class = metric(targets, decisions, average=None, zero_division=0)
                values[key] = by_class.tolist()
                values["macro_" + key] = float(by_class.mean())
            classifications[name] = values
        study = audit["metrics"]["study"]
        output = {
            "cohort": {"name": "audit", "images": 4454, "studies": 2692, "patients": 1658},
            "stage": selected, "gates": gates.tolist(), **classifications,
            "auc": {"macro_auc": study["macro_auc"], "by_class": study["auc"]},
            "training_updates": 0,
            "note": "Threshold fitting affects F1, precision and recall; AUC is unchanged.",
        }
        exp.save_json(ROOT / "threshold_results.json", output)
    finally:
        model = None
        gc.collect()
        exp.torch.cuda.empty_cache()

    compact = {"stage": selected, "macro_auc": output["auc"]["macro_auc"], "training_updates": 0}
    for name in ("default", "calibrated"):
        compact[name] = {key: value for key, value in output[name].items() if key.startswith("macro_")}
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
