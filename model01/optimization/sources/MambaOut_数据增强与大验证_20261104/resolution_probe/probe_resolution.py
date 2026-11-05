"""Fixed champion, official validation, input 224 versus 320; never train or read test."""
import os
import sys
import hashlib
import json
import gc
from pathlib import Path
from datetime import datetime
from time import perf_counter
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
TEMP = ROOT / "temp"
TEMP.mkdir(exist_ok=True)
for key, value in {
    "TEMP": TEMP, "TMP": TEMP, "TMPDIR": TEMP,
    "CUDA_CACHE_PATH": TEMP / "cuda", "HF_HOME": TEMP / "huggingface",
    "TORCH_HOME": TEMP / "torch", "XDG_CACHE_HOME": TEMP,
}.items():
    os.environ[key] = str(value)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "MambaOut_泛化优化"))
import train_generalization as core
sys.path.insert(0, str(BASE / "MambaOut_100轮"))
import train_and_optimize as ref
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import GroupKFold

EXPECTED = "9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585"
WEIGHT = BASE / "MambaOut_ASL对照" / "best_model.pth"
BASELINE_DIR = BASE / "MambaOut_F1正则化优化_20261104"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()
def status(state, **kwargs):
    core.save_json(ROOT / "status.json", dict(state=state,pid=os.getpid(),
        time=datetime.now().astimezone().isoformat(),**kwargs))
def scores(y,p):
    auc = roc_auc_score(y,p,average=None)
    ap = average_precision_score(y,p,average=None)
    f1 = f1_score(y,p>=.5,average=None,zero_division=0)
    return dict(macro_auc=float(auc.mean()),macro_ap=float(ap.mean()),macro_f1=float(f1.mean()),
        auc=auc.tolist(),ap=ap.tolist(),f1=f1.tolist())
def detailed_gates(y,p,s):
    decisions=np.zeros_like(y)
    folds=[]
    for index,(fit,held) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s),1):
        assert not set(s[fit]).intersection(s[held])
        gates=ref.fit_gates(y[fit],p[fit])
        decisions[held]=p[held]>=gates
        folds.append(dict(fold=index,fit_studies=len(fit),held_out_studies=len(held),
            fit_patients=len(np.unique(s[fit])),held_out_patients=len(np.unique(s[held])),
            thresholds=gates.tolist()))
    f1=f1_score(y,decisions,average=None,zero_division=0)
    return dict(oof_macro_f1=float(f1.mean()),oof_f1_by_class_id=f1.tolist(),
        final_gates_by_class_id=ref.fit_gates(y,p).tolist(),patients=len(np.unique(s)),folds=folds),decisions

def main():
    start=perf_counter()
    status("preparing")
    assert digest(WEIGHT)==EXPECTED
    baseline_file=BASELINE_DIR / "baseline_predictions.npz"
    with np.load(baseline_file,allow_pickle=False) as archive:
        old_y=archive["targets"];old_p=archive["probabilities"];old_s=archive["subjects"]
    assert old_y.shape==old_p.shape==(859,10) and old_s.shape==(859,)
    assert len(np.unique(old_s))==228
    original=json.loads((BASELINE_DIR / "baseline_metrics.json").read_text(encoding="utf-8"))
    old_scores=scores(old_y,old_p)
    assert abs(old_scores["macro_auc"]-original["validation"]["study"]["macro_auc"])<1e-12
    assert abs(old_scores["macro_f1"]-original["validation"]["study"]["macro_f1"])<1e-12
    old_gates,old_decisions=detailed_gates(old_y,old_p,old_s)
    assert abs(old_gates["oof_macro_f1"]-original["calibration"]["oof_macro_f1"])<1e-12
    config=yaml.safe_load((BASE / "本地跑通" / "data_local.yaml").read_text(encoding="utf-8"))
    checkpoint=torch.load(WEIGHT,map_location="cpu",weights_only=True)
    settings=checkpoint["training"]
    assert settings["drop_rate"]==.2 and settings["drop_path_rate"]==.1
    assert digest(Path(config["manifest"]))==settings["manifest_sha256"]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    model=core.build_model(settings,checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    data=ref.CXRDataset(config["manifest"],config["data_root"],"val",ref.build_transform(False,320),cache_dir=None)
    assert len(data)==1422
    ref.ARGS.batch_size=12
    gpu=torch.cuda.get_device_name()
    status("inference320",images=1422,batch_size=12)
    print("Running fixed champion at 320; raw official validation images=1422; batch=12.",flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    infer_start=perf_counter()
    metrics,predictions=ref.validate(model,data,keep_predictions=True)
    torch.cuda.synchronize()
    infer_seconds=perf_counter()-infer_start
    memory=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    y,p,s=predictions
    assert np.array_equal(y,old_y) and np.array_equal(s,old_s)
    np.savez_compressed(ROOT / "predictions.npz",targets=y,probabilities=p,subjects=s)
    del model,checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    status("gpu_released",inference_seconds=infer_seconds,**memory)
    print("GPU_RELEASED",flush=True)
    new_scores=scores(y,p)
    calibration,decisions=detailed_gates(y,p,s)
    np.savez_compressed(ROOT / "threshold_oof_decisions.npz",targets=y,subjects=s,
        decisions224=old_decisions,decisions320=decisions)
    interval=core.paired_auc_interval(y,old_p,p,s,seed=20261104,repeats=400)
    assert digest(WEIGHT)==EXPECTED
    result=dict(
        purpose="Frozen S7 champion input-resolution probe; no training and no test data accessed.",
        checkpoint=str(WEIGHT),checkpoint_sha256=EXPECTED,training_settings=settings,
        baseline_predictions=str(baseline_file),baseline_predictions_sha256=digest(baseline_file),
        manifest_sha256=digest(Path(config["manifest"])),
        source="official val; raw images; cache_dir=None; 1422 images, 859 studies, 228 patients",
        protocol="Same ref.validate; float16 autocast logits converted to float32 before sigmoid; study probability mean; batch12; only transform size changed to320.",
        baseline224=dict(study=old_scores,calibration=old_gates),
        candidate320=dict(validation=metrics,study=new_scores,calibration=calibration),
        paired_patient_macro_auc_bootstrap=interval,
        alignment=dict(targets_identical=True,subjects_identical=True),
        runtime=dict(gpu=gpu,inference_seconds=infer_seconds,**memory,total_seconds=perf_counter()-start),
        limitations=[
            "Official validation has already been reused for model selection; this is not independent test evidence.",
            "OOF threshold fitting separates patients within this validation set but does not undo previous model-selection reuse.",
            "Patient bootstrap conditions on fixed weights and this cohort; no training-seed uncertainty.",
            "Runtime is this RTX5060 Laptop GPU end-to-end validation, not competition T4 latency certification.",
            "Higher resolution without retraining can change feature statistics; this only tests a direct input-size change."
        ],
        decision="Keep224; do not scan additional resolutions." if interval["delta"]<=0 else "Exploratory320 improvement; requires independent confirmation; preserve224 champion."
    )
    core.save_json(ROOT / "metrics.json",result)
    lines=["# Fixed-weight resolution probe","","| Input | Study macro AUC | Macro AP | F1 @0.5 | Patient 5-fold OOF F1 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for size,sc,ga in [(224,old_scores,old_gates),(320,new_scores,calibration)]:
        lines.append(f"| {size} | {sc['macro_auc']:.8f} | {sc['macro_ap']:.8f} | {sc['macro_f1']:.8f} | {ga['oof_macro_f1']:.8f} |")
    lines += ["",f"Paired patient bootstrap (400 requested): {interval}","",result["decision"],"",
        "| Class ID | Positive studies | AUC224 | AUC320 | AP224 | AP320 | F1@0.5 224 | F1@0.5 320 | OOF F1 224 | OOF F1 320 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in range(10):
        vals=[old_scores['auc'][k],new_scores['auc'][k],old_scores['ap'][k],new_scores['ap'][k],
            old_scores['f1'][k],new_scores['f1'][k],old_gates['oof_f1_by_class_id'][k],calibration['oof_f1_by_class_id'][k]]
        lines.append("| "+str(k+1)+" | "+str(int(y[:,k].sum()))+" | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines += ["",f"Checkpoint SHA256: {EXPECTED}",f"Runtime: {result['runtime']}",""]+result["limitations"]
    (ROOT / "README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    status("complete",macro_auc224=old_scores["macro_auc"],macro_auc320=new_scores["macro_auc"],
        delta=interval["delta"],gpu_released=True)
    print(json.dumps(dict(macro_auc224=old_scores["macro_auc"],macro_auc320=new_scores["macro_auc"],
        macro_f1_320=new_scores["macro_f1"],oof_f1_320=calibration["oof_macro_f1"],
        paired_bootstrap=interval,runtime=result["runtime"]),ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        status("failed",error=repr(exc))
        raise


