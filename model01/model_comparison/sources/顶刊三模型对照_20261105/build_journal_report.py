"""CPU-only verified backbone report. --help performs no experiment reads/writes.

A final report requires three successful five-epoch journal models. A terminal failure
run is documented as incomplete, including complete epochs from failed stages.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
MODELS=("caformer_s18","densenet121","hrnet_w18")
NAMES={"caformer_s18":"CAFormer-S18","densenet121":"DenseNet-121","hrnet_w18":"HRNet-W18"}
ARCH={"caformer_s18":"caformer_s18.sail_in1k","densenet121":"densenet121.ra_in1k","hrnet_w18":"hrnet_w18.ms_aug_in1k"}
LICENSES={"caformer_s18":"apache-2.0","densenet121":"apache-2.0","hrnet_w18":"mit"}
CLASSES=("纵隔心影增大","气胸","肺部实变","肺炎","肺水肿","心影扩大","肺不张","肺部阴影","胸腔积液","无异常发现")
ORIGINAL_AUC=.8019769550500871
GAMMA_AUC=.8021774782195852
ORIGINAL=DATA/"MambaOut_ASL对照"
GAMMA=DATA/"MambaOut_二次数据增强_20261104"
REQUIRED_PROTECTED={
    str(ORIGINAL/"best_model.pth"):"9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585",
    str(ORIGINAL/"gates.json"):"1ef20f260ee1ea2e3e7737a8aa472537ebb8f43dbdf3198cd3e5264acba5f380",
    str(GAMMA/"C_gamma_noise/best.pth"):"b0066e479b1b0c383fa7daa2067ff0d6c954d20fbf81d1202af95481d67cc5f9",
    str(DATA/"提交结果/MambaOut_S7微调第1轮_20261104/submission.csv"):"d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8",
}
FINGERPRINTS={}
CHECKS=[]

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def read(path):
    path=Path(path);FINGERPRINTS[str(path)]=sha(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def near(a,b,message,tol=1e-12):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=0,atol=tol),message)

def dependencies():
    global np,roc_auc_score,average_precision_score,f1_score,GroupKFold
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="4"
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["PYTHONDONTWRITEBYTECODE"]="1";sys.dont_write_bytecode=True
    for key in ("TEMP","TMP","TMPDIR","XDG_CACHE_HOME"):os.environ[key]=str(ROOT/"temp")
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
    from sklearn.model_selection import GroupKFold

def prediction(path,reference=None):
    path=Path(path);FINGERPRINTS[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as archive:out={key:archive[key].copy() for key in archive.files}
    require(all(key in out for key in ("targets","probabilities","subjects","study_ids")),"Missing NPZ fields: "+str(path))
    y,p,s,ids=(out[key] for key in ("targets","probabilities","subjects","study_ids"))
    require(y.shape==p.shape==(859,10) and s.shape==ids.shape==(859,),"Wrong validation shape: "+str(path))
    require(len(np.unique(s))==228 and len(np.unique(ids))==859,"Wrong patient/study coverage")
    require(np.isin(y,(0,1)).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),"Invalid labels/probabilities")
    require(((y.sum(0)>0)&(y.sum(0)<len(y))).all(),"Class lacks both labels")
    if reference is not None:
        for key in ("targets","subjects","study_ids"):require(np.array_equal(out[key],reference[key]),"Prediction alignment: "+str(path)+" "+key)
    return out

def fit_gates(y,p):
    candidates=sorted(np.arange(1,40)/40,key=lambda g:(abs(g-.5),g))
    scores=[f1_score(y,p>=g,average=None,zero_division=0) for g in candidates]
    return np.asarray(candidates)[np.argmax(scores,axis=0)]

def calibration(arrays):
    y,p,s=(arrays[key] for key in ("targets","probabilities","subjects"))
    z=np.zeros_like(y,dtype=bool)
    for fit,held in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s):
        require(not set(s[fit])&set(s[held]),"Patient leakage in threshold CV")
        z[held]=p[held]>=fit_gates(y[fit],p[fit])
    f=f1_score(y,z,average=None,zero_division=0)
    return dict(oof_macro_f1=float(f.mean()),oof_f1_by_class_id=f.tolist(),
        final_gates_by_class_id=fit_gates(y,p).tolist(),patients=228),z

def confusion(y,z):
    return dict(tp=np.sum(z&(y==1),0).astype(int).tolist(),
        fp=np.sum(z&(y==0),0).astype(int).tolist(),fn=np.sum(~z&(y==1),0).astype(int).tolist())

def metrics(arrays):
    y,p=arrays["targets"],arrays["probabilities"];z=p>=.5
    return dict(auc=roc_auc_score(y,p,average=None).tolist(),f1=f1_score(y,z,average=None,zero_division=0).tolist(),
        macro_auc=float(roc_auc_score(y,p,average="macro")),macro_f1=float(f1_score(y,z,average="macro",zero_division=0)),
        macro_ap=float(average_precision_score(y,p,average="macro")),ap_by_class=average_precision_score(y,p,average=None).tolist(),**confusion(y,z))

def check_calibration(measured,recorded,description):
    for key in measured:near(measured[key],recorded[key],description+" "+key)

def verify_hashes(mapping,description):
    require(isinstance(mapping,dict) and mapping,description+" empty")
    for path,expected in mapping.items():
        require(isinstance(expected,str) and re.fullmatch(r"[0-9a-f]{64}",expected),description+" invalid SHA")
        require(sha(path)==expected,description+" changed: "+path)

def score(row):return row["validation"]["study"]["macro_auc"],row["validation"]["study"]["macro_f1"]

def check_interval(record,delta,description):
    near(record["delta"],delta,description+" point delta")
    ci=record["ci95"]
    require(len(ci)==2 and all(math.isfinite(float(v)) for v in ci) and ci[0]<=ci[1],description+" invalid CI")
    require(record["repeats"]==400 and 300<=record["valid_repeats"]<=400,description+" repeats differ")

def provenance_check(manifest):
    require(set(manifest["models"])==set(MODELS),"Wrong pretrained model set")
    license_sources=read(ROOT/"pretrained/license_sources.json")
    expected_sources={
        "LICENSE.apache-2.0.txt":"https://www.apache.org/licenses/LICENSE-2.0.txt",
        "LICENSE.hrnet-MIT.txt":"https://raw.githubusercontent.com/HRNet/HRNet-Image-Classification/master/LICENSE"}
    license_hashes={}
    for filename,url in expected_sources.items():
        record=license_sources[filename];path=ROOT/"pretrained"/filename
        require(record["url"]==url and sha(path)==record["sha256"],"License text/source differs: "+filename)
        license_hashes[filename]=record["sha256"]
    records={}
    for name in MODELS:
        entry=manifest["models"][name]
        require(entry["architecture"]==ARCH[name],"Wrong pretrained architecture")
        require(entry["license"].lower()==LICENSES[name] and entry["pretraining_data"].lower()=="imagenet-1k","Unexpected license/domain")
        require(re.fullmatch(r"[0-9a-f]{40}",entry["revision"]),"Model revision is not pinned")
        directory=(ROOT/"pretrained"/entry["directory"]).resolve()
        require(directory.is_relative_to((ROOT/"pretrained").resolve()),"Pretrained directory escaped root")
        metadata=read(directory/"repository_metadata.json");config=read(directory/"config.json")
        require(metadata["sha"]==entry["revision"] and metadata["id"]==entry["repository"],"Model-card revision mismatch")
        require(metadata.get("private") is False and metadata.get("gated") is False,"Private/gated model")
        require(metadata["cardData"]["license"].lower()==LICENSES[name],"Model-card license mismatch")
        ds=metadata["cardData"]["datasets"];ds=[ds] if isinstance(ds,str) else ds
        require("imagenet-1k" in [str(v).lower() for v in ds],"Model-card dataset mismatch")
        lfs=[v for v in metadata["siblings"] if v["rfilename"]=="model.safetensors"]
        require(len(lfs)==1 and lfs[0]["lfs"]["sha256"]==entry["sha256"],"Pretrained LFS provenance mismatch")
        weights=directory/"model.safetensors"
        require(sha(weights)==entry["sha256"] and weights.stat().st_size==entry["bytes"],"Pretrained file changed")
        require(config==entry["config"],"Downloaded config differs from manifest")
        card=directory/"README.md";require(card.is_file(),"Archived model card missing")
        filename="LICENSE.hrnet-MIT.txt" if name=="hrnet_w18" else "LICENSE.apache-2.0.txt"
        records[name]=dict(**entry,repository_metadata_sha256=sha(directory/"repository_metadata.json"),
            model_card_sha256=sha(card),license_text_sha256=license_hashes[filename],license_text_file=filename)
    CHECKS.append("pretrained_weights_fixed_revisions_and_archived_license_metadata")
    return records


def audit_construction(name,info,checked):
    expected_kwargs={
        "caformer_s18":dict(num_classes=1000,drop_rate=.2,drop_path_rate=.1,proj_drop_rate=0.),
        "densenet121":dict(num_classes=1000,drop_rate=.2,proj_drop_rate=0.),
        "hrnet_w18":dict(num_classes=1000,drop_rate=.2)}
    paths={"caformer_s18":"head.fc.fc2","densenet121":"classifier","hrnet_w18":"classifier"}
    require(info["constructor_kwargs"]==expected_kwargs[name],"Actual constructor kwargs differ")
    require(info["classifier_path"]==paths[name] and info["output_classes"]==10 and info["original_num_classes"]==1000,"Classifier mapping differs")
    require(info["classifier_adaptation"]=="Only final Linear replaced; pretrained hidden head/norm retained","Classifier adaptation differs")
    require(info["classifier_lr_scope"]=="Only new final Linear has classifier_lr; all other pretrained parameters use backbone_lr","Classifier LR scope differs")
    near(info["head_drop_rate"],.2,"Head dropout differs")
    near(info["actual_drop_path_rate"],.1 if name=="caformer_s18" else 0.,"Actual DropPath differs")
    if name=="densenet121":near(info["dense_layer_drop_rate"],0.,"Dense-layer dropout differs")
    audit=checked["construction_audit"]
    require(audit["source"]=="ImageNet" and audit["final_linear_only_replaced"] is True and audit["all_nonfinal_pretrained_tensors_bit_equal"] is True,"Pretrained tensor retention audit failed")
    require(audit["retained_tensor_count"]>0,"No retained pretrained tensors")
    if name=="caformer_s18":
        special=audit["caformer_hidden_head_preservation"]
        for prefix in ("head.fc.fc1.","head.fc.norm.","head.norm."):
            require(prefix in special and special[prefix]["bit_equal_to_pretrained"] is True,"CAFormer hidden head/norm not retained")
        require(special["head.fc.fc1."]["tensors"]>0,"CAFormer pretrained hidden Linear missing")
    dropout=checked["dropout_sites"];paths_audit=checked["drop_path_sites"]
    require(all(0<=float(value)<1 for value in list(dropout.values())+list(paths_audit.values())),"Invalid dropout sites")
    if name!="caformer_s18":require(not paths_audit,"Unexpected DropPath in CNN candidate")
    CHECKS.append(name+":final_linear_only,constructor_kwargs,pretrained_retention,actual_regularization")
    return dict(constructor_kwargs=info["constructor_kwargs"],classifier_path=info["classifier_path"],
        classifier_adaptation=info["classifier_adaptation"],classifier_lr_scope=info["classifier_lr_scope"],
        head_drop_rate=info["head_drop_rate"],actual_drop_path_rate=info["actual_drop_path_rate"],
        dense_layer_drop_rate=info["dense_layer_drop_rate"],dropout_sites=dropout,drop_path_sites=paths_audit,construction_audit=audit)

def audit_stage(name,stage,common,protected,reference,provenance):
    folder=ROOT/name;successful=stage.get("state")=="complete"
    hp=folder/"history.json";history=read(hp) if hp.is_file() else []
    require(isinstance(history,list) and len(history)<=5,"Invalid history: "+name)
    require([r["epoch"] for r in history]==list(range(1,len(history)+1)),"Non-contiguous history")
    require(stage.get("state") in ("complete","failed","not_recorded"),"Unexpected stage state")
    if successful:
        require(len(history)==stage["epochs_completed"]==5,"Successful model must have exactly five epochs")
        require(stage==read(folder/"results.json") and history==stage["history"],"History/result differs")
    parameters=None;construction=None
    if history or successful:
        plan=read(folder/"plan.json");checked=read(ROOT/f"preflight_{name}.json")

        require(plan["common"]==common==checked["common"],"Frozen recipe differs")
        require(plan["protected_hashes"]==protected==checked["protected_hashes"],"Plan protected hashes differ")
        require(plan["source_hashes"]==checked["source_hashes"],"Run and preflight source hashes differ")
        verify_hashes(plan["source_hashes"],"Frozen experiment source")
        require(plan["model"]==checked["model"],"Preflight model metadata differs")
        info=plan["model"];config=provenance[name]["config"]["pretrained_cfg"]
        construction=audit_construction(name,info,checked)
        require(info["name"]==name and info["architecture"]==ARCH[name],"Wrong stage model")
        require(info["sha256"]==provenance[name]["sha256"] and info["pretrained_config"]==provenance[name]["config"],"Stage pretraining differs")
        near(info["mean"],config["mean"],"Normalization mean differs");near(info["std"],config["std"],"Normalization std differs")
        require(checked["state"]=="passed" and checked["cpu_weight_load_strict"] is True,"CPU preflight did not pass")
        for key,value in dict(train_images=30344,train_studies=18513,train_patients=11055,validation_images=1422,validation_studies=859,validation_patients=228).items():
            require(checked["data_audit"][key]==value,"Preflight cohort differs: "+key)
        require(checked["data_audit"]["patient_splits_disjoint"] is True,"Patient split audit failed")
        parameters=checked["parameters"]
    rows=[];saved={}
    for row in history:
        epoch=row["epoch"]
        require(row["images"]==30344 and row["validation"]["image"]["count"]==1422 and row["validation"]["study"]["count"]==859,"Incomplete epoch coverage")
        require(math.isfinite(row["train_loss"]) and row["train_loss"]>=0,"Invalid train BCE")
        for key in ("train_seconds","validation_seconds","peak_allocated_mib"):
            require(math.isfinite(row[key]) and row[key]>0,"Invalid timing or memory")
        drops=row["grad_scaler_scale_decreases"]
        require(isinstance(drops,int) and 0<=drops<=math.ceil(30344/12),"Invalid scaler adjustment count")
        near(row["learning_rates"],[common["backbone_lr"]*common["lr_factors"][epoch-1],common["classifier_lr"]*common["lr_factors"][epoch-1]],"Recorded LR differs")
        arrays=prediction(folder/f"epoch_{epoch}_predictions.npz",reference)
        m=metrics(arrays);cal,z=calibration(arrays)
        for key in ("auc","f1","macro_auc","macro_f1"):near(m[key],row["validation"]["study"][key],name+" metrics "+key)
        for key in ("macro_ap","ap_by_class","tp","fp","fn"):near(m[key],row["extra"][key],name+" extras "+key)
        check_calibration(cal,row["calibration"],name+" patient threshold CV")
        require((folder/f"epoch_{epoch}.pth").is_file(),"Recorded epoch checkpoint missing")
        saved[epoch]=dict(metrics=m,calibration=cal,oof_confusion=confusion(arrays["targets"],z))
        rows.append(dict(model=name,epoch=epoch,stage_state=stage.get("state"),images=30344,train_loss=row["train_loss"],
            study_auc=m["macro_auc"],f1_at_0_5=m["macro_f1"],patient_oof_f1=cal["oof_macro_f1"],macro_ap=m["macro_ap"],
            train_seconds=row["train_seconds"],validation_seconds=row["validation_seconds"],
            peak_allocated_mib=row["peak_allocated_mib"],scaler_decreases=drops))
        CHECKS.append(f"{name}/epoch{epoch}:coverage,AUC,F1,AP,TP_FP_FN,patient_CV")
    best_row=max(history,key=score) if history else None
    best=saved[best_row["epoch"]] if best_row else None
    if successful:
        require(best_row["epoch"]==stage["best_epoch"],"Best epoch differs")
        require(stage["architecture"]==ARCH[name],"Completed architecture differs")
        require(stage["best_validation"]==best_row["validation"] and stage["best_calibration"]==best_row["calibration"],"Best metric selection differs")
        best_path=folder/"best.pth"
        require(Path(stage["best_checkpoint"]).resolve()==best_path.resolve() and sha(best_path)==stage["best_checkpoint_sha256"],"Best checkpoint path/SHA differs")
        v=stage["verification"]
        require(v["state"]=="passed" and v["epoch"]==stage["best_epoch"] and v["metrics_reproduced"] is True and v["predictions_reproduced"] is True and v["bn_buffers_unchanged"] is True,"Best GPU reload verification absent/failed")
        require(0<=v["max_abs_probability_difference"]<=1e-7,"Reload probability difference too large")
        require(stage["protected_hashes_after"]==protected and stage["no_test_inference"] is True and stage["github_uploaded"] is False,"Unexpected stage side effect")
        gate=read(folder/"candidate_gates.json")
        require(gate["checkpoint_sha256"]==stage["best_checkpoint_sha256"] and Path(gate["checkpoint"]).resolve()==best_path.resolve() and gate["class_ids"]==list(range(1,11)),"Candidate gates not bound to best checkpoint")
        check_calibration(best["calibration"],gate,name+" candidate gates")
        for key,auc in (("original_mambaout",ORIGINAL_AUC),("latest_gamma_candidate",GAMMA_AUC)):
            check_interval(stage["paired_auc_intervals"][key],best["metrics"]["macro_auc"]-auc,name+" "+key)
        CHECKS.append(name+":best_selection,SHA,GPU_verification,gates,bootstrap_points")
    return dict(name=name,state=stage.get("state"),actual_epochs=len(history),successful=successful,
        history=history,rows=rows,best_epoch=best_row["epoch"] if best_row else None,best=best,parameters=parameters,construction=construction,source=stage)

def ensemble_metrics_match(measured,recorded,description):
    keys={"auc_by_class":"auc","f1_at_0_5_by_class":"f1","macro_auc":"macro_auc",
        "macro_f1_at_0_5":"macro_f1","macro_ap":"macro_ap","ap_by_class":"ap_by_class",
        "tp_at_0_5":"tp","fp_at_0_5":"fp","fn_at_0_5":"fn"}
    for target,source in keys.items():near(measured[source],recorded[target],description+" "+target)


def audit_fixed_ensembles(audited,gamma,gm,gc,protected):
    plan_path=ROOT/"fixed_ensemble_plan.json";result_path=ROOT/"fixed_ensemble_results.json"
    appendix=ROOT/"等权组合对照.md"
    if not plan_path.is_file() and not result_path.is_file():return None
    require(plan_path.is_file() and result_path.is_file() and appendix.is_file(),
        "Fixed-ensemble plan exists: run evaluate_fixed_ensembles.py after all three models before final report")
    require(all(audited[name]["successful"] and audited[name]["actual_epochs"]==5 for name in MODELS),
        "Three fixed ensembles require three complete, GPU-verified five-epoch single models")
    plan=read(plan_path);result=read(result_path)
    require(result["state"]=="complete" and result["kind"]==plan["kind"]=="fixed_probability_ensembles_not_new_single_models","Unexpected ensemble result kind/state")
    require(result["plan"]==plan and plan["model_keys"]==list(MODELS),"Ensemble plan/model set differs")
    require(plan["predetermined_date"]=="2026-11-05" and plan["alpha"]==.5 and plan["source_weights"]==[.5,.5],"Fixed weights/date differ")
    require(plan["constraints"]==dict(require_three_models_times_five_epochs=True,search_mixing_weights=False,new_training_epochs=0,test_inference=False,github_upload=False),"Ensemble constraints differ")
    require(plan["cohort"]==dict(split="official validation",images=1422,studies=859,patients=228,classes=10,same_study_aggregation="mean image probabilities per source"),"Ensemble cohort differs")
    require(plan["threshold_protocol"]==dict(folds=5,shuffle=True,seed=42,grid="0.025..0.975 step0.025",tie_break="closest to 0.5, then smaller"),"Ensemble threshold protocol differs")
    require(plan["bootstrap"]==dict(repeats=400,seed=20261104,unit="patient cluster",percentiles=[.025,.975]),"Ensemble bootstrap protocol differs")
    arithmetic="float64 equal mean of saved Study mean image probabilities"
    require(plan["probability_arithmetic"]==arithmetic,"Ensemble arithmetic differs")
    baseline_path=GAMMA/"C_gamma_noise/best.pth";baseline_pred=GAMMA/"verified_candidate_predictions.npz"
    baseline_sha=REQUIRED_PROTECTED[str(baseline_path)]
    require(Path(plan["baseline"]["checkpoint"]).resolve()==baseline_path.resolve() and plan["baseline"]["checkpoint_sha256"]==sha(baseline_path)==baseline_sha,"Ensemble baseline checkpoint differs")
    require(Path(plan["baseline"]["predictions"]).resolve()==baseline_pred.resolve() and sha(baseline_pred)==plan["baseline"]["predictions_sha256"],"Ensemble baseline prediction differs")
    near(plan["baseline"]["reference_auc"],GAMMA_AUC,"Plan baseline AUC differs")
    require(result["completed_single_model_epochs"]==dict.fromkeys(MODELS,5) and result["new_training_epochs"]==0,"Ensemble evaluation changed epoch count")
    require(result["no_test_inference"] is True and result["github_uploaded"] is False and result["all_input_hashes_unchanged"] is True,"Ensemble side effects differ")
    require(result["protected_hashes_after"]==protected,"Ensemble protected receipt differs")
    fingerprints=result["input_fingerprints"];verify_hashes(fingerprints,"Fixed-ensemble evaluation input")
    require(fingerprints.get(str(plan_path))==sha(plan_path),"Fixed plan SHA is absent or differs")
    require(result["tool_sha256"]==sha(ROOT/"evaluate_fixed_ensembles.py"),"Fixed-ensemble tool changed since evaluation")
    ensemble_metrics_match(gm,result["baseline_metrics"],"Ensemble baseline metrics")
    check_calibration(gc,result["baseline_calibration"],"Ensemble baseline patient thresholds")
    require(set(result["ensembles"])==set(MODELS),"Expected exactly three fixed combinations")
    text=appendix.read_text(encoding="utf-8-sig");FINGERPRINTS[str(appendix)]=sha(appendix)
    verified={}
    for name in MODELS:
        item=result["ensembles"][name];single=audited[name];stage=single["source"]
        require(item["state"]=="complete","Incomplete fixed combination")
        binding=item["binding"]
        require(binding["kind"]=="probability_ensemble_not_new_single_model" and binding["weights"]==[.5,.5],"Combination is not fixed 50:50 probabilities")
        require(binding["single_model_epoch"]==single["best_epoch"] and binding["probability_arithmetic"]==arithmetic,"Combination epoch/arithmetic differs")
        require(binding["aggregation"]=="Each source first averages image probabilities within the same Study; then sources are averaged 50:50","Combination Study/source aggregation differs")
        sources=binding["source_checkpoints"]
        expected=[("existing_gamma_mambaout",baseline_path,baseline_sha),(name,ROOT/name/"best.pth",stage["best_checkpoint_sha256"])]
        require(len(sources)==2,"Ensemble must bind exactly two models")
        for entry,(role,path,digest) in zip(sources,expected):
            require(entry["role"]==role and Path(entry["path"]).resolve()==path.resolve() and entry["sha256"]==sha(path)==digest and entry["weight"]==.5,"Ensemble source checkpoint binding differs")
        folder=ROOT/"fixed_ensembles"/name;pred_path=folder/"predictions.npz";gate_path=folder/"candidate_gates.json"
        require(Path(item["predictions"]).resolve()==pred_path.resolve() and sha(pred_path)==item["predictions_sha256"],"Ensemble output predictions SHA/path differs")
        require(Path(item["candidate_gates"]).resolve()==gate_path.resolve() and sha(gate_path)==item["candidate_gates_sha256"],"Ensemble output gates SHA/path differs")
        arrays=prediction(pred_path,gamma)
        prior=prediction(ROOT/name/f"epoch_{single['best_epoch']}_predictions.npz",gamma)
        expected_p=.5*gamma["probabilities"].astype(np.float64)+.5*prior["probabilities"].astype(np.float64)
        require(arrays["probabilities"].dtype==np.dtype("float64") and np.array_equal(arrays["probabilities"],expected_p),"Saved ensemble is not the declared exact equal probability mean")
        measured=metrics(arrays);cal,z=calibration(arrays)
        ensemble_metrics_match(measured,item["metrics"],"Recomputed fixed combination "+name)
        check_calibration(cal,item["calibration"],"Recomputed fixed-combination patient thresholds")
        require("oof_decisions" in arrays and np.array_equal(arrays["oof_decisions"],z),"Fixed-ensemble OOF decisions differ")
        ensemble_metrics_match(single["best"]["metrics"],item["original_single_metrics"],"Combination source single-model score")
        check_calibration(single["best"]["calibration"],item["original_single_calibration"],"Combination source single calibration")
        gates=read(gate_path)
        require(gates["binding"]==binding and gates["class_ids"]==list(range(1,11)) and gates["candidate_only"] is True,"Combination gates source binding differs")
        require(gates["decision_rule"]=="Fixed 50:50 Study probability >= class Gate","Combination gate comparison rule differs")
        require(gates["no_test_inference"] is True and gates["github_uploaded"] is False,"Combination gate side effects differ")
        check_calibration(cal,gates,"Combination final gates")
        check_interval(item["paired_auc_vs_existing_gamma"],measured["macro_auc"]-GAMMA_AUC,"Combination paired CI")
        require(item["paired_auc_vs_existing_gamma"]["seed"]==plan["bootstrap"]["seed"],"Ensemble bootstrap seed differs")
        require(item["exceeds_existing_gamma_auc"]==(measured["macro_auc"]>GAMMA_AUC),"Ensemble exceed flag differs")
        row=f"| MambaOut＋{NAMES[name]} | {measured['macro_auc']:.8f} | {measured['macro_f1']:.6f} | {cal['oof_macro_f1']:.6f} |"
        require(row in text,"Ensemble appendix table differs from audited JSON")
        verified[name]=dict(metrics=measured,calibration=cal,binding=binding,
            paired_auc_vs_existing_gamma=item["paired_auc_vs_existing_gamma"],predictions_sha256=item["predictions_sha256"],
            candidate_gates_sha256=item["candidate_gates_sha256"],exceeds_existing_gamma_auc=item["exceeds_existing_gamma_auc"])
        CHECKS.append(name+":fixed50_50_probability_identity,source_bindings,metrics,patient_CV,gates,CI_points,appendix")
    winner=max(MODELS,key=lambda name:(verified[name]["metrics"]["macro_auc"],verified[name]["metrics"]["macro_f1"]))
    require(result["highest_point_estimate_combination"]==winner,"Highest combination selection differs")
    verify_hashes(fingerprints,"Fixed-ensemble input after report audit")
    return dict(state="passed",kind=result["kind"],combinations=verified,highest_point_estimate_combination=winner,
        plan_sha256=sha(plan_path),result_sha256=sha(result_path),appendix_path=str(appendix),appendix_sha256=sha(appendix),
        new_training_epochs=0,paired_bootstrap_recomputed=False,probabilities_and_patient_cv_recomputed=True)


def table(headers,rows):
    def clean(value):return str(value).replace("|","\\|").replace("\n"," ")
    return ["| "+" | ".join(map(clean,headers))+" |","| "+" | ".join("---" for _ in headers)+" |"]+["| "+" | ".join(map(clean,row))+" |" for row in rows]

def number(value,digits=6):return "—" if value is None else f"{value:.{digits}f}"
def ci_text(r):return f"{r['delta']:+.6f}，95%区间[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"
def write_atomic(path,value,encoding="utf-8"):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(value,encoding=encoding,newline="");temp.replace(path)

def build(result):
    protected=read(ROOT/"protected_inputs.json")
    require(len(protected)==10,"Expected ten protected original/previous-model files")
    for path,expected in REQUIRED_PROTECTED.items():require(protected.get(path)==expected,"Required protection missing/changed: "+path)
    require(str(GAMMA/"candidate_gates.json") in protected and str(DATA/"喻数据产物/manifest.csv") in protected,"Protected gamma gates/manifest missing")
    verify_hashes(protected,"Protected original")
    require(result["protected_hashes_after"]==protected,"Completion protection receipt differs")
    require(result["no_test_inference"] is True and result["github_uploaded"] is False,"Unexpected test inference or new-model upload")
    require(read(ROOT/"status.json")["state"]==result["state"],"Status not terminal or differs")
    common=result["common"]
    for key,expected in dict(epochs=5,seed=42,image_size=224,batch_size=12,workers=0,train_images=30344,validation_images=1422,validation_studies=859,validation_patients=228).items():
        require(common[key]==expected,"Unexpected setting: "+key)
    provenance=provenance_check(read(ROOT/"pretrained_manifest.json"))
    note=ROOT/"文献与选型.md";require(note.is_file(),"Literature note missing");FINGERPRINTS[str(note)]=sha(note)
    medical_note=ROOT/"医学顶刊权重适用性.md"
    require(medical_note.is_file(),"Medical-journal applicability note missing");FINGERPRINTS[str(medical_note)]=sha(medical_note)
    upload=read(ROOT/"upload_receipt.json") if (ROOT/"upload_receipt.json").is_file() else None
    if upload is not None:
        require(upload["repository"]=="https://github.com/yuqirui618/computer-competence" and upload["branch"]=="feat/model01-mambaout" and upload["remote_verified"] is True,"Optimization upload receipt differs")
        require(re.fullmatch(r"[0-9a-f]{40}",upload["commit"]) and upload["main_before"]==upload["main_after"],"Upload receipt does not preserve main")
    base=prediction(GAMMA/"baseline_predictions.npz")
    gamma=prediction(GAMMA/"verified_candidate_predictions.npz",base)
    bm,gm=metrics(base),metrics(gamma)
    bc,_=calibration(base);gc,_=calibration(gamma)
    near(bm["macro_auc"],ORIGINAL_AUC,"Original MambaOut score differs")
    near(gm["macro_auc"],GAMMA_AUC,"Latest gamma score differs")
    near(result["original_mambaout_auc"],ORIGINAL_AUC,"Root original AUC differs")
    near(result["latest_gamma_auc"],GAMMA_AUC,"Root gamma AUC differs")
    check_calibration(bc,read(ORIGINAL/"gates.json"),"Original MambaOut gates")
    check_calibration(gc,read(GAMMA/"candidate_gates.json"),"Latest gamma gates")
    near(base["targets"].sum(0),[24,37,46,68,115,177,193,214,260,306],"Class positive coverage differs")
    old=read(DATA/"MambaOut_100轮/baseline_100/history.json")
    first5=[r for r in old if r["epoch"]<=5]
    require([r["epoch"] for r in first5]==list(range(1,6)),"Historical first five epochs missing")
    old_best=max(first5,key=score)
    recorded5=result["original_mambaout_first5_best"]
    require(recorded5["epoch"]==old_best["epoch"] and recorded5["validation"]==old_best["validation"],"Historical first-five reference differs")
    CHECKS.append("MambaOut_original_gamma_metrics_thresholds_and_historical_first5")
    previous=read(DATA/"新模型对照_20261105/results.json")
    previous_checks=read(DATA/"新模型对照_20261105/report_checks.json")
    require(previous["state"]=="complete" and previous_checks["state"]=="passed" and previous_checks["actual_complete_epochs_from_history"]==10,"Previous two-model result not audited complete")
    previous_summary={}
    for old_name,label in (("convnext_tiny","ConvNeXt-Tiny"),("efficientnetv2_s","EfficientNetV2-S")):
        prior=previous["stages"][old_name]
        require(prior["state"]=="complete" and prior["epochs_completed"]==5,"Prior model not five epochs")
        prior_best=max(prior["history"],key=score)
        require(prior["best_epoch"]==prior_best["epoch"] and prior["best_validation"]==prior_best["validation"],"Prior best selection differs")
        require(sha(prior["best_checkpoint"])==prior["best_checkpoint_sha256"],"Prior model weight changed")
        require(protected.get(str(Path(prior["best_checkpoint"])))==prior["best_checkpoint_sha256"],"Prior weight missing from protected inputs")
        prior_gate_path=DATA/"新模型对照_20261105"/old_name/"candidate_gates.json"
        require(str(prior_gate_path) in protected,"Prior model gates missing from protected inputs")
        prior_gates=read(prior_gate_path)
        require(prior_gates["checkpoint_sha256"]==prior["best_checkpoint_sha256"],"Prior model gate binding changed")
        check_calibration(prior["best_calibration"],prior_gates,"Prior model preserved gates")

        previous_summary[old_name]=dict(label=label,best_epoch=prior["best_epoch"],best_validation=prior["best_validation"],best_calibration=prior["best_calibration"],checkpoint_sha256=prior["best_checkpoint_sha256"])
    CHECKS.append("previous_two_model_aggregates_and_unchanged_best_weights")
    stages=result["stages"]

    require(isinstance(stages,dict) and set(stages).issubset(MODELS),"Unexpected result model set")
    audited={name:audit_stage(name,stages.get(name,dict(state="not_recorded",name=name)),common,protected,base,provenance) for name in MODELS}
    successful=[e for e in audited.values() if e["successful"]]
    actual=sum(e["actual_epochs"] for e in audited.values())
    runner_count=sum(e.get("epochs_completed",0) for e in stages.values())
    require(result["completed_epochs"]=={name:e["actual_epochs"] for name,e in audited.items()},"Root per-model epoch counts differ from histories")
    require(runner_count==result["total_new_epochs"],"Root summary count differs from its own stages")
    complete=result["state"]=="complete"
    if complete:require(set(stages)==set(MODELS) and len(successful)==3 and actual==runner_count==15,"Final report requires three successful five-epoch journal models")
    else:require(result["state"]=="complete_with_failures" and len(successful)<3,"Invalid terminal failure state")
    ensemble_audit=audit_fixed_ensembles(audited,gamma,gm,gc,protected) if complete else None
    best_new=max(successful,key=lambda e:(e["best"]["metrics"]["macro_auc"],e["best"]["metrics"]["macro_f1"])) if successful else None
    require(result["best_new_model"]==(best_new["name"] if best_new else None),"Best new model differs")
    rows=[r for name in MODELS for r in audited[name]["rows"]]
    verify_hashes(protected,"Protected original after report audit")
    title="# 顶刊三模型五轮全量对照总结" if complete else "# 新模型对照运行失败记录（未完成三模型各五轮）"
    lines=[title,"",f"报告时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。依据终态结果及逐轮预测独立复算，没有导入训练模块或使用GPU。",""]
    if complete:
        lines+=["三个模型均完成 **5轮全量训练，共15轮**；每轮30,344张官方训练图，完整验证1,422图、859次检查、228患者。",""]
    else:
        lines+=[f"这是终止运行的失败记录，**不是三模型各五轮完成的最终成绩**。按history独立核对，实际完整轮次共{actual}轮；runner根摘要计{runner_count}轮。失败stage也按实际history统计完整轮，不将已处理的半轮补成完整轮。",
            "十五轮明细.csv沿用计划文件名，但只包含实际核验的完整轮，不补齐虚拟行；未记入history的半轮或孤立文件不计完整轮。",""]
    if best_new:
        auc=best_new["best"]["metrics"]["macro_auc"]
        head=f"成功复核的新模型中，最高为{NAMES[best_new['name']]}第{best_new['best_epoch']}轮，AUC={auc:.8f}。"
        if auc>GAMMA_AUC:head+=f"其点估计超过最新MambaOut Gamma候选{GAMMA_AUC:.8f}。"
        elif auc>ORIGINAL_AUC:head+=f"其点估计高于原单模型冠军{ORIGINAL_AUC:.8f}，仍未超过最新Gamma候选{GAMMA_AUC:.8f}。"
        else:head+="本预算下未超过原单模型冠军或最新Gamma候选。"
        ci=best_new["source"]["paired_auc_intervals"]["latest_gamma_candidate"]["ci95"]
        lines+=[head,"相对最新候选的差值区间跨0，不能确认稳定提升或稳定下降。" if ci[0]<=0<=ci[1] else "相对最新候选的区间未跨0，但它仍条件于被选中的权重及反复使用的验证集。",""]
    else:
        lines+=["没有成功完成五轮并通过最佳权重复核的新模型，不能宣布新候选获胜。失败模型的已保存分数仅作故障记录。",""]
    if ensemble_audit is not None:
        top=ensemble_audit["highest_point_estimate_combination"];item=ensemble_audit["combinations"][top]
        if item["exceeds_existing_gamma_auc"]:
            interval=item["paired_auc_vs_existing_gamma"]["ci95"]
            certainty="差值区间跨0，尚不能确认稳定提升。" if interval[0]<=0<=interval[1] else "这份反复使用的验证集上有正向差值证据，仍需处理选择偏差与独立验证限制。"
            lines += [f"另有三组预先固定的50:50概率组合：最高为MambaOut＋{NAMES[top]}，AUC={item['metrics']['macro_auc']:.8f}，点估计超过现有Gamma候选；{certainty}这属于两个模型的组合，不能归为新单模型胜出。",""]
        else:
            lines += [f"三组预先固定的50:50概率组合也已完成，最高为MambaOut＋{NAMES[top]}，AUC={item['metrics']['macro_auc']:.8f}，未超过现有Gamma候选。组合是额外的CPU概率评估，没有新增训练轮次。",""]
    lines += ["本次实际测试的是具有正式TPAMI期刊版本的三种通用分类骨干，将公开ImageNet-1K权重适配为10类。它们不是现成的胸片专用模型；Nature Ark+、TMI CheXFound和NMI RAD-DINO的医学权重没有进入本实验，也不列为实验失败。",
        f"文献与正式发表凭据见[文献与选型.md]({note.as_posix()})；医学基础模型排查见[医学顶刊权重适用性.md]({medical_note.as_posix()})。", "",
        "## 适配方式与实际正则化", ""]
    recipe_rows=[]
    for name in MODELS:
        config=audited[name]["construction"]
        if config is None:
            recipe_rows.append([NAMES[name],"未形成已审计完整轮","—","—","—"]);continue
        recipe_rows.append([NAMES[name],config["classifier_path"],json.dumps(config["constructor_kwargs"],ensure_ascii=False,sort_keys=True),config["head_drop_rate"],config["actual_drop_path_rate"]])
    lines += table(["模型","仅替换的末层","实际timm构造kwargs","头部Dropout声明","实际DropPath"],recipe_rows)
    lines += ["", "CAFormer只将head.fc.fc2的1000类末层替换为10类，保留预训练head.fc.fc1、隐藏归一化和外层head.norm（原来无参数的模块也保留）。只有新末层使用分类头学习率，预训练隐藏MLP与其余骨干使用骨干学习率；CPU构造审计核对全部非末层张量与预训练逐值相同。",
        "DenseNet与HRNet本次不传DropPath参数，也没有审计到DropPath模块；DenseNet的密集层内部dropout为0。三者没有完全相同的正则化结构。nn.Dropout/DropPath模块位置快照保存在preflight中，不把模块枚举当成全部functional运算的完整证明。",
        "共享训练部分为224输入、batch12、workers0、种子42、逐图BCE、AdamW、weight_decay0.05、骨干初始LR3e-5/新末层3e-4，五轮余弦系数由1降到0.1。归一化遵循各自模型配置；同Study各图概率取均值。这里只比较本次适配配方，不把ImageNet或原论文分数当成本题成绩。", ""]
    lines+=["## 实际轮次、资源与最佳结果",""]
    model_rows=[]
    for name in MODELS:
        e=audited[name];source=e["source"];m=e["best"];rs=e["rows"]
        vs=source["verification"]["validation_seconds"] if e["successful"] else 0
        model_rows.append([NAMES[name],e["state"],e["actual_epochs"],e["best_epoch"] or "—",
            number(m["metrics"]["macro_auc"] if m else None,8),number(m["calibration"]["oof_macro_f1"] if m else None),
            number(sum(r["train_seconds"] for r in rs)/60,2),
            number((sum(r["train_seconds"]+r["validation_seconds"] for r in rs)+vs)/60,2),
            number(max((r["peak_allocated_mib"] for r in rs),default=None),1)])
        if not e["successful"]:
            lines+=[f"{NAMES[name]}状态：{e['state']}；原因：{source.get('error_type','未提供类型')} {source.get('error','无成功结果记录')}。已完整保存{e['actual_epochs']}轮，不声称满五轮或最佳权重已复验。",""]
    lines+=table(["模型","状态","实际完整轮","AUC最佳轮","最佳AUC","同轮患者OOF F1","训练分钟","记录计时合计分钟","训练峰值MiB"],model_rows)
    lines+=["","记录计时合计是逐轮训练、逐轮验证及成功模型最佳权重复验的计时之和，不包含下载、预检、bootstrap、写盘等全部墙钟开销。失败任务分数只指已完整保存的轮次。","","## 逐轮明细",""]
    epoch_table=[]
    for r in rows:
        mark=" ★" if r["epoch"]==audited[r["model"]]["best_epoch"] else ""
        epoch_table.append([NAMES[r["model"]],str(r["epoch"])+mark,number(r["train_loss"]),number(r["study_auc"],8),number(r["f1_at_0_5"]),number(r["patient_oof_f1"]),number(r["macro_ap"]),r["images"],r["scaler_decreases"]])
    lines+=table(["模型","轮次","训练BCE","Study AUC","F1@0.5","患者OOF F1","Macro AP","训练图数","Loss-scale回退次数"],epoch_table)
    lines+=["","★按检查级Macro AUC优先、固定0.5的Macro F1次优选择。每轮全部图像经过训练循环不等于每批都有效更新参数；GradScaler回退可能对应溢出跳步。上表列实际次数，少量起始动态缩放调整本身不等于实验失败；持续较多回退时应进一步核对有效更新及梯度。","","## MambaOut同口径比较",""]
    comparison=[
        ["历史MambaOut前5轮最佳",old_best["epoch"],number(old_best["validation"]["study"]["macro_auc"],8),number(old_best["validation"]["study"]["macro_f1"]),"本报告未重算","初期训练；优化器方案不同"],
        ["保留的原单模型冠军","原保护性微调权重",number(bm["macro_auc"],8),number(bm["macro_f1"]),number(bc["oof_macro_f1"]),"经历多次方案与轮次选择"],
        ["最新Gamma＋轻噪声候选","该实验第1轮",number(gm["macro_auc"],8),number(gm["macro_f1"]),number(gc["oof_macro_f1"]),"已有AUC最高单模型候选"]]
    for old_name in ("convnext_tiny","efficientnetv2_s"):
        prior=previous_summary[old_name]
        comparison.append([prior["label"]+"（前次）",prior["best_epoch"],number(prior["best_validation"]["study"]["macro_auc"],8),number(prior["best_validation"]["study"]["macro_f1"]),number(prior["best_calibration"]["oof_macro_f1"]),"前次独立5轮；仅聚合历史参照"])
    for e in successful:
        comparison.append([NAMES[e["name"]],e["best_epoch"],number(e["best"]["metrics"]["macro_auc"],8),number(e["best"]["metrics"]["macro_f1"]),number(e["best"]["calibration"]["oof_macro_f1"]),"本次ImageNet预训练＋5轮全量"])
    lines+=table(["比较对象","最佳轮次","Study AUC","F1@0.5","同粗网格患者OOF F1","预算边界"],comparison)
    lines+=["","OOF统一用GroupKFold患者五折、shuffle=True、seed42，阈值网格0.025至0.975；每折只在拟合患者选阈值，并列选距离0.5最近再较小值。它与历史嵌套策略选择F1不同，不能混为同一估计。全验证集拟合阈值后的拟合内F1不作为泛化成绩。",
        "历史MambaOut前五轮仅描述训练历史差异，不能当成同优化器、同配方的架构因果对照；前次ConvNeXt/EfficientNet也只引用已审计聚合记录，本报告未重跑其完整实验；原冠军与最新候选拥有更多历史选择机会，新模型只有五轮初筛。","","## 患者层面AUC差异",""]
    ci_rows=[]
    for e in successful:
        for key,label in (("original_mambaout","原单模型冠军"),("latest_gamma_candidate","最新Gamma候选")):
            ci_rows.append([NAMES[e["name"]],label,ci_text(e["source"]["paired_auc_intervals"][key])])
    lines+=table(["新模型最佳轮","参照","AUC差值与95%区间"],ci_rows)
    lines+=["","区间引用runner的400次患者整组配对重采样，本报告独立核对点差、次数与范围，没有宣称重新执行全部bootstrap。区间条件于已选权重；五轮选最好、模型选择、历史反复查看验证集的偏差未被消除，也不包含重训种子的随机性。跨0时不能称稳定突破。",
        "赛题补充规则的主分先四舍五入到小数后4位，展示主分相同才比较F1；本报告保留完整内部AUC。正式排序及端到端延迟以主办方实现和设备复测为准。","","## 三组固定等权概率组合",""]
    if ensemble_audit is None:
        lines += ["本报告没有已完成的固定组合评估记录，不推断任何组合分数；未完成的组合也不列为训练失败。",""]
    else:
        lines += ["每组固定为50%现有Gamma MambaOut＋50%对应新模型的原AUC最佳轮。两条模型各自先做同Study图像概率均值，再用float64作等权平均；未混合模型参数、未搜索比例、未按组合成绩重新挑单模型轮次，也没有新增训练。",""]
        combo_rows=[["现有Gamma MambaOut","—",number(gm["macro_auc"],8),number(gm["macro_f1"]),number(gc["oof_macro_f1"]),"—","—"]]
        for name in MODELS:
            item=ensemble_audit["combinations"][name];m=item["metrics"];cal=item["calibration"];ci=item["paired_auc_vs_existing_gamma"]
            combo_rows.append(["MambaOut＋"+NAMES[name],number(audited[name]["best"]["metrics"]["macro_auc"],8),number(m["macro_auc"],8),number(m["macro_f1"]),number(cal["oof_macro_f1"]),ci_text(ci),"是" if item["exceeds_existing_gamma_auc"] else "否"])
        lines += table(["方案","其中新单模型AUC","组合Study AUC","组合F1@0.5","组合患者OOF F1","相对Gamma ΔAUC与95%区间","AUC点估计超过Gamma"],combo_rows)
        top=ensemble_audit["highest_point_estimate_combination"];item=ensemble_audit["combinations"][top]
        lines += ["",f"最高组合点估计为MambaOut＋{NAMES[top]}：AUC={item['metrics']['macro_auc']:.8f}，患者OOF F1={item['calibration']['oof_macro_f1']:.6f}。单模型与双模型是不同推理方案，组合改善不能表述成某个新单模型改善。",
            f"固定方案SHA256={ensemble_audit['plan_sha256']}；三组来源权重SHA、预测算术、患者分折阈值和附录表行均独立核对。详见[等权组合对照.md]({Path(ensemble_audit['appendix_path']).as_posix()})。其组合JSON与绑定gates仍保留在E盘。",
            "组合CI引用固定评估工具的400次患者配对重采样，本报告重算组合概率与指标但不重新采样；固定比例不能消除选轮次、历史Gamma选择或三组再择优的偏差。仍须在赛事T4环境实测两模型合计的文件规模及端到端耗时，不能仅因AUC升高宣布100ms与500MB约束已满足。",""]
    lines += ["## 最佳单模型轮次十类表现",""]
    positives=base["targets"].sum(0).astype(int)
    for e in audited.values():
        if e["best"] is None:continue
        b=e["best"];m=b["metrics"]
        lines+=[f"### {NAMES[e['name']]}第{e['best_epoch']}轮"+("（失败任务已保存轮次，未作复验通过候选）" if not e["successful"] else ""),""]
        per_class=[]
        for j,label in enumerate(CLASSES):
            counts="/".join(str(m[k][j]) for k in ("tp","fp","fn"))
            oof="/".join(str(b["oof_confusion"][k][j]) for k in ("tp","fp","fn"))
            per_class.append([j+1,label,positives[j],number(m["auc"][j]),number(m["ap_by_class"][j]),number(m["f1"][j]),number(b["calibration"]["oof_f1_by_class_id"][j]),counts,oof])
        lines+=table(["类ID","类别","阳性检查","AUC","AP","F1@0.5","患者OOF F1","TP/FP/FN@0.5","TP/FP/FN OOF"],per_class)+[""]
    lines += ["## 相对现有最高AUC候选的主要类别差距", ""]
    for e in successful:
        delta=np.asarray(e["best"]["metrics"]["auc"])-np.asarray(gm["auc"])
        near(float(delta.mean()),e["best"]["metrics"]["macro_auc"]-gm["macro_auc"],"Per-class AUC differences do not sum to macro difference")
        worst=np.argsort(delta)[:3]
        detail="；".join(f"{CLASSES[j]}：{e['best']['metrics']['auc'][j]:.6f} 对 {gm['auc'][j]:.6f}，差{delta[j]:+.6f}" for j in worst)
        lines += [f"{NAMES[e['name']]}相对现有Gamma候选，AUC差值最小的三类为：{detail}。十类差值取均值等于总体Macro AUC差值{delta.mean():+.6f}。", ""]
    lines += ["这些类别分数说明本次差距出现在哪里，不能单靠它们断定学习率、病灶尺寸、标注或架构是哪一个原因。", ""]
    lines+=["固定0.5下F1为0表示该判定点没有正确阳性命中，不等于所有分数完全无信息。阈值改变精确率/召回率的折中，不会改变固定概率的AUC排序；应结合AP、OOF F1及误报漏检理解。","","## 有证据的事实与待检验解释",""]
    subjects=base["subjects"]
    rare=[len(np.unique(subjects[base["targets"][:,j]==1])) for j in range(10)]
    counts=np.unique(subjects,return_counts=True)[1]
    lines+=[f"已确认：859次验证检查仅来自228位患者，第1类24次阳性检查来自{rare[0]}位患者，单人最多{int(counts.max())}次检查。重复检查有关联，稀有类结果可能受少数患者影响；这支持患者分折及整组重采样。"]
    for e in audited.values():
        if len(e["history"])>=2:
            a,b=e["history"][0],e["history"][-1]
            lines+=[f"{NAMES[e['name']]}实测首轮→末个完整轮：训练BCE {a['train_loss']:.6f}→{b['train_loss']:.6f}，验证AUC {a['validation']['study']['macro_auc']:.8f}→{b['validation']['study']['macro_auc']:.8f}，最好在第{e['best_epoch']}轮。这是本次曲线事实，五轮总分不能证明全部误差的原因。"]
    lines+=["","若本预算未超过既有最佳，只能确认指定ImageNet初始化、224、单种子和五轮共同配方下未得到更高相应分数，不能推断该架构永久弱于MambaOut。",
        "尚未验证的可能原因：共同学习率不一定适合每个骨干；短程适配不足；224可能限制细节；类别不平衡或检查内多图权重问题限制排序。现有实验未把这些因素逐一控制，不能当成失败原因定论，也没有证据宣布标签错误。",
        "本次三者均固定224、单个种子和5轮预算，没有完整复刻原期刊论文的长训练、分辨率、多任务或跨数据集设置。更长训练、分辨率、学习率与种子是否改善，需要另外受控验证，不能由此次分数推定架构的性能上限。",
        "后续若继续，应预先固定预算与更独立的患者验证，按错误类型复核稀有类，再考虑学习率/输入尺寸或检查级目标；不围绕同一验证集无限挑选。","","## 来源、保存和上传范围","",
        f"文献与赛规选型见[文献与选型.md]({note.as_posix()})。已核对实际文件SHA、固定revision、归档模型卡与相应Apache/MIT许可文本，不只依据模型名字判断来源。",""]
    provenance_rows=[]
    for name in MODELS:
        p=provenance[name];provenance_rows.append([NAMES[name],p["architecture"],p["repository"],p["revision"],p["license"],p["sha256"]])
    lines+=table(["模型","timm名称","公开仓库","固定revision","许可","预训练SHA256"],provenance_rows)
    lines+=["","所有原最佳权重、两套阈值、原submission和manifest的SHA均核对未变。新模型best.pth和专属candidate_gates.json保存在各模型目录。CPU报告核对文件SHA与runner的GPU重载收据，不自行解包或执行权重。",""]
    for e in successful:
        lines+=[f"{NAMES[e['name']]}候选：{e['source']['best_checkpoint']}；SHA256={e['source']['best_checkpoint_sha256']}；阈值：{ROOT/e['name']/'candidate_gates.json'}。"]
    if upload is not None:
        url=upload["repository"]+"/tree/"+upload["branch"]
        lines += ["",f"此前MambaOut优化代码已上传[个人分支{upload['branch']}]({url})，提交{upload['commit']}，旧upload_receipt核实当时main未改。该历史上传不代表本次新任务已上传。"]
    lines += ["", "本次顶刊三模型实验没有自动上传GitHub。环境、数据、权重、缓存及产物均沿用E盘；十五轮明细CSV只含聚合指标，不含患者标识或逐样本预测。",
        "报告独立重算逐轮Study AUC、F1、AP、TP/FP/FN与患者五折阈值，并核对完整图数、最佳选择、SHA与复验状态。BCE、耗时、显存和loss-scale回退引用运行记录，不能由Study均值预测反推；没有编造验证loss曲线或隐藏测试结果。"]

    fields=("model","epoch","stage_state","images","train_loss","study_auc","f1_at_0_5","patient_oof_f1","macro_ap","train_seconds","validation_seconds","peak_allocated_mib","scaler_decreases")
    csv_buffer=io.StringIO(newline="");writer=csv.DictWriter(csv_buffer,fieldnames=fields,lineterminator=chr(10));writer.writeheader();writer.writerows(rows)
    checks=dict(state="passed",experiment_state=result["state"],full_three_by_five_completed=complete,previous_two_model_summary=previous_summary,
        actual_complete_epochs_from_history=actual,runner_summary_epochs=runner_count,
        per_model={name:dict(state=e["state"],actual_epochs=e["actual_epochs"],successful=e["successful"],best_epoch=e["best_epoch"],construction=e["construction"]) for name,e in audited.items()},
        checks=CHECKS,all_saved_study_predictions_recomputed=True,patient_threshold_cv_recomputed=True,
        paired_bootstrap_recomputed=False,paired_bootstrap_point_differences_checked=True,
        protected_hashes_verified=protected,pretraining_provenance=provenance,new_models_uploaded=False,
        previous_optimization_upload=upload,fixed_ensemble_evaluation=ensemble_audit,source_fingerprints=FINGERPRINTS,
        report_generator_sha256=sha(Path(__file__)),generated_at=datetime.now().astimezone().isoformat())
    write_atomic(ROOT/"顶刊三模型实验总结.md","\n".join(lines)+"\n")
    write_atomic(ROOT/"十五轮明细.csv",csv_buffer.getvalue(),"utf-8-sig")
    write_atomic(ROOT/"report_checks.json",json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(report="complete" if complete else "incomplete_failure_record",actual_complete_epochs=actual,
        successful_models=[e["name"] for e in successful],best_new_model=result["best_new_model"],fixed_combinations_audited=3 if ensemble_audit else 0),ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        require(ROOT.drive.upper()=="E:","Report must execute from E drive")
        result=read(ROOT/"results.json")
        require(result.get("state") in ("complete","complete_with_failures"),"Training not terminal; no report generated")
        dependencies();build(result);return 0
    except (OSError,ValueError,KeyError,TypeError,IndexError,AssertionError) as error:
        print("Report not generated: "+str(error),file=sys.stderr);return 1

if __name__=="__main__":raise SystemExit(main())


