"""Evaluate three predeclared 50:50 probability ensembles on validation only.

CPU-only: never imports torch, runner or experiment_metrics. Requires all three
journal candidates to have completed five epochs and passed GPU reload checks.
--help does not read experiments, import scientific packages or write files.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
MODELS=('caformer_s18','densenet121','hrnet_w18')
NAMES={'caformer_s18':'CAFormer-S18','densenet121':'DenseNet-121','hrnet_w18':'HRNet-W18'}
CLASSES=('纵隔心影增大','气胸','肺部实变','肺炎','肺水肿','心影扩大','肺不张','肺部阴影','胸腔积液','无异常发现')
GAMMA=DATA/'MambaOut_二次数据增强_20261104'
BASELINE_CHECKPOINT=GAMMA/'C_gamma_noise/best.pth'
BASELINE_PREDICTIONS=GAMMA/'verified_candidate_predictions.npz'
BASELINE_SHA='b0066e479b1b0c383fa7daa2067ff0d6c954d20fbf81d1202af95481d67cc5f9'
BASELINE_AUC=.8021774782195852
PLAN=ROOT/'fixed_ensemble_plan.json'
FINGERPRINTS={}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    path=Path(path);track(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def track(path,expected=None):
    path=Path(path);actual=sha(path)
    if expected is not None:require(actual==expected,'Input SHA changed: '+str(path))
    if str(path) in FINGERPRINTS:
        require(FINGERPRINTS[str(path)]==actual,'Input changed between reads: '+str(path))
    FINGERPRINTS[str(path)]=actual
    return actual


def configure_cpu():
    global np,roc_auc_score,average_precision_score,f1_score,GroupKFold
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='4'
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):os.environ[key]=str(ROOT/'temp')
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
    from sklearn.model_selection import GroupKFold


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def near(a,b,description,tolerance=1e-12):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=0,atol=tolerance),description)


def load_predictions(path,reference=None):
    track(path)
    with np.load(path,allow_pickle=False) as data:
        out={key:data[key].copy() for key in ('targets','probabilities','subjects','study_ids')}
    y,p,s,ids=(out[key] for key in ('targets','probabilities','subjects','study_ids'))
    require(y.shape==p.shape==(859,10) and s.shape==ids.shape==(859,),'Unexpected Study prediction dimensions')
    require(len(np.unique(s))==228 and len(np.unique(ids))==859,'Unexpected patient/Study counts')
    require(np.isin(y,(0,1)).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid predictions/labels')
    require(((y.sum(0)>0)&(y.sum(0)<859)).all(),'Every validation class requires positive and negative examples')
    if reference is not None:
        for key in ('targets','subjects','study_ids'):
            require(np.array_equal(out[key],reference[key]),'Study/subject/target order differs: '+str(path)+' '+key)
    return out


def fit_gates(y,p):
    candidates=sorted(np.arange(1,40)/40,key=lambda value:(abs(value-.5),value))
    scores=[f1_score(y,p>=value,average=None,zero_division=0) for value in candidates]
    return np.asarray(candidates)[np.argmax(scores,axis=0)]


def calibrate(arrays):
    y,p,s=(arrays[key] for key in ('targets','probabilities','subjects'))
    decisions=np.zeros_like(y,dtype=bool)
    for fit,held in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s):
        require(not set(s[fit])&set(s[held]),'Patient crossed threshold fold')
        decisions[held]=p[held]>=fit_gates(y[fit],p[fit])
    per_class=f1_score(y,decisions,average=None,zero_division=0)
    return dict(oof_macro_f1=float(per_class.mean()),oof_f1_by_class_id=per_class.tolist(),
        final_gates_by_class_id=fit_gates(y,p).tolist(),patients=228),decisions


def metrics(arrays):
    y,p=arrays['targets'],arrays['probabilities'];decisions=p>=.5
    return dict(macro_auc=float(roc_auc_score(y,p,average='macro')),
        auc_by_class=roc_auc_score(y,p,average=None).tolist(),
        macro_f1_at_0_5=float(f1_score(y,decisions,average='macro',zero_division=0)),
        f1_at_0_5_by_class=f1_score(y,decisions,average=None,zero_division=0).tolist(),
        macro_ap=float(average_precision_score(y,p,average='macro')),
        ap_by_class=average_precision_score(y,p,average=None).tolist(),
        tp_at_0_5=np.sum(decisions&(y==1),0).astype(int).tolist(),
        fp_at_0_5=np.sum(decisions&(y==0),0).astype(int).tolist(),
        fn_at_0_5=np.sum(~decisions&(y==1),0).astype(int).tolist())


def paired_auc_interval(y,baseline,candidate,subjects,repeats=400,seed=20261104):
    # Preserve first-appearance patient ordering, as in the original metric helper.
    groups=OrderedDict()
    for index,subject in enumerate(subjects):groups.setdefault(str(subject),[]).append(index)
    require(len(groups)==228,'Unexpected bootstrap patient count')
    groups=[np.asarray(indices,dtype=np.intp) for indices in groups.values()]
    rng=np.random.default_rng(seed);differences=[]
    point=float(roc_auc_score(y,candidate,average='macro')-roc_auc_score(y,baseline,average='macro'))
    for _ in range(repeats):
        sampled=rng.integers(0,len(groups),size=len(groups))
        indices=np.concatenate([groups[index] for index in sampled])
        labels=y[indices]
        if not ((labels.sum(0)>0)&(labels.sum(0)<len(labels))).all():continue
        differences.append(float(roc_auc_score(labels,candidate[indices],average='macro')-
            roc_auc_score(labels,baseline[indices],average='macro')))
    require(len(differences)>=300,'Too few defined patient-bootstrap replicates')
    return dict(delta=point,ci95=np.quantile(differences,[.025,.975]).tolist(),
        repeats=repeats,valid_repeats=len(differences),seed=seed,
        scope='Patient-cluster paired percentile CI, conditional on fixed selected models; not model-selection corrected; no thresholds refit.')


def verify_fingerprints():
    for path,expected in FINGERPRINTS.items():require(sha(path)==expected,'Input changed during evaluation: '+path)


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='')
    temporary.replace(path)


def stage_score(row):return row['validation']['study']['macro_auc'],row['validation']['study']['macro_f1']


def verify_terminal_inputs():
    require(ROOT.drive.upper()=='E:','CPU outputs must stay on E drive')
    plan=read(PLAN)
    require(plan['model_keys']==list(MODELS) and plan['alpha']==.5 and plan['predetermined_date']=='2026-11-05','Frozen ensemble choices differ')
    require(plan['kind']=='fixed_probability_ensembles_not_new_single_models','Unexpected ensemble kind')
    require(plan['baseline']['checkpoint_sha256']==BASELINE_SHA and Path(plan['baseline']['checkpoint'])==BASELINE_CHECKPOINT,'Frozen baseline changed')
    require(Path(plan['baseline']['predictions'])==BASELINE_PREDICTIONS,'Frozen baseline prediction path changed')
    require(plan['threshold_protocol']==dict(folds=5,shuffle=True,seed=42,grid='0.025..0.975 step0.025',tie_break='closest to 0.5, then smaller'),'Threshold protocol changed')
    require(plan['bootstrap']==dict(repeats=400,seed=20261104,unit='patient cluster',percentiles=[.025,.975]),'Bootstrap protocol changed')
    require(plan['probability_arithmetic']=='float64 equal mean of saved Study mean image probabilities','Probability arithmetic changed')
    result=read(ROOT/'results.json');status=read(ROOT/'status.json')
    require(result['state']=='complete' and status['state']=='complete','Three-model run is not complete')
    require(set(result['stages'])==set(MODELS) and result['total_new_epochs']==15,'All three models must complete five epochs')
    require(result['completed_epochs']==dict.fromkeys(MODELS,5),'Root completed epoch counts differ')
    require(status['completed_epochs']==dict.fromkeys(MODELS,5),'Status contains incomplete model counts')
    require(result['no_test_inference'] is True and result['github_uploaded'] is False,'Unexpected test inference/upload')
    protected=read(ROOT/'protected_inputs.json')
    require(protected.get(str(BASELINE_CHECKPOINT))==BASELINE_SHA,'Protected baseline checkpoint differs')
    require(result['protected_hashes_after']==protected,'Root protection receipt differs')
    for path,expected in protected.items():track(path,expected)
    track(BASELINE_CHECKPOINT,BASELINE_SHA)
    track(BASELINE_PREDICTIONS,plan['baseline']['predictions_sha256'])
    base=load_predictions(BASELINE_PREDICTIONS)
    near(base['targets'].sum(0),[24,37,46,68,115,177,193,214,260,306],'Official validation class counts changed')
    base_metrics=metrics(base);near(base_metrics['macro_auc'],BASELINE_AUC,'Baseline AUC does not reproduce')
    base_cal,_=calibrate(base)
    singles={}
    for name in MODELS:
        folder=ROOT/name;stage=read(folder/'results.json');history=read(folder/'history.json')
        require(stage==result['stages'][name] and stage['state']=='complete' and stage['epochs_completed']==5,'Stage results differ: '+name)
        require(history==stage['history'] and [row['epoch'] for row in history]==[1,2,3,4,5],'Stage history differs: '+name)
        require(all(row['images']==30344 and row['validation']['image']['count']==1422 and row['validation']['study']['count']==859 for row in history),'Incomplete full-data epoch: '+name)
        chosen=max(history,key=stage_score)
        require(chosen['epoch']==stage['best_epoch'] and chosen['validation']==stage['best_validation'],'Best epoch selection differs: '+name)
        verification=stage['verification']
        require(verification['state']=='passed' and verification['epoch']==stage['best_epoch'] and verification['metrics_reproduced'] is True and verification['predictions_reproduced'] is True,'GPU reload not verified: '+name)
        require(0<=verification['max_abs_probability_difference']<=1e-7 and verification['bn_buffers_unchanged'] is True,'GPU reload/BN evidence differs: '+name)
        checkpoint=folder/'best.pth'
        require(Path(stage['best_checkpoint']).resolve()==checkpoint.resolve(),'Unexpected checkpoint path')
        track(checkpoint,stage['best_checkpoint_sha256'])
        arrays=load_predictions(folder/f"epoch_{stage['best_epoch']}_predictions.npz",base)
        measured=metrics(arrays);calibration,_=calibrate(arrays)
        near(measured['macro_auc'],stage['best_validation']['study']['macro_auc'],'Single-model best AUC differs')
        near(measured['macro_f1_at_0_5'],stage['best_validation']['study']['macro_f1'],'Single-model best F1 differs')
        near(calibration['oof_macro_f1'],stage['best_calibration']['oof_macro_f1'],'Single-model OOF F1 differs')
        gates=read(folder/'candidate_gates.json')
        require(gates['checkpoint_sha256']==stage['best_checkpoint_sha256'],'Single-model gates binding differs')
        near(gates['final_gates_by_class_id'],calibration['final_gates_by_class_id'],'Single-model final gates differ')
        singles[name]=dict(stage=stage,arrays=arrays,metrics=measured,calibration=calibration)
    verify_fingerprints()
    return plan,base,base_metrics,base_cal,singles,protected


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]


def evaluate():
    plan,base,bm,bc,singles,protected=verify_terminal_inputs()
    computed={}
    # Finish computations and verify all inputs before writing any new results.
    for name in MODELS:
        single=singles[name];source=single['stage']
        probability=.5*base['probabilities'].astype(np.float64)+.5*single['arrays']['probabilities'].astype(np.float64)
        arrays=dict(targets=base['targets'].copy(),probabilities=probability,subjects=base['subjects'].copy(),study_ids=base['study_ids'].copy())
        measured=metrics(arrays);calibration,decisions=calibrate(arrays)
        interval=paired_auc_interval(arrays['targets'],base['probabilities'],probability,arrays['subjects'])
        near(interval['delta'],measured['macro_auc']-bm['macro_auc'],'Ensemble point difference differs')
        binding=dict(kind='probability_ensemble_not_new_single_model',weights=[.5,.5],
            source_checkpoints=[dict(role='existing_gamma_mambaout',path=str(BASELINE_CHECKPOINT),sha256=BASELINE_SHA,weight=.5),
                dict(role=name,path=source['best_checkpoint'],sha256=source['best_checkpoint_sha256'],weight=.5)],
            single_model_epoch=source['best_epoch'],probability_arithmetic=plan['probability_arithmetic'],
            aggregation='Each source first averages image probabilities within the same Study; then sources are averaged 50:50')
        computed[name]=dict(arrays=arrays,decisions=decisions,metrics=measured,calibration=calibration,
            paired_auc_vs_existing_gamma=interval,binding=binding,
            exceeds_existing_gamma_auc=bool(measured['macro_auc']>bm['macro_auc']))
    verify_fingerprints()
    output=ROOT/'fixed_ensembles'
    output.mkdir(exist_ok=True)
    summaries={}
    for name,item in computed.items():
        folder=output/name;folder.mkdir(exist_ok=True)
        prediction_path=folder/'predictions.npz'
        np.savez_compressed(prediction_path,**item['arrays'],oof_decisions=item['decisions'])
        with np.load(prediction_path,allow_pickle=False) as saved:
            for key in ('targets','probabilities','subjects','study_ids'):
                require(np.array_equal(saved[key],item['arrays'][key]),'Saved ensemble prediction differs')
            require(np.array_equal(saved['oof_decisions'],item['decisions']),'Saved OOF decisions differ')
        gate_path=folder/'candidate_gates.json'
        write_json(gate_path,dict(**item['calibration'],binding=item['binding'],class_ids=list(range(1,11)),
            decision_rule='Fixed 50:50 Study probability >= class Gate',candidate_only=True,
            fit_source='Final gates fit all reused validation; OOF gates fit only respective patient-training folds',
            no_test_inference=True,github_uploaded=False))
        summaries[name]=dict(state='complete',metrics=item['metrics'],calibration=item['calibration'],
            paired_auc_vs_existing_gamma=item['paired_auc_vs_existing_gamma'],binding=item['binding'],
            exceeds_existing_gamma_auc=item['exceeds_existing_gamma_auc'],predictions=str(prediction_path),
            predictions_sha256=sha(prediction_path),candidate_gates=str(gate_path),candidate_gates_sha256=sha(gate_path),
            original_single_metrics=singles[name]['metrics'],original_single_calibration=singles[name]['calibration'])
    verify_fingerprints()
    best=max(MODELS,key=lambda name:(summaries[name]['metrics']['macro_auc'],summaries[name]['metrics']['macro_f1_at_0_5']))
    result=dict(state='complete',kind=plan['kind'],generated_at=now(),plan=plan,baseline_metrics=bm,
        baseline_calibration=bc,ensembles=summaries,highest_point_estimate_combination=best,
        completed_single_model_epochs=dict.fromkeys(MODELS,5),new_training_epochs=0,no_test_inference=True,
        github_uploaded=False,all_input_hashes_unchanged=True,protected_hashes_after=protected,
        input_fingerprints=dict(FINGERPRINTS),tool_sha256=sha(Path(__file__)))
    write_json(ROOT/'fixed_ensemble_results.json',result)
    lines=['# 三组固定等权概率组合对照','',
        '三个顶刊单模型均已完成全量5轮并通过最佳权重GPU重载核对。本工具只在CPU上组合保存的验证概率，没有新增训练轮次，没有读取测试图像或测试标签。','',
        '预先固定每个组合为50%现有Gamma MambaOut＋50%对应顶刊模型；共3组，没有搜索混合比例。单模型最佳轮按其原有AUC选择，组合阈值按患者五折拟合。**组合结果不能称为某一个新单模型超过MambaOut。**','']
    rows=[['现有Gamma MambaOut',f"{bm['macro_auc']:.8f}",f"{bm['macro_f1_at_0_5']:.6f}",f"{bc['oof_macro_f1']:.6f}",f"{bm['macro_ap']:.6f}",'—']]
    for name in MODELS:
        row=summaries[name];m=row['metrics'];ci=row['paired_auc_vs_existing_gamma']
        rows.append(['MambaOut＋'+NAMES[name],f"{m['macro_auc']:.8f}",f"{m['macro_f1_at_0_5']:.6f}",f"{row['calibration']['oof_macro_f1']:.6f}",f"{m['macro_ap']:.6f}",f"{ci['delta']:+.6f} [{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}]"])
    lines+=table(['方案','Study AUC','F1@0.5','患者OOF F1','Macro AP','AUC差值及95%区间'],rows)
    winner=summaries[best]
    if winner['exceeds_existing_gamma_auc']:
        lines+=['',f"最高数值来自MambaOut＋{NAMES[best]}，AUC={winner['metrics']['macro_auc']:.8f}，高于现有Gamma候选。"]
        ci=winner['paired_auc_vs_existing_gamma']['ci95']
        lines+=['该组合的差值区间跨0，尚不能确认稳定提升。' if ci[0]<=0<=ci[1] else '该固定组合在当前验证集上有正向差值证据，但仍需面对选择偏差及独立验证限制。']
    else:
        lines+=['','三组固定组合的AUC均未超过现有Gamma候选；原有权重、阈值和提交文件保留。']
    lines+=['','## 十类结果','']
    for name in MODELS:
        item=summaries[name];m=item['metrics'];cal=item['calibration']
        lines+=['### MambaOut＋'+NAMES[name],'']
        lines+=table(['类ID','类别','AUC','AP','F1@0.5','患者OOF F1'],[[j+1,CLASSES[j],f"{m['auc_by_class'][j]:.6f}",f"{m['ap_by_class'][j]:.6f}",f"{m['f1_at_0_5_by_class'][j]:.6f}",f"{cal['oof_f1_by_class_id'][j]:.6f}"] for j in range(10)])
        lines+=['']
    lines+=['## 如何解释这些结果','',
        '患者五折阈值采用GroupKFold(shuffle=True, seed=42)，网格0.025至0.975；并列优先距0.5近，再选较小阈值。每折只在其训练患者拟合阈值。最终gates由全部验证集拟合，不能将拟合内F1当作泛化结果。',
        '95%区间来自400次按患者整组配对重采样，条件于固定的单模型权重和这份验证集。最佳轮已经用过完整验证集，Gamma也经历过历史选择；三组合之间再择优仍有选择偏差，区间没有对此校正，也没有纳入重训种子的变化。',
        '这里只组合概率，并未平均模型参数。未来推理需要同时加载两份绑定权重，并复用两条各自的预处理、Sigmoid、Study概率均值与float64等权运算。每组candidate_gates绑定两份SHA及0.5/0.5系数，不能拿去配另一个单模型。',
        '未在比赛T4设备上测过组合的端到端延迟；额外模型会增加计算和文件规模，不能因为验证AUC增加就宣布已满足100ms与500MB要求。',
        '所有产物仅写入本轮fixed_ensembles目录与两个汇总文件，没有覆盖原Gamma、三个单模型目录或原submission。没有自动上传GitHub。',
        '预测NPZ含患者/检查标识，仅保留本地E盘；公开分享应另行提取聚合报告。']
    report=ROOT/'等权组合对照.md';temporary=report.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='');temporary.replace(report)
    print(json.dumps(dict(state='complete',combinations=3,new_training_epochs=0,highest_point_estimate=best,
        highest_auc=winner['metrics']['macro_auc'],any_auc_exceeds_baseline=any(x['exceeds_existing_gamma_auc'] for x in summaries.values())),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        configure_cpu();evaluate();return 0
    except (OSError,ValueError,KeyError,TypeError,IndexError,AssertionError) as exc:
        print('Fixed ensembles not evaluated: '+str(exc),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
