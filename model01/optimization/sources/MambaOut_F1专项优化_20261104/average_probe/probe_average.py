"""One fixed 50:50 checkpoint average and one fixed prediction average; no training."""
import os, sys, csv, json, gc, hashlib, time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
TMP=ROOT/'temp'
TMP.mkdir(exist_ok=True)
for key,part in {'TEMP':'','TMP':'','TMPDIR':'','XDG_CACHE_HOME':'','CUDA_CACHE_PATH':'cuda','HF_HOME':'hf','TORCH_HOME':'torch'}.items():
    path=TMP/part;path.mkdir(exist_ok=True);os.environ[key]=str(path)
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'MambaOut_泛化优化'))
import train_generalization as core
sys.path.insert(0,str(BASE/'MambaOut_100轮'))
import train_and_optimize as ref
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import GroupKFold

OLD=BASE/'MambaOut_F1正则化优化_20261104'
CHAMP=BASE/'MambaOut_ASL对照/best_model.pth'
SECOND=OLD/'A_current_regularization/epoch_2.pth'
SUBMISSION=BASE/'提交结果/MambaOut_S7微调第1轮_20261104/submission.csv'
GATES=BASE/'MambaOut_ASL对照/gates.json'
EXPECTED_CHAMP='9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585'
EXPECTED_SUBMISSION='d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8'
MANIFEST=BASE/'喻数据产物/manifest.csv'
EXPECTED_MANIFEST='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):core.save_json(ROOT/name,value)
def status(state,**fields):save('status.json',dict(state=state,pid=os.getpid(),time=datetime.now().astimezone().isoformat(),**fields))
def classification(y,d):
    y=np.asarray(y,dtype=bool);d=np.asarray(d,dtype=bool)
    tp=(y&d).sum(0);fp=(~y&d).sum(0);fn=(y&~d).sum(0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(10),where=2*tp+fp+fn!=0)
    precision=np.divide(tp,tp+fp,out=np.zeros(10),where=tp+fp!=0)
    recall=np.divide(tp,tp+fn,out=np.zeros(10),where=tp+fn!=0)
    return dict(macro_f1=float(f1.mean()),f1=f1.tolist(),macro_precision=float(precision.mean()),precision=precision.tolist(),macro_recall=float(recall.mean()),recall=recall.tolist(),tp=tp.tolist(),fp=fp.tolist(),fn=fn.tolist())
def load_pred(path):
    with np.load(path,allow_pickle=False) as z:return tuple(z[k].copy() for k in ('targets','probabilities','subjects'))
def oof_gates(y,p,s):
    d=np.zeros_like(y,dtype=bool);folds=[]
    for k,(fit,held) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s),1):
        assert not set(s[fit])&set(s[held])
        gates=ref.fit_gates(y[fit],p[fit]);d[held]=p[held]>=gates
        folds.append(dict(fold=k,fit_studies=len(fit),held_studies=len(held),fit_patients=len(np.unique(s[fit])),held_patients=len(np.unique(s[held])),gates=gates.tolist()))
    return dict(**classification(y,d),folds=folds,final_gates=ref.fit_gates(y,p).tolist()),d
def scores(y,p):
    auc=roc_auc_score(y,p,average=None);ap=average_precision_score(y,p,average=None)
    return dict(macro_auc=float(auc.mean()),auc=auc.tolist(),macro_ap=float(ap.mean()),ap=ap.tolist(),fixed_0_5=classification(y,p>=.5))
def f1_bootstrap(y,a,b,s):
    groups=[np.flatnonzero(s==subject) for subject in np.unique(s)]
    rng=np.random.default_rng(20261104);values=[]
    for _ in range(400):
        selected=rng.integers(0,len(groups),len(groups));idx=np.concatenate([groups[i] for i in selected])
        values.append(classification(y[idx],b[idx])['macro_f1']-classification(y[idx],a[idx])['macro_f1'])
    return dict(delta=classification(y,b)['macro_f1']-classification(y,a)['macro_f1'],ci95=np.quantile(values,[.025,.975]).tolist(),repeats=400,seed=20261104,scope='Conditional on the fixed OOF decisions; not a full refit bootstrap or independent test proof.')

def main():
    started=time.perf_counter();status('checking_sources')
    protected={str(p):sha(p) for p in (CHAMP,SECOND,SUBMISSION,GATES,MANIFEST)}
    assert protected[str(CHAMP)]==EXPECTED_CHAMP
    assert protected[str(SUBMISSION)]==EXPECTED_SUBMISSION
    assert protected[str(MANIFEST)]==EXPECTED_MANIFEST
    sources=[dict(path=str(CHAMP),sha256=protected[str(CHAMP)],weight=.5),dict(path=str(SECOND),sha256=protected[str(SECOND)],weight=.5)]
    y,p0,s=load_pred(OLD/'baseline_predictions.npz')
    y1,p1,s1=load_pred(OLD/'A_current_regularization/epoch_2_predictions.npz')
    assert np.array_equal(y,y1) and np.array_equal(s,s1)
    assert y.shape==p0.shape==p1.shape==(859,10) and len(np.unique(s))==228
    assert abs(roc_auc_score(y,p0,average='macro')-.8019769550500871)<1e-12
    with MANIFEST.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='val']
    order={}
    for row in rows:
        labels=tuple(sorted(int(v) for v in row['label_ids'].split('|')))
        value=(row['subject_id'],labels)
        if row['study_id'] in order:assert order[row['study_id']]==value
        order[row['study_id']]=value
    ids=np.array(list(order));ys=np.zeros((859,10),dtype=np.float32)
    for i,(_,labs) in enumerate(order.values()):ys[i,np.array(labs)-1]=1
    assert len(rows)==1422 and np.array_equal(ys,y) and np.array_equal(s,np.array([v[0] for v in order.values()]))
    left=torch.load(CHAMP,map_location='cpu',weights_only=True)
    right=torch.load(SECOND,map_location='cpu',weights_only=True)
    for loaded in (left,right):
        meta=loaded['training']
        for field,expected in [('architecture','mambaout_tiny'),('img_size',224),('drop_rate',.2),('drop_path_rate',.1)]:
            assert meta[field]==expected,(field,meta[field])
    assert right['training']['best_epoch']==2 and right['training']['name']=='A_current_regularization'
    a,b=left['state_dict'],right['state_dict'];assert set(a)==set(b)
    assert a['head.fc.weight'].shape[0]==a['head.fc.bias'].shape[0]==10
    average={};float_keys=0;integer_keys=0
    for key in a:
        assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype
        if a[key].is_floating_point():
            assert torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all()
            average[key]=.5*a[key].float()+.5*b[key].float();float_keys+=1
        else:
            assert torch.equal(a[key],b[key]);average[key]=a[key].clone();integer_keys+=1
    averaged=ROOT/'averaged_model.pth'
    meta={**left['training'],'name':'fixed_50_50_checkpoint_average','best_epoch':0,'source_checkpoints':sources,'new_training_epochs':0}
    core.save_checkpoint(averaged,dict(state_dict=average,training=meta,averaging=dict(sources=sources,ratio=[.5,.5],floating_state_count=float_keys,nonfloating_state_count=integer_keys,note='Fixed checkpoint average; not a complete SWA training procedure.')))
    average_sha=sha(averaged)
    del a,b,left,right,average;gc.collect()
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    state=torch.load(averaged,map_location='cpu',weights_only=True)
    model=core.build_model(meta,state['state_dict']);model.eval().requires_grad_(False)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    cfg=yaml.safe_load((BASE/'本地跑通/data_local.yaml').read_text(encoding='utf-8'))
    data=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'val',ref.build_transform(False,224),cache_dir=None)
    ref.ARGS.batch_size=12
    status('gpu_validation',images=1422,study_count=859,checkpoint_sha256=average_sha)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
    measured,pred=ref.validate(model,data,keep_predictions=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    memory={'allocated_mib':torch.cuda.max_memory_allocated()/2**20,'reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    yy,pw,ss=pred;ss=np.asarray(ss)
    assert np.array_equal(y,yy) and np.array_equal(s,ss)
    del model,state;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    status('gpu_released',gpu_released=True,inference_seconds=elapsed,**memory)
    print('GPU_RELEASED',flush=True)
    probs={'champion':p0,'epoch2':p1,'probability_average':.5*p0+.5*p1,'weight_average':pw}
    candidates={};decisions={}
    bindings={'champion':dict(kind='checkpoint',checkpoint=dict(path=str(CHAMP),sha256=protected[str(CHAMP)])),'epoch2':dict(kind='checkpoint',checkpoint=dict(path=str(SECOND),sha256=protected[str(SECOND)])),'probability_average':dict(kind='probability_ensemble',source_checkpoints=sources,weights=[.5,.5]),'weight_average':dict(kind='checkpoint',checkpoint=dict(path=str(averaged),sha256=average_sha))}
    for name,probs_i in probs.items():
        np.savez_compressed(ROOT/(name+'_predictions.npz'),targets=y,probabilities=probs_i,subjects=s,study_ids=ids)
        calibration,d=oof_gates(y,probs_i,s);decisions[name]=d
        candidates[name]=dict(binding=bindings[name],metrics=scores(y,probs_i),calibration=calibration,predictions_sha256=sha(ROOT/(name+'_predictions.npz')))
        gates=dict(**bindings[name],gate_by_class_id=calibration['final_gates'],class_ids=list(range(1,11)),rule='Probability >= Gate after Study probability mean',fit_source='All reused official validation; performance estimated by patient-group OOF threshold fitting',threshold_policy='coarse grid .025 to .975; maximum class F1, tie closest .5 then lower',oof_macro_f1=calibration['macro_f1'])
        save(name+'_gates.json',gates)
    assert abs(candidates['champion']['calibration']['macro_f1']-.451054963682343)<1e-12
    assert abs(candidates['epoch2']['calibration']['macro_f1']-.4568632242034646)<1e-12
    np.savez_compressed(ROOT/'oof_decisions.npz',targets=y,subjects=s,study_ids=ids,**decisions)
    intervals={name:f1_bootstrap(y,decisions['champion'],d,s) for name,d in decisions.items() if name!='champion'}
    assert protected=={path:sha(path) for path in protected}
    limitations=['Official validation has been reused for model/epoch selection; these are exploratory F1 results, not blind test performance.','Epoch2 was proposed because its previous OOF F1 was high. This selection is not undone by refitting thresholds in patient folds.','Thresholds for each held fold exclude that fold, but model training/selection history is not fully cross-fitted.','Averaging ratio was fixed at 50:50, with no weight search. This is checkpoint averaging, not full SWA training.','Bootstrap conditions on already generated OOF predictions and ignores model-selection and refitting uncertainty.','A probability ensemble needs two network passes; a parameter average needs one.']
    result=dict(state='complete',new_training_epochs=0,candidates=candidates,paired_oof_f1_intervals=intervals,protected_hashes_before_and_after=protected,weight_averaging=dict(float_keys=float_keys,nonfloat_keys=integer_keys,ratio=[.5,.5],checkpoint_sha256=average_sha),runtime=dict(inference_seconds=elapsed,**memory,total_seconds=time.perf_counter()-started),raw_validation=measured,limitations=limitations,original_files_modified=False,test_data_evaluated=False)
    save('results.json',result)
    lines=['# 固定50:50平均：F1探测','','无新增训练；同一官方验证集1422图、859检查、228患者。原冠军及提交文件未改。','','|方案|AUC|F1@0.5|患者五折阈值OOF F1|OOF Precision|OOF Recall|','|---|---:|---:|---:|---:|---:|']
    for name,r in candidates.items():
        m=r['metrics'];c=r['calibration'];lines.append(f"|{name}|{m['macro_auc']:.8f}|{m['fixed_0_5']['macro_f1']:.8f}|{c['macro_f1']:.8f}|{c['macro_precision']:.8f}|{c['macro_recall']:.8f}|")
    lines+=['','比较基准是已经校准过阈值的冠军OOF F1=0.451055，不能仅比较默认0.5阈值。','']
    for name,v in intervals.items():lines.append(f"- {name} − champion：OOF F1差值 {v['delta']:+.8f}；400次患者配对95%区间 {v['ci95']}。")
    lines+=['','## 限制','']+['- '+v for v in limitations]
    lines+=['','权重平均适用的两份模型已检查所有state key、shape和dtype相同，浮点state以float32平均，非浮点state必须完全相同。无BatchNorm统计需额外估计。','每份gates.json均单独绑定相应权重SHA或概率集成输入，不要混用。','源冠军SHA：'+EXPECTED_CHAMP,'原提交SHA：'+EXPECTED_SUBMISSION]
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status('complete',gpu_released=True,summary={name:dict(auc=r['metrics']['macro_auc'],f1_05=r['metrics']['fixed_0_5']['macro_f1'],oof_f1=r['calibration']['macro_f1']) for name,r in candidates.items()})
    print(json.dumps(dict(summary={name:dict(auc=r['metrics']['macro_auc'],f1_05=r['metrics']['fixed_0_5']['macro_f1'],oof_f1=r['calibration']['macro_f1']) for name,r in candidates.items()},intervals=intervals),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('failed',error=repr(exc));raise
