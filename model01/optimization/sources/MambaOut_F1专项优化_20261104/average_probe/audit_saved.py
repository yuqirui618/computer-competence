"""Independent CPU recomputation of frozen checkpoint-average predictions."""
import os,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
for k in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):os.environ[k]=str(R/'temp')
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

def f1(y,d):
    yy=y.astype(bool);dd=d.astype(bool);tp=(yy&dd).sum(0);fp=(~yy&dd).sum(0);fn=(yy&~dd).sum(0)
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros(10),where=den!=0)
def fit(y,p):
    gates=sorted(np.arange(1,40)/40,key=lambda g:(abs(g-.5),g))
    scores=np.stack([f1(y,p>=g) for g in gates])
    return np.array(gates)[np.argmax(scores,axis=0)]
r=json.loads((R/'results.json').read_text(encoding='utf-8'))
ps={}
for name,entry in r['candidates'].items():
    path=R/(name+'_predictions.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['predictions_sha256']
    with np.load(path,allow_pickle=False) as z: ps[name]={k:z[k].copy() for k in z.files}
    q=ps[name];assert abs(roc_auc_score(q['targets'],q['probabilities'],average='macro')-entry['metrics']['macro_auc'])<1e-12
baseline=ps['champion']
y,p0,s=baseline['targets'],baseline['probabilities'],baseline['subjects']
for q in ps.values():
    for key in ('targets','subjects','study_ids'):assert np.array_equal(q[key],baseline[key])
assert np.array_equal(ps['probability_average']['probabilities'],.5*p0+.5*ps['epoch2']['probabilities'])
repeated={}
for seed in (42,43,44):
    measured={}
    for name,q in ps.items():
        p=q['probabilities'];d=np.zeros_like(y,dtype=bool)
        for a,b in GroupKFold(n_splits=5,shuffle=True,random_state=seed).split(p,y,s):d[b]=p[b]>=fit(y[a],p[a])
        measured[name]=float(f1(y,d).mean())
        if seed==42:assert abs(measured[name]-r['candidates'][name]['calibration']['macro_f1'])<1e-12
    repeated[str(seed)]=measured
rng=np.random.default_rng(20261104)
groups=[np.flatnonzero(s==x) for x in np.unique(s)]
p1=ps['probability_average']['probabilities'];diffs=[]
for _ in range(400):
    ids=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);yy=y[ids]
    if not ((yy==0).any(0)&(yy==1).any(0)).all():continue
    diffs.append(float(roc_auc_score(yy,p1[ids],average='macro')-roc_auc_score(yy,p0[ids],average='macro')))
assert len(diffs)>=300
auc=dict(delta=float(roc_auc_score(y,p1,average='macro')-roc_auc_score(y,p0,average='macro')),ci95=np.quantile(diffs,[.025,.975]).tolist(),repeats=400,valid_repeats=len(diffs),seed=20261104)
out=dict(state='verified',source_predictions_recomputed=True,repeat_fold_oof_f1=repeated,probability_average_minus_champion_auc=auc,limitation='Exploratory fixed-model cross-validation. Seeds43/44 are sensitivity checks after seed42 results, not independent new test sets; no threshold or ratio chosen using them.')
(R/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['','## CPU独立复算与折分敏感性','',f"概率平均−原冠军的AUC差值 {auc['delta']:+.8f}；患者配对95%区间 {auc['ci95']}，{len(diffs)}/400次有效。微小AUC增加也未证实稳定。",'','|患者fold seed|champion|epoch2|probability_average|weight_average|','|---|---:|---:|---:|---:|']
for seed,items in repeated.items():lines.append('|'+seed+'|'+'|'.join(f'{v:.8f}' for v in items.values())+'|')
lines+=['','43/44仅为结果后的折分敏感性检查，不是新增独立测试；未利用它们改权重平均比例或阈值策略。']
with (R/'README.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(out,ensure_ascii=False))
