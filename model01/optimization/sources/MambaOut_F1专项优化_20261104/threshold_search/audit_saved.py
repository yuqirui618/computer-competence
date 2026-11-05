"""Independent saved-output checks and anonymized per-fold support counts."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parent
for k in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'): os.environ[k]=str(ROOT/'temp')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
sys.dont_write_bytecode=True
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import GroupKFold
import run_thresholds as run

def support(y,s):
    return {'studies':len(y),'patients':len(np.unique(s)),
            'positive_studies':y.sum(0).astype(int).tolist(),
            'positive_patients':[len(np.unique(s[y[:,j]==1])) for j in range(10)]}

result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
with np.load(run.SOURCE,allow_pickle=False) as a:
    y=a['targets'].astype(np.int8);p=a['probabilities'].astype(np.float64);s=a['subjects'].astype(str)
summaries={}
checks=0
for seed in run.OUTER_SEEDS:
    records=result['seeds'][str(seed)]
    with np.load(ROOT/f'oof_seed_{seed}.npz',allow_pickle=False) as archive:
        assert np.array_equal(y,archive['targets']) and np.array_equal(p,archive['probabilities']) and np.array_equal(s,archive['subjects'])
        checks+=3
        for method in run.METHODS+('nested_selection',):
            decisions=archive[method]
            expected=records['metrics'][method]
            for field,func in [('f1_by_class',f1_score),('precision_by_class',precision_score),('recall_by_class',recall_score)]:
                assert np.allclose(func(y,decisions,average=None,zero_division=0),expected[field],atol=1e-14,rtol=0)
                checks+=1
        summaries[str(seed)]=[]
        outer=GroupKFold(n_splits=5,shuffle=True,random_state=seed)
        for fold,(tr,te) in enumerate(outer.split(p,y,s),1):
            assert np.all(archive['fold_ids'][te]==fold)
            rec=records['folds'][fold-1]
            for method in run.METHODS:
                expected=p[te]>=np.asarray(rec['final_outer_train_gates'][method])
                assert np.array_equal(expected,archive[method][te]); checks+=1
            selected=rec['inner_selection_strategy']
            assert selected==max(run.METHODS,key=lambda m:rec['inner_macro_f1'][m]);checks+=1
            assert np.array_equal(archive['nested_selection'][te],archive[selected][te]);checks+=1
            # Reproduce original coarse fitter independently, exactly as project used.
            grid=sorted(np.arange(1,40)/40,key=lambda x:(abs(x-.5),x))
            scores=[f1_score(y[tr],p[tr]>=g,average=None,zero_division=0) for g in grid]
            original=np.asarray(grid)[np.argmax(scores,axis=0)]
            assert np.array_equal(original,np.asarray(rec['final_outer_train_gates']['coarse']));checks+=1
            inner=[]
            for idx,(fit,held) in enumerate(GroupKFold(n_splits=4,shuffle=True,random_state=seed+fold*100).split(p[tr],y[tr],s[tr]),1):
                inner.append({'inner_fold':idx,'fit':support(y[tr][fit],s[tr][fit]),'held':support(y[tr][held],s[tr][held])})
            summaries[str(seed)].append({'outer_fold':fold,'fit':support(y[tr],s[tr]),'held':support(y[te],s[te]),'inner':inner})
# Independently brute-force observed-score exact threshold for a meaningful small subset.
fit=run.ThresholdFit(y[:60],p[:60],s[:60])
for j,actual in enumerate(fit.base_fit('exact')):
    if y[:60,j].sum()==0:
        assert actual==.5
    else:
        candidates=np.unique(p[:60,j])
        scored=[f1_score(y[:60,j],p[:60,j]>=gate,zero_division=0) for gate in candidates]
        mx=max(scored)
        expected=min((float(g) for g,f in zip(candidates,scored) if f==mx),key=lambda g:(abs(g-.5),g))
        assert actual==expected
    checks+=1
run.save_json(ROOT/'fold_support.json',summaries)
run.save_json(ROOT/'independent_checks.json',{'passed':True,'checks':checks,'methods':'sklearn metrics and original coarse fitter, saved gate reconstruction, exact brute-force small subset',
    'coarse_original_three_seeds_reproduced':True,'fold_counts_only_no_patient_ids':True,
    'bootstrap_implementation_note':'Exact bootstrap uses the fixed score support of its fit cohort with patient-bootstrap multiplicity weights. Non-drawn patients have zero objective weight, but their observed score values remain allowed threshold candidates. This uses no outer-held scores or labels; tie preference is closest .5 then lower.'})
print(json.dumps({'passed':True,'checks':checks,'class1_outer_held_positive_patients_seed42':[f['held']['positive_patients'][0] for f in summaries['42']]},ensure_ascii=False))
