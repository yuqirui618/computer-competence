# Derived without threshold algorithm changes; only seed42 coarse reproduction expectation changed for probability averaging.
"""CPU-only prespecified threshold strategy evaluation with nested patient CV."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'temp'
CACHE.mkdir(exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):
    os.environ[key]=str(CACHE)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, hashlib, time
from datetime import datetime
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

DATA=ROOT.parent.parent
SOURCE=DATA/'MambaOut_F1正则化优化/baseline_predictions.npz'
SOURCE=DATA/'MambaOut_F1正则化优化_20261104/baseline_predictions.npz'
GATES=DATA/'MambaOut_ASL对照/gates.json'
CHAMPION=DATA/'MambaOut_ASL对照/best_model.pth'
SUBMISSION=DATA/'提交结果/MambaOut_S7微调第1轮_20261104/submission.csv'
CHAMPION_SHA='9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585'
SUBMISSION_SHA='d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8'
METHODS=('coarse','fine','exact','bootstrap_exact_median','bootstrap_fine_median')
OUTER_SEEDS=(42,43,44)
BOOTSTRAPS=40
GRIDS={'coarse':np.arange(1,40)/40, 'fine':np.arange(1,200)/200}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save_json(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def progress(**kw):
    obj={'time':datetime.now().astimezone().isoformat(),'pid':os.getpid(),**kw}
    save_json(ROOT/'status.json',obj)
    print(json.dumps(obj,ensure_ascii=False),flush=True)


def f1_metrics(y,decisions,weights=None):
    if weights is None:
        weights=np.ones(len(y),dtype=float)
    weights=np.asarray(weights)[:,None]
    tp=((y==1)&decisions)*weights
    fp=((y==0)&decisions)*weights
    fn=((y==1)&~decisions)*weights
    tp,fp,fn=tp.sum(0),fp.sum(0),fn.sum(0)
    den=2*tp+fp+fn
    f=np.divide(2*tp,den,out=np.zeros(10),where=den>0)
    prec=np.divide(tp,tp+fp,out=np.zeros(10),where=tp+fp>0)
    rec=np.divide(tp,tp+fn,out=np.zeros(10),where=tp+fn>0)
    return dict(macro_f1=float(f.mean()),f1_by_class=f.tolist(),macro_precision=float(prec.mean()),
                precision_by_class=prec.tolist(),macro_recall=float(rec.mean()),recall_by_class=rec.tolist(),
                tp_by_class=tp.astype(int).tolist(),fp_by_class=fp.astype(int).tolist(),fn_by_class=fn.astype(int).tolist())


class ThresholdFit:
    def __init__(self,y,p,subjects):
        self.y=np.asarray(y,dtype=np.int8)
        self.p=np.asarray(p,dtype=np.float64)
        self.subjects=np.asarray(subjects,dtype=str)
        self.people,self.patient_index=np.unique(self.subjects,return_inverse=True)
        self.sorted=[]
        for j in range(10):
            order=np.argsort(self.p[:,j],kind='stable')
            scores=self.p[order,j]
            unique=np.unique(scores)
            candidates={'coarse':GRIDS['coarse'],'fine':GRIDS['fine'],'exact':unique}
            self.sorted.append((order,{k:(g,np.searchsorted(scores,g,side='left')) for k,g in candidates.items()}))

    def base_fit(self,method,weights=None):
        w=np.ones(len(self.y),dtype=float) if weights is None else np.asarray(weights,dtype=float)
        gates=[]
        for j,(order,candidates) in enumerate(self.sorted):
            yw=self.y[order,j]*w[order]
            total=yw.sum()
            if total<=0:
                gates.append(.5)
                continue
            prefix_tp=np.r_[0,np.cumsum(yw)]
            prefix_n=np.r_[0,np.cumsum(w[order])]
            g,start=candidates[method]
            tp=total-prefix_tp[start]
            predicted=prefix_n[-1]-prefix_n[start]
            f1=2*tp/(total+predicted)
            best=np.flatnonzero(f1==f1.max())
            # Same deterministic preference as existing project's grid fitter.
            selected=min(best,key=lambda idx:(abs(float(g[idx])-.5),float(g[idx])))
            gates.append(float(g[selected]))
        return np.asarray(gates)

    def all_methods(self,seed):
        fitted={m:self.base_fit(m) for m in ('coarse','fine','exact')}
        bag_exact,bag_fine=[],[]
        rng=np.random.default_rng(seed)
        for _ in range(BOOTSTRAPS):
            frequencies=np.bincount(rng.integers(len(self.people),size=len(self.people)),minlength=len(self.people))
            weights=frequencies[self.patient_index]
            bag_exact.append(self.base_fit('exact',weights))
            bag_fine.append(self.base_fit('fine',weights))
        fitted['bootstrap_exact_median']=np.median(bag_exact,axis=0)
        fitted['bootstrap_fine_median']=np.median(bag_fine,axis=0)
        return fitted


def choose(metrics):
    # One shared strategy for all ten classes; no per-class strategy search.
    return max(METHODS,key=lambda m:metrics[m]['macro_f1'])


def check_alignment(y,p,s):
    if y.shape!=(859,10) or p.shape!=y.shape or s.shape!=(859,) or len(np.unique(s))!=228:
        raise ValueError('Champion source cohort differs')
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid probability or target')
    if not np.isclose(roc_auc_score(y,p,average='macro'),.8019769550500871,atol=1e-14,rtol=0):
        raise ValueError('Source AUC differs')


def evaluate_seed(y,p,subjects,outer_seed):
    outer=GroupKFold(n_splits=5,shuffle=True,random_state=outer_seed)
    decisions={method:np.zeros_like(y,dtype=bool) for method in METHODS}
    nested=np.zeros_like(y,dtype=bool)
    fold_ids=np.full(len(y),-1,dtype=int)
    records=[]
    for fold,(train,test) in enumerate(outer.split(p,y,subjects),1):
        if set(subjects[train])&set(subjects[test]):
            raise ValueError('Outer patient overlap')
        inner_outputs={method:np.zeros_like(y[train],dtype=bool) for method in METHODS}
        inner=GroupKFold(n_splits=4,shuffle=True,random_state=outer_seed+fold*100)
        for inner_fold,(fit,held) in enumerate(inner.split(p[train],y[train],subjects[train]),1):
            if set(subjects[train][fit])&set(subjects[train][held]):
                raise ValueError('Inner patient overlap')
            fitted=ThresholdFit(y[train][fit],p[train][fit],subjects[train][fit]).all_methods(
                seed=outer_seed*100000+fold*1000+inner_fold)
            for method,gates in fitted.items():
                inner_outputs[method][held]=p[train][held]>=gates
        inner_metrics={method:f1_metrics(y[train],inner_outputs[method]) for method in METHODS}
        selected=choose(inner_metrics)
        fitted=ThresholdFit(y[train],p[train],subjects[train]).all_methods(seed=outer_seed*100000+fold*1000+99)
        for method,gates in fitted.items():
            decisions[method][test]=p[test]>=gates
        nested[test]=decisions[selected][test]
        fold_ids[test]=fold
        records.append(dict(fold=fold,train_studies=len(train),held_studies=len(test),
                            train_patients=len(np.unique(subjects[train])),held_patients=len(np.unique(subjects[test])),
                            inner_selection_strategy=selected,
                            inner_macro_f1={m:inner_metrics[m]['macro_f1'] for m in METHODS},
                            final_outer_train_gates={m:g.tolist() for m,g in fitted.items()},
                            outer_held_scores={m:f1_metrics(y[test],decisions[m][test]) for m in METHODS}))
        progress(state='running',seed=outer_seed,completed_outer_folds=fold,selected=selected)
    if (fold_ids<1).any():
        raise ValueError('OOF coverage failure')
    metrics={method:f1_metrics(y,decision) for method,decision in decisions.items()}
    metrics['nested_selection']=f1_metrics(y,nested)
    metrics['fixed_0_5']=f1_metrics(y,p>=.5)
    baseline=metrics['coarse']['macro_f1']
    if outer_seed==42 and not np.isclose(baseline,0.4529390017268306,atol=1e-14,rtol=0):
        raise ValueError(f'Original OOF F1 failed to reproduce: {baseline}')
    save_json(ROOT/f'seed_{outer_seed}.json',dict(seed=outer_seed,metrics=metrics,folds=records))
    np.savez_compressed(ROOT/f'oof_seed_{outer_seed}.npz',targets=y,probabilities=p,subjects=subjects,
                        fold_ids=fold_ids,nested_selection=nested,**decisions)
    return dict(seed=outer_seed,metrics=metrics,folds=records),decisions,nested


def paired_f1_interval(y,a,b,subjects,seed=20261104,repeats=400):
    people,inv=np.unique(subjects,return_inverse=True)
    rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(repeats):
        freq=np.bincount(rng.integers(len(people),size=len(people)),minlength=len(people))
        w=freq[inv]
        deltas.append(f1_metrics(y,b,w)['macro_f1']-f1_metrics(y,a,w)['macro_f1'])
    return dict(delta=f1_metrics(y,b)['macro_f1']-f1_metrics(y,a)['macro_f1'],
                ci95=np.quantile(deltas,[.025,.975]).tolist(),repeats=repeats,
                scope='Conditional on saved cross-fitted decisions. No threshold refitting within bootstrap; no model-selection or training-seed uncertainty.')


def main():
    start=time.perf_counter()
    protected={str(path):sha(path) for path in (SOURCE,GATES,CHAMPION,SUBMISSION)}
    if protected[str(CHAMPION)]!=CHAMPION_SHA or protected[str(SUBMISSION)]!=SUBMISSION_SHA:
        raise ValueError('Protected champion or submission differs')
    plan=dict(methods=list(METHODS),outer_seeds=list(OUTER_SEEDS),outer_folds=5,inner_folds=4,
              threshold_bootstraps=BOOTSTRAPS,patient_bootstrap=True,
              selection='Global strategy by inner pooled OOF Macro F1, strategy ties in listed order',
              seeds='Inner split seed=outerseed+fold*100; fit seed=outerseed*100000+fold*1000+innerfold; outer fit seed ends99',
              exact_thresholds='Unique observed per-class score values; >= decision; equal optimum chooses closest .5 then lower; all-negative bootstrap class returns .5',
              fine_grid='np.arange(1,200)/200',coarse_grid='np.arange(1,40)/40',
              final_policy='Choose strategy from seed42 full 5-fold OOF scores, fit all validation; nested OOF estimates the method-selection procedure',
              original_model_was_selected_on_validation=True,source_hashes=protected)
    if (ROOT/'plan.json').is_file():
        if json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))!=plan:
            raise ValueError('Prespecified plan differs')
    else:
        save_json(ROOT/'plan.json',plan)
    with np.load(SOURCE,allow_pickle=False) as archive:
        y=archive['targets'].astype(np.int8)
        p=archive['probabilities'].astype(np.float64)
        subjects=archive['subjects'].astype(str)
    check_alignment(y,p,subjects)
    # Small meaningful implementation check: numpy confusion agrees with sklearn.
    if not np.allclose(f1_metrics(y,p>=.5)['f1_by_class'],f1_score(y,p>=.5,average=None,zero_division=0),atol=1e-14,rtol=0):
        raise ValueError('Confusion metric check failed')
    original=json.loads(GATES.read_text(encoding='utf-8-sig'))
    coarse=ThresholdFit(y,p,subjects).base_fit('coarse')
    if not np.array_equal(coarse,np.asarray(original['final_gates_by_class_id'])):
        raise ValueError('Original full-validation gates do not reproduce')
    seeds={}
    first_decisions=first_nested=None
    for seed in OUTER_SEEDS:
        record,decisions,nested=evaluate_seed(y,p,subjects,seed)
        seeds[str(seed)]=record
        if seed==42:
            first_decisions,first_nested=decisions,nested
    selected=choose(seeds['42']['metrics'])
    all_fits=ThresholdFit(y,p,subjects).all_methods(seed=42000000+99)
    gates=all_fits[selected]
    fixed_boot=paired_f1_interval(y,first_decisions['coarse'],first_nested,subjects)
    method_boot={m:paired_f1_interval(y,first_decisions['coarse'],first_decisions[m],subjects) for m in METHODS if m!='coarse'}
    final_gate_record=dict(checkpoint=str(CHAMPION),checkpoint_sha256=CHAMPION_SHA,fit_prediction_sha256=protected[str(SOURCE)],
                           strategy=selected,gate_by_class_id=gates.tolist(),class_order=list(range(1,11)),
                           fit_cohort='Entire official validation only (859 studies / 228 patients), no test labels',
                           final_fitted_in_sample_f1=f1_metrics(y,p>=gates)['macro_f1'],
                           nested_selection_oof_f1=seeds['42']['metrics']['nested_selection']['macro_f1'],
                           original_coarse_oof_f1=seeds['42']['metrics']['coarse']['macro_f1'],
                           warning='Candidate only. Do not overwrite prior gates or submission; reused validation and model selection preclude blind-test claims.')
    after={path:sha(path) for path in protected}
    if after!=protected:
        raise ValueError('A protected source file changed')
    result=dict(plan=plan,source_auc=float(roc_auc_score(y,p,average='macro')),auc_unchanged=True,seeds=seeds,
                seed42_nested_minus_coarse_bootstrap=fixed_boot,seed42_fixed_method_minus_coarse_bootstraps=method_boot,
                selected_final_strategy=selected,candidate_gates=final_gate_record,
                repeated_nested_minus_coarse=[seeds[str(s)]['metrics']['nested_selection']['macro_f1']-seeds[str(s)]['metrics']['coarse']['macro_f1'] for s in OUTER_SEEDS],
                original_sources_unchanged=True,protected_hashes_after=after,seconds=time.perf_counter()-start,
                limitations=['Same official validation used earlier for model and epoch selection; outer threshold folds do not remove that bias',
                'Only 228 patients and 16 class1-positive patients; multiple threshold choices have selection variance',
                'Best standalone OOF strategy score is exploratory after comparing methods, not an unbiased improvement estimate',
                'Nested cross-fitting evaluates selection of a shared threshold method; not a new external cohort',
                'Bootstrap CI conditions on saved OOF decisions and does not rerun nested training or threshold fitting',
                'Repeated split seeds measure partition sensitivity, not independent replications',
                'No new training, no GPU, no test inference or labels, no original gates/weights/submission changed'])
    save_json(ROOT/'candidate_gates.json',final_gate_record)
    save_json(ROOT/'results.json',result)
    lines=['# F1阈值专项：嵌套患者交叉验证','',
           '仅修改各类别判定阈值，概率与AUC保持不变。原冠军、原阈值、原提交文件均通过前后SHA核对。',
           '所有方法及三个划分种子在计算前固定；outer5/inner4按患者分组，inner选全类别共同策略，outer只评分。','',
           '| 方法 | seed42 F1 | seed43 F1 | seed44 F1 |','|---|---:|---:|---:|']
    for method in ('fixed_0_5',)+METHODS+('nested_selection',):
        lines.append('| '+method+' | '+' | '.join(f"{seeds[str(s)]['metrics'][method]['macro_f1']:.8f}" for s in OUTER_SEEDS)+' |')
    lines+=['',f"seed42嵌套策略相对原coarse：{fixed_boot['delta']:+.8f}，95%患者条件区间 {fixed_boot['ci95']}。",
            f"完整验证集CV最终选择策略：{selected}。拟合完整验证集的候选阈值另存candidate_gates.json，未覆盖任何原文件。",
            f"全体拟合内F1={final_gate_record['final_fitted_in_sample_f1']:.8f}，不可当作泛化提升。",
            '最佳单项OOF存在多方案选择偏差；应优先看嵌套结果，以及三个划分种子的方向是否一致。',
            '本模型早先已在同一验证集选权重，因此嵌套阈值验证仍不是独立盲测。置信区间只对已保存OOF决策重采样，未包含重新拟合阈值或训练的不确定性。',
            '',f"耗时 {result['seconds']:.2f} 秒；CPU-only；源AUC={result['source_auc']:.10f}。"]
    (ROOT/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    progress(state='complete',seconds=result['seconds'],selected=selected,nested_delta=fixed_boot['delta'],nested_ci95=fixed_boot['ci95'])


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        progress(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
