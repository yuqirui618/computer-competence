"""Fixed 50:50 probability ensemble, unchanged nested patient threshold protocol."""
from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent
for k in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):os.environ[k]=str(ROOT/'temp')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score
from sklearn.model_selection import GroupKFold
import threshold_core as run
BASE=ROOT.parent
OLD=BASE/'threshold_search'
PROBE=BASE/'average_probe'
SOURCE=PROBE/'probability_average_predictions.npz'
NOTE='Bootstrap exact uses fixed unique score support of the fit cohort with patient-bootstrap weights; zero-weight patient score breakpoints remain eligible. No held-out scores or labels define the support.'


def load(path):
    with np.load(path,allow_pickle=False) as a:
        return {key:a[key].copy() for key in a.files}


def main():
    start=time.perf_counter()
    previous=json.loads((OLD/'results.json').read_text(encoding='utf-8'))
    probe=json.loads((PROBE/'results.json').read_text(encoding='utf-8'))
    candidate=probe['candidates']['probability_average']
    binding=candidate['binding']
    assert binding['kind']=='probability_ensemble' and binding['weights']==[.5,.5]
    checkpoints=binding['source_checkpoints']
    assert len(checkpoints)==2 and all(x['weight']==.5 for x in checkpoints)
    protected=[SOURCE,PROBE/'results.json',OLD/'results.json',run.CHAMPION,run.GATES,run.SUBMISSION,
               *(Path(x['path']) for x in checkpoints),*(OLD/f'oof_seed_{seed}.npz' for seed in run.OUTER_SEEDS)]
    hashes={str(path):run.sha(path) for path in protected}
    for checkpoint in checkpoints:
        assert hashes[checkpoint['path']]==checkpoint['sha256']
    assert hashes[str(run.CHAMPION)]==run.CHAMPION_SHA and hashes[str(run.SUBMISSION)]==run.SUBMISSION_SHA
    arrays=load(SOURCE)
    y=arrays['targets'].astype(np.int8);p=arrays['probabilities'].astype(np.float64);subjects=arrays['subjects'].astype(str)
    assert y.shape==p.shape==(859,10) and len(np.unique(subjects))==228
    assert np.isin(y,(0,1)).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    source_auc=float(roc_auc_score(y,p,average='macro'))
    assert abs(source_auc-candidate['metrics']['macro_auc'])<1e-14
    plan=dict(methods=list(run.METHODS),outer_seeds=list(run.OUTER_SEEDS),outer_folds=5,inner_folds=4,
              bootstrap_fits=40,bootstrap_support=NOTE,source_sha256=hashes[str(SOURCE)],binding=binding,
              source_auc=source_auc,expected_coarse_f1_seed42=candidate['calibration']['macro_f1'],
              protocol='Exactly original nested threshold strategy/seed protocol, fixed probability ensemble only',
              comparisons=['ensemble_nested minus original_champion_coarse','ensemble_nested minus original_champion_nested','ensemble_nested minus ensemble_coarse'],
              original_fitter_sha256=run.sha(OLD/'run_thresholds.py'),derived_fitter_sha256=run.sha(ROOT/'threshold_core.py'),
              model_choice_uses_reused_validation=True,no_new_model_or_weight_search=True)
    if (ROOT/'plan.json').is_file():
        assert json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))==plan
    else:
        run.save_json(ROOT/'plan.json',plan)
    seeds={};comparisons={};checks=0;fold_support={}
    for seed in run.OUTER_SEEDS:
        old=load(OLD/f'oof_seed_{seed}.npz')
        assert np.array_equal(y,old['targets']) and np.array_equal(subjects,old['subjects']);checks+=2
        record,decisions,nested=run.evaluate_seed(y,p,subjects,seed)
        seeds[str(seed)]=record
        comparisons[str(seed)]={
            'versus_original_coarse':run.paired_f1_interval(y,old['coarse'],nested,subjects),
            'versus_original_nested':run.paired_f1_interval(y,old['nested_selection'],nested,subjects),
            'versus_ensemble_coarse':run.paired_f1_interval(y,decisions['coarse'],nested,subjects)}
        for method,decision in {**decisions,'nested_selection':nested}.items():
            expected=record['metrics'][method]
            for field,func in [('f1_by_class',f1_score),('precision_by_class',precision_score),('recall_by_class',recall_score)]:
                assert np.allclose(func(y,decision,average=None,zero_division=0),expected[field],atol=1e-14,rtol=0)
                checks+=1
        fold_support[str(seed)]=[]
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=seed).split(p,y,subjects),1):
            assert np.all(old['fold_ids'][te]==fold);checks+=1
            rec=record['folds'][fold-1]
            for method in run.METHODS:
                assert np.array_equal(p[te]>=np.asarray(rec['final_outer_train_gates'][method]),decisions[method][te]);checks+=1
            assert np.array_equal(nested[te],decisions[rec['inner_selection_strategy']][te]);checks+=1
            fold_support[str(seed)].append({'fold':fold,'held_studies':len(te),'held_patients':len(np.unique(subjects[te])),
                'held_positive_studies':y[te].sum(0).astype(int).tolist(),
                'held_positive_patients':[len(np.unique(subjects[te][y[te,j]==1])) for j in range(10)]})
    selected=run.choose(seeds['42']['metrics'])
    fits=run.ThresholdFit(y,p,subjects).all_methods(seed=42000000+99)
    gates=fits[selected]
    old_gates=json.loads((PROBE/'probability_average_gates.json').read_text(encoding='utf-8'))['gate_by_class_id']
    assert np.array_equal(fits['coarse'],np.asarray(old_gates));checks+=1
    gate_record=dict(binding=binding,gate_by_class_id=gates.tolist(),class_ids=list(range(1,11)),
        rule='Probability >= Gate after same-Study probability averaging and fixed 0.5:0.5 model probability average',
        strategy=selected,bootstrap_support=NOTE,fit_source='All 859 reused official validation Studies, 228 patients; no test labels',
        source_prediction_sha256=hashes[str(SOURCE)],source_auc=source_auc,auc_unchanged_by_threshold=True,
        nested_selection_oof_macro_f1=seeds['42']['metrics']['nested_selection']['macro_f1'],
        selected_strategy_oof_macro_f1=seeds['42']['metrics'][selected]['macro_f1'],
        full_fit_in_sample_macro_f1=run.f1_metrics(y,p>=gates)['macro_f1'],
        caution='Threshold candidate for this two-checkpoint ensemble only. Nested threshold CV does not undo model/ensemble selection using this validation. No blind-test improvement claim.')
    after={path:run.sha(path) for path in hashes}
    assert after==hashes;checks+=1
    result=dict(state='complete',plan=plan,source_auc=source_auc,auc_unchanged_by_threshold=True,seeds=seeds,
        comparisons=comparisons,selected_strategy=selected,candidate_gates=gate_record,checks=checks,
        original_threshold_results=json.loads((OLD/'results.json').read_text(encoding='utf-8'))['seeds'],
        original_source_hashes_unchanged=True,protected_hashes_after=after,seconds=time.perf_counter()-start,
        limits=['Ensemble/model selection relied on previously inspected validation; nested threshold folds do not remove that bias',
               'Same five strategies, outer seeds42/43/44 and inner4 patient folds as original champion threshold study',
               'Pointwise paired bootstrap conditions on saved OOF decisions; no refitting within bootstrap, no full procedure or training-seed uncertainty',
               'Repeated split seeds are not independent datasets',
               'Full-fit F1 uses the same labels that fit gates and is not a generalization estimate',
               'No additional model or averaging-weight search, no training/GPU/test inference, no source checkpoint/gates/submission edits'])
    run.save_json(ROOT/'results.json',result)
    run.save_json(ROOT/'candidate_gates.json',gate_record)
    run.save_json(ROOT/'fold_support.json',fold_support)
    run.save_json(ROOT/'independent_checks.json',dict(passed=True,checks=checks,coarse_probe_reproduced=True,
        original_fold_alignment_verified=True,sklearn_metrics_recomputed=True,protected_sources_unchanged=True))
    lines=['# 固定双模型概率平均：F1阈值追加','',
        '固定两个指定权重的0.5:0.5概率平均，没有扫描新模型或权重。原单模型阈值不能直接复用。',
        f'平均模型AUC={source_auc:.10f}；本次所有阈值方法保持该AUC不变。','',
        '| 对照 | seed42 F1 | seed43 F1 | seed44 F1 |','|---|---:|---:|---:|']
    for old_name in ('coarse','nested_selection'):
        lines.append('| 原冠军 '+old_name+' | '+' | '.join(f"{previous['seeds'][str(s)]['metrics'][old_name]['macro_f1']:.8f}" for s in run.OUTER_SEEDS)+' |')
    for method in run.METHODS+('nested_selection',):
        lines.append('| 概率平均 '+method+' | '+' | '.join(f"{seeds[str(s)]['metrics'][method]['macro_f1']:.8f}" for s in run.OUTER_SEEDS)+' |')
    lines+=['','## 概率平均嵌套策略的患者配对差值','',
        '| seed | 对照 | F1差值 | 95%条件区间 |','|---|---|---:|---|']
    for seed in run.OUTER_SEEDS:
        for key,entry in comparisons[str(seed)].items():
            lines.append(f"| {seed} | {key} | {entry['delta']:+.8f} | [{entry['ci95'][0]:+.8f}, {entry['ci95'][1]:+.8f}] |")
    lines+=['',f'完整验证集CV预选阈值策略：{selected}。候选阈值已绑定两个checkpoint SHA及0.5:0.5概率均值。',
        f"拟合内F1={gate_record['full_fit_in_sample_macro_f1']:.8f}，不得当作泛化提升。",
        '模型/平均组合本身源自已看过的验证集历史，nested只能隔离阈值策略选择，不能消除模型选择偏差。所有bootstrap区间仅对现成OOF决策重采样，未重新拟合完整流程。',
        NOTE,'',f'CPU复算通过{checks}项检查；所有原文件SHA保持不变。']
    (ROOT/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    run.progress(state='complete',selected=selected,nested_f1=gate_record['nested_selection_oof_macro_f1'],
                 versus_original_coarse=comparisons['42']['versus_original_coarse'],seconds=result['seconds'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        run.progress(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
