"""Frozen threshold procedure transfer; no training, no test data, no GPU."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TEMP=ROOT/'temp'; TEMP.mkdir(exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'): os.environ[key]=str(TEMP)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'threshold_search'))
from run_thresholds import ThresholdFit,METHODS,f1_metrics,choose,paired_f1_interval,save_json,sha
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
DATA=ROOT.parent.parent
MODEL=DATA/'MambaOut_数据增强与大验证_20261104/A_current_affine'
EARLY=MODEL/'early_epoch_3.npz'
AUDIT=MODEL/'audit_predictions.npz'
GATE_FILE=ROOT.parent/'threshold_search/candidate_gates.json'

def read_prediction(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k].copy() for k in ('targets','probabilities','subjects','study_ids')}

def main():
    dev=read_prediction(EARLY)
    y,p,s=dev['targets'],dev['probabilities'],dev['subjects']
    assert y.shape==p.shape==(2646,10) and len(np.unique(s))==1658
    champion_method=json.loads(GATE_FILE.read_text(encoding='utf-8'))['strategy']
    decisions={m:np.zeros_like(y,dtype=bool) for m in METHODS}
    fold_records=[]
    for fold,(fit,held) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s),1):
        assert not set(s[fit])&set(s[held])
        fits=ThresholdFit(y[fit],p[fit],s[fit]).all_methods(seed=20261105*100+fold)
        for m,gates in fits.items(): decisions[m][held]=p[held]>=gates
        fold_records.append({'fold':fold,'fit_patients':len(np.unique(s[fit])),'held_patients':len(np.unique(s[held])),'gates':{m:g.tolist() for m,g in fits.items()}})
    cv={m:f1_metrics(y,decisions[m]) for m in METHODS}
    selected=choose(cv)
    fitted=ThresholdFit(y,p,s).all_methods(seed=20261105+99)
    old=json.loads((MODEL/'results.json').read_text(encoding='utf-8'))
    assert np.array_equal(fitted['coarse'],np.asarray(old['gates']))
    selection={'fit_source':str(EARLY),'fit_source_sha256':sha(EARLY),'development_cv':cv,'folds':fold_records,
               'selected_on_development':selected,'champion_selected_method_frozen':champion_method,
               'gates':{m:g.tolist() for m,g in fitted.items()},'audit_not_used_to_fit_or_select':True,
               'historical_audit_metrics_already_seen':True,'bootstrap_fit_seed':20261105+99}
    save_json(ROOT/'selection.json',selection)
    # Only open the audit predictions after writing the frozen decision record.
    audit=read_prediction(AUDIT)
    ay,ap,subjects=audit['targets'],audit['probabilities'],audit['subjects']
    assert ay.shape==ap.shape==(2842,10) and len(np.unique(subjects))==1658
    assert not set(subjects)&set(s)
    methods=list(dict.fromkeys(('coarse',selected,champion_method)))
    outputs={m:ap>=fitted[m] for m in methods}
    metrics={m:f1_metrics(ay,outputs[m]) for m in methods}
    assert abs(metrics['coarse']['macro_f1']-.43411780452790394)<1e-12
    intervals={m:paired_f1_interval(ay,outputs['coarse'],outputs[m],subjects) for m in methods if m!='coarse'}
    for interval in intervals.values(): interval['scope']='Conditional on fixed model and frozen development-fitted thresholds; no refitting, model selection, or external population uncertainty.'
    np.savez_compressed(ROOT/'audit_decisions.npz',targets=ay,probabilities=ap,subjects=subjects,study_ids=audit['study_ids'],**outputs)
    result={'selection':selection,'audit_source_sha256':sha(AUDIT),'source_model_sha256':sha(MODEL/'best.pth'),
            'audit_macro_auc_unchanged':float(roc_auc_score(ay,ap,average='macro')),
            'audit_metrics':metrics,'paired_f1_intervals':intervals,'evaluated_methods':methods,
            'interpretation':'Internal transfer on a different model/patient cohort. Earlier audit was already inspected; not external or fresh blind validation. Champion gates themselves are never transferred to this differently trained model.',
            'no_weights_or_original_gates_modified':True}
    save_json(ROOT/'results.json',result)
    lines=['# 较大患者队列：阈值策略迁移复核','',
           '使用上一轮只在训练子集拟合的A组第3轮模型。开发集2646次检查/1658患者，保留集2842次检查/1658患者，患者互斥。',
           '算法在开发集选定并拟合阈值后冻结；另将本轮冠军选中的阈值算法在此开发集重新拟合，不能直接复制冠军阈值。',
           '这个保留集在上一轮已经查看过，因此仅为内部复核，不是新的盲测。','',
           '| 阈值算法 | 开发患者CV F1 | 保留集 F1 | 保留集精确率 | 保留集召回率 |','|---|---:|---:|---:|---:|']
    for m in methods:
        v=metrics[m];lines.append(f"| {m} | {cv[m]['macro_f1']:.8f} | {v['macro_f1']:.8f} | {v['macro_precision']:.8f} | {v['macro_recall']:.8f} |")
    lines+=['',f'开发集预选算法：{selected}；冠军队列预选算法：{champion_method}。',f"概率未变化，保留集AUC仍为{result['audit_macro_auc_unchanged']:.8f}。"]
    for m,interval in intervals.items():lines.append(f"{m}相对coarse差值{interval['delta']:+.8f}，400次患者条件95%区间{interval['ci95']}。")
    (ROOT/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'selected':selected,'champion_method':champion_method,'audit_macro_f1':{m:v['macro_f1'] for m,v in metrics.items()},'intervals':intervals},ensure_ascii=False))

if __name__=='__main__':main()
