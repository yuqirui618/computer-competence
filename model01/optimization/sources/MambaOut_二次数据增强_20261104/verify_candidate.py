"""Reload the final candidate after training, and compare frozen predictions."""
import os,sys,json,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DATA=ROOT.parent
TEMP=ROOT/'temp';TEMP.mkdir(exist_ok=True)
for k,v in {'TEMP':TEMP,'TMP':TEMP,'TMPDIR':TEMP,'CUDA_CACHE_PATH':TEMP/'cuda','HF_HOME':TEMP/'huggingface','TORCH_HOME':TEMP/'torch','XDG_CACHE_HOME':TEMP}.items():
    Path(v).mkdir(parents=True,exist_ok=True);os.environ[k]=str(v)
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
sys.path.insert(0,str(DATA/'本地跑通'));sys.path.insert(0,str(DATA/'MambaOut_泛化优化'));sys.path.insert(0,str(DATA/'MambaOut_100轮'))
import train_generalization as core
import train_and_optimize as ref
import numpy as np
import torch,yaml
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_pred(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def oof(y,p,s):
    d=np.zeros_like(y,dtype=bool)
    for fit,held in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s):d[held]=p[held]>=ref.fit_gates(y[fit],p[fit])
    return d

def main():
    result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
    assert result['state']=='complete'
    name=result['winner_including_original']
    assert name!='original_champion','No new weight needs this verification'
    stage=result['stages'][name];epoch=stage['best_epoch_including_original'];path=Path(result['selected_checkpoint'])
    assert sha(path)==result['selected_sha256']
    protected=json.loads((ROOT/'protected_inputs.json').read_text(encoding='utf-8'))
    assert all(sha(p)==v for p,v in protected.items())
    saved=load_pred(ROOT/name/f'epoch_{epoch}_predictions.npz')
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['training']['best_epoch']==epoch and state['training']['img_size']==224
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;ref.ARGS.batch_size=12
    model=core.build_model(state['training'],state['state_dict']).eval();del state
    cfg=yaml.safe_load((DATA/'本地跑通/data_local.yaml').read_text(encoding='utf-8'))
    val=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'val',ref.build_transform(False,224))
    measured,pred=ref.validate(model,val,keep_predictions=True)
    expected=stage['best_validation']
    for level in ('image','study'):
        for k in ('auc','f1','macro_auc','macro_f1'):np.testing.assert_allclose(measured[level][k],expected[level][k],rtol=0,atol=1e-8)
    assert np.array_equal(pred[0],saved['targets']) and np.array_equal(pred[2],saved['subjects'])
    delta=float(np.max(np.abs(pred[1]-saved['probabilities'])))
    np.testing.assert_allclose(pred[1],saved['probabilities'],rtol=0,atol=1e-7)
    del model;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    print('GPU_RELEASED; candidate predictions verified',flush=True)
    y,p,s=pred;cal=ref.evaluate_gates(pred)
    np.savez_compressed(ROOT/'verified_candidate_predictions.npz',targets=y,probabilities=p,subjects=s,study_ids=saved['study_ids'])
    old=load_pred(ROOT/'baseline_predictions.npz')
    previous=load_pred(DATA/'MambaOut_F1专项优化_20261104/average_probe/probability_average_predictions.npz')
    assert np.array_equal(y,previous['targets']) and np.array_equal(s,previous['subjects'])
    auc_prior=core.paired_auc_interval(y,previous['probabilities'],p,s,repeats=400)
    baseline_decisions=oof(y,old['probabilities'],s);new_decisions=oof(y,p,s)
    people,inverse=np.unique(s,return_inverse=True);rng=np.random.default_rng(20261104);diffs=[]
    for _ in range(400):
        frequencies=np.bincount(rng.integers(len(people),size=len(people)),minlength=len(people));w=frequencies[inverse]
        diffs.append(float(f1_score(y,new_decisions,average='macro',sample_weight=w,zero_division=0)-f1_score(y,baseline_decisions,average='macro',sample_weight=w,zero_division=0)))
    f1_delta=float(f1_score(y,new_decisions,average='macro',zero_division=0)-f1_score(y,baseline_decisions,average='macro',zero_division=0))
    gate={**cal,'checkpoint':str(path),'checkpoint_sha256':sha(path),'class_ids':list(range(1,11)),
          'gate_fit_source':'Reused official validation; final gates fit all validation; OOF fits only respective patient-training folds',
          'decision_rule':'Probability >= Gate after same-Study mean probabilities','candidate_only':True}
    core.save_json(ROOT/'candidate_gates.json',gate)
    assert all(sha(source)==digest for source,digest in protected.items())
    record=dict(state='passed',checkpoint=str(path),checkpoint_sha256=sha(path),stage=name,epoch=epoch,
                metrics_reproduced=True,probabilities_reproduced=True,max_abs_probability_difference=delta,
                probabilities_bit_equal=bool(np.array_equal(p,saved['probabilities'])),validation=measured,calibration=cal,
                paired_auc_vs_previous_ensemble=auc_prior,
                paired_coarse_oof_f1_vs_original=dict(delta=f1_delta,ci95=np.quantile(diffs,[.025,.975]).tolist(),repeats=400,
                  scope='Conditional on fixed cross-fitted decisions and selected checkpoints; thresholds not refitted in bootstrap'),
                protected_hashes_after=protected,no_test_images_used=True,original_submission_unchanged=True)
    core.save_json(ROOT/'candidate_verification.json',record)
    print(json.dumps({'state':'passed','stage':name,'epoch':epoch,'auc':measured['study']['macro_auc'],'oof_f1':cal['oof_macro_f1'],'max_probability_diff':delta,'paired_oof_f1_delta':f1_delta},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
