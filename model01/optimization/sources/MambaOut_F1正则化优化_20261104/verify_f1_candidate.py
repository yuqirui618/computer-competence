import gc,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_regularization as run
stage=ROOT/'C_stronger_regularization_rare_weight'
p=stage/'epoch_2.pth'
if json.loads((ROOT/'status.json').read_text(encoding='utf-8'))['state']!='complete':raise RuntimeError('Training must finish first')
history=json.loads((stage/'history.json').read_text(encoding='utf-8'))
expected=history[1]
state=torch.load(p,map_location='cpu',weights_only=True)
s=state['training']
assert s['name']=='C_stronger_regularization_rare_weight' and s['best_epoch']==2 and s['pos_weight']==[3.,3.,3.,3.,1.,1.,1.,1.,1.,1.]
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
model=run.core.build_model(s,state['state_dict']).eval();del state
cfg=run.old.yaml.safe_load((run.ref.SMOKE/'data_local.yaml').read_text(encoding='utf-8'))
val=run.ref.CXRDataset(cfg['manifest'],cfg['data_root'],'val',run.ref.build_transform(False,224))
metrics,pred=run.ref.validate(model,val,keep_predictions=True)
for level in ['image','study']:
 for k in ['auc','f1','macro_auc','macro_f1']:
  if not np.allclose(metrics[level][k],expected['validation'][level][k],rtol=0,atol=1e-8):raise ValueError('Reloaded F1 candidate metrics mismatch')
saved=np.load(stage/'epoch_2_predictions.npz',allow_pickle=False)
for key,value in zip(['targets','probabilities','subjects'],pred):
 if not np.array_equal(saved[key],value):raise ValueError('Saved prediction mismatch after reloading weight')
gates=run.calibration(pred)
if gates!=expected['calibration']:raise ValueError('Candidate OOF threshold results mismatch')
extra=run.extra_metrics(pred)
assert extra==expected['extra']
result=dict(verified=True,checkpoint=str(p),sha256=run.sha(p),epoch=2,validation=metrics,extra=extra,calibration=gates,
 exact_prediction_reproduction=True,training_updates=0,scope='Official development validation, not independent test')
run.save(ROOT/'f1_candidate_verification.json',result)
run.save(ROOT/'f1_candidate_gates.json',gates)
print(json.dumps({'verified':True,'auc':metrics['study']['macro_auc'],'f1':metrics['study']['macro_f1'],'oof_f1':gates['oof_macro_f1'],'sha256':result['sha256']},ensure_ascii=False),flush=True)
del model;gc.collect();torch.cuda.empty_cache()
