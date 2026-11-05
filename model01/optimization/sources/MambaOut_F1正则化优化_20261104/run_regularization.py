"""Controlled, local-only MambaOut regularization and rare-positive loss pilots."""
import gc, hashlib, json, math, os, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'MambaOut_ASL对照'))
import run_optimization as old
core,ref=old.core,old.reference
BASE=ROOT/'preserved_champion/best_model.pth'
EXPECTED='9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585'
MAX_EPOCHS=2
RECIPES=[
 dict(name='A_current_regularization',drop_rate=.2,drop_path_rate=.1,loss='BCE'),
 dict(name='B_stronger_regularization',drop_rate=.35,drop_path_rate=.2,loss='BCE'),
 dict(name='C_stronger_regularization_rare_weight',drop_rate=.35,drop_path_rate=.2,loss='rare_weighted_BCE')]

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):core.save_json(Path(p),d)
def status(**d):save(ROOT/'status.json',dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),**d))
def pair(m):return (m['study']['macro_auc'],m['study']['macro_f1'])
def from_npz(p):
 d=np.load(p,allow_pickle=False)
 return (d['targets'],d['probabilities'],d['subjects'])
def calibration(pred):return ref.evaluate_gates(pred)
def extra_metrics(pred):
 from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
 y,p,s=pred;z=p>=.5
 old_gate=np.asarray(json.loads((ROOT/'preserved_champion/gates.json').read_text(encoding='utf-8'))['final_gates_by_class_id'])
 frozen=p>=old_gate
 return dict(frozen_previous_gate_macro_f1=float(f1_score(y,frozen,average='macro',zero_division=0)),
  frozen_previous_gate_f1_by_class=f1_score(y,frozen,average=None,zero_division=0).tolist(),positive_studies=y.sum(0).astype(int).tolist(),predicted_positive_studies=z.sum(0).astype(int).tolist(),
  tp=np.sum(z&(y==1),0).astype(int).tolist(),fp=np.sum(z&(y==0),0).astype(int).tolist(),fn=np.sum((~z)&(y==1),0).astype(int).tolist(),
  average_precision_by_class=average_precision_score(y,p,average=None).tolist(),macro_ap=float(average_precision_score(y,p,average='macro')),
  macro_precision=float(precision_score(y,z,average='macro',zero_division=0)),macro_recall=float(recall_score(y,z,average='macro',zero_division=0)),
  zero_f1_classes=(np.flatnonzero(f1_score(y,z,average=None,zero_division=0)==0)+1).tolist())
def save_pred(p,pred):np.savez_compressed(p,targets=pred[0],probabilities=pred[1],subjects=pred[2])

def main():
 if not torch.cuda.is_available() or ROOT.drive.upper()!='E:':raise RuntimeError('E-drive CUDA required')
 if sha(BASE)!=EXPECTED:raise ValueError('Preserved champion changed')
 initial=json.loads((ROOT/'preserved_champion/results.json').read_text(encoding='utf-8'))
 expected=initial['winner_validation']
 old_gates=initial['threshold_calibration']
 cfg=old.yaml.safe_load((ref.SMOKE/'data_local.yaml').read_text(encoding='utf-8'))
 manifest_hash=sha(cfg['manifest'])
 train=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'train',ref.build_transform(True,224),cfg['cache_dir'])
 val=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'val',ref.build_transform(False,224))
 if len(train)!=30344 or len(val)!=1422:raise ValueError('Cohort changed')
 if {r['subject_id'] for r in train.rows}&{r['subject_id'] for r in val.rows}:raise ValueError('Patient leakage')
 positives=np.zeros(10,dtype=np.int64)
 for labels in train.labels:positives[list(labels)]+=1
 prevalence=positives/len(train)
 # Weight only classes whose training image prevalence is under 10%; derive entirely from training labels.
 pos_weight=np.where(prevalence<.10,np.minimum(np.sqrt((len(train)-positives)/positives),3.0),1.0).astype(np.float32)
 settings_common=dict(architecture='mambaout_tiny',img_size=224,batch_size=12,seed=42,train_images=30344,
  backbone_lr=1e-6,classifier_lr=1e-5,weight_decay=.05,manifest_sha256=manifest_hash,
  initial_checkpoint=str(BASE),initial_checkpoint_sha256=EXPECTED,max_epochs=MAX_EPOCHS,
  lr_schedule='constant first epoch, half rate second epoch',loss_weight_rule='prevalence<0.10: min(sqrt(negative/positive),3), else 1; train images only')
 plan=dict(common=settings_common,recipes=RECIPES,positive_counts_train_images=positives.tolist(),pos_weight=pos_weight.tolist(),
  selection='Study macro AUC primary, F1@0.5 secondary; preserve initial checkpoint; report OOF F1 and class outcomes separately',
  stop='Maximum 2 complete epochs per trial; stop that trial if AUC below initial by more than 0.01 or first epoch not improved by second',
  limitations='Single seed and reused official development validation; joint Dropout/DropPath intervention; no test data used')
 if (ROOT/'experiment_plan.json').exists():
  if json.loads((ROOT/'experiment_plan.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Plan changed')
 else:save(ROOT/'experiment_plan.json',plan)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 original=torch.load(BASE,map_location='cpu',weights_only=True)
 if original['training']['img_size']!=224 or original['training']['best_epoch']!=1:raise ValueError('Initial metadata mismatch')
 if original['training']['manifest_sha256']!=manifest_hash:raise ValueError('Training manifest bytes differ from champion')
 weights=original['state_dict'];del original
 baseline_model=core.build_model(dict(drop_rate=.2,drop_path_rate=.1),weights).eval()
 baseline_metrics,baseline_pred=ref.validate(baseline_model,val,keep_predictions=True)
 for level in ['image','study']:
  for k in ['auc','f1','macro_auc','macro_f1']:
   if not np.allclose(baseline_metrics[level][k],expected[level][k],atol=1e-8,rtol=0):raise ValueError('Initial score does not reproduce')
 save_pred(ROOT/'baseline_predictions.npz',baseline_pred)
 save(ROOT/'baseline_metrics.json',dict(validation=baseline_metrics,calibration=old_gates,extra=extra_metrics(baseline_pred)))
 del baseline_model;gc.collect();torch.cuda.empty_cache()
 outcomes={}
 for recipe in RECIPES:
  folder=ROOT/recipe['name'];folder.mkdir(exist_ok=True)
  settings={**settings_common,**recipe,'pos_weight':pos_weight.tolist() if recipe['loss']!='BCE' else [1.0]*10}
  if (folder/'results.json').is_file():
   outcomes[recipe['name']]=json.loads((folder/'results.json').read_text(encoding='utf-8'));continue
  model=core.build_model(recipe,weights)
  dropout=[float(m.p) for m in model.modules() if isinstance(m,torch.nn.Dropout)]
  paths=[float(m.drop_prob) for m in model.modules() if hasattr(m,'drop_prob') and type(m).__name__=='DropPath']
  if not any(abs(p-recipe['drop_rate'])<1e-7 for p in dropout) or not paths or abs(max(paths)-recipe['drop_path_rate'])>1e-6:
   raise ValueError('Requested dropout or DropPath not actually instantiated')
  save(folder/'regularization_modules.json',dict(dropout_rates=dropout,drop_path_rates=paths,eval_disables_stochastic_layers=True))
  optimizer=core.make_optimizer(model,settings);scaler=torch.amp.GradScaler('cuda')
  loss_weight=torch.tensor(settings['pos_weight'],device='cuda',dtype=torch.float32)
  history=[];best_metrics=baseline_metrics;best_epoch=0;best_gates=old_gates;best_pred=baseline_pred
  last=folder/'last_training.pth';best=folder/'best.pth'
  if last.exists():
   s=torch.load(last,map_location='cpu',weights_only=True)
   if s['settings']!=settings:raise ValueError('Resume settings mismatch')
   model.load_state_dict(s['state_dict'],strict=True);optimizer.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler'])
   history=s['history'];best_epoch=s['best_epoch'];best_metrics=s['best_metrics'];best_gates=s['best_gates']
   best_pred=from_npz(folder/'best_predictions.npz')
   check=torch.load(best,map_location='cpu',weights_only=True,mmap=True)
   if check['training']['best_epoch']!=best_epoch:raise ValueError('Interrupted checkpoint commit; recover before resuming')
   from sklearn.metrics import roc_auc_score,f1_score
   pred_score=(float(roc_auc_score(best_pred[0],best_pred[1],average='macro')),float(f1_score(best_pred[0],best_pred[1]>=.5,average='macro',zero_division=0)))
   if not np.allclose(pred_score,pair(best_metrics),atol=1e-8,rtol=0):raise ValueError('Saved prediction and checkpoint metadata mismatch')
   del s,check
  else:
   core.save_best(best,weights,settings,0);save_pred(folder/'best_predictions.npz',baseline_pred)
  for epoch in range(len(history)+1,MAX_EPOCHS+1):
   if history and history[-1]['validation']['study']['macro_auc']<pair(baseline_metrics)[0]-.01:break
   core.seed(42+epoch);model.train()
   if not all(m.training for m in model.modules() if isinstance(m,torch.nn.Dropout) or type(m).__name__=='DropPath'):raise ValueError('Random regularization disabled during training')
   factor=1.0 if epoch==1 else .5
   optimizer.param_groups[0]['lr']=settings['backbone_lr']*factor
   optimizer.param_groups[1]['lr']=settings['classifier_lr']*factor
   loader=DataLoader(train,batch_size=12,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(42+epoch))
   seen=0;total=0.;started=time.perf_counter()
   for step,batch in enumerate(loader,1):
    optimizer.zero_grad(set_to_none=True)
    x=batch['image'].to('cuda',non_blocking=True,memory_format=torch.channels_last);y=batch['target'].to('cuda',non_blocking=True)
    with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y,pos_weight=loss_weight)
    value=loss.item()
    if not math.isfinite(value):raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    seen+=len(x);total+=value*len(x)
    if step==1 or step%250==0:
     status(state='training',stage=recipe['name'],epoch=epoch,images=seen,total_images=len(train),completed_epochs={k:v['epochs_completed'] for k,v in outcomes.items()})
     print(f"TRAIN {recipe['name']} epoch={epoch} images={seen}/{len(train)} loss={total/seen:.6f}",flush=True)
   torch.cuda.synchronize()
   if seen!=30344:raise ValueError('Incomplete epoch')
   train_seconds=time.perf_counter()-started
   status(state='validation',stage=recipe['name'],epoch=epoch)
   start=time.perf_counter();metrics,pred=ref.validate(model,val,keep_predictions=True);eval_seconds=time.perf_counter()-start
   gates=calibration(pred);extra=extra_metrics(pred)
   record=dict(epoch=epoch,images=seen,train_loss=total/seen,train_seconds=train_seconds,validation_seconds=eval_seconds,
    learning_rates=[g['lr'] for g in optimizer.param_groups],validation=metrics,calibration=gates,extra=extra)
   save_pred(folder/f'epoch_{epoch}_predictions.npz',pred)
   improved=pair(metrics)>pair(best_metrics)
   if improved:best_epoch=epoch;best_metrics=metrics;best_gates=gates;best_pred=pred
   history.append(record)
   current={k:v.detach().cpu() for k,v in model.state_dict().items()}
   core.save_checkpoint(last,dict(state_dict=current,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),settings=settings,
    history=history,best_epoch=best_epoch,best_metrics=best_metrics,best_gates=best_gates))
   core.save_best(folder/f'epoch_{epoch}.pth',current,settings,epoch)
   if improved:core.save_best(best,current,settings,epoch);save_pred(folder/'best_predictions.npz',pred)
   save(folder/'history.json',history)
   print(f"EPOCH_DONE {recipe['name']} epoch={epoch} AUC={pair(metrics)[0]:.10f} F1={pair(metrics)[1]:.10f} OOF_F1={gates['oof_macro_f1']:.10f} zero_classes={extra['zero_f1_classes']} best_epoch={best_epoch}",flush=True)
   del current
   if (ROOT/'pause_after_epoch.flag').exists():status(state='paused_after_epoch',stage=recipe['name'],epoch=epoch);return
  outcome=dict(settings=settings,epochs_completed=len(history),best_epoch=best_epoch,best_validation=best_metrics,
   threshold_calibration=best_gates,best_checkpoint=str(best),best_extra=extra_metrics(best_pred),
   stop_reason='auc_drop_guard' if len(history)<MAX_EPOCHS else 'pilot_epoch_budget')
  outcomes[recipe['name']]=outcome;save(folder/'results.json',outcome)
  del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
 winner=max(outcomes,key=lambda k:pair(outcomes[k]['best_validation']))
 chosen=outcomes[winner];improved=pair(chosen['best_validation'])>pair(baseline_metrics)
 selected=Path(chosen['best_checkpoint']) if improved else BASE
 state=torch.load(selected,map_location='cpu',weights_only=True)
 settings=state['training'];model=core.build_model(settings,state['state_dict']).eval();del state
 status(state='final_verification',stage=winner)
 final_metrics,final_pred=ref.validate(model,val,keep_predictions=True)
 target=chosen['best_validation'] if improved else baseline_metrics
 for level in ['image','study']:
  for k in ['auc','f1','macro_auc','macro_f1']:
   if not np.allclose(final_metrics[level][k],target[level][k],rtol=0,atol=1e-8):raise ValueError('Saved selected weights do not reproduce')
 if not np.array_equal(baseline_pred[0],final_pred[0]) or not np.array_equal(baseline_pred[2],final_pred[2]):raise ValueError('Paired evaluation ordering changed')
 interval=core.paired_auc_interval(baseline_pred[0],baseline_pred[1],final_pred[1],baseline_pred[2],repeats=400)
 if sha(BASE)!=EXPECTED or sha(ROOT.parent/'MambaOut_ASL对照/best_model.pth')!=EXPECTED:raise ValueError('Protected champion changed')
 for item in json.loads((ROOT/'preserved_champion/preservation_receipt.json').read_text(encoding='utf-8-sig')):
  if sha(item['source'])!=item['sha256'] or sha(item['backup'])!=item['sha256']:raise ValueError('A protected artifact changed')
 import shutil
 candidate=ROOT/'selected_candidate.pth';shutil.copyfile(selected,candidate)
 if sha(candidate)!=sha(selected):raise ValueError('Candidate copy mismatch')
 final_gates=calibration(final_pred)
 save(ROOT/'selected_gates.json',final_gates)
 result=dict(initial_validation=baseline_metrics,initial_calibration=old_gates,trials=outcomes,
  winner=winner if improved else 'preserved_champion',improved_on_development_auc=pair(final_metrics)[0]>pair(baseline_metrics)[0],
  final_validation=final_metrics,final_extra=extra_metrics(final_pred),final_calibration=final_gates,
  paired_auc_interval=interval,selected_checkpoint=str(candidate),selected_sha256=sha(candidate),
  preserved_checkpoint=str(BASE),preserved_sha256=EXPECTED,completed_training_epochs=sum(v['epochs_completed'] for v in outcomes.values()),
  training_image_visits=sum(v['epochs_completed'] for v in outcomes.values())*30344,
  scope='Official validation reused for selection; threshold OOF is internal; bootstrap conditional on chosen fixed models; test dataset not used')
 save(ROOT/'results.json',result);save_pred(ROOT/'selected_predictions.npz',final_pred)
 status(state='complete',winner=result['winner'],epochs=result['completed_training_epochs'],auc=pair(final_metrics)[0],f1=pair(final_metrics)[1])
 print('COMPLETE '+json.dumps(dict(winner=result['winner'],epochs=result['completed_training_epochs'],auc=pair(final_metrics)[0],f1=pair(final_metrics)[1],oof_f1=final_gates['oof_macro_f1'],ci=interval),ensure_ascii=False),flush=True)

if __name__=='__main__':
 import msvcrt,ctypes
 with (ROOT/'training.lock').open('a+b') as lock:
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
  try:main()
  except Exception as e:
   status(state='failed',error=f'{type(e).__name__}: {e}')
   raise
  finally:
   if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
