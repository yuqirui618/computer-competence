"""Fresh ImageNet384 controlled augmentation -> ASL -> S36 trials, E drive only."""
from __future__ import annotations
import argparse
def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('mode',choices=('preflight','profile','run'))
 return p.parse_args()
if __name__=='__main__':ARGS=arguments()
import copy,ctypes,gc,hashlib,math,msvcrt,os,time,traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from experiment_common import ROOT,DATA,OLD,GLOBAL_LOCK,CONFIG,COMMON,RAW,MANIFEST,INCUMBENT,INCUMBENT_SHA,INCUMBENT_PRED,INCUMBENT_PRED_SHA,INCUMBENT_AUC,sha,read,save,freeze,save_npz,status,sources,check_protected,load_inputs
import numpy as np
import torch,timm
from torch.nn import functional as F
from data_pipeline import RawStudyDataset,collate_studies
from augmentation import AugmentedStudyDataset
from asl_study import ASLStudyLoss
from models import build_model,optimizer_for,source_info
from training_utils import seed,checkpoint,copy_checkpoint,calibration,metrics,loader,worst_batches,predict
STEPS=math.ceil(18513/32)
ARMS=('A0_BCE_EMA','A1_noise','A2_photo','A3_photo_noise','B_ASL','C_S36')
GROUPS=('stem.','stages.0.','stages.1.','stages.2.','stages.3.','head.norm.','head.fc.fc1.','head.fc.norm.','head.fc.fc2.')


def note(event,**fields):
    value=dict(time=datetime.now().astimezone().isoformat(),event=event,**fields)
    line=__import__('json').dumps(value,ensure_ascii=False,allow_nan=False)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)


def setup():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')


def lr_factor(u):
    if u<STEPS:return .1+.9*u/max(1,STEPS-1)
    t=(u-STEPS)/max(1,4*STEPS-1);return .1+.9*(1+math.cos(math.pi*min(1.,t)))/2


def log_probabilities(logits,lengths):
    if sum(lengths)!=len(logits) or not lengths or min(lengths)<1:raise ValueError('Invalid Study bags')
    lp=[];lq=[]
    for z in logits.float().split(lengths):
        lp.append(torch.logsumexp(F.logsigmoid(z),0)-math.log(len(z)))
        lq.append(torch.logsumexp(F.logsigmoid(-z),0)-math.log(len(z)))
    return torch.stack(lp),torch.stack(lq)


class ShadowEMA:
    def __init__(self,model):
        self.model=copy.deepcopy(model).eval().requires_grad_(False);self.model.set_grad_checkpointing(False);self.updates=0;self.last_decay=None
    @torch.no_grad()
    def update(self,model):
        self.updates+=1;beta=min(.999,(1+self.updates)/(10+self.updates));self.last_decay=beta
        source_state=model.state_dict()
        for key,e in self.model.state_dict().items():
            value=source_state[key].detach()
            if e.is_floating_point():e.mul_(beta).add_(value,alpha=1-beta)
            else:e.copy_(value)
    def load(self,value):
        self.model.load_state_dict(value['state_dict'],strict=True);self.updates=value['updates'];self.last_decay=value['last_decay']
    def state(self):return dict(state_dict={k:v.detach().cpu() for k,v in self.model.state_dict().items()},updates=self.updates,last_decay=self.last_decay)


def cpu_math(audit):
    gen=torch.Generator().manual_seed(42);z=torch.randn(7,10,generator=gen,requires_grad=True);lengths=[2,1,3,1];y=torch.randint(0,2,(4,10),generator=gen).float()
    direct=F.binary_cross_entropy(torch.stack([v.sigmoid().mean(0) for v in z.split(lengths)]),y,reduction='none').mean(1)
    stable=losses_for('A_BCE_EMA',z,lengths,y,None);error=float((direct-stable).abs().max())
    if error>1e-6:raise ValueError('BCE mathematical mismatch')
    stable.mean().backward()
    if not torch.isfinite(z.grad).all():raise ValueError('Nonfinite stable BCE gradients')
    # EMA update is exactly the declared recurrence and never changes the raw model.
    model=torch.nn.Linear(2,1);model.set_grad_checkpointing=lambda enabled:None
    ema=ShadowEMA(model);original={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():p.add_(.5)
    expected={k:v.clone()*(2/11)+model.state_dict()[k].detach()*(9/11) for k,v in original.items()}
    raw={k:v.detach().clone() for k,v in model.state_dict().items()};ema.update(model)
    for k in expected:
        torch.testing.assert_close(ema.model.state_dict()[k],expected[k],rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(model.state_dict()[k],raw[k],rtol=0,atol=0)
    return dict(BCE_mean_probability_max_difference=error,EMA_first_update_decay=ema.last_decay,EMA_raw_model_unchanged=True,Study_count=18513)


def key(evaluation):return evaluation['metrics']['macro_auc'],evaluation['metrics']['macro_f1']


def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}


def best_evaluations(history):
    best={}
    for row in history:
        for name,e in row['evaluations'].items():
            if name not in best or key(e)>key(best[name]):best[name]=e
    return best


def training_epoch(arm,model,dataset,opt,scaler,ema,criterion,epoch,profile,initial,completed):
    seed(42+epoch);model.train();model.set_grad_checkpointing(True);batch=profile['real_batch_size'];seen=[];image_count=0;loss_sum=0.;micro=0;attempts=0;successful=0;skips=0;window=0;counts=[];factors=[];order=hashlib.sha256();start=time.perf_counter()
    opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    for image,lengths,y,index in loader(dataset,batch,True,epoch,collate_fn=collate_studies):
        if window==0:
            denominator=min(32,len(dataset)-len(seen));factor=lr_factor((epoch-1)*STEPS+attempts);factors.append(factor)
            for g in opt.param_groups:g['lr']=g['base_lr']*factor
        with torch.autocast('cuda',dtype=torch.float16):logits=model(image.to('cuda',memory_format=torch.channels_last,non_blocking=True))
        values=losses_for(arm,logits,lengths,y.to('cuda',non_blocking=True),criterion);scaler.scale(values.sum()/denominator).backward()
        n=len(y);window+=n;loss_sum+=float(values.detach().sum());seen.extend(index.tolist());image_count+=sum(lengths);micro+=1
        for i in index.tolist():order.update((str(dataset.subjects[i])+'|'+str(dataset.study_ids[i])+'\n').encode('utf-8'))
        if window==denominator:
            old=scaler.get_scale();scaler.step(opt);scaler.update();did_step=scaler.get_scale()>=old;attempts+=1;successful+=int(did_step);skips+=int(not did_step)
            if did_step and ema:ema.update(model)
            if scaler.get_scale()<1e-12:raise FloatingPointError('GradScaler collapsed')
            counts.append(window);window=0;opt.zero_grad(set_to_none=True)
            if attempts%10==0 or len(seen)==len(dataset):
                fields=dict(state='training',arm=arm,epoch=epoch,source_total_epoch=epoch,epochs_per_arm=5,arm_completed_epochs=epoch-1,completed_new_epochs=completed,studies=len(seen),total_studies=18513,images=image_count,total_images=30344,optimizer_attempts=attempts,optimizer_successful=successful,AMP_skips=skips,train_loss=loss_sum/len(seen),epoch_elapsed_seconds=time.perf_counter()-start,EMA_updates=ema.updates if ema else 0)
                status(**fields)
                if attempts%50==0 or len(seen)==len(dataset):note('TRAIN_PROGRESS',**{k:fields[k] for k in ('arm','epoch','studies','images','optimizer_attempts','train_loss','AMP_skips')})
        elif window>denominator:raise ValueError('Study window overflow')
    torch.cuda.synchronize()
    if len(seen)!=18513 or len(set(seen))!=18513 or sorted(seen)!=list(range(18513)) or image_count!=30344 or attempts!=579 or counts!=[32]*578+[17] or window or micro!=math.ceil(18513/batch):raise ValueError('Incomplete/duplicate train epoch or wrong Study normalization')
    return dict(epoch=epoch,source_total_epoch=epoch,train_loss=loss_sum/18513,train_seconds=time.perf_counter()-start,train_studies=len(seen),train_images=image_count,real_batch_size=batch,accumulation_steps=32//batch,effective_study_batch_size=32,micro_batches=micro,optimizer_steps_attempted=attempts,optimizer_steps_completed=successful,AMP_skips=skips,scaler_final=float(scaler.get_scale()),update_Study_counts=counts,lr_factors=factors,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,dataloader_seed=42+epoch,dataloader_order_sha256=order.hexdigest(),full_group_updates=group_updates(model,initial,arm),EMA_successful_updates=ema.updates if ema else 0)

def losses_for(arm,logits,lengths,y,criterion):
 lp,lq=log_probabilities(logits,lengths)
 values=criterion(lp,lq,y.float()) if criterion is not None else -(y.float()*lp+(1-y.float())*lq).mean(1)
 if values.shape!=(len(y),) or not torch.isfinite(values).all():raise FloatingPointError('Invalid Study loss')
 return values
def group_updates(model,initial,arm):
 state=model.state_dict();out={}
 for prefix in GROUPS:
  keys=[k for k in state if k.startswith(prefix)]
  if not keys:raise ValueError('Missing full model group '+prefix)
  changed=sum(not torch.equal(state[k].detach().cpu(),initial[k]) for k in keys)
  if changed==0:raise ValueError('No actual training update in '+prefix)
  out[prefix]=dict(tensors=len(keys),changed_tensors=changed)
 return out
def completed():
 return sum(len(read(ROOT/a/'history.json')) for a in ARMS if (ROOT/a/'history.json').exists())
def spec(arm,policy='none',loss='bce'):
 return dict(arm=arm,architecture='caformer_s36' if arm=='C_S36' else 'caformer_s18',
             augmentation=policy,loss=loss,seed=42,registered_max_epochs=5)
def source_bundle():
 return {a:source_info(a) for a in ('caformer_s18','caformer_s36')}
def check_runtime(src,guard):
 if sources()!=src or check_protected()!=guard:raise ValueError('Frozen source or historical protection changed')
 for a,m in source_bundle().items():
  if sha(m['weights'])!=m['sha256']:raise ValueError('ImageNet weights changed: '+a)
def cpu_preflight():
 setup();cfg,records,audit=load_inputs();src=sources();guard=check_protected();models_meta=source_bundle()
 for a,m in models_meta.items():
  if sha(m['weights'])!=m['sha256']:raise ValueError('Wrong pretrained SHA '+a)
 receipts={name:read(ROOT/name) for name in ('augmentation_checks.json','asl_checks.json','model_checks.json')}
 aug=receipts['augmentation_checks.json'];asl=receipts['asl_checks.json'];mod=receipts['model_checks.json']
 if aug['state']!='passed' or not all(aug['checks'].values()) or aug['source_sha256'][str(ROOT/'augmentation.py')]!=sha(ROOT/'augmentation.py'):raise ValueError('Augmentation CPU receipt/source mismatch')
 if asl['status']!='passed' or asl['device']!='cpu' or asl['configuration']!=CONFIG['ASL'] or asl['module_sha256']!=sha(ROOT/'asl_study.py') or asl['passed_checks']!=11 or len(asl['checks'])!=11 or not all(x['passed'] is True for x in asl['checks'].values()):raise ValueError('ASL CPU receipt/source mismatch')
 if mod['state']!='passed' or mod['source_file_sha256']!=sha(ROOT/'models.py'):raise ValueError('Model CPU receipt/source mismatch')
 checks=cpu_math(audit)
 model=build_model('caformer_s18',audit['train_priors'],'cpu')
 opt=optimizer_for(model)
 covered=[id(p) for g in opt.param_groups for p in g['params']]
 if sorted(covered)!=sorted(id(p) for p in model.parameters()):raise ValueError('Optimizer does not cover all trainable tensors exactly once')
 if any('base_lr' not in g for g in opt.param_groups):raise ValueError('Optimizer base LR missing')
 train=AugmentedStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True,policy='none',seed=42)
 val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
 if (len(train),train.images,len(val),val.images)!=(18513,30344,859,1422):raise ValueError('Dataset coverage mismatch')
 if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
 save(ROOT/'cpu_preflight.json',dict(state='passed',config=CONFIG,data=audit,math_checks=checks,source_hashes=src,protection=guard,pretrained=models_meta,CPU_only=True))
 note('CPU_PREFLIGHT_PASSED',train_studies=len(train),train_images=train.images,val_studies=len(val))
 del model,opt;gc.collect()
 return cfg,records,audit,src,guard
def resource_profile():
 cfg,records,audit,src,guard=cpu_preflight()
 train=AugmentedStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True,policy='photo_noise',seed=42)
 attempts=[];chosen=None
 for batch in CONFIG['common_real_batch_candidates']:
  worst,table=worst_batches(train,batch);all_ok=True;per_arch={}
  for arch in ('caformer_s18','caformer_s36'):
   model=opt=scaler=ema=images=y=logits=values=None
   try:
    train.set_epoch(worst['epoch']);seed(42)
    model=build_model(arch,audit['train_priors'],'cuda')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=CONFIG['amp_init_scale']);ema=ShadowEMA(model)
    images,lengths,y,index=collate_studies([train[i] for i in worst['indices']])
    images=images.to('cuda',memory_format=torch.channels_last);y=y.to('cuda')
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
     opt.zero_grad(set_to_none=True);model.train()
     for micro in range(32//batch):
      with torch.autocast('cuda',dtype=torch.float16):logits=model(images)
      values=losses_for('resource',logits,lengths,y,None)
      scaler.scale(values.sum()/32).backward()
     scaler.unscale_(opt)
     invalid=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
     if invalid:raise FloatingPointError('Missing/nonfinite GPU smoke gradients '+str(invalid[:3]))
     old=scaler.get_scale();scaler.step(opt);scaler.update()
     if scaler.get_scale()<old:raise FloatingPointError('Smoke optimizer step skipped')
     ema.update(model)
    changed=group_updates(model,initial,'resource')
    peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
    total=torch.cuda.get_device_properties(0).total_memory/2**20
    ok=max(peak,reserved)<=total*CONFIG['memory_margin'];all_ok=all_ok and ok
    item=dict(architecture=arch,real_batch=batch,peak_allocated_mib=peak,peak_reserved_mib=reserved,total_mib=total,
              state='passed' if ok else 'insufficient_margin',full32_window_steps=2,EMA_updates=ema.updates,all_body_group_updates=changed)
    attempts.append(item);per_arch[arch]=item
    note('RESOURCE_SMOKE',architecture=arch,real_batch=batch,state=item['state'],peak_mib=peak,reserved_mib=reserved)
   except torch.OutOfMemoryError as exc:
    all_ok=False;attempts.append(dict(architecture=arch,real_batch=batch,state='OOM',error=str(exc)));note('RESOURCE_OOM',architecture=arch,real_batch=batch)
   finally:
    del model,opt,scaler,ema,images,y,logits,values;gc.collect();torch.cuda.empty_cache()
   if not all_ok:break
  if all_ok:
   chosen=dict(real_batch_size=batch,accumulation_steps=32//batch,effective_study_batch_size=32,worst_batch=worst,profiles=per_arch)
   break
 if chosen is None:raise RuntimeError('No safe shared real batch for S18 and S36')
 # A full incumbent reload proves identical evaluation and Study aggregation.
 val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
 seed(42);m=timm.create_model('caformer_s18',pretrained=False,num_classes=10,drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
 m.head.fc.head_drop.p=.2
 stored=torch.load(INCUMBENT,map_location='cpu',weights_only=True)
 if stored['epoch']!=1 or stored['variant']!='ema' or stored['arm']!='A_BCE_EMA':raise ValueError('Wrong incumbent identity')
 m.load_state_dict(stored['state_dict'],strict=True);m.to(device='cuda',memory_format=torch.channels_last)
 p,extra=predict(m,val,cfg,True)
 with np.load(INCUMBENT_PRED,allow_pickle=False) as d:
  for name,x in (('targets',val.targets),('subjects',val.subjects),('study_ids',val.study_ids),('image_paths',extra['image_paths'])):np.testing.assert_array_equal(d[name],x)
  delta=float(np.max(np.abs(p-d['probabilities'])));idelta=float(np.max(np.abs(extra['image_probabilities']-d['image_probabilities'])))
  np.testing.assert_allclose(p,d['probabilities'],rtol=0,atol=1e-7);np.testing.assert_allclose(extra['image_probabilities'],d['image_probabilities'],rtol=0,atol=1e-7)
 measured=metrics(val.targets,p)
 if measured['macro_auc']!=INCUMBENT_AUC:raise ValueError('Incumbent validation AUC does not reproduce')
 del m,stored;gc.collect();torch.cuda.empty_cache()
 check_runtime(src,guard)
 registration=dict(config=CONFIG,source_hashes=src,protection=guard,pretrained=source_bundle(),
                   incumbent_checkpoint_sha256=INCUMBENT_SHA,incumbent_prediction_sha256=INCUMBENT_PRED_SHA,incumbent_AUC=INCUMBENT_AUC)
 freeze(ROOT/'registration.json',registration)
 freeze(ROOT/'resource_profile.json',dict(state='passed',source_hashes=src,protection=guard,attempts=attempts,
   incumbent_reloaded=dict(study_max_abs_difference=delta,image_max_abs_difference=idelta,metrics=measured),**chosen))
 status(state='GPU_resource_preflight_passed',completed_new_epochs=0,real_batch_size=chosen['real_batch_size'])
 note('RESOURCE_READY',real_batch=chosen['real_batch_size'],accumulation=chosen['accumulation_steps'],incumbent_AUC=INCUMBENT_AUC)
def evaluate_checkpoint(arm,variant,epoch,model,val,cfg,folder,registration):
 began=time.perf_counter();p,extra=predict(model,val,cfg,True);measured=metrics(val.targets,p)
 pp=folder/'predictions'/f'{variant}_epoch_{epoch:02d}.npz';cp=folder/'checkpoints'/f'{variant}_epoch_{epoch:02d}.pth'
 if cp.exists() or pp.exists():raise ValueError('Uncommitted epoch artifact exists: '+str(cp))
 save_npz(pp,targets=val.targets,probabilities=p,subjects=val.subjects,study_ids=val.study_ids,**extra)
 checkpoint(cp,dict(state_dict=cpu_state(model),epoch=epoch,source_total_epoch=epoch,arm=arm,variant=variant,
   architecture=registration['spec']['architecture'],pretrained_sha256=registration['pretrained']['sha256'],
   initialization_kind='fresh ImageNet1k native384',full_backbone_training=True,registration=registration))
 return dict(epoch=epoch,variant=variant,metrics=measured,validation_seconds=time.perf_counter()-began,
             checkpoint=str(cp),checkpoint_sha256=sha(cp),prediction=str(pp),prediction_sha256=sha(pp))
def verify_best(arm,history,arm_spec,cfg,val,folder,src,guard,phase):
 best=best_evaluations(history);out={}
 for variant,e in best.items():
  alias=folder/f'best_{variant}.pth';copy_checkpoint(e['checkpoint'],alias)
  m=build_model(arm_spec['architecture'],TRAIN_PRIORS,'cuda')
  stored=torch.load(alias,map_location='cpu',weights_only=True)
  if stored['epoch']!=e['epoch'] or stored['arm']!=arm or stored['variant']!=variant or stored['architecture']!=arm_spec['architecture']:raise ValueError('Selected weight identity mismatch')
  m.load_state_dict(stored['state_dict'],strict=True)
  p,extra=predict(m,val,cfg,True)
  with np.load(e['prediction'],allow_pickle=False) as d:
   for name,x in (('targets',val.targets),('subjects',val.subjects),('study_ids',val.study_ids),('image_paths',extra['image_paths'])):np.testing.assert_array_equal(d[name],x)
   delta=float(np.max(np.abs(p-d['probabilities'])));idelta=float(np.max(np.abs(extra['image_probabilities']-d['image_probabilities'])))
   np.testing.assert_allclose(p,d['probabilities'],rtol=0,atol=1e-7);np.testing.assert_allclose(extra['image_probabilities'],d['image_probabilities'],rtol=0,atol=1e-7)
  if metrics(val.targets,p)!=e['metrics']:raise ValueError('Selected reloaded metric mismatch')
  cal=calibration(val.targets,p,val.subjects)
  gates=dict(**cal,class_ids=list(range(1,11)),arm=arm,architecture=arm_spec['architecture'],variant=variant,best_epoch=e['epoch'],
             checkpoint=str(alias),checkpoint_sha256=e['checkpoint_sha256'],gate_fit_source='full official validation; patient OOF F1 separate')
  save(folder/f'gates_{variant}.json',gates)
  out[variant]=dict(**e,best_alias=str(alias),calibration=cal,reloaded_max_abs_study_difference=delta,reloaded_max_abs_image_difference=idelta,all_1422_validation_images_rechecked=True)
  del m,stored;gc.collect();torch.cuda.empty_cache()
 selected=max(out,key=lambda x:key(out[x]))
 copy_checkpoint(out[selected]['checkpoint'],folder/'best.pth')
 gates=read(folder/f'gates_{selected}.json');gates['checkpoint']=str(folder/'best.pth');save(folder/'gates.json',gates)
 check_runtime(src,guard)
 result=dict(state=phase,arm=arm,spec=arm_spec,completed_new_epochs=len(history),variants=out,best_variant=selected,
             best_epoch=out[selected]['epoch'],best_AUC=out[selected]['metrics']['macro_auc'],
             source_hashes=src,protected_inputs=guard,incumbent_AUC=INCUMBENT_AUC,no_test_inference=True,no_GitHub_upload=True)
 save(folder/'results.json',result);note('ARM_PHASE_COMPLETE',arm=arm,phase=phase,completed_epochs=len(history),
                                      best_variant=selected,best_epoch=result['best_epoch'],best_AUC=result['best_AUC'])
 return result
def run_arm(arm_spec,until,cfg,records,audit,profile,src,guard):
 global TRAIN_PRIORS
 TRAIN_PRIORS=audit['train_priors']
 arm=arm_spec['arm'];folder=ROOT/arm;folder.mkdir(exist_ok=True)
 registration=dict(config=CONFIG,spec=arm_spec,source_hashes=src,protection=guard,pretrained=source_info(arm_spec['architecture']),
                   real_batch_size=profile['real_batch_size'],accumulation_steps=profile['accumulation_steps'])
 freeze(folder/'registration.json',registration)
 if (folder/'results.json').exists():
  result=read(folder/'results.json')
  if result['spec']!=arm_spec or result['source_hashes']!=src:raise ValueError('Completed arm provenance changed')
  if result['completed_new_epochs']>=until:
   for e in result['variants'].values():
    if sha(e['checkpoint'])!=e['checkpoint_sha256'] or sha(e['prediction'])!=e['prediction_sha256']:raise ValueError('Selected arm artifact changed')
   return result
 model=build_model(arm_spec['architecture'],audit['train_priors'],'cuda')
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 opt=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=CONFIG['amp_init_scale']);ema=ShadowEMA(model)
 criterion=ASLStudyLoss(**CONFIG['ASL']).to('cuda') if arm_spec['loss']=='asl' else None
 train=AugmentedStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True,policy=arm_spec['augmentation'],seed=42)
 val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
 history=[];last=folder/'last_training.pth'
 if last.exists():
  resume=torch.load(last,map_location='cpu',weights_only=True)
  if resume['registration']!=registration:raise ValueError('Resume registration mismatch')
  history=resume['history']
  if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>5:raise ValueError('Invalid committed epoch history')
  for row in history:
   for e in row['evaluations'].values():
    if sha(e['checkpoint'])!=e['checkpoint_sha256'] or sha(e['prediction'])!=e['prediction_sha256']:raise ValueError('Committed epoch SHA mismatch')
  model.load_state_dict(resume['state_dict'],strict=True);opt.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler']);ema.load(resume['ema'])
  del resume;save(folder/'history.json',history)
 elif (folder/'history.json').exists():raise ValueError('History without full atomic resume checkpoint')
 for epoch in range(len(history)+1,until+1):
  if (ROOT/'pause_after_epoch.flag').exists():
   status(state='paused_before_epoch',arm=arm,epoch=epoch,completed_new_epochs=completed());return None
  if sources()!=src:raise ValueError('Training source changed')
  if sha(registration['pretrained']['weights'])!=registration['pretrained']['sha256']:raise ValueError('Pretrained source changed')
  for variant in ('raw','ema'):
   if (folder/'checkpoints'/f'{variant}_epoch_{epoch:02d}.pth').exists() or (folder/'predictions'/f'{variant}_epoch_{epoch:02d}.npz').exists():raise ValueError('Archive interrupted uncommitted epoch artifacts before retry')
  train.set_epoch(epoch)
  row=training_epoch(arm,model,train,opt,scaler,ema,criterion,epoch,profile,initial,completed())
  status(state='validation',arm=arm,epoch=epoch,completed_new_epochs=completed())
  row['evaluations']={'raw':evaluate_checkpoint(arm,'raw',epoch,model,val,cfg,folder,registration),
                      'ema':evaluate_checkpoint(arm,'ema',epoch,ema.model,val,cfg,folder,registration)}
  history.append(row)
  checkpoint(last,dict(state_dict=cpu_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),ema=ema.state(),history=history,registration=registration))
  save(folder/'history.json',history)
  for name,e in best_evaluations(history).items():copy_checkpoint(e['checkpoint'],folder/f'best_{name}.pth')
  status(state='epoch_completed',arm=arm,epoch=epoch,arm_completed_epochs=len(history),completed_new_epochs=completed(),
         validation={k:v['metrics'] for k,v in row['evaluations'].items()})
  note('EPOCH_COMPLETE',arm=arm,epoch=epoch,raw_AUC=row['evaluations']['raw']['metrics']['macro_auc'],
       EMA_AUC=row['evaluations']['ema']['metrics']['macro_auc'],train_seconds=row['train_seconds'],AMP_skips=row['AMP_skips'])
  if (ROOT/'pause_after_epoch.flag').exists():
   status(state='paused_after_epoch',arm=arm,epoch=epoch,completed_new_epochs=completed());return None
 del model,opt,scaler,ema,criterion;gc.collect();torch.cuda.empty_cache()
 return verify_best(arm,history,arm_spec,cfg,val,folder,src,guard,'complete' if until==5 else 'screened')
def screen_score(arm,n=2):
 h=read(ROOT/arm/'history.json')[:n]
 if len(h)!=n:raise ValueError('Screen budget not complete')
 values=[e for row in h for e in row['evaluations'].values()]
 e=max(values,key=key)
 return dict(arm=arm,epoch=e['epoch'],variant=e['variant'],AUC=e['metrics']['macro_auc'],F1_at05=e['metrics']['macro_f1'],checkpoint_sha256=e['checkpoint_sha256'],prediction_sha256=e['prediction_sha256'])
def run_all():
 setup();cfg,records,audit=load_inputs();src=sources();guard=check_protected();reg=read(ROOT/'registration.json')
 if reg!=dict(config=CONFIG,source_hashes=src,protection=guard,pretrained=source_bundle(),incumbent_checkpoint_sha256=INCUMBENT_SHA,incumbent_prediction_sha256=INCUMBENT_PRED_SHA,incumbent_AUC=INCUMBENT_AUC):raise ValueError('Run registration differs')
 profile=read(ROOT/'resource_profile.json')
 if profile['state']!='passed' or profile['source_hashes']!=src or profile['protection']!=guard:raise ValueError('Resource profile invalid')
 base=run_arm(spec('A0_BCE_EMA'),5,cfg,records,audit,profile,src,guard)
 if base is None:return
 policies={'A1_noise':'noise','A2_photo':'photo','A3_photo_noise':'photo_noise'}
 for arm,policy in policies.items():
  if run_arm(spec(arm,policy),2,cfg,records,audit,profile,src,guard) is None:return
 table=[screen_score(arm) for arm in policies]
 winner=max(table,key=lambda x:(x['AUC'],x['F1_at05']))
 decision=dict(state='selected',screen_epochs=2,matched_control=screen_score('A0_BCE_EMA'),candidates=table,winner=winner,
               rule=CONFIG['augmentation_screen_selection'],fifth_epoch_not_seen_in_screen=True)
 freeze(ROOT/'augmentation_screen_selection.json',decision)
 selected=run_arm(spec(winner['arm'],policies[winner['arm']]),5,cfg,records,audit,profile,src,guard)
 if selected is None:return
 use_aug=key(selected['variants'][selected['best_variant']])>key(base['variants'][base['best_variant']])
 loss_policy=policies[winner['arm']] if use_aug else 'none'
 policy_decision=dict(state='selected',augmentation_for_loss=loss_policy,matched_BCE_arm=winner['arm'] if use_aug else 'A0_BCE_EMA',
                      baseline_best_AUC=base['best_AUC'],augmentation_best_AUC=selected['best_AUC'])
 freeze(ROOT/'loss_policy_selection.json',policy_decision)
 loss=run_arm(spec('B_ASL',loss_policy,'asl'),5,cfg,records,audit,profile,src,guard)
 if loss is None:return
 control=selected if use_aug else base
 recipe=loss if key(loss['variants'][loss['best_variant']])>key(control['variants'][control['best_variant']]) else control
 depth_spec=spec('C_S36',recipe['spec']['augmentation'],recipe['spec']['loss'])
 freeze(ROOT/'depth_recipe_selection.json',dict(state='selected',S18_control_arm=recipe['arm'],S18_control_best_AUC=recipe['best_AUC'],spec=depth_spec))
 depth=run_arm(depth_spec,5,cfg,records,audit,profile,src,guard)
 if depth is None:return
 results={arm:read(ROOT/arm/'results.json') for arm in ARMS}
 counts={arm:len(read(ROOT/arm/'history.json')) for arm in ARMS}
 if sum(counts.values())!=24 or sorted(counts[a] for a in policies)!=[2,2,5]:raise ValueError('Final epoch budget mismatch')
 check_runtime(src,guard)
 save(ROOT/'training_complete.json',dict(state='complete',completed_new_epochs=24,validation_evaluations=48,
      epochs_by_arm=counts,results=results,source_hashes=src,protected_inputs_after=guard,incumbent_AUC=INCUMBENT_AUC,
      no_test_inference=True,no_GitHub_upload=True))
 status(state='all_training_complete',completed_new_epochs=24,validation_evaluations=48)
 note('THREE_DIRECTIONS_COMPLETE',epochs=24,validation_evaluations=48)
@contextmanager
def gpu_lock():
 paths=(GLOBAL_LOCK,ROOT/'gpu.lock');handles=[]
 try:
  for p in paths:
   f=p.open('a+b')
   if f.tell()==0:f.write(b'0');f.flush()
   f.seek(0)
   try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   except BaseException:f.close();raise
   handles.append(f)
  yield
 finally:
  for f in reversed(handles):
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()
if __name__=='__main__':
 awake=0
 try:
  with gpu_lock():
   if ARGS.mode in ('profile','run'):awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
   if ARGS.mode=='preflight':cpu_preflight()
   elif ARGS.mode=='profile':resource_profile()
   else:run_all()
 except KeyboardInterrupt:
  status(state='interrupted_partial_epoch_not_counted',completed_new_epochs=completed());raise
 except Exception as exc:
  status(state='failed',error=str(exc),traceback=traceback.format_exc(),completed_new_epochs=completed());raise
 finally:
  if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
