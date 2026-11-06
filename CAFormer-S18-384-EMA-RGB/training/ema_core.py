"""BCE+EMA-only continuation; mathematical/update functions preserve real A code."""
from __future__ import annotations
import copy,ctypes,gc,hashlib,math,os,random,time,traceback
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
if os.name=='nt':import msvcrt
else:import fcntl
from ema_common import ROOT,DATA,SOURCE,INITIAL,REFERENCE,INITIAL_SHA,NATIVE_SHA,CONFIG,COMMON,ARMS,sha,read,save,freeze,save_npz,status,sources,check_protected,load_inputs
import numpy as np
import torch,timm
from torch.nn import functional as F
from data_pipeline import RawStudyDataset,collate_studies
from training_utils import seed,checkpoint,copy_checkpoint,calibration,metrics,loader,worst_batches,predict
from losses import build_loss
STEPS=math.ceil(18513/32)


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
    rgb_criterion=build_loss('RGB',num_classes=10)
    rgb_z=torch.randn(7,10,generator=gen,requires_grad=True)
    rgb_loss=losses_for('RGB_EMA',rgb_z,lengths,y,rgb_criterion).mean()
    rgb_loss.backward()
    if not torch.isfinite(rgb_loss) or not torch.isfinite(rgb_z.grad).all():raise ValueError('Nonfinite RGB loss/gradient')
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
    return dict(BCE_mean_probability_max_difference=error,EMA_first_update_decay=ema.last_decay,EMA_raw_model_unchanged=True,RGB_loss_finite=True,Study_count=18513)

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
                fields=dict(state='training',arm=arm,epoch=epoch,source_total_epoch=5+epoch,epochs_per_arm=5,arm_completed_epochs=epoch-1,completed_new_epochs=completed,studies=len(seen),total_studies=18513,images=image_count,total_images=30344,optimizer_attempts=attempts,optimizer_successful=successful,AMP_skips=skips,train_loss=loss_sum/len(seen),epoch_elapsed_seconds=time.perf_counter()-start,EMA_updates=ema.updates if ema else 0)
                status(**fields)
                if attempts%50==0 or len(seen)==len(dataset):note('TRAIN_PROGRESS',**{k:fields[k] for k in ('arm','epoch','studies','images','optimizer_attempts','train_loss','AMP_skips')})
        elif window>denominator:raise ValueError('Study window overflow')
    torch.cuda.synchronize()
    if len(seen)!=18513 or len(set(seen))!=18513 or sorted(seen)!=list(range(18513)) or image_count!=30344 or attempts!=579 or counts!=[32]*578+[17] or window or micro!=math.ceil(18513/batch):raise ValueError('Incomplete/duplicate train epoch or wrong Study normalization')
    return dict(epoch=epoch,source_total_epoch=5+epoch,train_loss=loss_sum/18513,train_seconds=time.perf_counter()-start,train_studies=len(seen),train_images=image_count,real_batch_size=batch,accumulation_steps=32//batch,effective_study_batch_size=32,micro_batches=micro,optimizer_steps_attempted=attempts,optimizer_steps_completed=successful,AMP_skips=skips,scaler_final=float(scaler.get_scale()),update_Study_counts=counts,lr_factors=factors,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,dataloader_seed=42+epoch,dataloader_order_sha256=order.hexdigest(),full_group_updates=group_updates(model,initial,arm),EMA_successful_updates=ema.updates if ema else 0)

def key(evaluation):return evaluation['metrics']['macro_auc'],evaluation['metrics']['macro_f1']

def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}

def evaluate_checkpoint(arm,variant,epoch,model,val,cfg,folder,registration):
    began=time.perf_counter();p,extra=predict(model,val,cfg,True);measured=metrics(val.targets,p)
    pp=folder/'predictions'/f'{variant}_epoch_{epoch:02d}.npz';cp=folder/'checkpoints'/f'{variant}_epoch_{epoch:02d}.pth'
    if cp.exists() or pp.exists():raise ValueError('Uncommitted epoch artifact exists; archive interruption before retry')
    save_npz(pp,targets=val.targets,probabilities=p,subjects=val.subjects,study_ids=val.study_ids,**extra)
    checkpoint(cp,dict(state_dict=cpu_state(model),epoch=epoch,source_total_epoch=5+epoch,arm=arm,variant=variant,initial_checkpoint_sha256=INITIAL_SHA,pretrained_sha256=NATIVE_SHA,full_backbone_training=True,registration=registration))
    return dict(epoch=epoch,variant=variant,metrics=measured,validation_seconds=time.perf_counter()-began,checkpoint=str(cp),checkpoint_sha256=sha(cp),prediction=str(pp),prediction_sha256=sha(pp))

def best_evaluations(history):
    best={}
    for row in history:
        for name,e in row['evaluations'].items():
            if name not in best or key(e)>key(best[name]):best[name]=e
    return best

def verify_variants(arm,history,cfg,val,folder,src,guard):
    output={};best=best_evaluations(history)
    for name,e in best.items():
        alias=folder/f'best_{name}.pth';copy_checkpoint(e['checkpoint'],alias)
        if sha(alias)!=e['checkpoint_sha256']:raise ValueError('Best alias SHA differs')
        m=build(arm,'cuda');stored=torch.load(alias,map_location='cpu',weights_only=True);m.load_state_dict(stored['state_dict'],strict=True)
        if stored['epoch']!=e['epoch'] or stored['arm']!=arm or stored['variant']!=name:raise ValueError('Best checkpoint identity differs')
        p,extra=predict(m,val,cfg,True)
        with np.load(e['prediction'],allow_pickle=False) as saved:
            for k,v in (('targets',val.targets),('subjects',val.subjects),('study_ids',val.study_ids),('image_paths',extra['image_paths'])):np.testing.assert_array_equal(saved[k],v)
            delta=float(np.max(np.abs(p-saved['probabilities'])));image_delta=float(np.max(np.abs(extra['image_probabilities']-saved['image_probabilities'])))
            np.testing.assert_allclose(p,saved['probabilities'],rtol=0,atol=1e-7);np.testing.assert_allclose(extra['image_probabilities'],saved['image_probabilities'],rtol=0,atol=1e-7)
        if metrics(val.targets,p)!=e['metrics']:raise ValueError('Reloaded best metrics differ')
        cal=calibration(val.targets,p,val.subjects);gate=dict(**cal,checkpoint=str(alias),checkpoint_sha256=sha(alias),best_epoch=e['epoch'],class_ids=list(range(1,11)),pretrained_sha256=NATIVE_SHA,arm=arm,variant=name,candidate_only=True,gate_fit_source='Full official validation; patient OOF F1 separate')
        save(folder/f'gates_{name}.json',gate)
        output[name]=dict(**e,best_alias=str(alias),calibration=cal,reloaded_max_abs_study_difference=delta,reloaded_max_abs_image_difference=image_delta,all_1422_raw_validation_images_rechecked=True)
        del m,stored;gc.collect();torch.cuda.empty_cache()
    selected=max(output,key=lambda name:key(output[name]));copy_checkpoint(output[selected]['checkpoint'],folder/'best.pth');gates=read(folder/f'gates_{selected}.json');gates['checkpoint']=str(folder/'best.pth');save(folder/'gates.json',gates)
    if sources()!=src or check_protected()!=guard:raise ValueError('Source/history protection changed')
    result=dict(state='complete',arm=arm,completed_new_epochs=len(history),best_variant=selected,best_epoch=output[selected]['epoch'],best_AUC=output[selected]['metrics']['macro_auc'],variants=output,initial_CAFormer_AUC=initial_metrics()['macro_auc'],source_hashes=src,protected_inputs=guard,no_test_inference=True,no_GitHub_upload=True)
    save(folder/'results.json',result);note('ARM_COMPLETE',arm=arm,best_variant=selected,best_epoch=result['best_epoch'],best_auc=result['best_AUC'])
    return result

def run_arm(arm,cfg,records,audit,profile,src,guard,completed):
    folder=ROOT/arm;folder.mkdir(exist_ok=True);registration=dict(config=CONFIG,arm=arm,source_hashes=src,protected_inputs=guard,initial_checkpoint_sha256=INITIAL_SHA,real_batch_size=profile['real_batch_size'],accumulation_steps=profile['accumulation_steps'])
    freeze(folder/'registration.json',registration)
    if (folder/'results.json').exists():
        result=read(folder/'results.json')
        if result['state']=='complete' and result['completed_new_epochs']==5:return result
    train=RawStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True);val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
    model=build(arm,'cuda');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};opt=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=CONFIG['amp_init_scale']);criterion=criterion_for(arm,audit,'cuda');ema=ShadowEMA(model) if arm in ('A_BCE_EMA','RGB_EMA') else None
    history=[];last=folder/'last_training.pth'
    if last.exists():
        resume=torch.load(last,map_location='cpu',weights_only=True)
        if resume['registration']!=registration:raise ValueError('Resume registration changed')
        history=resume['history']
        if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>5:raise ValueError('Invalid resumed epochs')
        for row in history:
            for e in row['evaluations'].values():
                if sha(e['checkpoint'])!=e['checkpoint_sha256'] or sha(e['prediction'])!=e['prediction_sha256']:raise ValueError('Completed artifacts changed')
        model.load_state_dict(resume['state_dict'],strict=True);opt.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
        if ema:ema.load(resume['ema'])
        del resume;save(folder/'history.json',history)
    elif (folder/'history.json').exists():raise ValueError('History without atomic resume checkpoint')
    for epoch in range(len(history)+1,6):
        if (ROOT/'pause_after_epoch.flag').exists():status(state='paused_before_epoch',arm=arm,epoch=epoch,completed_new_epochs=completed+len(history));return None
        if sources()!=src or sha(INITIAL)!=INITIAL_SHA:raise ValueError('Frozen training source/start changed')
        variants=('raw','ema') if arm in ('A_BCE_EMA','RGB_EMA') else ('raw',)
        for variant in variants:
            if (folder/'checkpoints'/f'{variant}_epoch_{epoch:02d}.pth').exists() or (folder/'predictions'/f'{variant}_epoch_{epoch:02d}.npz').exists():
                raise ValueError('Uncommitted epoch artifacts exist; archive the interrupted attempt before retraining this epoch')
        status(state='training',arm=arm,epoch=epoch,studies=0,images=0,completed_new_epochs=completed+len(history))
        row=training_epoch(arm,model,train,opt,scaler,ema,criterion,epoch,profile,initial,completed+len(history));status(state='validation',arm=arm,epoch=epoch,completed_new_epochs=completed+len(history))
        evaluations={'raw':evaluate_checkpoint(arm,'raw',epoch,model,val,cfg,folder,registration)}
        if ema:evaluations['ema']=evaluate_checkpoint(arm,'ema',epoch,ema.model,val,cfg,folder,registration)
        row['evaluations']=evaluations;history.append(row)
        checkpoint(last,dict(state_dict=cpu_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),ema=ema.state() if ema else None,history=history,registration=registration))
        save(folder/'history.json',history)
        for name,e in best_evaluations(history).items():copy_checkpoint(e['checkpoint'],folder/f'best_{name}.pth')
        status(state='epoch_completed',arm=arm,epoch=epoch,arm_completed_epochs=len(history),completed_new_epochs=completed+len(history),validation={k:v['metrics'] for k,v in evaluations.items()});note('EPOCH_COMPLETE',arm=arm,epoch=epoch,raw_AUC=evaluations['raw']['metrics']['macro_auc'],EMA_AUC=evaluations.get('ema',{}).get('metrics',{}).get('macro_auc'),train_seconds=row['train_seconds'],AMP_skips=row['AMP_skips'])
        if (ROOT/'pause_after_epoch.flag').exists():status(state='paused_after_epoch',arm=arm,epoch=epoch,completed_new_epochs=completed+len(history));return None
    del model,opt,scaler,ema,criterion;gc.collect();torch.cuda.empty_cache()
    return verify_variants(arm,history,cfg,val,folder,src,guard)


def build(arm='A_BCE_EMA',device='cpu'):
    if arm not in ('A_BCE_EMA','RGB_EMA'):raise ValueError('Unsupported arm')
    seed(42);stored=torch.load(INITIAL,map_location='cpu',weights_only=True)
    if stored['epoch']!=5 or stored['pretrained_sha256']!=NATIVE_SHA or not stored['full_backbone_training']:
        raise ValueError('Expected complete native384 epoch5 source')
    model=timm.create_model('caformer_s18',pretrained=False,num_classes=10,drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
    model.head.fc.head_drop.p=.2;model.load_state_dict(stored['state_dict'],strict=True)
    model.requires_grad_(True);model.set_grad_checkpointing(True)
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):raise ValueError('Unexpected BN')
    if device=='cuda':model.to(device='cuda',memory_format=torch.channels_last)
    return model

def optimizer_for(model):
    groups={}
    for name,p in model.named_parameters():
        decay=p.ndim>1 and not name.endswith('.bias');key='existing_decay' if decay else 'existing_no_decay'
        if key not in groups:groups[key]=dict(params=[],lr=1e-5,base_lr=1e-5,weight_decay=.05 if decay else 0.,name=key)
        groups[key]['params'].append(p)
    if sum(len(group['params']) for group in groups.values())!=len(list(model.parameters())):raise ValueError('Optimizer coverage')
    return torch.optim.AdamW(list(groups.values()))

def criterion_for(arm,audit,device):
    if arm=='RGB_EMA':
        cfg=CONFIG.get('RGB',{})
        return build_loss('RGB',num_classes=10,tn_gamma_neg=float(cfg.get('tn_gamma_neg',3.0)),fp_gamma_neg=float(cfg.get('fp_gamma_neg',3.0)),ratio=float(cfg.get('ratio',1.0)),eps=float(cfg.get('eps',1e-4)))
    return None

def losses_for(arm,logits,lengths,y,criterion=None):
    if arm=='RGB_EMA':
        if criterion is None:raise ValueError('RGB_EMA requires criterion')
        probs=torch.stack([z.sigmoid().mean(0) for z in logits.float().split(lengths)])
        study_logits=torch.logit(probs.clamp(min=1e-6,max=1-1e-6))
        value=criterion(study_logits,y.float())
        if not torch.isfinite(value):raise FloatingPointError('Invalid Study RGB loss')
        return value/len(y)*torch.ones(len(y),device=value.device,dtype=value.dtype)
    lp,lq=log_probabilities(logits,lengths)
    if lp.shape!=y.shape or lp.shape[1]!=10:raise ValueError('Study shapes differ')
    values=-(y.float()*lp+(1-y.float())*lq).mean(1)
    if values.shape!=(len(y),) or not torch.isfinite(values).all():raise FloatingPointError('Invalid Study BCE')
    return values

def group_updates(model,initial,arm):
    groups=('stem.','stages.0.','stages.1.','stages.2.','stages.3.','head.norm.','head.fc.fc1.','head.fc.norm.','head.fc.fc2.')
    state=model.state_dict();result={}
    for prefix in groups:
        keys=[key for key in state if key.startswith(prefix)]
        count=sum(not torch.equal(state[key].detach().cpu(),initial[key]) for key in keys)
        if not keys or not count:raise ValueError('No full-network update: '+prefix)
        result[prefix]=dict(tensors=len(keys),changed_tensors=count)
    return result

def reference_predictions():
    with np.load(REFERENCE,allow_pickle=False) as stored:return {key:stored[key].copy() for key in stored.files}

def initial_metrics():
    value=reference_predictions();return metrics(value['targets'],value['probabilities'])

def preflight(cfg,records,audit,guard,src):
    mathematical=cpu_math(audit)
    model=build().eval();model.set_grad_checkpointing(False)
    dataset=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
    reference=reference_predictions()
    for key,value in (('targets',dataset.targets),('subjects',dataset.subjects),('study_ids',dataset.study_ids)):
        np.testing.assert_array_equal(reference[key],value)
    sample=dataset[0][0][:1]
    with torch.no_grad():output=model(sample)
    if output.shape!=(1,10) or not torch.isfinite(output).all():raise ValueError('Invalid complete model')
    train=RawStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True)
    worst,table=worst_batches(train,16)
    freeze(ROOT/'registration.json',dict(config=CONFIG,initial_checkpoint_sha256=INITIAL_SHA,native_pretrained_sha256=NATIVE_SHA,source_hashes=src,protection=guard))
    save(ROOT/'preflight.json',dict(state='passed',CPU_only=True,no_cuda_initialized=not torch.cuda.is_initialized(),data_audit=audit,math_checks=mathematical,worst_batch=worst,worst_by_epoch=table,source_hashes=src,protection=guard))
    status(state='CPU_preflight_passed',completed_new_epochs=0)

def smoke(cfg,records,audit,guard,src,arm='A_BCE_EMA'):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    checked=read(ROOT/'preflight.json')
    if checked['source_hashes']!=src or checked['protection']!=guard:raise ValueError('Stale preflight')
    dataset=RawStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True)
    worst=checked['worst_batch'];seed(42);model=build(arm,device='cuda').train()
    initial={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    opt=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=1024);ema=ShadowEMA(model)
    images,lengths,y,_=collate_studies([dataset[i] for i in worst['indices']])
    images=images.to('cuda',memory_format=torch.channels_last);y=y.to('cuda');torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(images)
        loss=losses_for(arm,logits,lengths,y,criterion_for(arm,audit,'cuda')).mean()
        scaler.scale(loss).backward();scaler.unscale_(opt)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Smoke gradients')
        old=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<old:raise FloatingPointError('AMP skipped smoke update')
        ema.update(model)
    groups=group_updates(model,initial,'A_BCE_EMA')
    peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
    total=torch.cuda.get_device_properties(0).total_memory/2**20
    if max(peak,reserved)>total*.85:raise RuntimeError('Fixed real16 recipe exceeds85%memory; do not silently alter batch')
    del model,ema,opt,scaler,images;gc.collect();torch.cuda.empty_cache()
    val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False);model=build(device='cuda')
    p,extra=predict(model,val,cfg,True);reference=reference_predictions()
    for key,value in (('targets',val.targets),('subjects',val.subjects),('study_ids',val.study_ids),('image_paths',extra['image_paths'])):
        np.testing.assert_array_equal(reference[key],value)
    np.testing.assert_allclose(p,reference['probabilities'],rtol=0,atol=1e-7)
    np.testing.assert_allclose(extra['image_probabilities'],reference['image_probabilities'],rtol=0,atol=1e-7)
    freeze(ROOT/'resource_profile.json',dict(state='passed',real_batch_size=16,accumulation_steps=2,effective_study_batch_size=32,peak_allocated_mib=peak,peak_reserved_mib=reserved,total_device_mib=total,two_successful_smoke_steps=True,EMA_updates=2,full_network_group_updates=groups,source_hashes=src,protection=guard))
    status(state='resource_preflight_passed',completed_new_epochs=0)

def main(args):
    setup();cfg,records,audit=load_inputs();src=sources();guard=check_protected()
    if args.check_only:preflight(cfg,records,audit,guard,src);return
    checked=read(ROOT/'preflight.json')
    if checked['state']!='passed' or checked['source_hashes']!=src or checked['protection']!=guard:raise ValueError('Preflight stale')
    if args.smoke:smoke(cfg,records,audit,guard,src,getattr(args,'arm','A_BCE_EMA'));return
    profile=read(ROOT/'resource_profile.json')
    if profile['state']!='passed' or profile['source_hashes']!=src or profile['protection']!=guard:raise ValueError('Resource profile stale')
    arm=getattr(args,'arm','A_BCE_EMA')
    result=run_arm(arm,cfg,records,audit,profile,src,guard,0)
    if result is not None:
        save(ROOT/'training_complete.json',dict(state='complete',completed_new_epochs=5,arm=arm,results=result,no_test_inference=True))
        status(state='EMA_continuation_complete',completed_new_epochs=5)

@contextmanager
def gpu_lock():
    with (ROOT/'gpu.lock').open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)
