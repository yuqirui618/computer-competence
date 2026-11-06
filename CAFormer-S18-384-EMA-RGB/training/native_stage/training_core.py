"""Fresh native384 CAFormer all-layer raw-image Study training, exactly five epochs.
--check-only: CPU provenance/data/loss checks. --smoke: throwaway worst real batch,
select32/16/8/4/2Study and check full-network gradients; neither counts as an epoch.
"""
from __future__ import annotations
import argparse

def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--check-only',action='store_true');group.add_argument('--smoke',action='store_true')
    return parser.parse_args()
if __name__=='__main__':ARGS=arguments()
import ctypes,gc,hashlib,math,os,random,shutil,time,traceback
if os.name=="nt":import msvcrt
else:import fcntl
from contextlib import contextmanager
from pathlib import Path
from common import ROOT,DATA,COMMON,EXPECTED,sha,read,save,freeze,save_npz,status,pretrained,load_inputs,source_hashes,check_protected
import numpy as np
import torch,timm
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_pipeline import RawStudyDataset,RawImageDataset,collate_studies,worker_init
EPOCH_STEPS=math.ceil(18513/32)
GROUPS=('stem.','stages.0.','stages.1.','stages.2.','stages.3.','head.norm.','head.fc.fc1.','head.fc.norm.','head.fc.fc2.')

def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
def checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def copy_checkpoint(source,target):
    target=Path(target);tmp=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(source,tmp);tmp.replace(target)

def build_model(priors,device='cpu'):
    meta=pretrained();seed(42)
    # Match the author's single hidden-head dropout. timm drop_rate would enable
    # both an outer dropout and a hidden-head dropout, so configure sites explicitly.
    model=timm.create_model('caformer_s18',pretrained=False,num_classes=1000,drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
    source=load_file(meta['weights'],device='cpu');model.load_state_dict(source,strict=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in source.items()):raise ValueError('Strict native1000-class source load failed')
    if model.head.fc.fc1.out_features!=2048 or model.head.fc.fc2.in_features!=2048:raise ValueError('Native CAFormer hidden MLP differs')
    model.head.fc.head_drop.p=.2
    if getattr(model.head.drop,'p',0.)!=0.:raise ValueError('Unexpected outer dropout')
    layer=torch.nn.Linear(2048,10);torch.nn.init.trunc_normal_(layer.weight,std=.02)
    with torch.no_grad():layer.bias.copy_(torch.logit(torch.tensor(priors,dtype=torch.float32)))
    model.head.fc.fc2=layer;model.num_classes=10;model.requires_grad_(True);model.set_grad_checkpointing(True)
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):raise ValueError('BatchNorm requires separate accumulation review')
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('Full backbone must be trainable')
    if device=='cuda':model.to(device='cuda',memory_format=torch.channels_last)
    return model

def trainability(model):
    return dict(full_backbone_training=True,all_parameters_trainable=all(p.requires_grad for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        original_head_hidden_trainable=True,gradient_checkpointing=bool(model.grad_checkpointing),no_BatchNorm=True,head_dropout_sites={name:float(m.p) for name,m in model.named_modules() if isinstance(m,torch.nn.Dropout)},
        actual_drop_path_rates=sorted(set(float(m.drop_prob) for m in model.modules() if hasattr(m,'drop_prob'))),source='native384 ImageNet only')

def optimizer_for(model):
    groups={}
    for name,param in model.named_parameters():
        head=name.startswith('head.fc.fc2.');decay=param.ndim>1 and not name.endswith('.bias')
        key=('classifier' if head else 'backbone')+('_decay' if decay else '_no_decay')
        if key not in groups:groups[key]=dict(params=[],lr=COMMON['classifier_lr'] if head else COMMON['backbone_lr'],weight_decay=.05 if decay else 0.,name=key,base_lr=COMMON['classifier_lr'] if head else COMMON['backbone_lr'])
        groups[key]['params'].append(param)
    if sum(len(g['params']) for g in groups.values())!=len(list(model.parameters())):raise ValueError('Optimizer coverage differs')
    return torch.optim.AdamW(list(groups.values()))

def lr_factor(global_update):
    if global_update<EPOCH_STEPS:return .1+.9*global_update/max(1,EPOCH_STEPS-1)
    progress=(global_update-EPOCH_STEPS)/max(1,4*EPOCH_STEPS-1)
    return .1+.9*(1+math.cos(math.pi*min(1.,progress)))/2

def study_losses(image_logits,lengths,targets):
    if sum(lengths)!=len(image_logits) or len(lengths)!=len(targets) or min(lengths)<1:raise ValueError('Invalid Study views')
    output=[]
    for scores,target in zip(image_logits.float().split(lengths),targets.float()):
        logp=torch.logsumexp(F.logsigmoid(scores),dim=0)-math.log(len(scores))
        logq=torch.logsumexp(F.logsigmoid(-scores),dim=0)-math.log(len(scores))
        output.append(-(target*logp+(1-target)*logq).mean())
    return torch.stack(output)

def math_checks():
    generator=torch.Generator().manual_seed(987)
    x=torch.randn(7,10,generator=generator,requires_grad=True);lengths=[2,1,3,1];y=torch.randint(0,2,(4,10),generator=generator).float()
    stable=study_losses(x,lengths,y)
    probs=torch.stack([z.sigmoid().mean(0) for z in x.split(lengths)])
    direct=F.binary_cross_entropy(probs,y,reduction='none').mean(1)
    loss_difference=float((stable-direct).abs().max().detach())
    if loss_difference>1e-6:raise ValueError('Stable Study probability-mean BCE differs')
    stable.mean().backward();full=x.grad.detach().clone()
    z=x.detach().clone().requires_grad_(True)
    (study_losses(z[:3],[2,1],y[:2]).sum()/4).backward()
    (study_losses(z[3:],[3,1],y[2:]).sum()/4).backward()
    grad_difference=float((full-z.grad).abs().max())
    # Explicit uneven last window: first micro has2Studies, last only1.
    u=x.detach()[:6].clone().requires_grad_(True);target=y[:3]
    study_losses(u,[2,1,3],target).mean().backward();expected=u.grad.detach().clone()
    v=u.detach().clone().requires_grad_(True)
    (study_losses(v[:3],[2,1],target[:2]).sum()/3).backward();(study_losses(v[3:],[3],target[2:]).sum()/3).backward()
    last_difference=float((expected-v.grad).abs().max())
    # Exercise the exact17-Study final optimizer window with8+8+1 microbatches.
    a=torch.randn(17,10,generator=generator,requires_grad=True);b=torch.randint(0,2,(17,10),generator=generator).float()
    study_losses(a,[1]*17,b).mean().backward();expected17=a.grad.detach().clone()
    c=a.detach().clone().requires_grad_(True)
    for start,end in ((0,8),(8,16),(16,17)):(study_losses(c[start:end],[1]*(end-start),b[start:end]).sum()/17).backward()
    last17_difference=float((expected17-c.grad).abs().max())
    extreme=torch.tensor([[1e4,-1e4]*5,[-1e4,1e4]*5],requires_grad=True)
    extreme_loss=study_losses(extreme,[1,1],torch.ones(2,10)).sum();extreme_loss.backward()
    if not torch.isfinite(extreme_loss) or not torch.isfinite(extreme.grad).all():raise ValueError('Extreme Study loss/gradient nonfinite')
    if max(grad_difference,last_difference,last17_difference)>1e-6:raise ValueError('Accumulated Study gradients differ')
    return dict(stable_probability_mean_BCE_matches_direct=True,max_abs_loss_difference=loss_difference,accumulated_gradient_matches_full_window=True,max_abs_gradient_difference=grad_difference,uneven_last_window_gradient_matches=True,max_abs_last_window_gradient_difference=last_difference,final17_study_window_gradient_matches=True,max_abs_final17_gradient_difference=last17_difference,extreme_logits_loss_and_gradient_finite=True,normalization='Actual Study count per optimizer window, not division by microbatch count')

def fit_gates(y,p):
    grid=sorted(np.arange(1,40)/40,key=lambda g:(abs(g-.5),g));scores=[f1_score(y,p>=g,average=None,zero_division=0) for g in grid]
    return np.asarray(grid)[np.argmax(scores,axis=0)]
def calibration(y,p,subjects):
    decisions=np.zeros_like(y)
    for fit,held in GroupKFold(5,shuffle=True,random_state=42).split(p,y,subjects):decisions[held]=p[held]>=fit_gates(y[fit],p[fit])
    f=f1_score(y,decisions,average=None,zero_division=0)
    return dict(oof_macro_f1=float(f.mean()),oof_f1_by_class_id=f.tolist(),final_gates_by_class_id=fit_gates(y,p).tolist(),patients=len(set(subjects)))
def metrics(y,p):
    if y.shape!=p.shape or y.shape!=(859,10) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Validation probabilities invalid/incomplete')
    a=roc_auc_score(y,p,average=None);f=f1_score(y,p>=.5,average=None,zero_division=0);ap=average_precision_score(y,p,average=None);z=p>=.5
    return dict(macro_auc=float(a.mean()),macro_f1=float(f.mean()),auc=a.tolist(),f1=f.tolist(),macro_ap=float(ap.mean()),ap=ap.tolist(),tp=np.sum(z&(y==1),0).tolist(),fp=np.sum(z&(y==0),0).tolist(),fn=np.sum(~z&(y==1),0).tolist(),studies=859,patients=228)

def loader(dataset,batch_size,shuffle=False,epoch=0,workers=None,collate_fn=None):
    n=COMMON['workers'] if workers is None else workers
    kwargs=dict(batch_size=batch_size,shuffle=shuffle,num_workers=n,pin_memory=True,collate_fn=collate_fn,generator=torch.Generator().manual_seed(42+epoch),worker_init_fn=worker_init)
    if n:kwargs.update(prefetch_factor=COMMON['prefetch_factor'],persistent_workers=False)
    return DataLoader(dataset,**kwargs)

def worst_batches(dataset,batch_size):
    best=None;table={}
    for epoch in range(1,6):
        trial=loader(dataset,batch_size,True,epoch,workers=0,collate_fn=collate_studies)
        torch.empty((),dtype=torch.int64).random_(generator=trial.generator)
        maximum=None
        for position,indices in enumerate(trial.batch_sampler):
            n=int(dataset.lengths[indices].sum())
            if maximum is None or n>maximum['images']:maximum=dict(epoch=epoch,batch_index=position,studies=len(indices),images=n,indices=indices)
        table[str(epoch)]={k:v for k,v in maximum.items() if k!='indices'}
        if best is None or maximum['images']>best['images']:best=maximum
    return best,table

def update_audit(model,initial):
    state=model.state_dict();result={}
    for prefix in GROUPS:
        keys=[k for k in state if k.startswith(prefix)]
        if not keys:raise ValueError('Missing trainable group: '+prefix)
        count=sum(not torch.equal(state[k].detach().cpu(),initial[k]) for k in keys)
        if count==0:raise ValueError('No update in full-network group: '+prefix)
        result[prefix]=dict(tensors=len(keys),changed_tensors=count)
    return result

@torch.inference_mode()
def predict(model,val_data,cfg,return_images=False):
    model.eval();model.set_grad_checkpointing(False)
    image_data=RawImageDataset(val_data.records,cfg['data_root'],cfg['mean'],cfg['std'])
    values=[];seen=[]
    for images,indices in loader(image_data,4,workers=COMMON['workers']):
        with torch.autocast('cuda',dtype=torch.float16):logits=model(images.to('cuda',memory_format=torch.channels_last,non_blocking=True))
        values.append(logits.float().sigmoid().cpu().numpy());seen.extend(indices.tolist())
    if seen!=list(range(1422)):raise ValueError('Validation image order/coverage changed')
    image_probabilities=np.concatenate(values);groups=[[] for _ in val_data.records]
    for i,group in enumerate(image_data.study_indices):groups[group].append(i)
    probabilities=np.stack([image_probabilities[group].mean(0) for group in groups])
    model.set_grad_checkpointing(True)
    if probabilities.shape!=(859,10):raise ValueError('Validation Study coverage differs')
    extra=dict(image_probabilities=image_probabilities,image_paths=np.asarray(image_data.paths,dtype=str),image_study_indices=np.asarray(image_data.study_indices,np.int64))
    return (probabilities,extra) if return_images else probabilities

def preflight(cfg,records,audit,meta,protected,sources):
    train=RawStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True);val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
    sample=train[0]
    if tuple(sample[0].shape[1:])!=(3,384,384) or not torch.isfinite(sample[0]).all():raise ValueError('Native raw train preprocessing invalid')
    model=build_model(audit['train_study_priors']).eval();model.set_grad_checkpointing(False)
    with torch.no_grad():output=model(val[0][0][:1])
    if output.shape!=(1,10) or not torch.isfinite(output).all():raise ValueError('Native CPU image graph invalid')
    before_source=load_file(meta['weights'],device='cpu')
    unchanged=sum(torch.equal(v,model.state_dict()[k]) for k,v in before_source.items() if not k.startswith('head.fc.fc2.'))
    expected=sum(not k.startswith('head.fc.fc2.') for k in before_source)
    if unchanged!=expected:raise ValueError('Native body/hidden head changed before training')
    model.set_grad_checkpointing(True)
    mathematical=math_checks();worst={str(b):worst_batches(train,b)[0] for b in (32,16,8,4,2)}
    base=dict(common=COMMON,source_hashes=sources,pretrained_sha256=meta['sha256'],pretrained_manifest_sha256=sha(ROOT/'pretrained_manifest.json'),protected_hashes=protected)
    if source_hashes()!=sources or check_protected()!=protected:raise ValueError('Preflight inputs changed')
    freeze(ROOT/'preflight_plan.json',base)
    save(ROOT/'preflight.json',dict(state='passed',CPU_only=True,no_cuda_initialized=not torch.cuda.is_initialized(),data_audit=audit,loss_math_checks=mathematical,trainability=trainability(model),
        strict_pretrained_load_nonclassifier_tensors=unchanged,only_final_fc2_reinitialized=True,new_classifier_bias_priors=audit['train_study_priors'],worst_batches=worst,**base))
    print('NATIVE_CPU_PREFLIGHT_PASSED '+str({b:x['images'] for b,x in worst.items()}),flush=True)


def smoke(cfg,records,audit,meta,protected,sources):
    dataset=RawStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True)
    attempts=[];selected=None
    for batch_size in (32,16,8,4,2):
        seed(42);worst,table=worst_batches(dataset,batch_size);model=optimizer=scaler=images=logits=losses=None
        try:
            model=build_model(audit['train_study_priors'],'cuda');model.train();optimizer=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=1024)
            initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            images,lengths,targets,_=collate_studies([dataset[i] for i in worst['indices']])
            torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(images.to('cuda',memory_format=torch.channels_last))
            losses=study_losses(logits,lengths,targets.to('cuda'))
            scaler.scale(losses.mean()).backward();scaler.unscale_(optimizer)
            missing=[name for name,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            if missing:raise FloatingPointError('Full-network missing/nonfinite gradients: '+str(missing[:5]))
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old:raise FloatingPointError('Smoke optimizer step skipped')
            changes=update_audit(model,initial);peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20;total=torch.cuda.get_device_properties(0).total_memory/2**20
            success=dict(real_batch_size=batch_size,effective_study_batch_size=32,accumulation_steps=32//batch_size,worst_batch=worst,worst_batches_by_epoch=table,peak_allocated_mib=peak,peak_reserved_mib=reserved,total_device_mib=total,safety_limit_fraction=COMMON['gpu_memory_margin_fraction'],
                finite_full_network_backward=True,effective_optimizer_step=True,group_update_audit=changes,full_backbone_training=True,gradient_checkpointing=True,trainability=trainability(model))
            attempts.append(dict(batch_size=batch_size,state='passed',peak_allocated_mib=peak))
            # Keep a practical margin; resolution/weights/loss are never changed to fit.
            if max(peak,reserved)>total*COMMON['gpu_memory_margin_fraction']:
                attempts[-1]['state']='insufficient_margin';print('NATIVE_SMOKE_REDUCE_MARGIN '+str(batch_size),flush=True)
            else:selected=success
        except torch.OutOfMemoryError as exc:
            attempts.append(dict(batch_size=batch_size,state='out_of_memory',error=str(exc)));print('NATIVE_SMOKE_OOM '+str(batch_size),flush=True)
        finally:
            del model,optimizer,scaler,images,logits,losses;gc.collect();torch.cuda.empty_cache()
        if selected is not None:break
    if selected is None:raise RuntimeError('No safe native384 full-network batch among32/16/8/4/2')
    if source_hashes()!=sources or check_protected()!=protected:raise ValueError('Smoke source/protected changed')
    profile=dict(state='passed',scope='Throwaway source-initialized full-network worst real batch; zero training epochs',common=COMMON,source_hashes=sources,pretrained_sha256=meta['sha256'],pretrained_manifest_sha256=sha(ROOT/'pretrained_manifest.json'),protected_hashes=protected,attempts=attempts,**selected)
    freeze(ROOT/'resource_profile.json',profile);print('NATIVE_SMOKE_PASSED '+str(dict(real_batch=selected['real_batch_size'],accumulation=selected['accumulation_steps'],peak_mib=selected['peak_allocated_mib'])),flush=True)


def training_epoch(model,dataset,optimizer,scaler,epoch,profile,initial):
    seed(42+epoch);model.train();model.set_grad_checkpointing(True)
    batch_size=profile['real_batch_size'];accum=profile['accumulation_steps'];began=time.perf_counter();seen=[];images_seen=0;loss_sum=0.;micro=0;updates=0;skips=0;window_count=0;denominator=None;update_counts=[];lr_trace=[];order=hashlib.sha256();next_progress=3000
    optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    for images,lengths,targets,indices in loader(dataset,batch_size,True,epoch,collate_fn=collate_studies):
        if window_count==0:
            denominator=min(32,len(dataset)-len(seen));factor=lr_factor((epoch-1)*EPOCH_STEPS+updates)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            lr_trace.append(factor)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(images.to('cuda',memory_format=torch.channels_last,non_blocking=True))
        losses=study_losses(logits,lengths,targets.to('cuda',non_blocking=True))
        if not torch.isfinite(losses).all():raise FloatingPointError('Native Study loss nonfinite')
        scaler.scale(losses.sum()/denominator).backward();count=len(targets);window_count+=count;loss_sum+=float(losses.detach().sum());seen.extend(indices.tolist());images_seen+=sum(lengths);micro+=1
        for index in indices.tolist():order.update((str(dataset.subjects[index])+'|'+str(dataset.study_ids[index])+'\n').encode())
        if window_count==denominator:
            old=scaler.get_scale();scaler.step(optimizer);scaler.update();skips+=int(scaler.get_scale()<old);updates+=1;update_counts.append(window_count);window_count=0;optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale()<1e-12:raise FloatingPointError('GradScaler collapsed')
        elif window_count>denominator:raise ValueError('Accumulation overshot exact Study window')
        if images_seen>=next_progress or images_seen==30344:
            status(state='training',epoch=epoch,images=images_seen,total_images=30344,studies=len(seen),train_loss=loss_sum/len(seen),optimizer_updates=updates)
            print(f'NATIVE_TRAIN epoch={epoch}/5 images={images_seen}/30344 studies={len(seen)}/18513 updates={updates}/579 loss={loss_sum/len(seen):.7f}',flush=True);next_progress+=3000
    torch.cuda.synchronize()
    if len(seen)!=18513 or len(set(seen))!=18513 or sorted(seen)!=list(range(18513)) or images_seen!=30344 or micro!=math.ceil(18513/batch_size) or window_count!=0 or updates!=579 or update_counts!=[32]*578+[17]:raise ValueError('Incomplete/duplicated Study epoch or incorrect accumulation')
    return dict(epoch=epoch,images=images_seen,studies=len(seen),train_loss=loss_sum/len(seen),train_seconds=time.perf_counter()-began,micro_batches=micro,optimizer_steps_attempted=updates,optimizer_steps_completed=updates-skips,grad_scaler_scale_decreases=skips,scaler_final_scale=float(scaler.get_scale()),
        update_study_counts=update_counts,real_batch_size=batch_size,accumulation_steps=accum,effective_study_batch_size=32,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,lr_factors=lr_trace,
        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},dataloader_seed=42+epoch,dataloader_order_sha256=order.hexdigest(),group_update_audit=update_audit(model,initial))


def score(row):return row['macro_auc'],row['macro_f1']
def run(cfg,records,audit,meta,protected,sources,profile):
    plan=dict(common=COMMON,source_hashes=sources,pretrained_sha256=meta['sha256'],pretrained_manifest_sha256=sha(ROOT/'pretrained_manifest.json'),resource_profile=profile,protected_hashes=protected)
    freeze(ROOT/'plan.json',plan);(ROOT/'checkpoints').mkdir(exist_ok=True);(ROOT/'predictions').mkdir(exist_ok=True)
    train=RawStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True);val=RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
    source_model=build_model(audit['train_study_priors']);initial={k:v.detach().clone() for k,v in source_model.state_dict().items()};del source_model
    model=build_model(audit['train_study_priors'],'cuda');optimizer=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=1024);history=[]
    last=ROOT/'last_optimizer.pth'
    if last.exists():
        resumed=torch.load(last,map_location='cpu',weights_only=True)
        if resumed['plan']!=plan or resumed['pretrained_sha256']!=meta['sha256']:raise ValueError('Resume provenance differs')
        history=resumed['history']
        if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>5:raise ValueError('Invalid completed history')
        for row in history:
            if sha(row['checkpoint'])!=row['checkpoint_sha256'] or sha(row['prediction'])!=row['prediction_sha256']:raise ValueError('Completed artifact changed')
        previous=read(ROOT/'history.json') if (ROOT/'history.json').exists() else []
        if previous!=history[:len(previous)]:raise ValueError('History differs from atomic optimizer snapshot')
        model.load_state_dict(resumed['state_dict'],strict=True);optimizer.load_state_dict(resumed['optimizer']);scaler.load_state_dict(resumed['scaler']);del resumed;save(ROOT/'history.json',history)
    elif (ROOT/'history.json').exists():raise ValueError('History exists without optimizer snapshot')
    best=max(history,key=lambda r:score(r['validation'])) if history else None
    if best is not None:copy_checkpoint(best['checkpoint'],ROOT/'best.pth')
    for epoch in range(len(history)+1,6):
        if source_hashes()!=sources:raise ValueError('Frozen training source changed')
        if (ROOT/'pause_after_epoch.flag').exists():status(state='paused_before_epoch',epoch=epoch);return
        status(state='training',epoch=epoch,images=0,total_images=30344,studies=0)
        row=training_epoch(model,train,optimizer,scaler,epoch,profile,initial);status(state='validation',epoch=epoch)
        began=time.perf_counter();p,extra=predict(model,val,cfg,True);measured=metrics(val.targets,p);cal=calibration(val.targets,p,val.subjects)
        pp=ROOT/'predictions'/f'epoch_{epoch:02d}.npz';cp=ROOT/'checkpoints'/f'epoch_{epoch:02d}.pth'
        save_npz(pp,targets=val.targets,probabilities=p,subjects=val.subjects,study_ids=val.study_ids,**extra)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()};checkpoint(cp,dict(state_dict=state,epoch=epoch,plan=plan,pretrained_sha256=meta['sha256'],full_backbone_training=True))
        row.update(validation=measured,calibration=cal,validation_seconds=time.perf_counter()-began,prediction=str(pp),prediction_sha256=sha(pp),checkpoint=str(cp),checkpoint_sha256=sha(cp))
        if best is None or score(measured)>score(best['validation']):best=row;copy_checkpoint(cp,ROOT/'best.pth')
        history.append(row);checkpoint(last,dict(state_dict=state,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),plan=plan,pretrained_sha256=meta['sha256'],history=history,best_epoch=best['epoch']))
        save(ROOT/'history.json',history);status(state='checkpointed',epoch=epoch,auc=measured['macro_auc'],f1=measured['macro_f1'],oof_f1=cal['oof_macro_f1'],best_epoch=best['epoch'])
        print(f"NATIVE_EPOCH_DONE {epoch}/5 AUC={measured['macro_auc']:.10f} F1={measured['macro_f1']:.10f} OOF_F1={cal['oof_macro_f1']:.10f} train_s={row['train_seconds']:.2f} skipped={row['grad_scaler_scale_decreases']}",flush=True)
        del state,p,extra
        if (ROOT/'pause_after_epoch.flag').exists():status(state='paused_after_epoch',epoch=epoch);return
    del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache();verify_best(cfg,val,audit,meta,protected,sources,profile,plan,history,best)


def verify_best(cfg,val,audit,meta,protected,sources,profile,plan,history,best):
    path=ROOT/'best.pth'
    if sha(path)!=best['checkpoint_sha256']:raise ValueError('Best alias is not selected epoch bytes')
    stored=torch.load(path,map_location='cpu',weights_only=True)
    if stored['plan']!=plan or stored['epoch']!=best['epoch'] or stored['pretrained_sha256']!=meta['sha256']:raise ValueError('Best metadata differs')
    model=build_model(audit['train_study_priors'],'cuda');model.load_state_dict(stored['state_dict'],strict=True)
    p,extra=predict(model,val,cfg,True)
    with np.load(best['prediction'],allow_pickle=False) as saved:
        for key,value in (('targets',val.targets),('subjects',val.subjects),('study_ids',val.study_ids),('image_paths',extra['image_paths']),('image_study_indices',extra['image_study_indices'])):np.testing.assert_array_equal(saved[key],value)
        difference=float(np.max(np.abs(p-saved['probabilities'])));np.testing.assert_allclose(p,saved['probabilities'],rtol=0,atol=1e-7)
        image_difference=float(np.max(np.abs(extra['image_probabilities']-saved['image_probabilities'])));np.testing.assert_allclose(extra['image_probabilities'],saved['image_probabilities'],rtol=0,atol=1e-7)
    measured=metrics(val.targets,p);cal=calibration(val.targets,p,val.subjects)
    if measured!=best['validation'] or cal!=best['calibration']:raise ValueError('Best reloaded metrics differ')
    receipt=dict(state='passed',epoch=best['epoch'],checkpoint=str(path),checkpoint_sha256=sha(path),prediction=best['prediction'],prediction_sha256=sha(best['prediction']),max_abs_probability_difference=difference,max_abs_image_probability_difference=image_difference,all_raw_validation_images=1422,all_validation_studies=859)
    save(ROOT/'best_reload_verification.json',receipt)
    gate=ROOT/'gates.json';save(gate,dict(**cal,checkpoint=str(path),checkpoint_sha256=sha(path),best_epoch=best['epoch'],class_ids=list(range(1,11)),pretrained_sha256=meta['sha256'],candidate_only=True,gate_fit_source='Full official validation; patient OOF threshold metrics separately'))
    if source_hashes()!=sources or check_protected()!=protected:raise ValueError('Source/protected changed at completion')
    result=dict(state='complete',epochs_completed=5,total_new_epochs=5,history=history,best_epoch=best['epoch'],best_validation=measured,best_calibration=cal,best_checkpoint=str(path),best_checkpoint_sha256=sha(path),gates=str(gate),gates_sha256=sha(gate),verification=receipt,
        common=COMMON,resource_profile=profile,pretrained_sha256=meta['sha256'],pretrained_manifest_sha256=sha(ROOT/'pretrained_manifest.json'),source_hashes=sources,protected_hashes_after=protected,full_backbone_training=True,previous_cxr_checkpoint_used=False,old_image_or_feature_cache_used=False,no_test_inference=True,github_uploaded=False)
    save(ROOT/'results.json',result);status(state='complete');print('NATIVE_FIVE_EPOCH_COMPLETE '+str(dict(best_epoch=best['epoch'],auc=measured['macro_auc'],oof_f1=cal['oof_macro_f1'])),flush=True)


def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    protected=check_protected();sources=source_hashes();cfg,rows,records,audit,meta=load_inputs()
    if args.check_only:preflight(cfg,records,audit,meta,protected,sources);return
    checked=read(ROOT/'preflight.json')
    if checked['state']!='passed' or checked['source_hashes']!=sources or checked['protected_hashes']!=protected or checked['common']!=COMMON or checked['data_audit']!=audit:raise ValueError('CPU preflight stale')
    if not torch.cuda.is_available():raise RuntimeError('Refuse CPU training')
    if args.smoke:smoke(cfg,records,audit,meta,protected,sources);return
    profile=read(ROOT/'resource_profile.json')
    if profile['state']!='passed' or profile['source_hashes']!=sources or profile['protected_hashes']!=protected or profile['common']!=COMMON or profile['pretrained_sha256']!=meta['sha256']:raise ValueError('Resource smoke stale')
    run(cfg,records,audit,meta,protected,sources,profile)

@contextmanager
def lock():
    with (ROOT/'gpu.lock').open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

if __name__=='__main__':
    with lock():
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:main(ARGS)
        except KeyboardInterrupt:status(state='interrupted',partial_epoch_not_counted=True);raise
        except Exception as exc:status(state='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),partial_epoch_not_counted=True);raise
        finally:
            if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
