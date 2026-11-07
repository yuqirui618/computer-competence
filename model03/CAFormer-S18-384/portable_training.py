"""Portable training math copied from the frozen experiment; local output only.
No dependency on the original experiment folder, old champions or protection lists.
"""
from __future__ import annotations
import copy,csv,hashlib,json,math,os,random,shutil,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from sklearn.model_selection import GroupKFold
from data_pipeline import RawStudyDataset,RawImageDataset,collate_studies,worker_init
from augmentation import AugmentedStudyDataset
from portable_model import build_model,optimizer_for,SOURCE_SHA
ROOT=None
COMMON=dict(workers=2,prefetch_factor=1,persistent_workers=False)
CONFIG=dict(mean=[.485,.456,.406],std=[.229,.224,.225])
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
STEPS=math.ceil(18513/32)
GROUPS=('stem.','stages.0.','stages.1.','stages.2.','stages.3.','head.norm.','head.fc.fc1.','head.fc.norm.','head.fc.fc2.')
POLICIES={'A0_BCE_EMA':'none','A2_photo':'photo'}


def configure_output(output):
    global ROOT
    ROOT=Path(output).expanduser().resolve();ROOT.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
        p=ROOT/'cache'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)

def freeze(path,value):
 if Path(path).exists():
  if read(path)!=value:raise ValueError('Frozen registration changed: '+str(path))
 else:save(path,value)

def save_npz(path,**arrays):
 import numpy as np
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('wb') as stream:np.savez_compressed(stream,**arrays)
 q.replace(p)

def status(**fields):save(ROOT/'status.json',dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),**fields))

def setup():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')

def note(event,**fields):
    value=dict(time=datetime.now().astimezone().isoformat(),event=event,**fields)
    line=__import__('json').dumps(value,ensure_ascii=False,allow_nan=False)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)

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

def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}

def best_evaluations(history):
    best={}
    for row in history:
        for name,e in row['evaluations'].items():
            if name not in best or key(e)>key(best[name]):best[name]=e
    return best

def key(evaluation):return evaluation['metrics']['macro_auc'],evaluation['metrics']['macro_f1']

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

def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)

def checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def copy_checkpoint(source,target):
    target=Path(target);tmp=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(source,tmp);tmp.replace(target)

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

def load_inputs(manifest,raw_root):
 MANIFEST=Path(manifest).expanduser().resolve();RAW=Path(raw_root).expanduser().resolve()
 if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Expected the official manifest with the pinned SHA256')
 rows={s:[] for s in ('train','val','test')};groups={s:{} for s in rows};patients={s:set() for s in rows};paths=set()
 with MANIFEST.open(encoding='utf-8-sig',newline='') as stream:
  for r in csv.DictReader(stream):
   s=r['split'];path=r['image_path'];sub=r['subject_id'];study=r['study_id']
   labels=[int(v)-1 for v in r['label_ids'].split('|') if v]
   if s not in rows or len(labels)!=len(set(labels)) or any(v<0 or v>=10 for v in labels) or bool(labels)!=(s!='test'):raise ValueError('Invalid label/split')
   p=(RAW/path).resolve()
   if path in paths or not p.is_relative_to(RAW.resolve()):raise ValueError('Duplicate/escaping image')
   if s!='test' and not p.is_file():raise FileNotFoundError(p)
   paths.add(path);rows[s].append(r);patients[s].add(sub);target=[float(i in labels) for i in range(10)]
   if study not in groups[s]:groups[s][study]=dict(subject_id=sub,study_id=study,image_paths=[],image_positions=[],target=target)
   rec=groups[s][study]
   if rec['subject_id']!=sub or rec['target']!=target:raise ValueError('Study identity/label mismatch')
   rec['image_paths'].append(path);rec['image_positions'].append(len(rows[s])-1)
 records={s:list(g.values()) for s,g in groups.items()}
 expected={'train':(30344,18513,11055),'val':(1422,859,228),'test':(2218,1414,136)}
 for s in rows:
  if (len(rows[s]),len(records[s]),len(patients[s]))!=expected[s]:raise ValueError('Official coverage mismatch')
 for a,b in (('train','val'),('train','test'),('val','test')):
  if patients[a]&patients[b]:raise ValueError('Patient leakage')
 pos=[sum(r['target'][i] for r in records['train']) for i in range(10)]
 if pos!=[622,900,942,1436,2404,3885,3968,4508,4654,6683]:raise ValueError('Train Study class counts mismatch')
 priors=[max(.001,min(.999,n/18513)) for n in pos]
 audit=dict(coverage={s:dict(images=len(rows[s]),studies=len(records[s]),patients=len(patients[s])) for s in rows},patient_splits_disjoint=True,
  train_positive_Study_counts=pos,train_priors=priors,test_metadata_only=True)
 cfg=dict(data_root=str(RAW),mean=CONFIG['mean'],std=CONFIG['std'],image_size=384)
 return cfg,records,audit
