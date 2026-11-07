"""Unmodified numerical utilities extracted from original native384 training; isolated runtime config."""
from __future__ import annotations
import math,random,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from data_pipeline import RawImageDataset,collate_studies,worker_init
from experiment_common import COMMON


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

