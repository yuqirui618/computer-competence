"""Gamma/noise pilots on the full official training split; E-drive outputs only.

CPU-only --check-only is the launch gate. Each recipe starts at the same protected
champion, trains at most two epochs, and evaluates deterministic raw validation.
"""
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TEMP=ROOT/'temp';TEMP.mkdir(exist_ok=True)
for key,value in {'TEMP':TEMP,'TMP':TEMP,'TMPDIR':TEMP,'XDG_CACHE_HOME':TEMP,
                  'CUDA_CACHE_PATH':TEMP/'cuda','HF_HOME':TEMP/'huggingface','TORCH_HOME':TEMP/'torch'}.items():
    Path(value).mkdir(parents=True,exist_ok=True);os.environ[key]=str(value)
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import argparse,csv,ctypes,gc,hashlib,json,math,msvcrt,time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
DATA=ROOT.parent
sys.path.insert(0,str(DATA/'本地跑通'))
sys.path.insert(0,str(DATA/'MambaOut_泛化优化'))
sys.path.insert(0,str(DATA/'MambaOut_100轮'))
import train_generalization as core
import train_and_optimize as ref
from dataset.tools.build_cache import cache_image_path
CHAMPION=DATA/'MambaOut_ASL对照/best_model.pth'
GATES=DATA/'MambaOut_ASL对照/gates.json'
SUBMISSION=DATA/'提交结果/MambaOut_S7微调第1轮_20261104/submission.csv'
PREVIOUS=DATA/'MambaOut_F1正则化优化_20261104'
EXPECTED_CHAMPION='9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585'
EXPECTED_SUBMISSION='d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8'
EXPECTED_MANIFEST='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
BASELINE_AUC=.8019769550500871
PREVIOUS_ENSEMBLE_AUC=.8020562026911335
COMMON=dict(architecture='mambaout_tiny',img_size=224,batch_size=12,workers=0,seed=42,
            train_images=30344,backbone_lr=1e-6,classifier_lr=1e-5,weight_decay=.05,
            drop_rate=.2,drop_path_rate=.1,loss='BCE',pos_weight=[1.0]*10,max_epochs=2,
            lr_factors=[1.0,.5],initial_checkpoint=str(CHAMPION),initial_checkpoint_sha256=EXPECTED_CHAMPION,
            manifest_sha256=EXPECTED_MANIFEST,train_source='cache512_JPEG92',validation_source='raw',
            extra_augmentation_domain='float32 RGB tensor [0,1] after affine/ToTensor, before Normalize',
            private_rng='gamma numpy seed420000+epoch; noise choice numpy seed430000+epoch; noise torch CPU seed440000+epoch')
RECIPES={
 'A_control':dict(gamma_probability=0.0,gamma_range=[.85,1.15],noise_probability=0.0,noise_sigma=[.005,.015]),
 'B_gamma':dict(gamma_probability=.5,gamma_range=[.85,1.15],noise_probability=0.0,noise_sigma=[.005,.015]),
 'C_gamma_noise':dict(gamma_probability=.5,gamma_range=[.85,1.15],noise_probability=.25,noise_sigma=[.005,.015])}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,obj):core.save_json(Path(path),obj)
def status(**fields):save(ROOT/'status.json',dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),**fields))
def freeze(path,obj):
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8'))!=obj:raise ValueError(f'Frozen settings changed: {path.name}')
    else:save(path,obj)
def pair(metrics):return metrics['study']['macro_auc'],metrics['study']['macro_f1']
def protected_hashes(cfg):
    actual={str(path):sha(path) for path in (CHAMPION,GATES,SUBMISSION,Path(cfg['manifest']),PREVIOUS/'baseline_predictions.npz',PREVIOUS/'baseline_metrics.json')}
    for path,digest in ((CHAMPION,EXPECTED_CHAMPION),(SUBMISSION,EXPECTED_SUBMISSION),(Path(cfg['manifest']),EXPECTED_MANIFEST)):
        if actual[str(path)]!=digest:raise ValueError(f'Protected SHA differs: {path.name}')
    receipt=ROOT/'protected_inputs.json'
    freeze(receipt,actual)
    return actual

class ExposureAugment:
    """Private RNGs preserve original affine/global dropout RNG consumption."""
    def __init__(self,recipe):self.recipe=recipe;self.reset(0)
    def reset(self,epoch):
        self.gamma_rng=np.random.default_rng(420000+epoch)
        self.noise_rng=np.random.default_rng(430000+epoch)
        self.pixel_rng=torch.Generator(device='cpu').manual_seed(440000+epoch)
        self.counts=dict(images=0,gamma_applied=0,noise_applied=0,gamma_sum=0.0,noise_sigma_sum=0.0)
    def __call__(self,x):
        self.counts['images']+=1
        if self.recipe['gamma_probability'] and self.gamma_rng.random()<self.recipe['gamma_probability']:
            gamma=float(self.gamma_rng.uniform(*self.recipe['gamma_range']))
            x=TF.adjust_gamma(x,gamma=gamma,gain=1)
            self.counts['gamma_applied']+=1;self.counts['gamma_sum']+=gamma
        if self.recipe['noise_probability'] and self.noise_rng.random()<self.recipe['noise_probability']:
            sigma=float(self.noise_rng.uniform(*self.recipe['noise_sigma']))
            noise=torch.randn((1,x.shape[-2],x.shape[-1]),generator=self.pixel_rng,dtype=x.dtype)*sigma
            mask=(x.amax(dim=0,keepdim=True)>0).to(x.dtype)
            x=(x+noise*mask).clamp(0,1)
            self.counts['noise_applied']+=1;self.counts['noise_sigma_sum']+=sigma
        return x

def training_transform(recipe):
    transform=ref.build_transform(True,224)
    effect=ExposureAugment(recipe)
    position=next(i for i,t in enumerate(transform.transforms) if isinstance(t,transforms.Normalize))
    transform.transforms.insert(position,effect)
    return transform,effect

def load_data(cfg):
    train=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'train',ref.build_transform(True,224),cfg['cache_dir'])
    val=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'val',ref.build_transform(False,224))
    if len(train)!=30344 or len(val)!=1422:raise ValueError('Official image counts changed')
    with Path(cfg['manifest']).open(encoding='utf-8-sig',newline='') as f:
        patient_sets={name:set() for name in ('train','val','test')}
        for row in csv.DictReader(f):patient_sets[row['split']].add(row['subject_id'])
    for a,b in (('train','val'),('train','test'),('val','test')):
        if patient_sets[a]&patient_sets[b]:raise ValueError('Official patient split leakage')
    if tuple(len(patient_sets[k]) for k in ('train','val','test'))!=(11055,228,136):raise ValueError('Patient counts differ')
    for row in train.rows:
        if not cache_image_path(Path(cfg['cache_dir']),row['image_path']).is_file():raise FileNotFoundError('Training cache absent')
    grouped=OrderedDict()
    for row,labels in zip(val.rows,val.labels):
        info=(row['subject_id'],labels)
        if row['study_id'] in grouped and grouped[row['study_id']]!=info:raise ValueError('Validation study inconsistent')
        grouped[row['study_id']]=info
    if len(grouped)!=859:raise ValueError('Validation Study count differs')
    ids=np.array(list(grouped),dtype=str)
    expected_y=np.zeros((859,10),dtype=np.float32)
    for i,(_,labels) in enumerate(grouped.values()):expected_y[i,list(labels)]=1
    expected_s=np.array([v[0] for v in grouped.values()],dtype=str)
    with np.load(PREVIOUS/'baseline_predictions.npz',allow_pickle=False) as a:
        baseline=(a['targets'].copy(),a['probabilities'].copy(),a['subjects'].astype(str))
    if not np.array_equal(expected_y,baseline[0]) or not np.array_equal(expected_s,baseline[2]):raise ValueError('Validation order/labels changed')
    if abs(roc_auc_score(baseline[0],baseline[1],average='macro')-BASELINE_AUC)>1e-12:raise ValueError('Baseline AUC differs')
    return train,val,ids,baseline

def cpu_check(cfg,train,val,ids,baseline,protected):
    torch.set_num_threads(4)
    # Synthetic shape/range/channel/padding checks, forced probability only for unit check.
    ramp=torch.linspace(.1,.9,224).reshape(1,1,224).expand(3,224,224).clone();ramp[:,:,:24]=0
    forced=ExposureAugment(dict(RECIPES['C_gamma_noise'],gamma_probability=1.,noise_probability=1.))
    global_before=torch.get_rng_state().clone();out=forced(ramp)
    if not torch.equal(global_before,torch.get_rng_state()):raise ValueError('Extra augmentation consumed global CPU RNG')
    if not torch.equal(out[0],out[1]) or not torch.equal(out[1],out[2]):raise ValueError('Gray noise introduced RGB channel differences')
    if out[:,:,:24].count_nonzero().item() or out.min()<0 or out.max()>1:raise ValueError('Padding/range corrupted')
    gamma_a=ExposureAugment(RECIPES['B_gamma']);gamma_b=ExposureAugment(dict(RECIPES['C_gamma_noise'],noise_probability=0.))
    for _ in range(100):
        if not torch.equal(gamma_a(ramp),gamma_b(ramp)):raise ValueError('B and C gamma RNG streams differ')
    maes={};shapes={};coverage={}
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    picks=np.linspace(0,len(train)-1,24,dtype=int)
    original=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'train',ref.build_transform(True,224),cfg['cache_dir'])
    for name,recipe in RECIPES.items():
        transform,effect=training_transform(recipe);effect.reset(1)
        altered=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'train',transform,cfg['cache_dir'])
        diffs=[]
        for n,index in enumerate(picks):
            torch.manual_seed(7000+n);before=original[int(index)]
            torch.manual_seed(7000+n);after=altered[int(index)]
            if not torch.equal(before['target'],after['target']):raise ValueError('Augmentation altered labels')
            x=after['image']*std+mean
            if after['image'].shape!=(3,224,224) or not torch.isfinite(x).all() or x.min() < -1e-6 or x.max()>1+1e-6:raise ValueError('Augmentation output invalid')
            diffs.append(float(((after['image']-before['image'])*std).abs().mean()))
        maes[name]=dict(mean_absolute_difference_0_1=float(np.mean(diffs)),max_image_mean_absolute_difference=float(np.max(diffs)),paired_images=len(picks))
        shapes[name]=[3,224,224];coverage[name]=effect.counts.copy()
    if maes['A_control']['max_image_mean_absolute_difference']!=0:raise ValueError('Control preprocessing differs')
    check=ExposureAugment(RECIPES['C_gamma_noise']);check.reset(1)
    small=torch.ones((3,8,8))*.5
    for _ in range(5000):check(small)
    if not 2200<check.counts['gamma_applied']<2800 or not 1000<check.counts['noise_applied']<1500:raise ValueError('Augmentation application frequency unexpected')
    old=json.loads((PREVIOUS/'A_current_regularization/results.json').read_text(encoding='utf-8'))['settings']
    matched={key:old[key]==COMMON[key] for key in ('architecture','img_size','batch_size','seed','train_images','backbone_lr','classifier_lr','weight_decay','drop_rate','drop_path_rate','pos_weight','manifest_sha256','initial_checkpoint_sha256','max_epochs')}
    if not all(matched.values()):raise ValueError('Historical control configuration mismatch')
    result=dict(state='passed',CPU_only=True,protected_hashes=protected,train_images=30344,val_images=1422,val_studies=859,val_patients=228,
                train_cache_complete=True,patient_disjoint=True,labels_unchanged=True,private_rng_verified=True,
                gray_channels_shared=True,padding_zero_preserved=True,range_0_1=True,
                augmentation_MAE=maes,sample_application_counts=coverage,synthetic_5000_application_counts=check.counts,
                historical_control_settings_match=matched,torch=str(torch.__version__))
    if protected_hashes(cfg)!=protected:raise ValueError('Protected file changed during preflight')
    save(ROOT/'preflight.json',result)
    print(json.dumps(dict(preflight='passed',CPU_only=True,augmentation_MAE=maes,synthetic_counts=check.counts),ensure_ascii=False),flush=True)

def save_pred(path,pred,ids):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,targets=pred[0],probabilities=pred[1],subjects=pred[2],study_ids=ids)
    core.replace_file(tmp,path)
def load_pred(path):
    with np.load(path,allow_pickle=False) as a:return a['targets'].copy(),a['probabilities'].copy(),a['subjects'].copy()
def extras(pred,old_gates):
    y,p,s=pred;z=p>=.5
    return dict(macro_ap=float(average_precision_score(y,p,average='macro')),
      ap_by_class=average_precision_score(y,p,average=None).tolist(),
      macro_precision=float(precision_score(y,z,average='macro',zero_division=0)),macro_recall=float(recall_score(y,z,average='macro',zero_division=0)),
      tp=np.sum(z&(y==1),axis=0).tolist(),fp=np.sum(z&(y==0),axis=0).tolist(),fn=np.sum(~z&(y==1),axis=0).tolist(),
      f1_by_class=f1_score(y,z,average=None,zero_division=0).tolist(),
      old_frozen_gates_f1=float(f1_score(y,p>=old_gates,average='macro',zero_division=0)),
      old_gate_evaluation_note='Old champion gates on reused validation; not refitted and not an independent test.')

def train_epoch(model,data,effect,optimizer,scaler,name,epoch,completed):
    core.seed(42+epoch);effect.reset(epoch);model.train()
    factor=COMMON['lr_factors'][epoch-1]
    optimizer.param_groups[0]['lr']=COMMON['backbone_lr']*factor;optimizer.param_groups[1]['lr']=COMMON['classifier_lr']*factor
    loader=DataLoader(data,batch_size=12,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(42+epoch))
    weight=torch.ones(10,device='cuda',dtype=torch.float32)
    seen=0;total=0.;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step,batch in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        x=batch['image'].to('cuda',non_blocking=True,memory_format=torch.channels_last);y=batch['target'].to('cuda',non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y,pos_weight=weight)
        value=loss.item()
        if not math.isfinite(value):raise FloatingPointError('Nonfinite loss')
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        seen+=len(x);total+=value*len(x)
        if step==1 or step%250==0 or step==len(loader):
            status(state='training',stage=name,epoch=epoch,max_epochs=2,images=seen,total_images=len(data),completed_epochs=completed)
            print(f'TRAIN {name} epoch={epoch}/2 images={seen}/{len(data)} BCE={total/seen:.7f}',flush=True)
    torch.cuda.synchronize()
    if seen!=30344 or effect.counts['images']!=30344:raise ValueError('Incomplete epoch coverage')
    return dict(epoch=epoch,images=seen,train_loss=total/seen,train_seconds=time.perf_counter()-started,
                learning_rates=[g['lr'] for g in optimizer.param_groups],augmentation_counts=effect.counts.copy(),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)

def run_stage(name,recipe,cfg,val,ids,initial,baseline_metrics,baseline_pred,old_gates,completed):
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    settings={**COMMON,**recipe,'name':name};freeze(folder/'settings.json',settings)
    transform,effect=training_transform(recipe)
    train=ref.CXRDataset(cfg['manifest'],cfg['data_root'],'train',transform,cfg['cache_dir'])
    last=folder/'last_training.pth';state=torch.load(last,map_location='cpu',weights_only=True) if last.is_file() else None
    if state is not None and state['settings']!=settings:raise ValueError('Resume settings changed')
    model=core.build_model(COMMON,initial if state is None else state['state_dict'])
    optimizer=core.make_optimizer(model,COMMON);scaler=torch.amp.GradScaler('cuda')
    history=[];best_epoch=0;best_metrics=baseline_metrics
    if state is not None:
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        history=state['history'];best_epoch=state['best_epoch'];best_metrics=state['best_metrics']
        if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>2:raise ValueError('Invalid resume history')
        for row in history:
            if not (folder/f"epoch_{row['epoch']}.pth").is_file() or not (folder/f"epoch_{row['epoch']}_predictions.npz").is_file():raise ValueError('Missing committed epoch artifact')
        expected=max([(0,pair(baseline_metrics))]+[(r['epoch'],pair(r['validation'])) for r in history],key=lambda x:x[1])[0]
        if expected!=best_epoch:raise ValueError('Committed best epoch disagrees with history')
        del state
    if best_epoch:
        checkpoint=torch.load(folder/f'epoch_{best_epoch}.pth',map_location='cpu',weights_only=True)
        core.save_checkpoint(folder/'best.pth',checkpoint);del checkpoint
    else:core.save_best(folder/'best.pth',initial,settings,0)
    completed[name]=len(history)
    for epoch in range(len(history)+1,3):
        if history and history[-1]['validation']['study']['macro_auc']<BASELINE_AUC-.01:break
        record=train_epoch(model,train,effect,optimizer,scaler,name,epoch,completed)
        status(state='validation',stage=name,epoch=epoch,images=0,total_images=1422,completed_epochs=completed)
        started=time.perf_counter();metrics,pred=ref.validate(model,val,keep_predictions=True)
        if not np.array_equal(pred[0],baseline_pred[0]) or not np.array_equal(pred[2],baseline_pred[2]):raise ValueError('Prediction alignment changed')
        record.update(validation=metrics,validation_seconds=time.perf_counter()-started,calibration=ref.evaluate_gates(pred),extra=extras(pred,old_gates))
        weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
        core.save_best(folder/f'epoch_{epoch}.pth',weights,settings,epoch);save_pred(folder/f'epoch_{epoch}_predictions.npz',pred,ids)
        if pair(metrics)>pair(best_metrics):
            best_epoch=epoch;best_metrics=metrics;core.save_best(folder/'best.pth',weights,settings,epoch)
        history.append(record)
        core.save_checkpoint(last,dict(state_dict=weights,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),settings=settings,history=history,best_epoch=best_epoch,best_metrics=best_metrics))
        save(folder/'history.json',history);completed[name]=epoch
        status(state='checkpointed',stage=name,epoch=epoch,completed_epochs=completed,auc=pair(metrics)[0],f1=pair(metrics)[1],oof_f1=record['calibration']['oof_macro_f1'],best_epoch=best_epoch)
        print(f"EPOCH_DONE {name} epoch={epoch} AUC={pair(metrics)[0]:.10f} F1={pair(metrics)[1]:.10f} OOF_F1={record['calibration']['oof_macro_f1']:.10f} best_epoch={best_epoch}",flush=True)
        del weights,pred
        if (ROOT/'pause_after_epoch.flag').is_file():
            status(state='paused_after_epoch',stage=name,epoch=epoch,completed_epochs=completed);raise SystemExit(0)
    trained=max(history,key=lambda r:pair(r['validation']))
    result=dict(settings=settings,epochs_completed=len(history),best_epoch_including_original=best_epoch,best_validation=best_metrics,
                best_checkpoint=str(folder/'best.pth'),best_trained_epoch=trained['epoch'],best_trained_checkpoint=str(folder/f"epoch_{trained['epoch']}.pth"),
                best_trained_validation=trained['validation'],best_trained_calibration=trained['calibration'],history=history,
                stop_reason='auc_drop_guard' if len(history)<2 else 'fixed_two_epoch_budget')
    save(folder/'results.json',result)
    del model,optimizer,scaler,train;gc.collect();torch.cuda.empty_cache()
    return result

def run(check_only=False):
    if ROOT.drive.upper()!='E:':raise ValueError('E drive required')
    cfg=yaml.safe_load((DATA/'本地跑通/data_local.yaml').read_text(encoding='utf-8'))
    protected=protected_hashes(cfg)
    train,val,ids,baseline_pred=load_data(cfg)
    if check_only:cpu_check(cfg,train,val,ids,baseline_pred,protected);return
    if not (ROOT/'preflight.json').is_file() or json.loads((ROOT/'preflight.json').read_text(encoding='utf-8'))['state']!='passed':raise ValueError('CPU preflight is required')
    if (ROOT/'results.json').is_file():
        if json.loads((ROOT/'results.json').read_text(encoding='utf-8'))['state']!='complete':raise ValueError('Unexpected root result')
        print('ALREADY_COMPLETE; protected source hashes checked',flush=True);return
    if (ROOT/'pause_after_epoch.flag').is_file():status(state='paused_before_start');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; refusing accidental CPU training')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;ref.ARGS.batch_size=12
    plan=dict(common=COMMON,recipes=RECIPES,max_total_training_epochs=6,original_single_auc=BASELINE_AUC,
              earlier_exploratory_probability_ensemble_auc=PREVIOUS_ENSEMBLE_AUC,
              selection='Study Macro AUC primary, F1@0.5 secondary; original epoch0 remains protected; separately report best trained epoch',
              stop='At most two epochs per recipe; stop recipe if preceding AUC drops >0.01 from original',
              validation='Raw full official validation; Study mean probability; original coarse 5fold patient gates; no test inference',
              limits=['Single seed, short low-LR fine-tuning, repeated official validation; not an external test',
                      'Augmentation alone changes effective difficulty; null result is not an architecture upper bound',
                      'Epoch checkpoints label their own epoch; true best is in results.json'])
    freeze(ROOT/'plan.json',plan)
    checkpoint=torch.load(CHAMPION,map_location='cpu',weights_only=True)
    if checkpoint['training']['img_size']!=224 or checkpoint['training']['manifest_sha256']!=EXPECTED_MANIFEST:raise ValueError('Original weight metadata changed')
    initial=checkpoint['state_dict'];del checkpoint
    original_metrics=json.loads((PREVIOUS/'baseline_metrics.json').read_text(encoding='utf-8'))['validation']
    baseline_model=core.build_model(COMMON,initial).eval()
    status(state='baseline_verification',images=0,total_images=1422)
    baseline_metrics,new_pred=ref.validate(baseline_model,val,keep_predictions=True)
    for level in ('image','study'):
        for key in ('auc','f1','macro_auc','macro_f1'):
            if not np.allclose(baseline_metrics[level][key],original_metrics[level][key],rtol=0,atol=1e-8):raise ValueError('Original metrics failed to reproduce')
    if not np.allclose(new_pred[1],baseline_pred[1],rtol=0,atol=1e-7) or not np.array_equal(new_pred[0],baseline_pred[0]):raise ValueError('Baseline prediction failed to reproduce')
    baseline_pred=new_pred;save_pred(ROOT/'baseline_predictions.npz',baseline_pred,ids)
    gates=json.loads(GATES.read_text(encoding='utf-8'));old_gates=np.asarray(gates['final_gates_by_class_id'])
    save(ROOT/'baseline_metrics.json',dict(validation=baseline_metrics,calibration=ref.evaluate_gates(baseline_pred),extra=extras(baseline_pred,old_gates)))
    del baseline_model;gc.collect();torch.cuda.empty_cache()
    stages={};completed={}
    for name,recipe in RECIPES.items():stages[name]=run_stage(name,recipe,cfg,val,ids,initial,baseline_metrics,baseline_pred,old_gates,completed)
    selected=max(stages,key=lambda name:pair(stages[name]['best_validation']))
    improvement=pair(stages[selected]['best_validation'])>pair(baseline_metrics)
    chosen=stages[selected]['best_checkpoint'] if improvement else str(CHAMPION)
    control_name='A_control';control_epoch=stages[control_name]['best_trained_epoch']
    control_pred=load_pred(ROOT/control_name/f'epoch_{control_epoch}_predictions.npz')
    intervals={}
    for name,stage in stages.items():
        pred=load_pred(ROOT/name/f"epoch_{stage['best_trained_epoch']}_predictions.npz")
        intervals[name]={'versus_original':core.paired_auc_interval(baseline_pred[0],baseline_pred[1],pred[1],pred[2],repeats=400),
                         'versus_retrained_control':core.paired_auc_interval(control_pred[0],control_pred[1],pred[1],pred[2],repeats=400)}
    if protected_hashes(cfg)!=protected:raise ValueError('Protected original changed')
    result=dict(state='complete',new_training_epochs=sum(completed.values()),completed_epochs=completed,stages=stages,
                winner_including_original=selected if improvement else 'original_champion',selected_checkpoint=chosen,selected_sha256=sha(chosen),
                original_validation=baseline_metrics,original_calibration=gates,paired_auc_intervals=intervals,
                surpasses_original_single_auc=pair(stages[selected]['best_validation'])[0]>BASELINE_AUC,
                surpasses_prior_exploratory_ensemble_auc=pair(stages[selected]['best_validation'])[0]>PREVIOUS_ENSEMBLE_AUC,
                preserved_hashes_after=protected,original_files_overwritten=False,new_test_inference=False,github_uploaded=False,
                final_candidate_gpu_reload_verification='Root performs if a trained candidate improves; this runner preserves epoch weights and predictions')
    save(ROOT/'results.json',result)
    status(state='complete',completed_epochs=completed,total_new_epochs=sum(completed.values()),winner=result['winner_including_original'],auc=pair(stages[selected]['best_validation'])[0])
    print('COMPLETE '+json.dumps(dict(winner=result['winner_including_original'],epochs=result['new_training_epochs'],auc=pair(stages[selected]['best_validation'])[0]),ensure_ascii=False),flush=True)

@contextmanager
def lock():
    with (ROOT/'training.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    with lock():
        awake=0 if args.check_only else ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            if not args.check_only:status(state='starting')
            run(args.check_only)
        except Exception as exc:
            status(state='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
