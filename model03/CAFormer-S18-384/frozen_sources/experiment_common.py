"""E-drive-only registered augmentation, loss and depth experiments."""
from __future__ import annotations
import csv,hashlib,json,os,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
if ROOT.drive.upper()!='E:':raise RuntimeError('Experiment must remain on E drive')
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','HF_XET_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
 p=ROOT/'temp'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
COMMON=dict(workers=2,prefetch_factor=1,persistent_workers=False)
RAW=DATA/'raw/cxr_first';MANIFEST=DATA/'喻数据产物/manifest.csv'
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
OLD=DATA/'CAFormer384_三项优化_20261106'
INCUMBENT=OLD/'A_BCE_EMA/best_ema.pth'
INCUMBENT_SHA='f1f862bd6c76235e59e96b17b4bdae92aa497846b4eef42e8fb33b2b34ddee85'
INCUMBENT_PRED=OLD/'A_BCE_EMA/predictions/ema_epoch_01.npz'
INCUMBENT_PRED_SHA='84de88e32b80f1240f6f1e476d6eda6d6ca4850fae95ab10f122d855f15f99cc'
INCUMBENT_AUC=.8061307765133356
GLOBAL_LOCK=OLD/'gpu.lock'
CONFIG=dict(seed=42,image_size=384,mean=[.485,.456,.406],std=[.229,.224,.225],epochs_per_full_arm=5,
 initial='Fresh pinned author ImageNet1K native384 per architecture, only final fc2 reset10 with official train Study priors; no CXR checkpoint initialization',
 baseline_arm='A0_BCE_EMA',augmentation_arms=['A1_noise','A2_photo','A3_photo_noise'],
 screening_epochs=2,augmentation_winner_total_epochs=5,loss_arm='B_ASL',depth_arm='C_S36',
 expected_full_new_epochs=24,expected_validation_evaluations=48,
 effective_study_batch=32,common_real_batch_candidates=[16,8,4,2],memory_margin=.85,
 backbone_hidden_lr=5e-5,new_classifier_lr=5e-4,weight_decay=.05,bias_norm_weight_decay=0.,
 hidden_dropout=.2,outer_dropout=0.,drop_path=.1,gradient_checkpointing=True,
 schedule='579-attempt epoch1 linear .1..1; next4epochs cosine1 to0.1; AMP skipped attempts still advance schedule',
 amp_init_scale=1024,amp='FP16 forward; stable FP32 Study mean-probability loss',
 EMA='min(.999,(1+successful_updates)/(10+successful_updates)); independent shadow, successful optimizer steps only',
 aggregation='Complete per-image classifier -> FP32 sigmoid -> FP32 Study mean',
 selection='MacroAUC primary, F1@.5 tie, earlier epoch final tie; raw/EMA selected independently',
 augmentation_screen_selection='max best raw/EMA AUC over2 full epochs; tie F1@.5 then fixed noise/photo/combo order; winner resumed to5 without resetting optimizer/EMA/schedule',
 policy_for_ASL='winner augmentation only if its best5 beats baseline best5, else original affine; common fresh initialization and matched BCE control',
 depth_recipe='S36 ImageNet1K native384, full5, augmentation/loss of best full5 S18 recipe, matched effective batch/schedule/EMA',
 ASL=dict(gamma_neg=2.,gamma_pos=0.,clip=.05,disable_focal_grad=True,reduction='none'),
 F1='Patient GroupKFold5 shuffleTrue seed42, grid.025...975, tie closest.5 then lower; fullval Gates separate',
 bootstrap=dict(attempts=400,seed=20261104,unit='patient cluster paired',selection_repeated=False),
 train=dict(images=30344,studies=18513,patients=11055),
 val=dict(images=1422,studies=859,patients=228),
 no_test_inference=True,no_GitHub_upload=True,loader=COMMON)
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
def sources():
 import timm
 paths=[ROOT/x for x in ('experiment_common.py','trainer.py','training_utils.py','data_pipeline.py','augmentation.py','asl_study.py','models.py')]
 paths.append(Path(timm.__file__).parent/'models/metaformer.py')
 return {str(p):sha(p) for p in paths}
def check_protected():
 values=read(ROOT/'protected_inputs.json')
 if len(values)<600:raise ValueError('Historical protection snapshot incomplete')
 for p,h in values.items():
  if sha(p)!=h:raise ValueError('Historical model, code or submission changed: '+p)
 return dict(files=len(values),manifest_sha256=sha(ROOT/'protected_inputs.json'),all_sha256_matched=True)
def load_inputs():
 if sha(MANIFEST)!=MANIFEST_SHA or sha(INCUMBENT)!=INCUMBENT_SHA or sha(INCUMBENT_PRED)!=INCUMBENT_PRED_SHA:raise ValueError('Official manifest or preserved incumbent changed')
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
