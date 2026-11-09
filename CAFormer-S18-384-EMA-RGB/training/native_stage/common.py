"""Portable runtime paths and bounded source guards; no historical machine paths."""
from __future__ import annotations
import csv,hashlib,json,os,sys
from pathlib import Path
from datetime import datetime
PACKAGE=Path(__file__).resolve().parent
RECIPE=json.loads((PACKAGE/'recipe.json').read_text(encoding='utf-8'))
COMMON=RECIPE['COMMON'].copy();EXPECTED=RECIPE['EXPECTED']
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
NATIVE_SHA='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
ROOT=DATA=RAW=MANIFEST=CONFIG=WEIGHTS=None
_INPUTS={}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Frozen input differs: '+str(path))
    else:save(path,value)
def save_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)

def configure(args):
    global ROOT,DATA,RAW,MANIFEST,CONFIG,WEIGHTS,_INPUTS
    ROOT=args.output_dir.expanduser().resolve();RAW=args.data_root.expanduser().resolve()
    MANIFEST=args.manifest.expanduser().resolve();WEIGHTS=args.pretrained_weights.expanduser().resolve();CONFIG=args.pretrained_config.expanduser().resolve();DATA=RAW.parent
    if ROOT==PACKAGE or ROOT.is_relative_to(PACKAGE):raise ValueError('Use a separate run output directory outside the source package')
    if ROOT==RAW or ROOT.is_relative_to(RAW) or RAW.is_relative_to(ROOT):raise ValueError('Output and raw data directories must be separate')
    if ROOT in (WEIGHTS.parent,MANIFEST.parent,CONFIG.parent):raise ValueError('Output must not overwrite an input directory')
    marker=ROOT/'portable_run.json'
    if ROOT.exists() and not marker.exists() and any(ROOT.iterdir()):
        raise ValueError('Existing nonportable experiment directory; choose a new output directory')
    for q in (MANIFEST,WEIGHTS,CONFIG):
        if not q.is_file():raise FileNotFoundError(q)
    if not RAW.is_dir():raise FileNotFoundError(RAW)
    if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Expected the frozen official manifest; check its SHA in weights_manifest.json')
    if sha(WEIGHTS)!=NATIVE_SHA:raise ValueError('Only the native384 ImageNet source is accepted; never a CXR-trained checkpoint')
    cfg=read(CONFIG);pc=cfg['pretrained_cfg']
    if cfg!=read(PACKAGE/'pretrained_config.json'):raise ValueError('External config must exactly match the supplied native384 config')
    if cfg['architecture']!='caformer_s18' or pc['tag']!='sail_in1k_384' or pc['input_size']!=[3,384,384] or pc['interpolation']!='bicubic':raise ValueError('Wrong native384 config')
    ROOT.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','HF_XET_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
        q=ROOT/'temp'/key.lower();q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    COMMON['workers']=args.workers
    _INPUTS={str(q):sha(q) for q in (MANIFEST,WEIGHTS,CONFIG)}
    freeze(ROOT/'input_hashes.json',_INPUTS)
    freeze(marker,dict(schema_version=1,recipe='native384_full_backbone_5epochs',manifest_sha256=MANIFEST_SHA,pretrained_sha256=NATIVE_SHA))
    meta=dict(architecture='caformer_s18.sail_in1k_384',weights=str(WEIGHTS),sha256=NATIVE_SHA,pretraining_data='ImageNet-1k',license='apache-2.0',config=cfg,config_path=str(CONFIG))
    freeze(ROOT/'pretrained_manifest.json',meta)

def status(**fields):
    history=read(ROOT/'history.json') if (ROOT/'history.json').exists() else []
    save(ROOT/'status.json',dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),epochs_completed=len(history),total_new_epochs=len(history),**fields))

def pretrained():
    meta=read(ROOT/'pretrained_manifest.json')
    if meta['architecture']!='caformer_s18.sail_in1k_384' or meta['pretraining_data']!='ImageNet-1k' or meta['license']!='apache-2.0' or sha(meta['weights'])!=NATIVE_SHA:raise ValueError('Native384 source changed')
    pc=meta['config']['pretrained_cfg']
    if pc['mean']!=[.485,.456,.406] or pc['std']!=[.229,.224,.225] or meta['config']['num_classes']!=1000:raise ValueError('Native normalization/classes differ')
    if pc['input_size']!=[3,384,384] or pc['interpolation']!='bicubic' or pc['classifier']!='head.fc.fc2':raise ValueError('Native config differs')
    return meta

def source_hashes():
    import timm
    paths=[PACKAGE/n for n in ('train.py','common.py','data_pipeline.py','training_core.py','recipe.json','pretrained_config.json')]
    paths += [MANIFEST,CONFIG,ROOT/'pretrained_manifest.json',Path(timm.__file__).resolve().parent/'models/metaformer.py']
    return {str(q):sha(q) for q in paths}

def check_protected():
    if not _INPUTS:raise RuntimeError('Call configure before importing the training core')
    for path,digest in _INPUTS.items():
        if sha(path)!=digest:raise ValueError('Input changed: '+path)
    return _INPUTS.copy()

def load_inputs():
    if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Official manifest changed')
    meta=pretrained();rows={k:[] for k in EXPECTED};groups={k:{} for k in EXPECTED};patients={k:set() for k in EXPECTED};seen_images=set();image_positives=[0]*10
    with MANIFEST.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            split=row['split'];labels=[int(x)-1 for x in row['label_ids'].split('|') if x]
            if split not in rows or len(set(labels))!=len(labels) or any(x<0 or x>=10 for x in labels) or bool(labels)!=(split!='test'):raise ValueError('Invalid official labels/split')
            path=row['image_path']
            relative=Path(path.replace('\\','/'))
            if relative.is_absolute() or '..' in relative.parts or not (RAW/relative).resolve().is_relative_to(RAW):raise ValueError('Unsafe official image path')
            subject=row['subject_id'];study=row['study_id'];target=[float(i in labels) for i in range(10)]
            if path in seen_images:raise ValueError('Duplicate image in official manifest')
            seen_images.add(path)
            if split!='test' and not (RAW/path).is_file():raise ValueError('Missing raw image: '+path)
            rows[split].append(row);patients[split].add(subject)
            if study not in groups[split]:groups[split][study]=dict(study_id=study,subject_id=subject,image_paths=[],image_positions=[],target=target)
            record=groups[split][study]
            if record['subject_id']!=subject or record['target']!=target:raise ValueError('Study labels/patient inconsistent')
            record['image_paths'].append(path);record['image_positions'].append(len(rows[split])-1)
            if split=='train':
                for index in labels:image_positives[index]+=1
    coverage={split:dict(images=len(rows[split]),studies=len(groups[split]),patients=len(patients[split])) for split in rows}
    if coverage!=EXPECTED or image_positives!=[894,1238,1265,2293,3220,5590,5649,6703,6637,12649]:raise ValueError('Official coverage/class counts differ')
    for a,b in (('train','val'),('train','test'),('val','test')):
        if patients[a]&patients[b]:raise ValueError('Patient leakage across official splits')
    records={split:list(value.values()) for split,value in groups.items()}
    positives=[sum(r['target'][i] for r in records['train']) for i in range(10)]
    priors=[min(.999,max(.001,count/18513)) for count in positives]
    audit=dict(coverage=coverage,patient_splits_disjoint=True,all_raw_train_val_files_present=True,train_positive_image_counts=image_positives,train_positive_study_counts=positives,train_study_priors=priors,test_metadata_only=True,no_old_image_or_feature_cache=True)
    cfg=dict(manifest=str(MANIFEST),data_root=str(RAW),mean=meta['config']['pretrained_cfg']['mean'],std=meta['config']['pretrained_cfg']['std'],image_size=384)
    return cfg,rows,records,audit,meta
