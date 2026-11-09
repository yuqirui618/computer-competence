"""Explicit portable paths, fixed official split and immutable run-input hashes."""
from __future__ import annotations
import csv,hashlib,json,os,sys
from pathlib import Path
from datetime import datetime
PACKAGE=Path(__file__).resolve().parent
CONFIG=json.loads((PACKAGE/'ema_recipe.json').read_text(encoding='utf-8'))
COMMON=CONFIG['loader'].copy()
ARMS=('A_BCE_EMA','RGB_EMA')
NATIVE_SHA='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
ROOT=DATA=SOURCE=INITIAL=REFERENCE=RAW=MANIFEST=None
INITIAL_SHA=REFERENCE_SHA=None
_INPUTS={}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def freeze(path,value):
    if Path(path).exists():
        if read(path)!=value:raise ValueError('Frozen run metadata differs: '+str(path))
    else:save(path,value)
def save_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)
def configure(args):
    global ROOT,DATA,SOURCE,INITIAL,REFERENCE,RAW,MANIFEST,INITIAL_SHA,REFERENCE_SHA,_INPUTS
    ROOT=args.output_dir.expanduser().resolve();RAW=args.data_root.expanduser().resolve()
    MANIFEST=args.manifest.expanduser().resolve();INITIAL=args.initial_checkpoint.expanduser().resolve()
    REFERENCE=args.initial_predictions.expanduser().resolve();SOURCE=INITIAL.parent;DATA=RAW.parent
    INITIAL_SHA=args.initial_sha256;REFERENCE_SHA=args.initial_predictions_sha256
    if any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in (INITIAL_SHA,REFERENCE_SHA)):
        raise ValueError('Explicit lowercase SHA256 required')
    source_root=PACKAGE.parent
    if ROOT==source_root or ROOT.is_relative_to(source_root) or source_root.is_relative_to(ROOT):
        raise ValueError('Use a new output outside the source package')
    if ROOT==RAW or ROOT.is_relative_to(RAW) or RAW.is_relative_to(ROOT):
        raise ValueError('Raw data and output directories must be separate')
    for value in (MANIFEST,INITIAL,REFERENCE):
        if not value.is_file():raise FileNotFoundError(value)
        if ROOT==value.parent or value.is_relative_to(ROOT):raise ValueError('Output must not contain input files')
    if not RAW.is_dir():raise FileNotFoundError(RAW)
    if sha(MANIFEST)!=MANIFEST_SHA or sha(INITIAL)!=INITIAL_SHA or sha(REFERENCE)!=REFERENCE_SHA:
        raise ValueError('Official manifest, initialization or validation reference SHA differs')
    marker=ROOT/'portable_EMA_run.json'
    if ROOT.exists() and any(ROOT.iterdir()) and not marker.exists():
        raise ValueError('Refuse existing nonportable output; choose a new directory')
    ROOT.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','HF_XET_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
        path=ROOT/'temp'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    COMMON['workers']=args.workers;CONFIG['loader']=COMMON.copy()
    _INPUTS={str(MANIFEST):MANIFEST_SHA,str(INITIAL):INITIAL_SHA,str(REFERENCE):REFERENCE_SHA}
    freeze(marker,dict(schema_version=1,recipe='native384_BCE_or_RGB_EMA_5continuation_epochs_real16_accum2',
        initial_sha256=INITIAL_SHA,reference_sha256=REFERENCE_SHA,manifest_sha256=MANIFEST_SHA))
    freeze(ROOT/'input_hashes.json',_INPUTS)
def status(**fields):save(ROOT/'status.json',dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),**fields))
def sources():
    import timm
    paths=[PACKAGE/name for name in ('ema_continue.py','ema_common.py','ema_core.py','training_utils.py','data_pipeline.py','ema_recipe.json')]
    paths.append(Path(timm.__file__).resolve().parent/'models/metaformer.py')
    return {str(path):sha(path) for path in paths}
def check_protected():
    if not _INPUTS:raise RuntimeError('Configure runtime before loading training core')
    for path,digest in _INPUTS.items():
        if sha(path)!=digest:raise ValueError('Input changed: '+path)
    return _INPUTS.copy()
def load_inputs():
    if sha(INITIAL)!=INITIAL_SHA or sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Initial weight or official manifest changed')
    rows={s:[] for s in ('train','val','test')};groups={s:{} for s in rows};patients={s:set() for s in rows};paths=set()
    with MANIFEST.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            s=r['split'];path=r['image_path'];sub=r['subject_id'];study=r['study_id']
            labels=[int(v)-1 for v in r['label_ids'].split('|') if v]
            if s not in rows or len(labels)!=len(set(labels)) or any(v<0 or v>=10 for v in labels) or bool(labels)!=(s!='test'):raise ValueError('Invalid label/split')
            p=(RAW/path).resolve()
            if path in paths or not p.is_relative_to(RAW.resolve()):raise ValueError('Duplicate/escaping image')
            if s!='test' and not p.is_file():raise FileNotFoundError(p)
            paths.add(path);rows[s].append(r);patients[s].add(sub);target=[float(i in labels) for i in range(10)]
            if study not in groups[s]:groups[s][study]=dict(subject_id=sub,study_id=study,image_paths=[],image_positions=[],target=target)
            rec=groups[s][study]
            if rec['subject_id']!=sub or rec['target']!=target:raise ValueError('Study identity/labels inconsistent')
            rec['image_paths'].append(path);rec['image_positions'].append(len(rows[s])-1)
    records={s:list(g.values()) for s,g in groups.items()}
    expected={'train':(30344,18513,11055),'val':(1422,859,228),'test':(2218,1414,136)}
    for s in rows:
        if (len(rows[s]),len(records[s]),len(patients[s]))!=expected[s]:raise ValueError('Official coverage changed')
    for a,b in (('train','val'),('train','test'),('val','test')):
        if patients[a]&patients[b]:raise ValueError('Patient leakage')
    pos=[sum(r['target'][i] for r in records['train']) for i in range(10)]
    if pos!=[622,900,942,1436,2404,3885,3968,4508,4654,6683]:raise ValueError('Study class counts differ')
    cfg=dict(data_root=str(RAW),mean=CONFIG['mean'],std=CONFIG['std'],image_size=384)
    audit=dict(coverage={s:dict(images=len(rows[s]),studies=len(records[s]),patients=len(patients[s])) for s in rows},patient_splits_disjoint=True,train_positive_Study_counts=pos,train_negative_Study_counts=[18513-x for x in pos],test_metadata_only=True,no_test_image_access=True)
    return cfg,records,audit
