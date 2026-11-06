"""Portable complete-checkpoint inference: raw384 -> image sigmoid -> FP32 Study mean."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

NATIVE_SHA='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
TEST_SHA='63e107d840cbf27aee7c1c0ced1157aef454baa1141fdc3851e6c3011471529e'
COLUMNS=['Subject_id','Study_id','Predict_class','Probability','Gate']

def args():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','gates','data-root','manifest','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--official-test-csv',type=Path)
    p.add_argument('--split',choices=('test','val'),default='test')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--workers',type=int,choices=(0,1,2),default=0)
    p.add_argument('--check-only',action='store_true',help='CPU input/model check, one validation image forward; no GPU or predictions written')
    return p.parse_args()

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main(a):
    out=a.output_dir.expanduser().resolve();package=Path(__file__).resolve().parent
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),'Refuse to overwrite any existing nonempty output directory; choose a new empty directory')
    require(out!=package and not out.is_relative_to(package),'Choose output outside the source package')
    require(out not in (a.checkpoint.expanduser().resolve().parent,a.gates.expanduser().resolve().parent,a.manifest.expanduser().resolve().parent),'Output must be separate from checkpoint/Gates/manifest directories')
    raw=a.data_root.expanduser().resolve();require(not out.is_relative_to(raw) and not raw.is_relative_to(out),'Keep output separate from raw data')
    out.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
        q=out/'temp'/key.lower();q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    if a.check_only or a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch,timm
    from torch.utils.data import DataLoader
    from data_pipeline import RawImageDataset
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    require(sha(a.manifest)==MANIFEST_SHA,'Official manifest SHA differs')
    gate=read(a.gates);weight_sha=sha(a.checkpoint)
    require(gate['checkpoint_sha256']==weight_sha and gate['pretrained_sha256']==NATIVE_SHA,'Gates do not bind this native-source checkpoint')
    require(gate['class_ids']==list(range(1,11)),'Wrong Gate class ordering')
    gates=np.asarray(gate['final_gates_by_class_id'],dtype=np.float64)
    require(gates.shape==(10,) and np.isfinite(gates).all() and ((gates>=0)&(gates<=1)).all(),'Invalid Gates')
    config=read(package/'pretrained_config.json')['pretrained_cfg'];groups=OrderedDict();image_keys=set()
    with a.manifest.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['split']==a.split]
    for pos,r in enumerate(rows):
        path=Path(r['image_path'].replace('\\','/'));absolute=(raw/path).resolve()
        require(not path.is_absolute() and '..' not in path.parts and absolute.is_relative_to(raw) and absolute.is_file(),'Unsafe or missing raw image path')
        key=(r['subject_id'],r['study_id']);image_key=(*key,path.stem)
        require(image_key not in image_keys,'Duplicate image identity');image_keys.add(image_key)
        if a.split=='test':require(not r['label_ids'],'Hidden test must be unlabeled')
        labels=[int(v)-1 for v in r['label_ids'].split('|') if v];target=[float(i in labels) for i in range(10)]
        if key not in groups:groups[key]=dict(subject_id=key[0],study_id=key[1],image_paths=[],image_positions=[],target=target)
        require(groups[key]['target']==target,'Inconsistent Study labels')
        groups[key]['image_paths'].append(r['image_path']);groups[key]['image_positions'].append(pos)
    expected={'test':(2218,1414,136),'val':(1422,859,228)}[a.split]
    require((len(rows),len(groups),len({k[0] for k in groups}))==expected,'Official split coverage differs')
    order=list(groups)
    if a.split=='test':
        official=a.official_test_csv or raw/'test.csv'
        require(official.is_file() and sha(official)==TEST_SHA,'Provide the original official test.csv')
        with official.open(encoding='utf-8-sig',newline='') as f:official_rows=list(csv.DictReader(f))
        require({(r['Subject_id'],r['Study_id'],r['Dicom_id']) for r in official_rows}==image_keys and len(official_rows)==2218,'Official image IDs differ')
        order=list(OrderedDict(((r['Subject_id'],r['Study_id']),None) for r in official_rows))
        require(set(order)==set(groups),'Official Study IDs differ')
    stored=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    require(stored['pretrained_sha256']==NATIVE_SHA and stored.get('full_backbone_training') is True,'Wrong complete model source/scope')
    require(stored['epoch']==gate['best_epoch'],'Checkpoint and Gate epochs differ')
    model=timm.create_model('caformer_s18',pretrained=False,num_classes=10,drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
    model.head.fc.head_drop.p=.2;model.load_state_dict(stored['state_dict'],strict=True);model.set_grad_checkpointing(False);model.eval().requires_grad_(False)
    dataset=RawImageDataset(list(groups.values()),raw,config['mean'],config['std'])
    if a.check_only:
        with torch.inference_mode():z=model(dataset[0][0].unsqueeze(0))
        require(z.shape==(1,10) and torch.isfinite(z).all() and not torch.cuda.is_initialized(),'CPU model/input check failed')
        print('INFERENCE_CPU_CHECK_PASSED',{'images':len(rows),'studies':len(groups),'checkpoint_sha256':weight_sha});return
    device=torch.device(a.device)
    if device.type=='cuda':require(torch.cuda.is_available(),'CUDA unavailable; choose --device cpu explicitly')
    model.to(device=device,memory_format=torch.channels_last);values=[];seen=[]
    with torch.inference_mode():
        for image,index in DataLoader(dataset,batch_size=4,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda'):
            image=image.to(device=device,memory_format=torch.channels_last)
            context=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
            with context:logits=model(image)
            values.append(logits.float().sigmoid().cpu().numpy());seen.extend(index.tolist())
    require(seen==list(range(len(rows))),'Image inference coverage differs')
    image_p=np.concatenate(values)
    require(image_p.dtype==np.float32 and image_p.shape==(len(rows),10) and np.isfinite(image_p).all() and ((image_p>=0)&(image_p<=1)).all(),'Invalid image probability array')
    keys=list(groups);indices=OrderedDict((k,[]) for k in groups)
    for i,study_index in enumerate(dataset.study_indices):indices[keys[study_index]].append(i)
    probabilities={key:image_p[v].mean(0,dtype=np.float32) for key,v in indices.items()}
    require(all(v.shape==(10,) and v.dtype==np.float32 and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in probabilities.values()),'Invalid Study probability array')
    output=out/'submission.csv'
    with output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(COLUMNS)
        for key in order:
            for c in range(10):writer.writerow([key[0],key[1],c+1,format(float(probabilities[key][c]),'.12f'),format(float(gates[c]),'.12g')])
    np.savez_compressed(out/'image_probabilities.npz',image_paths=np.asarray(dataset.paths,dtype=str),subjects=np.asarray([keys[i][0] for i in dataset.study_indices],dtype=str),study_ids=np.asarray([keys[i][1] for i in dataset.study_indices],dtype=str),probabilities=image_p)
    np.savez_compressed(out/'study_probabilities.npz',subjects=np.asarray([k[0] for k in order],dtype=str),study_ids=np.asarray([k[1] for k in order],dtype=str),probabilities=np.stack([probabilities[k] for k in order]))
    receipt=dict(state='generated',split=a.split,rows=len(order)*10,images=len(rows),studies=len(order),checkpoint_sha256=weight_sha,gates_sha256=sha(a.gates),manifest_sha256=MANIFEST_SHA,submission_sha256=sha(output),device=str(device),validation_image_microbatch=4,aggregation='float32_mean_image_sigmoid_probabilities_by_Study',test_labels_available=False if a.split=='test' else True,test_AUC_or_F1_computed=False)
    (out/'inference_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('INFERENCE_COMPLETE',str(output),receipt['submission_sha256'])

if __name__=='__main__':main(args())
