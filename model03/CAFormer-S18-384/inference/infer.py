"""Infer one of three frozen full CAFormer-S18-384 checkpoints; no training/download."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from contextlib import contextmanager
from pathlib import Path
from common import FIELDS,MEAN,STD,NATIVE_SHA,need,sha,profile,load_gates,official_metadata,manifest_order_check,image_paths,aggregate_probabilities,submission_bytes,validate_csv,atomic_bytes,atomic_json

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('a0_raw','a0_ema','photo_raw'),required=True)
    for name in ('checkpoint','gates','raw-root','official-test-csv','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest',type=Path,help='Optional original official manifest; verifies image order only')
    p.add_argument('--check-only',action='store_true',help='CPU-only strict model/data checks; no image forward or GPU')
    return p.parse_args()

def configure(a):
    out=a.output_dir.expanduser().resolve();raw=a.raw_root.expanduser().resolve();package=Path(__file__).resolve().parents[1]
    need(out!=raw and not out.is_relative_to(raw) and not raw.is_relative_to(out),'Keep output separate from raw data')
    need(out!=package and not out.is_relative_to(package) and not package.is_relative_to(out),'Choose output outside source package')
    for p in (a.checkpoint,a.gates,a.official_test_csv):need(not p.resolve().is_relative_to(out) and out!=p.resolve().parent,'Keep output separate from input files')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('submission.csv','image_probabilities.npz','study_probabilities.npz','prediction_receipt.json'):need(not (out/name).exists(),'Choose a fresh output; existing prediction must be preserved')
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','HF_XET_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
        q=out/'temp'/key.lower();q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    if a.check_only:os.environ['CUDA_VISIBLE_DEVICES']=''
    return out,raw

@contextmanager
def output_lock(out):
    path=out/'.inference.lock';fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,str(os.getpid()).encode('ascii'));yield
    finally:os.close(fd);path.unlink()

def build_model(checkpoint,p):
    import torch,timm
    stored=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(stored['epoch']==5 and stored['source_total_epoch']==5 and stored['arm']==p['arm'] and stored['variant']==p['variant'] and stored['architecture']=='caformer_s18','Checkpoint metadata belongs to another selected model')
    need(stored['pretrained_sha256']==NATIVE_SHA and stored['initialization_kind']=='fresh ImageNet1k native384' and stored['full_backbone_training'] is True,'Checkpoint source/scope differs')
    model=timm.create_model('caformer_s18',pretrained=False,num_classes=10,drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
    model.head.drop.p=0.;model.head.fc.head_drop.p=.2;model.load_state_dict(stored['state_dict'],strict=True)
    model.set_grad_checkpointing(False);model.eval().requires_grad_(False)
    need(tuple(model.head.fc.fc2.weight.shape)==(10,2048),'Full hidden2048 classification head required')
    return model,stored['state_dict']

def save_npz(path,**arrays):
    import numpy as np
    need(not path.exists(),'Prediction archive already exists');temporary=path.with_name(path.name+'.tmp');need(not temporary.exists(),'Uncommitted archive exists')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)

def main(a):
    out,raw=configure(a)
    with output_lock(out):
        p=profile(a.model);need(sha(a.checkpoint)==p['checkpoint_sha256'],'Wrong selected checkpoint SHA')
        gates=load_gates(a.gates,p);meta=official_metadata(a.official_test_csv);paths=image_paths(raw,meta)
        order_check=manifest_order_check(a.manifest,meta) if a.manifest else None
        import numpy as np
        import torch
        from PIL import Image,ImageOps
        from torchvision.transforms.functional import to_tensor,normalize
        from torch.utils.data import Dataset,DataLoader
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        source_paths=[a.checkpoint.resolve(),a.gates.resolve(),a.official_test_csv.resolve(),Path(__file__).resolve(),Path(__file__).with_name('common.py'),Path(__file__).with_name('profiles.json')]
        if a.manifest:source_paths.append(a.manifest.resolve())
        frozen={str(q):sha(q) for q in source_paths};model,initial=build_model(a.checkpoint,p)
        if a.check_only:
            need(not torch.cuda.is_initialized(),'CPU check initialized CUDA')
            report=dict(state='CPU_preflight_passed',model=a.model,checkpoint_sha256=p['checkpoint_sha256'],images=2218,studies=1414,patients=136,image_microbatch=4,parameter_count=sum(q.numel() for q in model.parameters()),output_shape=[1414,10],manifest_order_check=order_check,no_image_forward=True,CUDA_initialized=False,inputs_sha256=frozen)
            atomic_json(out/'preflight.json',report);print(json.dumps(report,ensure_ascii=False));return
        need(torch.cuda.is_available(),'This exact FP16 delivery path requires CUDA; --check-only is CPU-only')
        model.to('cuda',memory_format=torch.channels_last)
        class RawImages(Dataset):
            def __len__(self):return len(paths)
            def __getitem__(self,i):
                with Image.open(paths[i]) as image:
                    need(image.mode not in ('I','F') and not image.mode.startswith('I;16'),'High-bit-depth input needs explicit normalization')
                    image=ImageOps.pad(image.convert('RGB'),(384,384),method=Image.Resampling.BICUBIC,color=0)
                    return normalize(to_tensor(image),MEAN,STD)
        started=time.perf_counter();parts=[];seen=0
        with torch.inference_mode():
            for images in DataLoader(RawImages(),batch_size=4,shuffle=False,num_workers=0,pin_memory=True):
                with torch.autocast('cuda',dtype=torch.float16):logits=model(images.to('cuda',memory_format=torch.channels_last))
                need(torch.isfinite(logits).all().item(),'Nonfinite logits');parts.append(logits.float().sigmoid().cpu().numpy());seen+=len(images)
                if seen%300==0 or seen==2218:print('INFERENCE',a.model,seen,'/2218',flush=True)
        image_p=np.concatenate(parts);study_p=aggregate_probabilities(image_p,meta);data=submission_bytes(study_p,meta,gates)
        checked=validate_csv(data,meta,gates,study_p)
        for k,v in model.state_dict().items():need(torch.equal(v.detach().cpu(),initial[k]),'Inference changed trained state')
        need(all(sha(q)==h for q,h in frozen.items()),'Input/code changed during inference')
        image_file=out/'image_probabilities.npz';study_file=out/'study_probabilities.npz'
        save_npz(image_file,image_paths=np.asarray([q.relative_to(raw).as_posix() for q in paths],str),subjects=np.asarray([q[0] for q in meta['triples']],str),study_ids=np.asarray([q[1] for q in meta['triples']],str),probabilities=image_p)
        save_npz(study_file,subjects=np.asarray([q[0] for q in meta['study_order']],str),study_ids=np.asarray([q[1] for q in meta['study_order']],str),probabilities=study_p)
        atomic_bytes(out/'submission.csv',data)
        receipt=dict(state='predictions_generated',model=a.model,arm=p['arm'],variant=p['variant'],epoch=5,source_total_epoch=5,checkpoint_sha256=p['checkpoint_sha256'],gates_sha256=sha(a.gates),official_test_sha256=sha(a.official_test_csv),images=2218,studies=1414,patients=136,rows=14140,columns=FIELDS,image_microbatch=4,forward='CUDA float16',sigmoid_dtype='float32',aggregation='Study mean of float32 image sigmoid probabilities in input image row order',preprocessing=dict(size=384,mode='full-field RGB bicubic pad',mean=MEAN,std=STD),submission_sha256=hashlib.sha256(data).hexdigest(),prediction_artifact_sha256={q.name:sha(q) for q in (image_file,study_file)},inputs_sha256=frozen,manifest_order_check=order_check,CSV_rounding_check=checked,seconds=time.perf_counter()-started,no_training=True,no_download=True,no_test_label_or_threshold_fitting=True)
        atomic_json(out/'prediction_receipt.json',receipt);print('INFERENCE_COMPLETE',a.model,receipt['submission_sha256'],flush=True)

if __name__=='__main__':main(arguments())
