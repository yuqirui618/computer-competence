"""Portable CAFormer-S18-384-EMA selected-epoch1 official-test prediction exporter.

All model inputs are explicit CLI paths. No historical experiment protection list
or previous training runner is imported. The complete trained checkpoint is enough
for inference; native ImageNet weights are provenance, not inference dependencies.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

EXPECTED_WEIGHT='f1f862bd6c76235e59e96b17b4bdae92aa497846b4eef42e8fb33b2b34ddee85'
EXPECTED_INITIAL='e4396db33d9fa25751205fc4d848766ff894ec4d104d884b77847e4afd01b250'
EXPECTED_VALP='84de88e32b80f1240f6f1e476d6eda6d6ca4850fae95ab10f122d855f15f99cc'
EXPECTED_NATIVE='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
EXPECTED_TEST='63e107d840cbf27aee7c1c0ced1157aef454baa1141fdc3851e6c3011471529e'
EXPECTED_MANIFEST='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
EXPECTED_GATES=[.1,.25,.15,.175,.2,.175,.225,.25,.225,.3]
FIELDS=['Subject_id','Study_id','Predict_class','Probability','Gate']

def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('checkpoint','gates-json','pretrained-config','manifest','data-root','official-test-csv','validation-predictions','output-dir'):
  p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--check-only',action='store_true');p.add_argument('--batch-size',type=int,default=4)
 return p.parse_args()

def require(ok,msg):
 if not ok:raise ValueError(msg)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
 path=Path(path);part=path.with_suffix(path.suffix+'.tmp')
 part.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');part.replace(path)

def npz(path,**arrays):
 path=Path(path);part=path.with_suffix(path.suffix+'.tmp')
 with part.open('wb') as f:np.savez_compressed(f,**arrays)
 part.replace(path)

def prepare(args):
 global np,torch,timm,Image,ImageOps,to_tensor,normalize,DataLoader,Dataset,roc_auc_score,f1_score,WORK,OUT
 OUT=args.output_dir.resolve()
 raw=args.data_root.resolve()
 require(OUT!=raw and not OUT.is_relative_to(raw) and not raw.is_relative_to(OUT),'Output must be separate from official raw data')
 for key in ('checkpoint','gates_json','pretrained_config','manifest','official_test_csv','validation_predictions'):
  require(OUT!=Path(getattr(args,key)).resolve().parent,'Output must be separate from input directories')
 require(not (OUT/'CAFormer-S18-384-EMA.csv').exists(),'Refuse to overwrite an existing model submission; choose a new output directory')
 WORK=OUT/'work';WORK.mkdir(parents=True,exist_ok=True)
 for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','HF_XET_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
  q=WORK/'temp'/key.lower();q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
 os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
 os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
 import numpy as np
 import torch,timm
 from PIL import Image,ImageOps
 from torchvision.transforms.functional import to_tensor,normalize
 from torch.utils.data import DataLoader,Dataset
 from sklearn.metrics import roc_auc_score,f1_score
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 require(args.batch_size==4,'This locked delivery uses global image microbatch4')

def inputs(args):
 files={key:Path(getattr(args,key)).resolve() for key in ('checkpoint','gates_json','pretrained_config','manifest','official_test_csv','validation_predictions')}
 sources={str(p):sha(p) for p in files.values()}
 require(sources[str(files['checkpoint'])]==EXPECTED_WEIGHT,'Wrong selected full checkpoint SHA')
 require(sources[str(files['manifest'])]==EXPECTED_MANIFEST,'Official manifest changed')
 require(sources[str(files['validation_predictions'])]==EXPECTED_VALP,'Wrong selected EMA validation predictions')
 require(sources[str(files['official_test_csv'])]==EXPECTED_TEST,'Official test metadata changed')
 gates=read(files['gates_json'])
 require(gates['checkpoint_sha256']==EXPECTED_WEIGHT and gates['best_epoch']==1 and gates['arm']=='A_BCE_EMA' and gates['variant']=='ema' and gates['pretrained_sha256']==EXPECTED_NATIVE,'Gates bind a different model')
 require(gates['class_ids']==list(range(1,11)) and gates['final_gates_by_class_id']==EXPECTED_GATES,'Unexpected class order or Gates')
 config=read(files['pretrained_config']);pc=config['pretrained_cfg']
 require(config['architecture']=='caformer_s18' and pc['input_size']==[3,384,384] and pc['tag']=='sail_in1k_384' and pc['interpolation']=='bicubic','Wrong native model configuration')
 require(pc['mean']==[.485,.456,.406] and pc['std']==[.229,.224,.225],'Native normalization differs')
 with files['manifest'].open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
 rows={s:[r for r in all_rows if r['split']==s] for s in ('train','val','test')}
 subjects={s:{r['subject_id'] for r in rrs} for s,rrs in rows.items()}
 for a,b in (('train','val'),('train','test'),('val','test')):require(not subjects[a]&subjects[b],'Official patient leakage')
 require(len(rows['val'])==1422 and len(rows['test'])==2218,'Wrong val/test image coverage')
 require(len(subjects['val'])==228 and len(subjects['test'])==136,'Wrong val/test patient coverage')
 with files['official_test_csv'].open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);require({'Dicom_id','Subject_id','Study_id'}<=set(reader.fieldnames or []),'Wrong official test columns');official=list(reader)
 official_keys=[(r['Subject_id'],r['Study_id'],r['Dicom_id']) for r in official]
 actual_keys=[(r['subject_id'],r['study_id'],Path(r['image_path']).stem) for r in rows['test']]
 require(len(official_keys)==len(set(official_keys))==2218 and len(actual_keys)==len(set(actual_keys))==2218 and set(official_keys)==set(actual_keys),'Official image identity mapping differs')
 order=OrderedDict()
 for r in official:
  key=r['Study_id'];subject=r['Subject_id'];require(key not in order or order[key]==subject,'Study assigned to multiple patients');order[key]=subject
 require(len(order)==1414 and len(set(order.values()))==136,'Wrong official Study coverage')
 raw=args.data_root.resolve()
 for split in ('val','test'):
  for r in rows[split]:
   path=(raw/r['image_path']).resolve();require(path.is_relative_to((raw/('valid' if split=='val' else split)).resolve()) and path.is_file(),'Missing or escaping raw '+split+' path')
 require(all(not r['label_ids'] for r in rows['test']),'Test unexpectedly labeled')
 return files,sources,gates,pc,rows,order

def construct(args):
 stored=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
 require(stored['epoch']==1 and stored['source_total_epoch']==6 and stored['arm']=='A_BCE_EMA' and stored['variant']=='ema' and stored['initial_checkpoint_sha256']==EXPECTED_INITIAL and stored['pretrained_sha256']==EXPECTED_NATIVE and stored['full_backbone_training'],'Selected EMA checkpoint metadata differs')
 model=timm.create_model('caformer_s18',pretrained=False,num_classes=10,drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
 model.head.drop.p=0.;model.head.fc.head_drop.p=.2
 model.load_state_dict(stored['state_dict'],strict=True);model.set_grad_checkpointing(False);model.eval().requires_grad_(False)
 require(tuple(model.head.fc.fc2.weight.shape)==(10,2048),'Wrong full classifier shape')
 return model,stored['state_dict']

def infer(model,raw,rows,pc,phase):
 class RawImages(Dataset):
  def __len__(self):return len(rows)
  def __getitem__(self,i):
   with Image.open(raw/rows[i]['image_path']) as im:
    require(im.mode not in ('I','F') and not im.mode.startswith('I;16'),'Unsupported high-bit-depth source')
    image=ImageOps.pad(im.convert('RGB'),(384,384),method=Image.Resampling.BICUBIC,color=0)
    return normalize(to_tensor(image),pc['mean'],pc['std'])
 values=[];count=0;started=time.perf_counter()
 with torch.inference_mode():
  for image in DataLoader(RawImages(),batch_size=4,shuffle=False,num_workers=0,pin_memory=True):
   with torch.autocast('cuda',dtype=torch.float16):logits=model(image.to('cuda',memory_format=torch.channels_last))
   require(torch.isfinite(logits).all().item(),'Nonfinite logits')
   values.append(logits.float().sigmoid().cpu().numpy());count+=len(image)
   if count%300==0 or count==len(rows):
    save(WORK/'status.json',dict(state='inference',phase=phase,images=count,total_images=len(rows)))
    print('EMA_EXPORT',phase,count,'/',len(rows),flush=True)
 result=np.concatenate(values);require(result.shape==(len(rows),10) and result.dtype==np.float32 and np.isfinite(result).all(),'Incomplete image predictions')
 return result,time.perf_counter()-started

def aggregate(rows,probabilities,order=None):
 groups=OrderedDict()
 for i,row in enumerate(rows):groups.setdefault(row['study_id'],[]).append(i)
 keys=list(groups) if order is None else list(order)
 require(set(keys)==set(groups),'Study order missing images')
 ids=[];subjects=[];ps=[];ys=[]
 for key in keys:
  idx=groups[key];subject=rows[idx[0]]['subject_id'];require(all(rows[i]['subject_id']==subject for i in idx),'Inconsistent Study subject')
  if order is not None:require(order[key]==subject,'Official Study subject mismatch')
  labels=rows[idx[0]]['label_ids'];require(all(rows[i]['label_ids']==labels for i in idx),'Inconsistent Study labels')
  ids.append(key);subjects.append(subject);ps.append(probabilities[idx].mean(0))
  y=np.zeros(10,np.float32)
  if labels:y[[int(x)-1 for x in labels.split('|') if x]]=1
  ys.append(y)
 return dict(study_ids=np.asarray(ids,str),subjects=np.asarray(subjects,str),probabilities=np.stack(ps),targets=np.stack(ys))

def validation_check(rows,image_p,study,path):
 with np.load(path,allow_pickle=False) as saved:
  for key in ('targets','study_ids','subjects'):np.testing.assert_array_equal(saved[key],study[key])
  np.testing.assert_array_equal(saved['image_paths'],np.asarray([r['image_path'] for r in rows],str))
  image_diff=float(np.max(np.abs(saved['image_probabilities']-image_p)))
  study_diff=float(np.max(np.abs(saved['probabilities']-study['probabilities'])))
  np.testing.assert_allclose(saved['image_probabilities'],image_p,rtol=0,atol=1e-7)
  np.testing.assert_allclose(saved['probabilities'],study['probabilities'],rtol=0,atol=1e-7)
 auc=float(roc_auc_score(study['targets'],study['probabilities'],average='macro'))
 f1=float(f1_score(study['targets'],study['probabilities']>=.5,average='macro',zero_division=0))
 require(abs(auc-.8061307765133356)<=1e-10 and abs(f1-.3147037443711392)<=1e-10,'Selected validation metrics do not reproduce')
 return dict(state='passed',images=1422,studies=859,patients=228,macro_auc=auc,macro_f1_at05=f1,max_abs_image_probability_difference=image_diff,max_abs_Study_probability_difference=study_diff)

def main(args):
 prepare(args);files,sources,gates,pc,rows,order=inputs(args);model,initial_state=construct(args)
 if args.check_only:
  save(WORK/'preflight.json',dict(state='passed',checkpoint_sha256=EXPECTED_WEIGHT,source_inputs=sources,test_images=2218,test_studies=1414,test_patients=136,submission_rows=14140,columns=FIELDS,cpu_only=True))
  print('EMA_EXPORT_CPU_PREFLIGHT_PASSED',flush=True);return
 preflight=read(WORK/'preflight.json');require(preflight['state']=='passed' and preflight['source_inputs']==sources,'Run matching CPU preflight first')
 require(torch.cuda.is_available(),'CUDA required for this frozen prediction path')
 model.to('cuda',memory_format=torch.channels_last);raw=args.data_root.resolve()
 val_images,val_seconds=infer(model,raw,rows['val'],pc,'val');val_study=aggregate(rows['val'],val_images)
 recheck=validation_check(rows['val'],val_images,val_study,files['validation_predictions']);save(WORK/'validation_recheck.json',recheck)
 test_images,test_seconds=infer(model,raw,rows['test'],pc,'test');test_study=aggregate(rows['test'],test_images,order)
 require(test_study['probabilities'].shape==(1414,10),'Wrong test Study count')
 for key,value in model.state_dict().items():require(torch.equal(value.detach().cpu(),initial_state[key]),'Inference mutated trained weights')
 image_path=WORK/'test_image_probabilities.npz';study_path=WORK/'test_study_probabilities.npz'
 npz(image_path,image_paths=np.asarray([r['image_path'] for r in rows['test']],str),subjects=np.asarray([r['subject_id'] for r in rows['test']],str),study_ids=np.asarray([r['study_id'] for r in rows['test']],str),probabilities=test_images)
 npz(study_path,subjects=test_study['subjects'],study_ids=test_study['study_ids'],probabilities=test_study['probabilities'])
 config=dict(schema_version=1,model=dict(name='CAFormer-S18-384-EMA',architecture='caformer_s18',variant='caformer_s18.sail_in1k_384',checkpoint=dict(path=str(files['checkpoint']),sha256=EXPECTED_WEIGHT,epoch=1,source_total_epoch=6,arm='A_BCE_EMA',track='ema'),native_pretrained_sha256=EXPECTED_NATIVE),
  gates=dict(path=str(files['gates_json']),sha256=sha(files['gates_json']),class_ids=list(range(1,11)),values=EXPECTED_GATES),
  preprocessing=dict(size=384,interpolation='bicubic',resize_mode='full_field_pad',RGB=True,mean=pc['mean'],std=pc['std'],center_crop=False),
  inference=dict(image_batch_size=4,autocast='float16',probability_dtype='float32',aggregation='float32_mean_image_sigmoid_probabilities_by_Study'),
  official_test_csv=dict(path=str(files['official_test_csv']),sha256=EXPECTED_TEST),manifest=dict(path=str(files['manifest']),sha256=EXPECTED_MANIFEST),source_config=dict(path=str(files['pretrained_config']),sha256=sha(files['pretrained_config'])),highest_validation_single_model_candidate=True,statistically_confirmed_AUC_improvement=False,auc_gain_vs_native_ci95=[-.0008183884988084588,.0034409896943986044])
 save(OUT/'model_config.json',config)
 destination=OUT/'CAFormer-S18-384-EMA.csv';part=destination.with_suffix('.csv.tmp')
 with part.open('w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f,lineterminator='\n');writer.writerow(FIELDS)
  for subject,study,p in zip(test_study['subjects'],test_study['study_ids'],test_study['probabilities']):
   for c in range(10):writer.writerow([str(subject),str(study),c+1,format(float(p[c]),'.12f'),format(EXPECTED_GATES[c],'.12g')])
 part.replace(destination)
 require(all(sha(path)==digest for path,digest in sources.items()),'Source inputs changed during inference')
 record=dict(state='generated_pending_independent_CPU_validation',created_at=datetime.now().astimezone().isoformat(),model_config_sha256=sha(OUT/'model_config.json'),source_inputs=sources,
  submission=dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,rows=14140,columns=FIELDS,encoding='UTF-8 without BOM',probability_decimal_places=12),validation_recheck=recheck,
  test=dict(images=2218,studies=1414,patients=136,labels_available=False),prediction_artifacts={str(q):sha(q) for q in (image_path,study_path)},inference_seconds=dict(validation=val_seconds,test=test_seconds),
  exporter_sha256=sha(Path(__file__)),no_training=True,no_test_threshold_fitting=True,github_uploaded=False,original_training_reports_unchanged=True)
 save(OUT/'校验记录.json',record);save(WORK/'status.json',dict(state=record['state'],rows=14140))
 print('EMA_TEST_EXPORTED_PENDING_CPU_VALIDATION',sha(destination),flush=True)

if __name__=='__main__':
 args=arguments()
 try:main(args)
 except Exception as exc:
  if 'WORK' in globals():save(WORK/'status.json',dict(state='failed',error=str(exc)))
  raise
