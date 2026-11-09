"""CPU-only independent validation of the CAFormer-S18-384-EMA selected epoch1 official test CSV."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,re,sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path,PurePosixPath

CHECKPOINT_SHA='f1f862bd6c76235e59e96b17b4bdae92aa497846b4eef42e8fb33b2b34ddee85'
TEST_SHA='63e107d840cbf27aee7c1c0ced1157aef454baa1141fdc3851e6c3011471529e'
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
NATIVE_SHA='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
GATES=[.1,.25,.15,.175,.2,.175,.225,.25,.225,.3]
COLUMNS=['Subject_id','Study_id','Predict_class','Probability','Gate']
PROB_RE=re.compile(r'(?:0\.[0-9]{12}|1\.000000000000)\Z')

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def string(value):
    return value.decode('utf-8') if isinstance(value,bytes) else str(value)

def args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--official-test-csv',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--gates-json',type=Path)
    return parser.parse_args()

def resolve_path(explicit,record,base):
    path=Path(explicit if explicit is not None else record['path']).expanduser()
    return (path if path.is_absolute() else base/path).resolve()

def path_identity(raw):
    raw=string(raw).replace('\\','/');path=PurePosixPath(raw)
    require(not path.is_absolute() and len(path.parts)==4 and path.parts[0]=='test' and '..' not in path.parts,'Invalid relative test image path')
    return raw,(path.parts[1],path.parts[2],path.stem)

def main(a):
    out=a.output_dir.expanduser().resolve();require(out.is_dir(),'Output directory missing')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):os.environ[key]=str(out/'work')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    cfg_path=out/'model_config.json';receipt_path=out/'校验记录.json';csv_path=out/'CAFormer-S18-384-EMA.csv'
    cfg=read(cfg_path);receipt=read(receipt_path)
    require(cfg.get('schema_version')==1,'Unsupported export config schema')
    model=cfg['model'];pre=cfg['preprocessing'];inf=cfg['inference'];gate_record=cfg['gates']
    require(model['architecture']=='caformer_s18' and model['variant']=='caformer_s18.sail_in1k_384','Wrong native CAFormer version')
    require(model['checkpoint']['sha256']==CHECKPOINT_SHA and model['checkpoint']['epoch']==1 and model['checkpoint']['source_total_epoch']==6 and model['checkpoint']['arm']=='A_BCE_EMA' and model['checkpoint']['track']=='ema','Wrong selected checkpoint metadata')
    require(model['native_pretrained_sha256']==NATIVE_SHA,'Wrong native ImageNet source')
    require(pre['size']==384 and pre['interpolation']=='bicubic' and pre['resize_mode']=='full_field_pad' and pre['RGB'] is True and pre['center_crop'] is False,'Native preprocessing config differs')
    require(pre['mean']==[.485,.456,.406] and pre['std']==[.229,.224,.225],'Native normalization differs')
    require(inf['image_batch_size']==4 and inf['autocast']=='float16' and inf['probability_dtype']=='float32','Inference precision or microbatch differs')
    require(inf['aggregation']=='float32_mean_image_sigmoid_probabilities_by_Study','Wrong Study aggregation')
    require(gate_record['class_ids']==list(range(1,11)) and gate_record['values']==GATES,'Export Gates differ')
    test_path=resolve_path(a.official_test_csv,cfg['official_test_csv'],out)
    manifest_path=resolve_path(a.manifest,cfg['manifest'],out)
    checkpoint_path=resolve_path(a.checkpoint,model['checkpoint'],out)
    gate_path=resolve_path(a.gates_json,gate_record,out)
    source_config=resolve_path(None,cfg['source_config'],out)
    image_npz=out/'work/test_image_probabilities.npz';study_npz=out/'work/test_study_probabilities.npz'
    receipt_sha_before=sha(receipt_path)
    inputs=[cfg_path,csv_path,test_path,manifest_path,checkpoint_path,gate_path,source_config,image_npz,study_npz]
    before={str(p):sha(p) for p in inputs}
    require(before[str(checkpoint_path)]==CHECKPOINT_SHA,'Selected checkpoint file SHA mismatch')
    require(before[str(test_path)]==TEST_SHA==cfg['official_test_csv']['sha256'],'Official test CSV SHA mismatch')
    require(before[str(manifest_path)]==MANIFEST_SHA==cfg['manifest']['sha256'],'Official manifest SHA mismatch')
    require(before[str(gate_path)]==gate_record['sha256'],'Gate file SHA mismatch')
    require(before[str(source_config)]==cfg['source_config']['sha256'],'Source config SHA mismatch')
    gates=read(gate_path)
    require(gates['checkpoint_sha256']==CHECKPOINT_SHA and gates['best_epoch']==1 and gates['arm']=='A_BCE_EMA' and gates['variant']=='ema' and gates['pretrained_sha256']==NATIVE_SHA,'Source Gate binding differs')
    require(gates['class_ids']==list(range(1,11)) and gates['final_gates_by_class_id']==GATES,'Source Gate values differ')
    require(receipt['state'] in ('generated_pending_independent_CPU_validation','passed'),'Export is not ready for independent validation')
    require(receipt['validation_recheck']['state']=='passed','Exporter original-validation replay did not pass')
    require(receipt['submission']['sha256']==before[str(csv_path)] and receipt['submission']['bytes']==csv_path.stat().st_size,'Export receipt CSV fingerprint differs')
    require(receipt['submission']['rows']==14140 and receipt['submission']['columns']==COLUMNS,'Export receipt schema differs')
    require(receipt['test']['images']==2218 and receipt['test']['studies']==1414 and receipt['test']['patients']==136 and receipt['test']['labels_available'] is False,'Export receipt cohort differs')
    official=OrderedDict();official_images=set();subjects=set()
    with test_path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);require(set(('Dicom_id','Subject_id','Study_id')).issubset(reader.fieldnames or []),'Official test columns missing')
        for row in reader:
            key=(row['Subject_id'],row['Study_id'],row['Dicom_id']);study=(key[0],key[1])
            require(all(key) and key not in official_images,'Duplicate/empty official image ID')
            official_images.add(key);subjects.add(key[0]);official.setdefault(study,[]).append(key)
    require(len(official_images)==2218 and len(official)==1414 and len(subjects)==136,'Official test coverage differs')
    require(len({key[1] for key in official})==1414,'Study ID maps to multiple patients')
    manifest_images={};manifest_studies=set()
    with manifest_path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split']!='test':continue
            path,key=path_identity(row['image_path'])
            require(row['label_ids']=='' and row['subject_id']==key[0] and row['study_id']==key[1],'Manifest labels or path IDs differ')
            require(key not in manifest_images,'Duplicate test image in manifest')
            manifest_images[key]=path;manifest_studies.add((key[0],key[1]))
    require(set(manifest_images)==official_images and manifest_studies==set(official),'Manifest/official test IDs differ')
    with np.load(image_npz,allow_pickle=False) as z:
        require(set(('image_paths','subjects','study_ids','probabilities')).issubset(z.files),'Image NPZ keys missing')
        paths=z['image_paths'].copy();image_subjects=z['subjects'].copy();image_studies=z['study_ids'].copy();image_p=z['probabilities'].copy()
    require(image_p.shape==(2218,10) and image_p.dtype==np.float32,'Image predictions must be float32[2218,10]')
    require(np.isfinite(image_p).all() and ((image_p>=0)&(image_p<=1)).all(),'Invalid image probabilities')
    require(paths.shape==image_subjects.shape==image_studies.shape==(2218,),'Image NPZ ID lengths differ')
    groups=OrderedDict((k,[]) for k in official);seen=set()
    for i,(path,subject,study) in enumerate(zip(paths,image_subjects,image_studies)):
        path,key=path_identity(path)
        require(key in official_images and key not in seen and manifest_images[key]==path,'Unknown/duplicate image prediction path')
        require(string(subject)==key[0] and string(study)==key[1],'Image NPZ path/ID mismatch')
        seen.add(key);groups[(key[0],key[1])].append(i)
    require(seen==official_images and all(groups.values()),'Image predictions omit official test images')
    # Use image NPZ row order within each Study, independently of Study output order.
    calculated=np.stack([image_p[indices].mean(0,dtype=np.float32) for indices in groups.values()])
    with np.load(study_npz,allow_pickle=False) as z:
        require(set(('subjects','study_ids','probabilities')).issubset(z.files),'Study NPZ keys missing')
        study_subjects=z['subjects'].copy();study_ids=z['study_ids'].copy();study_p=z['probabilities'].copy()
    expected_subjects=np.asarray([k[0] for k in official],dtype=str);expected_studies=np.asarray([k[1] for k in official],dtype=str)
    require(study_p.dtype==np.float32 and study_p.shape==(1414,10),'Study predictions must be float32[1414,10]')
    require(np.array_equal(np.asarray([string(v) for v in study_subjects]),expected_subjects) and np.array_equal(np.asarray([string(v) for v in study_ids]),expected_studies),'Study NPZ official first-seen order differs')
    require(np.array_equal(study_p,calculated),'Float32 Study means do not reproduce exactly')
    raw=csv_path.read_bytes();require(not raw.startswith(b'\xef\xbb\xbf'),'Submission must be UTF8 without BOM');raw.decode('utf-8',errors='strict')
    rows=0;max_error=0.;positive_counts=[0]*10;ordered_studies=list(official)
    with csv_path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);require(reader.fieldnames==COLUMNS,'Submission columns/order differ')
        for row in reader:
            require(rows<14140 and None not in row,'Extra CSV rows or columns')
            study_index,class_index=divmod(rows,10);subject,study=ordered_studies[study_index]
            require(row['Subject_id']==subject and row['Study_id']==study and row['Predict_class']==str(class_index+1),'CSV IDs/class/order differ')
            require(PROB_RE.fullmatch(row['Probability']) is not None,'Probability is not twelve-decimal [0,1] format')
            actual=float(row['Probability']);expected=float(calculated[study_index,class_index]);gate=float(row['Gate'])
            require(math.isfinite(actual) and 0<=actual<=1 and math.isfinite(gate) and gate==GATES[class_index],'CSV probability/Gate invalid')
            error=abs(actual-expected);max_error=max(max_error,error)
            require(error<=5.01e-13 and row['Probability']==format(expected,'.12f'),'CSV probability does not match recomputed Study mean')
            require((actual>=gate)==(expected>=GATES[class_index]),'Rounding changed threshold decision')
            positive_counts[class_index]+=int(actual>=gate);rows+=1
    require(rows==14140,'Missing CSV rows')
    for path,digest in before.items():require(sha(path)==digest,'Input changed during validation: '+path)
    require(sha(receipt_path)==receipt_sha_before,'Exporter receipt changed during independent validation')
    validator_sha=sha(__file__)
    report={'state':'passed','checked_at':datetime.now().astimezone().isoformat(),'validator':str(Path(__file__).resolve()),'validator_sha256':validator_sha,'submission':{'path':str(csv_path),'sha256':before[str(csv_path)],'bytes':csv_path.stat().st_size,'rows':rows,'columns':COLUMNS,'encoding':'UTF-8 without BOM'},'cohort':{'images':2218,'studies':1414,'patients':136,'classes_per_study':10},'checkpoint_sha256':CHECKPOINT_SHA,'native_pretrained_sha256':NATIVE_SHA,'best_epoch':1,'source_total_epoch':6,'model':'CAFormer-S18-384-EMA','variant':'ema','gates':GATES,'aggregation':'FP32 image probability mean, independently regrouped in official Study first-seen order','max_abs_csv_rounding_difference':max_error,'study_npz_recomputed_exactly':True,'all_paths_and_IDs_match_official_csv_and_manifest':True,'threshold_decisions_unchanged_by_rounding':True,'positive_study_counts_by_class':positive_counts,'test_labels_available':False,'test_AUC_or_F1_computed':False,'no_model_import_or_GPU':True,'export_receipt_sha256_before_independent_update':receipt_sha_before,'input_sha256':before}
    result_path=out/'独立校验.json';atomic_json(result_path,report)
    receipt['state']='passed';receipt['independent_validation']={'path':str(result_path),'sha256':sha(result_path),'validator_sha256':validator_sha,'state':'passed'}
    atomic_json(receipt_path,receipt)
    print('INDEPENDENT_CPU_SUBMISSION_PASSED '+json.dumps({'rows':rows,'images':2218,'studies':1414,'patients':136,'csv_sha256':before[str(csv_path)],'max_rounding_error':max_error},ensure_ascii=False),flush=True)

if __name__=='__main__':main(args())
