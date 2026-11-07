"""CPU-only regression checks; full checkpoints are optional and never forwarded."""
from __future__ import annotations
import argparse,ast,hashlib,json,os,sys,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
from common import need,sha,profile,load_gates,build_metadata,official_metadata,manifest_order_check,image_paths,aggregate_probabilities,submission_bytes,validate_csv,atomic_json

def rejects(function):
    try:function()
    except (ValueError,KeyError,TypeError,UnicodeError):return True
    raise AssertionError('Invalid data unexpectedly accepted')

def unit_checks():
    import numpy as np
    rows=[dict(Subject_id='subject_B',Study_id='study_B',Dicom_id='pic_0'),dict(Subject_id='subject_A',Study_id='study_A',Dicom_id='pic_1'),dict(Subject_id='subject_B',Study_id='study_B',Dicom_id='pic_2')]
    meta=build_metadata(rows,(3,2,2));x=np.arange(10,dtype=np.float32)/32
    images=np.stack([x+.125,x+.25,x+.625]);expected=np.stack([x+.375,x+.25]);p=aggregate_probabilities(images,meta)
    np.testing.assert_array_equal(p,expected);gates=[.5]*10;data=submission_bytes(p,meta,gates);checked=validate_csv(data,meta,gates,p)
    need(checked['rows']==20 and data.splitlines()[0]==b'Subject_id,Study_id,Predict_class,Probability,Gate','Header/count differs')
    tests={
      'duplicate_image_ID':lambda:build_metadata(rows+[rows[0]],(4,2,2)),
      'missing_image_ID':lambda:build_metadata(rows[:-1],(3,2,2)),
      'unsafe_image_ID':lambda:build_metadata([dict(Subject_id='../x',Study_id='study_B',Dicom_id='pic_0')],(1,1,1)),
      'Study_multiple_subjects':lambda:build_metadata([rows[0],dict(Subject_id='subject_A',Study_id='study_B',Dicom_id='pic_1')],(2,2,2)),
      'float64_probability_array':lambda:aggregate_probabilities(images.astype(np.float64),meta),
      'bad_header':lambda:validate_csv(data.replace(b'Probability',b'Score',1),meta,gates,p),
      'missing_CSV_class':lambda:validate_csv(b'\n'.join(data.splitlines()[:-1])+b'\n',meta,gates,p),
      'wrong_CSV_Study_order':lambda:validate_csv(data.replace(b'study_B',b'study_X',1),meta,gates,p),
      'wrong_Gate':lambda:validate_csv(data.replace(b',0.5\n',b',0.6\n',1),meta,gates,p),
      'wrong_probability_precision':lambda:validate_csv(data.replace(b'0.375000000000',b'0.375',1),meta,gates,p),
      'nonfinite_probability':lambda:validate_csv(data.replace(b'0.375000000000',b'nan',1),meta,gates,p),
      'wrong_12decimal_probability':lambda:validate_csv(data.replace(b'0.375000000000',b'0.400000000000',1),meta,gates,p),
      'unexpected_BOM':lambda:validate_csv(b'\xef\xbb\xbf'+data,meta,gates,p),
    }
    for name,fn in tests.items():need(rejects(fn),'Expected rejection failed: '+name)
    return dict(state='passed',analytical_FP32_Study_mean_exact=True,interleaved_image_and_firstseen_Study_order_correct=True,columns=5,fixture_rows=20,rejected_cases=list(tests),max_fixture_rounding_difference=checked['max_rounding_difference'])

def compare_references(reference_root,models,meta):
    import numpy as np
    results={}
    for model in models:
        p=profile(model);directory=Path(reference_root)/('CAFormer-S18-384-'+p['publish_label'])
        csv_file=directory/'submission.csv';image_file=directory/'work/test_image_probabilities.npz';study_file=directory/'work/test_study_probabilities.npz'
        need(csv_file.is_file() and image_file.is_file() and study_file.is_file(),'Requested reference export not ready: '+model)
        receipt_path=directory/'校验记录.json';receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        need(receipt['state']=='passed' and receipt['independent_validation']['state']=='passed','Reference export is not independently validated')
        need(any(digest==p['checkpoint_sha256'] for digest in receipt['source_inputs'].values()),'Reference export weight differs')
        with np.load(image_file,allow_pickle=False) as z:
            image_p=z['probabilities'].copy()
            need([(str(s),str(t),Path(str(q)).stem) for s,t,q in zip(z['subjects'],z['study_ids'],z['image_paths'])]==meta['triples'],'Reference image order differs from official test.csv')
        calculated=aggregate_probabilities(image_p,meta)
        with np.load(study_file,allow_pickle=False) as z:
            np.testing.assert_array_equal(calculated,z['probabilities']);np.testing.assert_array_equal(z['subjects'],np.asarray([q[0] for q in meta['study_order']],str));np.testing.assert_array_equal(z['study_ids'],np.asarray([q[1] for q in meta['study_order']],str))
        actual=submission_bytes(calculated,meta,p['gates']);need(actual==csv_file.read_bytes(),'Portable serialization differs from actual official export')
        checked=validate_csv(actual,meta,p['gates'],calculated)
        results[model]=dict(state='passed',images=2218,studies=1414,patients=136,rows=14140,FP32_image_to_Study_maxdiff=0.,CSV_byte_for_byte_equal=True,submission_sha256=hashlib.sha256(actual).hexdigest(),max_rounding_difference=checked['max_rounding_difference'],no_new_GPU_forward=True)
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('official-test-csv','manifest','raw-root','checkpoint-root','reference-root','output-report'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--reference-models',nargs='+',choices=('a0_raw','a0_ema','photo_raw'),default=[])
    parser.add_argument('--only-reference',action='store_true',help='Skip already-passed unit/checkpoint checks and compare only newly ready exports')
    a=parser.parse_args();here=Path(__file__).resolve().parent
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','HF_HOME','HF_XET_CACHE','TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR'):
        # The optional report location controls temporary caches; source package is never a runtime cache.
        base=(a.output_report.resolve().parent if a.output_report else Path.cwd())/'cpu_check_temp';q=base/key.lower();q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    result=dict(state='passed',scope='CPU-only metadata/numeric/model-load checks; no GPU/image forward/training',no_GPU=True,no_training=True,no_image_forward=True)
    if not a.only_reference:
        for path in here.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
        result['unit_checks']=unit_checks();result['AST_all_sources_passed']=True
    if a.official_test_csv:
        meta=official_metadata(a.official_test_csv);result['official_test_coverage']=dict(images=2218,studies=1414,patients=136)
        if a.manifest:result['manifest_test_image_order']=manifest_order_check(a.manifest,meta)
        if a.raw_root:result['raw_files_present']=len(image_paths(a.raw_root,meta))
        if a.reference_models:
            need(a.reference_root is not None,'--reference-root required');result['actual_export_replay']=compare_references(a.reference_root,a.reference_models,meta)
    else:need(not a.reference_models and not a.manifest and not a.raw_root,'Official metadata is required for data checks')
    if a.checkpoint_root and not a.only_reference:
        import torch
        from infer import build_model
        torch.set_num_threads(4);result['complete_checkpoint_CPU_load']={}
        for model_id in ('a0_raw','a0_ema','photo_raw'):
            p=profile(model_id);root=a.checkpoint_root/p['arm'];weight=root/('best_'+p['variant']+'.pth');gates=root/('gates_'+p['variant']+'.json')
            need(sha(weight)==p['checkpoint_sha256'],'Frozen checkpoint SHA differs');load_gates(gates,p);model,state=build_model(weight,p)
            need(not torch.cuda.is_initialized(),'CPU check initialized CUDA')
            result['complete_checkpoint_CPU_load'][model_id]=dict(state='passed',checkpoint_sha256=p['checkpoint_sha256'],parameters=sum(q.numel() for q in model.parameters()),all_parameters_frozen=not any(q.requires_grad for q in model.parameters()),training_mode=model.training,hidden_head_shape=list(model.head.fc.fc2.weight.shape))
            del model,state;gc.collect()
        result['CUDA_initialized']=torch.cuda.is_initialized()
    result['source_sha256']={q.name:sha(q) for q in here.iterdir() if q.is_file() and q.suffix in ('.py','.json') and q!=a.output_report and q.name not in ('cpu_checks.json','cpu_checks_ema.json')}
    if a.output_report:atomic_json(a.output_report,result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
