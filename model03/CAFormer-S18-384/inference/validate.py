"""CPU-only CSV validation; optional NPZ evidence independently recomputes Study means."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from common import need,sha,read,profile,load_gates,official_metadata,validate_csv

def verify(submission,model,gates_path,official_test_csv,prediction_dir=None,checkpoint=None):
    import numpy as np
    p=profile(model);gates=load_gates(gates_path,p);meta=official_metadata(official_test_csv)
    if checkpoint:need(sha(checkpoint)==p['checkpoint_sha256'],'Checkpoint SHA mismatch')
    expected=None;exact=False
    if prediction_dir:
        directory=Path(prediction_dir);receipt=read(directory/'prediction_receipt.json')
        need(receipt['model']==model and receipt['checkpoint_sha256']==p['checkpoint_sha256'] and receipt['arm']==p['arm'] and receipt['variant']==p['variant'],'Prediction evidence belongs to a different model')
        need(receipt['gates_sha256']==sha(gates_path) and receipt['official_test_sha256']==sha(official_test_csv),'Prediction evidence Gate/metadata differs')
        need(receipt['submission_sha256']==sha(submission),'CSV SHA differs from prediction receipt')
        image_file=directory/'image_probabilities.npz';study_file=directory/'study_probabilities.npz'
        for q in (image_file,study_file):need(sha(q)==receipt['prediction_artifact_sha256'][q.name],'Prediction NPZ SHA mismatch')
        with np.load(image_file,allow_pickle=False) as z:
            paths=z['image_paths'].copy();subjects=z['subjects'].copy();studies=z['study_ids'].copy();image_p=z['probabilities'].copy()
        need(image_p.shape==(2218,10) and image_p.dtype==np.float32 and np.isfinite(image_p).all() and ((image_p>=0)&(image_p<=1)).all(),'Invalid image probability archive')
        need(paths.shape==subjects.shape==studies.shape==(2218,),'Wrong image ID lengths')
        groups={key:[] for key in meta['study_order']};seen=set()
        for i,(path,subject,study) in enumerate(zip(paths,subjects,studies)):
            q=Path(str(path).replace('\\','/'));need(not q.is_absolute() and '..' not in q.parts and len(q.parts)==4 and q.parts[0]=='test','Unsafe image path in evidence')
            triple=(str(subject),str(study),q.stem)
            need(triple==meta['triples'][i] and q.parts[1:3]==triple[:2] and triple not in seen,'NPZ image ID/order mismatch')
            seen.add(triple);groups[(str(subject),str(study))].append(i)
        # Independent regrouping of original image rows, with exact float32 means.
        expected=np.stack([image_p[index].mean(axis=0,dtype=np.float32) for index in groups.values()])
        with np.load(study_file,allow_pickle=False) as z:
            need(z['probabilities'].dtype==np.float32,'Study probability dtype changed');np.testing.assert_array_equal(expected,z['probabilities'])
            np.testing.assert_array_equal(z['subjects'],np.asarray([q[0] for q in meta['study_order']],str))
            np.testing.assert_array_equal(z['study_ids'],np.asarray([q[1] for q in meta['study_order']],str))
        exact=True
    checked=validate_csv(Path(submission).read_bytes(),meta,gates,expected)
    return dict(state='passed',model=model,checkpoint_sha256=p['checkpoint_sha256'],checkpoint_file_checked=bool(checkpoint),rows=checked['rows'],images=2218,studies=1414,patients=136,submission_sha256=sha(submission),gates_sha256=sha(gates_path),official_test_sha256=sha(official_test_csv),scope='structure_Gates_and_exact_FP32_aggregation' if exact else 'structure_and_Gates_only',image_to_Study_FP32_recomputed=exact,rounding_max_difference=checked['max_rounding_difference'],no_model_import_or_GPU=True,test_labels_available=False,test_AUC_or_F1_computed=False)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--model',choices=('a0_raw','a0_ema','photo_raw'),required=True)
    for key in ('submission','gates','official-test-csv'):a.add_argument('--'+key,type=Path,required=True)
    a.add_argument('--prediction-dir',type=Path,help='Optional local infer output with image/Study NPZ and receipt; NPZ need not be committed')
    a.add_argument('--checkpoint',type=Path,help='Optional complete trained checkpoint for additional file-SHA check')
    v=a.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    print(json.dumps(verify(v.submission,v.model,v.gates,v.official_test_csv,v.prediction_dir,v.checkpoint),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
