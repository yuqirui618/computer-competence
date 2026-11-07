"""Portable, CPU-safe metadata and numeric contracts for three frozen CAFormer models."""
from __future__ import annotations
import csv,hashlib,io,json,math,re
from collections import OrderedDict
from pathlib import Path
FIELDS=['Subject_id','Study_id','Predict_class','Probability','Gate']
TEST_SHA='63e107d840cbf27aee7c1c0ced1157aef454baa1141fdc3851e6c3011471529e'
MANIFEST_SHA='b7c280dc0edc8b99d2aa1210651bab7ff9724d4c0a26b76b231c1935b275c142'
NATIVE_SHA='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
MEAN=[.485,.456,.406];STD=[.229,.224,.225]
ID_PATTERN=re.compile(r'[A-Za-z0-9_-]+\Z')
PROBABILITY_PATTERN=re.compile(r'(?:0\.[0-9]{12}|1\.000000000000)\Z')

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def profile(model):
    registry=read(Path(__file__).with_name('profiles.json'))
    need(model in registry['models'],'Unknown frozen model profile')
    return registry['models'][model]

def load_gates(path,p):
    g=read(path)
    need(g['checkpoint_sha256']==p['checkpoint_sha256'] and g['arm']==p['arm'] and g['variant']==p['variant'] and g['best_epoch']==5,'Gate/checkpoint/track binding mismatch')
    need(g['architecture']=='caformer_s18' and g['class_ids']==list(range(1,11)),'Wrong Gate architecture/class order')
    values=g['final_gates_by_class_id']
    need(values==p['gates'] and len(values)==10 and all(math.isfinite(x) and 0<=x<=1 for x in values),'Wrong frozen Gate values')
    if 'oof_macro_f1' in g:need(abs(g['oof_macro_f1']-p['validation']['patient_oof_F1'])<=1e-12,'Gate calibration belongs to another selected weight')
    return values

def build_metadata(rows,expected=(2218,1414,136)):
    rows=list(rows);triples=[];seen=set();order=OrderedDict();study_subject={}
    for r in rows:
        need({'Subject_id','Study_id','Dicom_id'}<=set(r),'Official test columns missing')
        triple=tuple(r[k] for k in ('Subject_id','Study_id','Dicom_id'))
        need(all(isinstance(x,str) and ID_PATTERN.fullmatch(x) for x in triple),'Unsafe/empty image identifier')
        subject,study,_=triple
        need(triple not in seen,'Duplicate official image identifier');seen.add(triple);triples.append(triple)
        need(study not in study_subject or study_subject[study]==subject,'Study belongs to multiple subjects')
        study_subject[study]=subject;order.setdefault((subject,study),[]).append(len(triples)-1)
    counts=(len(triples),len(order),len({x[0] for x in order}))
    need(counts==expected,'Official test coverage mismatch: '+str(counts))
    return dict(rows=rows,triples=triples,study_order=list(order),groups=order,counts=counts)

def official_metadata(path):
    need(sha(path)==TEST_SHA,'Provide the original frozen official test.csv; altered/reordered metadata is rejected')
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return build_metadata(csv.DictReader(f))

def manifest_order_check(path,meta):
    need(sha(path)==MANIFEST_SHA,'Optional manifest differs from the original official manifest')
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
    triples=[(r['subject_id'],r['study_id'],Path(r['image_path'].replace('\\','/')).stem) for r in rows]
    need(triples==meta['triples'],'Manifest and official test image order differ; fixed global batch4 must be preserved')
    need(all(r['label_ids']=='' for r in rows),'Test labels must be absent')
    return dict(state='identical',images=len(rows),manifest_sha256=MANIFEST_SHA)

def image_paths(raw,meta):
    root=Path(raw).resolve();test=(root/'test').resolve();result=[]
    for subject,study,dicom in meta['triples']:
        directory=test/subject/study
        matches=[directory/(dicom+s) for s in ('.jpg','.jpeg','.png','.JPG','.JPEG','.PNG')]
        matches=list(dict.fromkeys(x.resolve() for x in matches if x.is_file()))
        need(len(matches)==1,'Missing/ambiguous official image: '+dicom)
        p=matches[0];need(p.is_relative_to(test),'Image escapes official test directory')
        result.append(p)
    need(len(set(result))==2218,'Repeated raw image path')
    return result

def aggregate_probabilities(image_p,meta):
    import numpy as np
    need(image_p.dtype==np.float32 and image_p.shape==(len(meta['triples']),10),'Require full float32 image probabilities')
    need(np.isfinite(image_p).all() and ((image_p>=0)&(image_p<=1)).all(),'Invalid image probabilities')
    return np.stack([image_p[index].mean(axis=0,dtype=np.float32) for index in meta['groups'].values()])

def submission_bytes(study_p,meta,gates):
    import numpy as np
    need(study_p.dtype==np.float32 and study_p.shape==(len(meta['study_order']),10),'Require full float32 Study probabilities')
    need(np.isfinite(study_p).all() and ((study_p>=0)&(study_p<=1)).all(),'Invalid Study probabilities')
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow(FIELDS)
    for (subject,study),p in zip(meta['study_order'],study_p):
        for c in range(10):writer.writerow([subject,study,c+1,format(float(p[c]),'.12f'),format(gates[c],'.12g')])
    return stream.getvalue().encode('utf-8')

def validate_csv(data,meta,gates,probabilities=None):
    need(not data.startswith(b'\xef\xbb\xbf'),'CSV must be UTF8 without BOM')
    text=data.decode('utf-8',errors='strict');need('\r' not in text,'Canonical CSV uses LF newlines')
    reader=csv.DictReader(io.StringIO(text,newline=''));need(reader.fieldnames==FIELDS,'Wrong official five columns/order')
    count=0;error=0.;positive=[0]*10
    for row in reader:
        need(count<10*len(meta['study_order']) and None not in row,'Extra rows/columns')
        i,c=divmod(count,10);subject,study=meta['study_order'][i]
        need(row['Subject_id']==subject and row['Study_id']==study and row['Predict_class']==str(c+1),'Missing/duplicate/reordered Study or class')
        need(PROBABILITY_PATTERN.fullmatch(row['Probability']) is not None,'Probability must have12 decimals in[0,1]')
        p=float(row['Probability']);g=float(row['Gate'])
        need(math.isfinite(p) and math.isfinite(g) and 0<=p<=1 and g==gates[c],'Invalid probability/Gate')
        if probabilities is not None:
            expected=float(probabilities[i,c]);difference=abs(p-expected);error=max(error,difference)
            need(difference<=5.01e-13 and row['Probability']==format(expected,'.12f'),'CSV differs from original float32 Study probability')
            need((p>=g)==(expected>=gates[c]),'Decimal rounding changed threshold decision')
        positive[c]+=int(p>=g);count+=1
    need(count==10*len(meta['study_order']),'Missing submission rows')
    return dict(rows=count,max_rounding_difference=error if probabilities is not None else None,positive_studies_by_class=positive)

def atomic_bytes(path,data):
    path=Path(path);part=path.with_name(path.name+'.tmp');need(not path.exists() and not part.exists(),'Refuse to overwrite output: '+str(path))
    part.write_bytes(data);part.replace(path)

def atomic_json(path,value):atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
