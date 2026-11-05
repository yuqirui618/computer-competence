"""Export official-format VALIDATION CSVs from frozen local Study predictions.

No inference, no torch imports, no training. Local labels are used only for
verification and aggregate metrics; labels/NPZ/weights are never exported.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,os,re,sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]
HEADER=['Subject_id','Study_id','Predict_class','Probability','Gate']
ORIGINAL_SHA='9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585'
GAMMA_SHA='b0066e479b1b0c383fa7daa2067ff0d6c954d20fbf81d1202af95481d67cc5f9'
REFERENCE_SUBMISSION_SHA='d49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8'
FINGERPRINTS={}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def pin(path):
    path=Path(path).resolve();digest=sha(path)
    old=FINGERPRINTS.setdefault(str(path),digest)
    require(old==digest,'Input changed during export: '+str(path))
    return path

def read(path):return json.loads(pin(path).read_text(encoding='utf-8-sig'))

def near(a,b,message):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=0,atol=1e-12),message)

def labels(value):
    values=tuple(sorted(set(int(x) for x in re.split(r'[;|,\s]+',value.strip()) if x)))
    require(values and all(1<=x<=10 for x in values),'Invalid official validation labels')
    return values

def official_cohort(data):
    raw=pin(data/'raw/cxr_first/valid.csv');manifest=pin(data/'喻数据产物/manifest.csv')
    studies=OrderedDict();pictures={}
    with raw.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        require(set(('Subject_id','Study_id','Dicom_id','Predict_class')).issubset(reader.fieldnames or []),'Unexpected official valid.csv fields')
        for row in reader:
            subject,study,pic=row['Subject_id'],row['Study_id'],row['Dicom_id'];truth=labels(row['Predict_class'])
            require(subject and study and pic,'Empty official ID')
            record=(subject,truth)
            require(studies.setdefault(study,record)==record,'Study has conflicting subject/labels')
            key=(subject,study,pic);require(key not in pictures,'Duplicate official validation image');pictures[key]=truth
    require(len(pictures)==1422 and len(studies)==859 and len({v[0] for v in studies.values()})==228,'Official validation size changed')
    manifest_val={};split_subjects={x:set() for x in ('train','val','test')};split_studies={x:{} for x in split_subjects};image_counts=dict.fromkeys(split_subjects,0)
    with manifest.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            split=row['split'];subject,study=row['subject_id'],row['study_id']
            require(split in split_subjects,'Unknown official split')
            split_subjects[split].add(subject);image_counts[split]+=1
            require(split_studies[split].setdefault(study,subject)==subject,'Study subject conflicts in manifest')
            if split=='val':
                key=(subject,study,Path(row['image_path']).stem)
                require(key not in manifest_val,'Duplicate manifest validation image')
                manifest_val[key]=labels(row['label_ids'])
    require(manifest_val==pictures,'manifest val images/IDs/labels differ from official valid.csv')
    require(all(not(split_subjects[a]&split_subjects[b]) for a,b in (('train','val'),('train','test'),('val','test'))),'Official patient splits overlap')
    require(image_counts==dict(train=30344,val=1422,test=2218),'Official image counts changed')
    require(len(split_studies['train'])==18513 and len(split_studies['test'])==1414,'Official Study counts changed')
    order=list(studies);truth=np.zeros((859,10),dtype=np.int8)
    for i,study in enumerate(order):truth[i,np.asarray(studies[study][1])-1]=1
    near(truth.sum(0),[24,37,46,68,115,177,193,214,260,306],'Validation class support changed')
    return dict(order=order,subjects=[studies[s][0] for s in order],targets=truth,test_studies=split_studies['test'])

def prediction(path,cohort):
    with np.load(pin(path),allow_pickle=False) as archive:
        require(set(('targets','probabilities','subjects','study_ids')).issubset(archive.files),'Missing Study prediction fields')
        arrays={k:archive[k].copy() for k in ('targets','probabilities','subjects','study_ids')}
    require(arrays['targets'].shape==arrays['probabilities'].shape==(859,10),'Wrong prediction shape')
    ids=arrays['study_ids'].astype(str).tolist();subjects=arrays['subjects'].astype(str).tolist()
    require(len(ids)==len(subjects)==len(set(ids))==859,'Duplicate/missing Study prediction IDs')
    require(set(ids)==set(cohort['order']),'Prediction Study set differs from official valid.csv')
    lookup={study:i for i,study in enumerate(ids)};ix=np.asarray([lookup[s] for s in cohort['order']])
    require([subjects[i] for i in ix]==cohort['subjects'],'Prediction subject mapping differs from valid.csv')
    require(np.array_equal(arrays['targets'][ix],cohort['targets']),'NPZ targets differ from official valid.csv labels')
    p=arrays['probabilities'][ix].astype(np.float64)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probability values')
    return p

def fit_gates(y,p):
    grid=sorted(np.arange(1,40)/40,key=lambda t:(abs(t-.5),t))
    values=[f1_score(y,p>=gate,average=None,zero_division=0) for gate in grid]
    return np.asarray(grid)[np.argmax(values,axis=0)]

def calibration(y,p,subjects):
    decisions=np.zeros_like(y,dtype=bool)
    for fit,held in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,subjects):
        require(not(set(subjects[fit])&set(subjects[held])),'Patient crossed threshold fitting fold')
        decisions[held]=p[held]>=fit_gates(y[fit],p[fit])
    return dict(oof_macro_f1=float(f1_score(y,decisions,average='macro',zero_division=0)),final_gates_by_class_id=fit_gates(y,p).tolist())

def confusion(y,z):
    return {key:value.astype(int).tolist() for key,value in dict(tp=np.sum(z&(y==1),0),fp=np.sum(z&(y==0),0),fn=np.sum(~z&(y==1),0)).items()}

def scores(y,p,gates):
    return dict(study_macro_auc=float(roc_auc_score(y,p,average='macro')),macro_f1_at_0_5=float(f1_score(y,p>=.5,average='macro',zero_division=0)),csv_final_gate_macro_f1_in_sample=float(f1_score(y,p>=gates,average='macro',zero_division=0)),macro_ap=float(average_precision_score(y,p,average='macro')),confusion_at_0_5=confusion(y,p>=.5),confusion_at_final_gate=confusion(y,p>=gates))

def checkpoint(path,expected,role,weight=1.):
    path=pin(path);require(FINGERPRINTS[str(path)]==expected,'Checkpoint SHA binding changed: '+str(path))
    return dict(role=role,path=str(path),sha256=expected,weight=weight)

def gate_values(gate):
    values=np.asarray(gate['final_gates_by_class_id'],dtype=np.float64)
    require(values.shape==(10,) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'Invalid final class gates')
    if 'class_ids' in gate:require(gate['class_ids']==list(range(1,11)),'Wrong class-id order')
    return values

def make_candidate(key,label,epoch,pred_path,gate_path,binding,cohort,recorded_auc,recorded_f1,recorded_calibration):
    gate=read(gate_path);gates=gate_values(gate);p=prediction(pred_path,cohort);y=cohort['targets']
    m=scores(y,p,gates);cal=calibration(y,p,np.asarray(cohort['subjects']))
    near(m['study_macro_auc'],recorded_auc,'Source AUC differs: '+key)
    near(m['macro_f1_at_0_5'],recorded_f1,'Source default F1 differs: '+key)
    near(cal['oof_macro_f1'],recorded_calibration['oof_macro_f1'],'Source OOF F1 differs: '+key)
    near(cal['oof_macro_f1'],gate['oof_macro_f1'],'Gate OOF F1 differs: '+key)
    near(cal['final_gates_by_class_id'],gates,'Source final gates differ: '+key)
    return dict(key=key,label=label,best_epoch=epoch,predictions_path=str(pin(pred_path)),gates_path=str(pin(gate_path)),binding=binding,probabilities=p,gates=gates,metrics=m,patient_fivefold_oof_f1=cal['oof_macro_f1'])

def collect(data,cohort):
    result=[];gamma=data/'MambaOut_二次数据增强_20261104';old=data/'MambaOut_ASL对照'
    protected=read(gamma/'protected_inputs.json')
    require(protected[str(old/'best_model.pth')]==ORIGINAL_SHA,'Original checkpoint receipt differs')
    require(protected[str(old/'gates.json')]==sha(old/'gates.json'),'Original gates receipt differs')
    b=checkpoint(old/'best_model.pth',ORIGINAL_SHA,'original_mambaout')
    original=read(gamma/'baseline_metrics.json')
    result.append(make_candidate('mambaout_original','保留的原MambaOut单模型','protected_original',gamma/'baseline_predictions.npz',old/'gates.json',dict(kind='single_model',source_checkpoints=[b],legacy_gate_binding_evidence=str(gamma/'protected_inputs.json')),cohort,original['validation']['study']['macro_auc'],original['validation']['study']['macro_f1'],original['calibration']))
    v=read(gamma/'candidate_verification.json');require(v['state']=='passed' and v['metrics_reproduced'] and v['probabilities_reproduced'],'Gamma reload receipt failed')
    require(v['checkpoint_sha256']==GAMMA_SHA,'Gamma SHA differs')
    b=checkpoint(v['checkpoint'],GAMMA_SHA,'mambaout_gamma_noise')
    gate=read(gamma/'candidate_gates.json');require(gate['checkpoint_sha256']==GAMMA_SHA and Path(gate['checkpoint']).resolve()==Path(b['path']),'Gamma gate binding differs')
    result.append(make_candidate('mambaout_gamma','MambaOut Gamma/轻噪声候选',v['epoch'],gamma/'verified_candidate_predictions.npz',gamma/'candidate_gates.json',dict(kind='single_model',source_checkpoints=[b]),cohort,v['validation']['study']['macro_auc'],v['validation']['study']['macro_f1'],v['calibration']))
    all_single={}
    for relative,names in [('新模型对照_20261105',('convnext_tiny','efficientnetv2_s')),('顶刊三模型对照_20261105',('caformer_s18','densenet121','hrnet_w18'))]:
        run=data/relative;master=read(run/'results.json');require(master['state']=='complete','Source run is not complete')
        receipt=read(run/'protected_inputs.json')
        for path,digest in receipt.items():require(sha(pin(path))==digest,'Protected source changed')
        for name in names:
            folder=run/name;r=read(folder/'results.json');h=read(folder/'history.json')
            require(r['state']=='complete' and r['epochs_completed']==len(h)==5,'Source model lacks five complete epochs')
            require(r==master['stages'][name] and r['history']==h,'Source result/history differs')
            best=max(h,key=lambda x:(x['validation']['study']['macro_auc'],x['validation']['study']['macro_f1']))
            require(best['epoch']==r['best_epoch'] and r['best_validation']==best['validation'],'Best epoch selection differs')
            require(all(row['images']==30344 and row['validation']['study']['count']==859 for row in h),'Incomplete source training')
            verify=r['verification'];require(verify['state']=='passed' and verify['metrics_reproduced'] and verify['predictions_reproduced'] and verify['max_abs_probability_difference']<=1e-7,'Model reload verification failed')
            b=checkpoint(r['best_checkpoint'],r['best_checkpoint_sha256'],name)
            gate_path=folder/'candidate_gates.json';g=read(gate_path)
            require(g['checkpoint_sha256']==b['sha256'] and Path(g['checkpoint']).resolve()==Path(b['path']),'Model gate binding differs')
            item=make_candidate(name,name,r['best_epoch'],folder/f"epoch_{r['best_epoch']}_predictions.npz",gate_path,dict(kind='single_model',source_checkpoints=[b]),cohort,r['best_validation']['study']['macro_auc'],r['best_validation']['study']['macro_f1'],r['best_calibration'])
            result.append(item);all_single[name]=item
    run=data/'顶刊三模型对照_20261105';combinations=read(run/'fixed_ensemble_results.json');plan=read(run/'fixed_ensemble_plan.json')
    require(combinations['state']=='complete' and combinations['plan']==plan and plan['alpha']==.5 and plan['source_weights']==[.5,.5],'Fixed-ensemble plan differs')
    for name in ('caformer_s18','densenet121','hrnet_w18'):
        r=combinations['ensembles'][name];binding=r['binding'];gate_path=run/'fixed_ensembles'/name/'candidate_gates.json';g=read(gate_path)
        require(r['state']=='complete' and binding==g['binding'] and binding['weights']==[.5,.5],'Ensemble gates/weights binding differs')
        require(binding['single_model_epoch']==all_single[name]['best_epoch'],'Ensemble chose a different single epoch')
        require(len(binding['source_checkpoints'])==2,'Expected two ensemble checkpoints')
        for b in binding['source_checkpoints']:checkpoint(b['path'],b['sha256'],b['role'],b['weight'])
        pred=run/'fixed_ensembles'/name/'predictions.npz'
        require(sha(pin(pred))==r['predictions_sha256'] and sha(pin(gate_path))==r['candidate_gates_sha256'],'Ensemble frozen source SHA differs')
        item=make_candidate('ensemble_'+name,'MambaOut Gamma + '+name+' 50:50',binding['single_model_epoch'],pred,gate_path,binding,cohort,r['metrics']['macro_auc'],r['metrics']['macro_f1_at_0_5'],r['calibration'])
        require(np.array_equal(item['probabilities'],(result[1]['probabilities']+all_single[name]['probabilities'])/2),'Ensemble probabilities are not exact frozen 50:50 means')
        result.append(item)
    require(len(result)==10 and len({r['key'] for r in result})==10,'Expected ten distinct validation exports')
    return result


def reference_submission(data,cohort):
    path=pin(data/'提交结果/MambaOut_S7微调第1轮_20261104/submission.csv')
    require(FINGERPRINTS[str(path)]==REFERENCE_SUBMISSION_SHA,'Original test submission changed')
    seen=set();class_gates={};rows=0
    with path.open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle);require(reader.fieldnames==HEADER,'Original official-format header differs')
        for row in reader:
            subject,study=row['Subject_id'],row['Study_id'];c=int(row['Predict_class']);prob=float(row['Probability']);gate=float(row['Gate'])
            require(study in cohort['test_studies'] and cohort['test_studies'][study]==subject,'Original submission test IDs differ')
            require(1<=c<=10 and math.isfinite(prob) and 0<=prob<=1 and math.isfinite(gate) and 0<=gate<=1,'Original submission contains invalid value')
            key=(subject,study,c);require(key not in seen,'Original submission duplicates key');seen.add(key);rows+=1
            require(class_gates.setdefault(c,gate)==gate,'Original submission gate is not global by class')
    require(rows==14140 and len({x[1] for x in seen})==1414,'Original test submission coverage differs')
    return dict(path=str(path),sha256=REFERENCE_SUBMISSION_SHA,data_rows=rows,studies=1414,read_only=True)

def csv_text(item,cohort):
    buffer=io.StringIO(newline='');writer=csv.writer(buffer,lineterminator='\n');writer.writerow(HEADER)
    for i,(subject,study) in enumerate(zip(cohort['subjects'],cohort['order'])):
        for j in range(10):writer.writerow([subject,study,j+1,format(float(item['probabilities'][i,j]),'.17g'),format(float(item['gates'][j]),'.17g')])
    return buffer.getvalue()

def write_bytes(path,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(body);temporary.replace(path)

def write_json(path,value):
    write_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))

def readback(path,item,cohort):
    body=path.read_bytes();require(not body.startswith(b'\xef\xbb\xbf'),'CSV must have UTF8 without BOM')
    body.decode('utf-8',errors='strict')
    p=np.full((859,10),np.nan,dtype=np.float64);g=np.full((859,10),np.nan,dtype=np.float64)
    seen=set();lookup={study:i for i,study in enumerate(cohort['order'])};actual_order=[]
    with path.open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle);require(reader.fieldnames==HEADER,'Wrong exact CSV header/order')
        for row in reader:
            require(None not in row and all(row[k] is not None for k in HEADER),'Extra or missing CSV columns')
            subject,study=row['Subject_id'],row['Study_id'];raw_class=row['Predict_class']
            require(raw_class in [str(x) for x in range(1,11)],'Invalid class literal')
            c=int(raw_class);require(study in lookup,'Unexpected validation Study')
            i=lookup[study];require(subject==cohort['subjects'][i],'CSV subject mismatch')
            key=(subject,study,c);require(key not in seen,'Duplicate CSV key');seen.add(key);actual_order.append((study,c))
            prob,gate=float(row['Probability']),float(row['Gate'])
            require(math.isfinite(prob) and 0<=prob<=1 and math.isfinite(gate) and 0<=gate<=1,'Invalid CSV probability/Gate')
            p[i,c-1]=prob;g[i,c-1]=gate
    require(len(seen)==8590 and len({x[1] for x in seen})==859,'CSV rows/Study coverage differ')
    expected_order=[(study,c) for study in cohort['order'] for c in range(1,11)]
    require(actual_order==expected_order,'CSV does not follow official validation Study order/class1..10')
    require(np.isfinite(p).all() and np.isfinite(g).all(),'CSV missing class values')
    require(np.array_equal(p,item['probabilities']),'CSV changed probabilities during roundtrip')
    require(np.array_equal(g,np.broadcast_to(item['gates'],g.shape)),'CSV gates changed or are not global per class')
    measured=scores(cohort['targets'],p,item['gates'])
    for key,value in item['metrics'].items():
        if isinstance(value,dict):
            for k,v in value.items():near(measured[key][k],v,'CSV confusion counts differ')
        else:near(measured[key],value,'CSV roundtrip metric differs')
    return dict(state='passed',data_rows=8590,rows_including_header=8591,studies=859,patients=228,classes_per_study=10,
        exact_header=HEADER,utf8_without_bom=True,official_valid_IDs_and_targets_checked=True,unique_subject_study_class=True,
        row_order='Official valid.csv first Study occurrence; class IDs1..10 ascending',probabilities_bit_equal_after_float64_roundtrip=True,
        max_abs_probability_roundtrip_difference=float(np.max(np.abs(p-item['probabilities']))),gates_global_per_class=True,
        csv_metrics_from_independent_readback=measured,patient_fivefold_oof_f1_reference=item['patient_fivefold_oof_f1'],
        final_gate_f1_scope='In-sample: final class gates were fitted on all reused validation labels; not the patient-OOF score')

def export(data,output):
    require(data.is_absolute() and data.is_dir(),'--data-root must be an existing absolute data directory')
    require(output.is_absolute() and output.drive.upper()=='E:','Generated validation artifacts must stay on E drive')
    cohort=official_cohort(data);reference=reference_submission(data,cohort);items=collect(data,cohort)
    artifacts=[];checks={}
    for item in items:
        key=item['key'];csv_relative=Path('predictions')/(key+'.csv');gate_relative=Path('gates')/(key+'.json')
        csv_path=output/csv_relative;gate_path=output/gate_relative
        write_bytes(csv_path,csv_text(item,cohort).encode('utf-8'))
        checks[key]=readback(csv_path,item,cohort)
        gate_record=dict(schema_version=1,key=key,class_ids=list(range(1,11)),final_gates_by_class_id=item['gates'].tolist(),
            binding=item['binding'],source_gate_file=item['gates_path'],source_gate_sha256=FINGERPRINTS[item['gates_path']],
            final_gate_fit_scope='All official validation labels, reused for tuning; optimistic in-sample F1',
            patient_fivefold_oof_f1_reference=item['patient_fivefold_oof_f1'],decision_rule_for_local_metrics='Probability >= Gate',
            validation_studies=859,validation_patients=228,not_a_test_submission=True)
        write_json(gate_path,gate_record)
        artifacts.append(dict(key=key,label=item['label'],best_epoch=item['best_epoch'],csv=csv_relative.as_posix(),csv_sha256=sha(csv_path),
            gates=gate_relative.as_posix(),gates_sha256=sha(gate_path),source_predictions=item['predictions_path'],source_predictions_sha256=FINGERPRINTS[item['predictions_path']],
            source_gates=item['gates_path'],source_gates_sha256=FINGERPRINTS[item['gates_path']],binding=item['binding'],
            aggregate_validation_metrics=checks[key]['csv_metrics_from_independent_readback'],patient_fivefold_oof_f1=item['patient_fivefold_oof_f1']))
    require(all(sha(path)==digest for path,digest in FINGERPRINTS.items()),'A source input changed during export')
    manifest=dict(schema_version=1,exporter='export_validation.py',exporter_sha256=sha(Path(__file__)),generated_at=datetime.now().astimezone().isoformat(),
        kind='official_five_column_format_validation_predictions_not_test_submission',split='official validation',header=HEADER,files=artifacts,
        count=10,data_rows_per_csv=8590,studies_per_csv=859,patients_per_csv=228,probability_serialization='float64 .17g; exact independent readback equality',
        gate_source='Each source model/combo final_gates_by_class_id fitted using the entire validation set, not per-fold OOF gates',
        source_fingerprints=FINGERPRINTS,reference_test_submission=reference,source_files_modified=False,
        labels_exported=False,NPZ_exported=False,weights_exported=False,new_test_inference=False,new_training=False,
        metrics_caution='CSV final-gate F1 is in-sample. Patient fivefold OOF F1 is separately recomputed; models were also selected on reused validation, so neither is a hidden-test result.')
    write_json(output/'export_checks.json',dict(state='passed',kind='independent_CSV_readback_checks',all_ten_exports_verified=True,checks=checks,
        all_input_SHA_unchanged=True,official_labels_used_only_locally=True,labels_exported=False,new_test_inference=False))
    write_json(output/'export_manifest.json',manifest)
    lines=['# 验证集预测结果：官方五列格式','',
        '**这里是验证集结果，不是可直接交到赛方的测试集submission.csv。** 每份包含859次验证检查×10类别＝8,590条数据，另加1行表头；正式测试集是1,414次检查×10＝14,140条数据。没有进行新的测试集推理。','',
        '精确表头：`Subject_id,Study_id,Predict_class,Probability,Gate`。依据仓库赛题相关资料的任务描述.docx内嵌示例和已核验的原submission.csv。CSV为UTF-8无BOM；保留官方ID字符串，按valid.csv首次Study出现顺序及类别1至10排序。概率与Gate用17位有效数字保存，独立回读后与原冻结概率数值完全一致。','',
        '| 文件key | 对应AUC最佳轮 | Study AUC | CSV最终Gate的F1（拟合内） | 患者五折OOF F1（内部估计） |',
        '| --- | --- | --- | --- | --- |']
    for item in items:lines.append(f"| {item['key']} | {item['best_epoch']} | {item['metrics']['study_macro_auc']:.8f} | {item['metrics']['csv_final_gate_macro_f1_in_sample']:.8f} | {item['patient_fivefold_oof_f1']:.8f} |")
    lines+=['','`predictions/<key>.csv`保存五列预测；`gates/<key>.json`保存最终逐类阈值及权重SHA绑定；`export_manifest.json`记录全部输入/输出SHA和来源；`export_checks.json`记录独立回读的格式、AUC、F1及每类TP/FP/FN。','',
        '**F1口径必须分开。** CSV每一类在全部Study使用同一个最终Gate，来自该模型/组合在整个验证集上拟合的final_gates_by_class_id。它没有使用每折不同的OOF阈值，因此CSV回读的F1是拟合内分数，不能当作泛化提升。患者五折OOF F1另行列出，每折只在拟合患者选阈值；模型和轮次仍曾用同一验证集选择，所以也是内部估计，不能当作隐藏测试分数。AUC从原连续概率计算，不用Gate。','',
        '三个ensemble是现有MambaOut Gamma候选与对应顶刊模型的固定50:50概率均值，先各自按Study平均图像概率，再按模型均值。阈值JSON同时绑定两份checkpoint SHA及0.5/0.5权重，不能当成任何一个单模型的阈值。','',
        '导出前用本机官方valid.csv重新构建检查级10类真值，并逐项核对manifest和NPZ中的subject、Study、图像与标签。所有10份CSV均独立回读验证：无重复subject/Study/类别键，每个Study完整10类，概率有限且在[0,1]，各类Gate全局固定，指标和原预测一致。真值标签、逐患者NPZ和模型权重没有复制到此目录。','',
        '在原E盘实验产物齐全时，从仓库根目录重导出：','',
        '```powershell',
        r"& 'E:\比赛数据集\envs\cxr-smoke-cu128\Scripts\python.exe' -B model01\model_comparison\tools\export_validation.py --data-root 'E:\比赛数据集' --output-dir 'E:\比赛数据集\验证结果重导出'",
        '```','',
        '该工具仅使用numpy/scikit-learn及标准库，不导入训练模块、不使用GPU、不改原实验、不运行训练或测试推理。需要源实验的实际权重/NPZ以核对SHA，因此只克隆仓库不会自动得到这些本地产物。']
    write_bytes(output/'README.md',('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(state='complete',exports=10,rows_each=8590,output=str(output),all_CSV_readback_checks_passed=True),ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('E:/比赛数据集'))
    parser.add_argument('--output-dir',type=Path,default=PACKAGE/'validation')
    args=parser.parse_args()
    global np,f1_score,roc_auc_score,average_precision_score,GroupKFold
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
    import numpy as np
    from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
    from sklearn.model_selection import GroupKFold
    try:
        export(args.data_root.resolve(),args.output_dir.resolve());return 0
    except (OSError,ValueError,KeyError,TypeError,IndexError,AssertionError) as error:
        print('Validation export refused: '+str(error),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
