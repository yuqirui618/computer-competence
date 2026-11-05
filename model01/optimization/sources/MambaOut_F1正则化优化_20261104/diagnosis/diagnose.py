"""Read-only fixed checkpoint diagnosis on official development validation data."""
import csv, gc, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA_ROOT=Path(r'E:\比赛数据集')
sys.path.insert(0,str(DATA_ROOT/'MambaOut_100轮'))
sys.path.insert(0,str(DATA_ROOT/'MambaOut_泛化优化'))
import train_and_optimize as reference
import train_generalization as core
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def counts(y,d):
 tp=((y==1)&d).sum(axis=0)
 fp=((y==0)&d).sum(axis=0)
 fn=((y==1)&~d).sum(axis=0)
 tn=((y==0)&~d).sum(axis=0)
 precision=np.divide(tp,tp+fp,out=np.zeros(10,dtype=float),where=tp+fp>0)
 recall=np.divide(tp,tp+fn,out=np.zeros(10,dtype=float),where=tp+fn>0)
 f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(10,dtype=float),where=2*tp+fp+fn>0)
 assert np.allclose(f1,f1_score(y,d,average=None,zero_division=0),atol=1e-15)
 return dict(tp=tp.tolist(),fp=fp.tolist(),fn=fn.tolist(),tn=tn.tolist(),predicted_positives=(tp+fp).tolist(),precision=precision.tolist(),recall=recall.tolist(),f1=f1.tolist(),macro_f1=float(f1.mean()),macro_precision=float(precision.mean()),macro_recall=float(recall.mean()))


def main():
 torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False
 cfg=yaml.safe_load((reference.SMOKE/'data_local.yaml').read_text(encoding='utf-8'))
 dataset=reference.CXRDataset(cfg['manifest'],cfg['data_root'],'val',reference.build_transform(False,224))
 assert len(dataset)==1422
 official=list(csv.DictReader((Path(cfg['data_root'])/'valid.csv').open(encoding='utf-8-sig',newline='')))
 from dataset.tools.build_manifest import parse_label_ids
 expected={(r['Subject_id'],r['Study_id'],r['Dicom_id']):set(parse_label_ids(r['Predict_class'])) for r in official}
 for row,labels in zip(dataset.rows,dataset.labels):
  assert {i+1 for i in labels}==expected[(row['subject_id'],row['study_id'],Path(row['image_path']).stem)]
 classes=list(csv.DictReader((reference.SMOKE/'class_mapping.csv').open(encoding='utf-8-sig',newline='')))
 res=json.loads((DATA_ROOT/'MambaOut_ASL对照/results.json').read_text(encoding='utf-8'))
 gates_info=json.loads((DATA_ROOT/'MambaOut_ASL对照/gates.json').read_text(encoding='utf-8'))
 fixed_gates=np.asarray(gates_info['final_gates_by_class_id'])
 models=[('old_best3',DATA_ROOT/'MambaOut_100轮/baseline_100/best.pth','58ed3c464d51b4c187880d5fc456c17e84fc05d45e7cc5a4fcd7da4fd57a4279',res['candidates']['历史全量 BCE']['best_validation']),('current_best',DATA_ROOT/'MambaOut_ASL对照/best_model.pth','9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585',res['winner_validation'])]
 report={'cohort':'official development validation set','images':1422,'studies':859,'patients':228,'fixed_thresholds':fixed_gates.tolist(),'label_check':'All 1422 raw validation CSV labels match manifest labels and class_id - 1 tensor positions.','models':{},'scope':'The official development validation set has been used for checkpoint/model selection. OOF only holds out threshold fitting patients. Neither OOF nor full-validation fitted gates is an independent blind-test estimate. No test images or predictions used.'}
 original_eval=reference.evaluate_predictions
 capture={}
 def capture_eval(labels,probabilities,studies,*args,**kwargs):
  capture['labels']=labels.copy();capture['probs']=probabilities.copy();capture['studies']=list(studies)
  return original_eval(labels,probabilities,studies,*args,**kwargs)
 reference.evaluate_predictions=capture_eval
 for name,path,expected_hash,expected_metrics in models:
  started=time.perf_counter()
  print('Evaluating',name,flush=True)
  assert sha(path)==expected_hash
  state=torch.load(path,map_location='cpu',weights_only=True)
  info=state['training']
  model=core.build_model({'drop_rate':info.get('drop_rate',0.0),'drop_path_rate':info.get('drop_path_rate',0.0)},state['state_dict']).eval().requires_grad_(False)
  del state
  metrics,predictions=reference.validate(model,dataset,keep_predictions=True)
  del model
  gc.collect();torch.cuda.empty_cache()
  for level in ('image','study'):
   for key in ('auc','f1','macro_auc','macro_f1','count','auc_classes'):
    assert np.allclose(metrics[level][key],expected_metrics[level][key],atol=1e-12,rtol=0),(name,level,key)
  y,p,subjects=predictions
  assert y.shape==p.shape==(859,10) and len(set(subjects.tolist()))==228
  groups=defaultdict(list)
  for i,study in enumerate(capture['studies']):groups[study].append(i)
  study_ids=list(groups)
  independently_meaned=np.stack([capture['probs'][idx].astype(np.float64).sum(axis=0)/len(idx) for idx in groups.values()])
  aggregation_delta=float(np.max(np.abs(independently_meaned-p)))
  assert aggregation_delta<1e-7
  independently_targets=np.stack([capture['labels'][idx[0]] for idx in groups.values()])
  assert np.array_equal(independently_targets,y)
  np.savez_compressed(ROOT/('study_predictions.npz' if name=='current_best' else 'old_study_predictions.npz'),targets=y,probabilities=p,subjects=subjects,study_ids=np.asarray(study_ids))
  # Independent rank statistic AUC with ties worth one half.
  pair_auc=[]
  for c in range(10):
   pos=p[y[:,c]==1,c];neg=p[y[:,c]==0,c]
   pair_auc.append(float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg))))
  assert np.allclose(pair_auc,metrics['study']['auc'],atol=1e-12,rtol=0)
  default=counts(y,p>=.5)
  fixed=counts(y,p>=fixed_gates)
  oof=np.zeros_like(y,dtype=bool)
  fold_records=[]
  for fold,(fit,held) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,subjects),1):
   assert not set(subjects[fit])&set(subjects[held])
   gates=reference.fit_gates(y[fit],p[fit])
   oof[held]=p[held]>=gates
   fold_records.append({'fold':fold,'fit_studies':len(fit),'held_out_studies':len(held),'fit_patients':len(set(subjects[fit])),'held_out_patients':len(set(subjects[held])),'gates':gates.tolist()})
  oof_stats=counts(y,oof)
  if name=='current_best':
   assert np.allclose(oof_stats['f1'],gates_info['oof_f1_by_class_id'],rtol=0,atol=1e-12)
   assert np.array_equal(reference.fit_gates(y,p),fixed_gates)
  detail=[]
  ap=average_precision_score(y,p,average=None)
  for c,cls in enumerate(classes):
   positives=p[y[:,c]==1,c];negatives=p[y[:,c]==0,c]
   detail.append({'class_id':c+1,'class_name':cls['chinese_name'],'positive_studies':int(y[:,c].sum()),'negative_studies':int((y[:,c]==0).sum()),'prevalence':float(y[:,c].mean()),'auc':pair_auc[c],'average_precision':float(ap[c]),'score_min':float(p[:,c].min()),'score_max':float(p[:,c].max()),'positive_score_quantiles_0_25_50_75_100':np.quantile(positives,[0,.25,.5,.75,1]).tolist(),'negative_score_quantiles_0_25_50_75_100':np.quantile(negatives,[0,.25,.5,.75,1]).tolist(),'default_threshold':{k:default[k][c] for k in ['tp','fp','fn','tn','predicted_positives','precision','recall','f1']},'frozen_current_gate':float(fixed_gates[c]),'current_gates_same_validation_fit':{k:fixed[k][c] for k in ['tp','fp','fn','tn','predicted_positives','precision','recall','f1']},'patient_oof':{k:oof_stats[k][c] for k in ['tp','fp','fn','tn','predicted_positives','precision','recall','f1']}})
  entry={'checkpoint':str(path),'sha256':expected_hash,'settings':info,'metrics':metrics,'macro_average_precision':float(ap.mean()),'default_05':default,'frozen_current_gates_same_validation':fixed,'patient_oof_thresholds':oof_stats,'folds':fold_records,'classes':detail,'independent_aggregation_max_abs_delta':aggregation_delta,'seconds':time.perf_counter()-started,'weight_unchanged':sha(path)==expected_hash}
  report['models'][name]=entry
  write(ROOT/(name+'_diagnosis.json'),entry)
  write(ROOT/'diagnosis.json',report)
  for c in detail:
   if c['default_threshold']['f1']==0 or c['class_id'] in (1,3,4):
    print(json.dumps({'model':name,'class_id':c['class_id'],'name':c['class_name'],'positives':c['positive_studies'],'auc':c['auc'],'ap':c['average_precision'],'max_score':c['score_max'],'default':c['default_threshold'],'oof_f1':c['patient_oof']['f1']},ensure_ascii=False),flush=True)
 lines=['# F1为零诊断','',report['scope'],'','固定模型在官方1422图/859次检查/228患者上重新推理。未训练、未使用测试数据。原始标签、逐类指标、概率均值聚合及checkpoint均已复核。','', '|模型|类别|阳性检查|AUC|AP|TP@0.5|FP@0.5|FN@0.5|F1@0.5|患者5折阈值OOF F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,e in report['models'].items():
  for c in e['classes']:
   d=c['default_threshold'];o=c['patient_oof']
   lines.append(f"|{name}|{c['class_id']} {c['class_name']}|{c['positive_studies']}|{c['auc']:.6f}|{c['average_precision']:.6f}|{d['tp']}|{d['fp']}|{d['fn']}|{d['f1']:.6f}|{o['f1']:.6f}|")
 lines+=['','old_best3是旧第3轮，current_best是当前保护性微调新增第1轮。冻结阈值在整个官方验证集拟合，因此直接在同一验证集评分只用于解释阈值变化，不作为泛化收益证据。患者OOF将阈值拟合患者与评分患者隔开，但模型已经由整份验证集筛选，仍属于内部开发估计。','', 'AP为Average Precision，类别阳性比例是其随机排序参考。AUC和AP用于评估排序，F1受阈值影响。F1=0可能因为全部预测阴性或所有阳性预测均错误，需看TP/FP/FN，而不能仅凭F1认定类别标签错位。','']
 (ROOT/'诊断报告.md').write_text('\n'.join(lines),encoding='utf-8')
 write(ROOT/'complete.json',{'state':'complete','gpu_released':True,'training_updates':0})
 print('DIAGNOSIS_COMPLETE_GPU_RELEASED',flush=True)

if __name__=='__main__':main()
