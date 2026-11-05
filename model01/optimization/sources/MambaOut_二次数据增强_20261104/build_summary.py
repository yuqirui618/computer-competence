"""CPU-only, verified summary for the completed gamma/noise training pilot.

Never imports torch or runner. Refuses incomplete runs and verifies saved study
predictions, coarse patient cross-validation, protection hashes and evidence.
"""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):os.environ[key]=str(ROOT/'temp')
import csv,hashlib,io,json,math
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score
from sklearn.model_selection import GroupKFold
DATA=ROOT.parent
NAMES=('A_control','B_gamma','C_gamma_noise')
LABELS={'A_control':'A：原增强控制','B_gamma':'B：增加Gamma','C_gamma_noise':'C：Gamma＋轻噪声'}
FINGERPRINTS={};CHECKS=[]

def require(ok,message):
    if not ok:raise ValueError(message)
def near(a,b,context,tol=1e-12):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=0,atol=tol),context)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):
    path=Path(path);FINGERPRINTS[str(path)]=sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))
def prediction(path):
    path=Path(path);FINGERPRINTS[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as a:out={k:a[k].copy() for k in a.files}
    y,p,s=out['targets'],out['probabilities'],out['subjects']
    require(y.shape==p.shape==(859,10) and s.shape==(859,) and len(np.unique(s))==228,'Prediction cohort differs')
    require(np.isin(y,[0,1]).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid labels/scores')
    if 'study_ids' in out:require(len(np.unique(out['study_ids']))==859,'Study IDs repeat')
    return out

def gates(y,p):
    candidates=sorted(np.arange(1,40)/40,key=lambda g:(abs(g-.5),g))
    values=[f1_score(y,p>=g,average=None,zero_division=0) for g in candidates]
    return np.asarray(candidates)[np.argmax(values,axis=0)]
def calibration(pred):
    y,p,s=pred['targets'],pred['probabilities'],pred['subjects'];decisions=np.zeros_like(y,dtype=bool)
    for fit,held in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,s):
        require(not set(s[fit])&set(s[held]),'Patient crossed threshold fold')
        decisions[held]=p[held]>=gates(y[fit],p[fit])
    f=f1_score(y,decisions,average=None,zero_division=0)
    return dict(oof_macro_f1=float(f.mean()),oof_f1_by_class_id=f.tolist(),final_gates_by_class_id=gates(y,p).tolist(),patients=228)
def metrics(pred,old_gates):
    y,p=pred['targets'],pred['probabilities'];z=p>=.5
    return dict(auc=roc_auc_score(y,p,average=None),f1=f1_score(y,z,average=None,zero_division=0),
      macro_auc=float(roc_auc_score(y,p,average='macro')),macro_f1=float(f1_score(y,z,average='macro',zero_division=0)),
      macro_ap=float(average_precision_score(y,p,average='macro')),ap_by_class=average_precision_score(y,p,average=None),
      macro_precision=float(precision_score(y,z,average='macro',zero_division=0)),macro_recall=float(recall_score(y,z,average='macro',zero_division=0)),
      tp=np.sum(z&(y==1),0),fp=np.sum(z&(y==0),0),fn=np.sum(~z&(y==1),0),
      old_frozen_gates_f1=float(f1_score(y,p>=old_gates,average='macro',zero_division=0)))
def score(record):return record['validation']['study']['macro_auc'],record['validation']['study']['macro_f1']
def table(head,rows):return ['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]
def interval_text(interval):return f"{interval['delta']:+.6f}，95%区间[{interval['ci95'][0]:+.6f}, {interval['ci95'][1]:+.6f}]"
def write_atomic(path,text,encoding='utf-8'):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text,encoding=encoding);temp.replace(path)

def main():
    require(ROOT.drive.upper()=='E:','E output required')
    result=read(ROOT/'results.json');status=read(ROOT/'status.json');plan=read(ROOT/'plan.json')
    require(result['state']=='complete' and status['state']=='complete','Training incomplete; no report generated')
    require(set(result['stages'])==set(NAMES),'Missing trial')
    require(result['original_files_overwritten'] is False and result['new_test_inference'] is False and result['github_uploaded'] is False,'Unexpected side effect')
    evidence=read(ROOT/'historical_evidence.json')
    for item in evidence['sources'].values():require(sha(item['path'])==item['sha256'],'Historical source changed')
    preserved=read(ROOT/'protected_inputs.json')
    require(preserved==result['preserved_hashes_after'],'Protection receipt mismatch')
    for path,digest in preserved.items():require(sha(path)==digest,'Protected original changed: '+Path(path).name)
    champ=evidence['preserved_champion'];require(sha(champ['checkpoint'])==champ['sha256'],'Champion SHA mismatch')
    old_gate=read(DATA/'MambaOut_ASL对照/gates.json')
    old_gates=np.asarray(old_gate['final_gates_by_class_id'])
    base=prediction(ROOT/'baseline_predictions.npz');base_metrics=metrics(base,old_gates);base_cal=calibration(base)
    near(base_metrics['macro_auc'],champ['auc'],'Champion AUC reference');near(base_cal['oof_macro_f1'],champ['patient_oof_f1'],'Champion OOF reference')
    original_result=read(ROOT/'baseline_metrics.json')
    for key in ('auc','f1','macro_auc','macro_f1'):near(base_metrics[key],original_result['validation']['study'][key],'Baseline '+key)
    CHECKS.append('baseline_study_metrics_and_patient_threshold_folds')
    epoch_rows=[];trials=result['stages'];predictions={};control_history_comparison=[]
    for name in NAMES:
        trial=trials[name];history=trial['history']
        require(trial==read(ROOT/name/'results.json'),'Per-trial JSON mismatch')
        require(history==read(ROOT/name/'history.json'),'Per-epoch history mismatch')
        require(1<=len(history)<=2 and [r['epoch'] for r in history]==list(range(1,len(history)+1)),'Noncontiguous trial epochs')
        require(len(history)==trial['epochs_completed']==result['completed_epochs'][name],'Epoch total differs')
        if len(history)<2:require(trial['stop_reason']=='auc_drop_guard' and history[-1]['validation']['study']['macro_auc']<champ['auc']-.01,'Unjustified early stop')
        best_trained=max(history,key=score)
        require(trial['best_trained_epoch']==best_trained['epoch'],'Best trained epoch wrong')
        expected=max([(0,(champ['auc'],champ['f1_at_0_5']))]+[(r['epoch'],score(r)) for r in history],key=lambda x:x[1])
        require(trial['best_epoch_including_original']==expected[0],'Original retention selection wrong')
        near((trial['best_validation']['study']['macro_auc'],trial['best_validation']['study']['macro_f1']),expected[1],'Trial best metrics wrong')
        for row in history:
            epoch=row['epoch'];pred=prediction(ROOT/name/f'epoch_{epoch}_predictions.npz')
            for key in ('targets','subjects','study_ids'):require(np.array_equal(pred[key],base[key]),'Epoch labels/IDs misaligned')
            m=metrics(pred,old_gates);cal=calibration(pred);recorded=row['validation']['study']
            for key in ('auc','f1','macro_auc','macro_f1'):near(m[key],recorded[key],name+' '+str(epoch)+' '+key)
            for key in ('macro_ap','ap_by_class','macro_precision','macro_recall','tp','fp','fn','old_frozen_gates_f1'):
                near(m[key],row['extra'][key],name+' extra '+key)
            for key in cal:near(cal[key],row['calibration'][key],name+' calibration '+key)
            require(row['images']==30344 and row['augmentation_counts']['images']==30344,'Incomplete training image coverage')
            require(row['validation']['image']['count']==1422 and recorded['count']==859,'Validation coverage mismatch')
            require(math.isfinite(row['train_loss']) and row['train_loss']>=0,'Invalid recorded training loss')
            require((ROOT/name/f'epoch_{epoch}.pth').is_file(),'Missing epoch checkpoint')
            if name=='A_control':
                old=prediction(DATA/'MambaOut_F1正则化优化_20261104/A_current_regularization'/f'epoch_{epoch}_predictions.npz')
                for key in ('targets','subjects'):require(np.array_equal(old[key],pred[key]),'Historical control alignment differs')
                om=metrics(old,old_gates)
                control_history_comparison.append(dict(epoch=epoch,max_abs_probability_difference=float(np.max(np.abs(pred['probabilities']-old['probabilities']))),
                   new_auc=m['macro_auc'],historical_auc=om['macro_auc'],new_default_f1=m['macro_f1'],historical_default_f1=om['macro_f1']))
            predictions[(name,epoch)]=pred
            epoch_rows.append({'组别':name,'方案':LABELS[name],'轮次':epoch,'本组新增轮次中最佳':epoch==trial['best_trained_epoch'],
              '包含原冠军后的最佳轮次':trial['best_epoch_including_original'],'训练图像':row['images'],'验证图像':1422,'验证检查':859,'验证患者':228,
              '训练BCE':row['train_loss'],'Study_AUC':m['macro_auc'],'默认0.5_F1':m['macro_f1'],'患者五折阈值OOF_F1':cal['oof_macro_f1'],
              'Macro_AP':m['macro_ap'],'旧固定阈值F1_重复验证集':m['old_frozen_gates_f1'],'Gamma应用图像':row['augmentation_counts']['gamma_applied'],
              '噪声应用图像':row['augmentation_counts']['noise_applied'],'训练秒数':row['train_seconds'],'验证秒数':row['validation_seconds'],
              '训练峰值显存MiB':row['peak_allocated_mib']})
            CHECKS.append(f'{name}/epoch{epoch}: AUC,F1,AP,confusion,patient_OOF,gates,coverage')
    actual_total=sum(t['epochs_completed'] for t in trials.values())
    require(result['new_training_epochs']==actual_total==len(epoch_rows),'Total new epochs wrong')
    require(sha(result['selected_checkpoint'])==result['selected_sha256'],'Selected checkpoint file changed')
    if result['winner_including_original']=='original_champion':
        verification_note='原冠军已在本轮启动前重新加载复验，本轮无替换候选需要额外重载。'
        verification_record=dict(state='original_retained',original_reload_source=str(ROOT/'baseline_metrics.json'),new_candidate_reload_required=False)
    else:
        verification_path=ROOT/'candidate_verification.json'
        require(verification_path.is_file(),'New candidate requires candidate_verification.json before the final report')
        verification_record=read(verification_path)
        require(verification_record.get('state')=='passed','Candidate reload verification did not pass')
        require(verification_record.get('checkpoint_sha256')==result['selected_sha256'],'Candidate verification SHA differs from selected checkpoint')
        require(verification_record.get('metrics_reproduced') is True and verification_record.get('probabilities_reproduced') is True,'Candidate metrics/probabilities have not both reproduced')
        verification_note='原冠军已在本轮启动前重新加载复验；新增候选也已完成独立GPU重新加载，权重SHA、指标和预测均与candidate_verification.json核对通过。'
    # Validate bootstrap point estimates only; intervals are imported from runner.
    control=trials['A_control'];control_auc=control['best_trained_validation']['study']['macro_auc']
    for name,trial in trials.items():
        auc=trial['best_trained_validation']['study']['macro_auc']
        for key,reference in (('versus_original',champ['auc']),('versus_retrained_control',control_auc)):
            interval=result['paired_auc_intervals'][name][key]
            near(interval['delta'],auc-reference,'Bootstrap point estimate differs')
            require(interval['repeats']==400 and interval['valid_repeats']>=300 and interval['ci95'][0]<=interval['ci95'][1],'Invalid bootstrap record')
    best_name=max(NAMES,key=lambda n:(trials[n]['best_trained_validation']['study']['macro_auc'],trials[n]['best_trained_validation']['study']['macro_f1']))
    best=trials[best_name];best_auc=best['best_trained_validation']['study']['macro_auc']
    aug_name=max(('B_gamma','C_gamma_noise'),key=lambda n:(trials[n]['best_trained_validation']['study']['macro_auc'],trials[n]['best_trained_validation']['study']['macro_f1']))
    aug=trials[aug_name];aug_auc=aug['best_trained_validation']['study']['macro_auc']
    prior_ensemble_auc=evidence['f1_threshold_and_ensemble']['ensemble_auc']
    require(result['surpasses_original_single_auc']==(best_auc>champ['auc']),'Original exceed flag inconsistent')
    require(result['surpasses_prior_exploratory_ensemble_auc']==(best_auc>prior_ensemble_auc),'Ensemble exceed flag inconsistent')
    aug_ci=result['paired_auc_intervals'][aug_name]
    if aug_auc<=champ['auc']:
        conclusion='本轮Gamma/轻噪声增强没有超过保留的原单模型冠军；目前未建立F1调优或已测试增强能稳定突破的证据。'
    elif aug_ci['versus_original']['ci95'][0]<=0 or aug_ci['versus_retrained_control']['ci95'][0]<=0:
        conclusion='本轮增强出现高于原冠军的AUC点估计，但相对原冠军或同期控制的差值区间包含0，暂只能称数值改善，不能称稳定突破。'
    else:
        conclusion='本轮增强在重复使用的内部验证上出现正向证据；这是需要进一步复核的候选，不能直接当成新的独立测试突破。'
    old_ensemble_audit=read(DATA/'MambaOut_F1专项优化_20261104/average_probe/independent_audit.json')
    ensemble_coarse=old_ensemble_audit['repeat_fold_oof_f1']['42']['probability_average']
    lines=['# F1与数据增强优化总结','',conclusion,'',
      f"本轮实际完成 **{actual_total}轮全量短程微调**，每轮覆盖30,344张训练图。所有原冠军、阈值和提交文件SHA与实验前一致。实验结果留在E盘，未上传GitHub。",
      '这里的“暂未突破”只针对已经执行的配置和预算，不等于MambaOut已达到性能上限，也不等于数据增强永远无效。','',
      '## 1. 本轮实际尝试与结果','',
      '三组从同一保留冠军独立开始：A重跑原轻微仿射增强；B在归一化前，以50%概率对[0,1]浮点图像应用Gamma=0.85–1.15；C在B上以25%概率加入σ=0.005–0.015的灰度高斯噪声。噪声三个通道共享，保留黑色padding并裁回[0,1]；没有翻转、裁病灶、替换标签或加入外部数据。',
      '均使用224输入、batch12、单GPU顺序训练、BCE、dropout0.2/DropPath0.1、AdamW、骨干LR1e-6/分类头1e-5、weight_decay0.05；第二轮学习率减半。额外增强采用私有随机流，避免消耗原仿射与Dropout的随机流。训练读512 JPEG缓存，验证读原图，三组保持一致。',
      '验证完整覆盖1,422图/859次检查/228患者。同次检查取图像概率均值；AUC优先，固定0.5的F1仅作选择次序。另用患者五折、每折只在其拟合患者上选原0.025网格阈值，报告OOF F1。','']
    rows=[]
    for r in epoch_rows:rows.append([LABELS[r['组别']],str(r['轮次'])+(' ★' if r['本组新增轮次中最佳'] else ''),f"{r['训练BCE']:.6f}",f"{r['Study_AUC']:.8f}",f"{r['默认0.5_F1']:.6f}",f"{r['患者五折阈值OOF_F1']:.6f}",r['Gamma应用图像'],r['噪声应用图像']])
    lines+=table(['方案','轮次','训练BCE','Study AUC','F1@0.5','患者OOF F1','Gamma图数','噪声图数'],rows)
    lines+=['','★表示该组新增训练轮次中的最佳；如果包含原冠军后最佳轮次仍为0，表示新增训练没有超过起点，不能把epoch0算成新训练成果。若出现AUC下降超过0.01的保护早停，以表中实际轮数为准。','']
    rows=[['保留的原单模型冠军','原权重',f"{champ['auc']:.8f}",f"{champ['patient_oof_f1']:.6f}"],
          ['此前双模型概率均值候选','历史探索',f'{prior_ensemble_auc:.8f}',f'{ensemble_coarse:.6f}']]
    for name in NAMES:
        t=trials[name];rows.append([LABELS[name],t['best_trained_epoch'],f"{t['best_trained_validation']['study']['macro_auc']:.8f}",f"{t['best_trained_calibration']['oof_macro_f1']:.6f}"])
    lines+=table(['比较对象','新增最佳轮次','AUC','同粗网格患者OOF F1'],rows)
    lines+=['',f"本轮新增轮次整体最高来自 **{LABELS[best_name]}第{best['best_trained_epoch']}轮**，AUC={best_auc:.8f}；增强组中最高为{LABELS[aug_name]}第{aug['best_trained_epoch']}轮，AUC={aug_auc:.8f}。",
      f"增强组最高结果相对原单模型：{interval_text(aug_ci['versus_original'])}；相对本轮原增强控制：{interval_text(aug_ci['versus_retrained_control'])}。",
      f"相对此前双模型候选AUC的点差为{aug_auc-prior_ensemble_auc:+.8f}。双模型历史候选自身改善也很小且不稳定，不能只选择有利的参照。",
      '这些区间按患者整组重采样400次，条件于已经选出的权重。最佳轮次/方案是在同一验证集选的，区间没有消除这种选择偏差，也未包含重训随机性；区间跨0时不能说有稳定提升。']
    if best_name=='A_control':lines.append('本轮最高来自不加新增强的控制组，因此任何高于起点的变化都不能归因于Gamma或噪声。')
    best_oof_row=max(epoch_rows,key=lambda r:r['患者五折阈值OOF_F1'])
    lines+=['',f"本轮最高AUC候选的同口径患者OOF F1为{best['best_trained_calibration']['oof_macro_f1']:.8f}，原冠军为{champ['patient_oof_f1']:.8f}，差值{best['best_trained_calibration']['oof_macro_f1']-champ['patient_oof_f1']:+.8f}；AUC与F1没有同步改善。",
      f"若单独按OOF F1挑轮次，本轮最高是{best_oof_row['方案']}第{best_oof_row['轮次']}轮：{best_oof_row['患者五折阈值OOF_F1']:.8f}，其AUC为{best_oof_row['Study_AUC']:.8f}。本次实际最高F1来自原增强控制，且与历史同配置结果复现，不能算作新增Gamma或噪声的收益。",'',verification_note]
    if verification_record['state']=='passed':
        lines += [f"候选相对此前双模型概率均值的AUC差值：{interval_text(verification_record['paired_auc_vs_previous_ensemble'])}；同口径OOF F1相对原冠军的差值：{interval_text(verification_record['paired_coarse_oof_f1_vs_original'])}。F1区间条件于固定的分折预测，重采样时没有重新拟合阈值，同样不能消除重复选择带来的偏差。"]
    lines+=['','### 历史控制是否一致','']
    lines+=table(['轮次','新控制AUC','历史同配置AUC','预测最大绝对差'],[[r['epoch'],f"{r['new_auc']:.9f}",f"{r['historical_auc']:.9f}",f"{r['max_abs_probability_difference']:.9g}"] for r in control_history_comparison])
    lines+=['','这只是同配置历史复核。即使存在微小运行差异，也不能把它算作增强收益；主要因果比较仍是本轮重跑A与B/C。','',
      '## 2. 为什么F1曾改善，却不能据此认定模型突破','']
    f1e=evidence['f1_threshold_and_ensemble'];main=f1e['seeds'][0]
    lines+=table(['主划分方案','患者OOF F1','说明'],[
      ['原冠军＋原粗网格',f"{main['original_coarse_oof_f1']:.8f}",'已调过阈值的真正基线'],
      ['原冠军＋新阈值选择流程',f"{main['single_nested_oof_f1']:.8f}",'内外层患者分折，只改阈值'],
      ['双模型均值＋新阈值流程',f"{main['ensemble_nested_oof_f1']:.8f}",'同样分折；模型组合已有选择历史']])
    lines+=['',f"单模型F1差值{interval_text(f1e['single_main_ci'])}；双模型相对原粗网格{interval_text(f1e['ensemble_main_comparisons']['versus_original_coarse'])}。三个预设患者划分均出现两次上升、一次下降，不能只展示最好的划分。",
      f"更大患者队列上阈值迁移F1仅从{f1e['large_patient_transfer']['baseline_f1']:.8f}变为{f1e['large_patient_transfer']['new_f1']:.8f}，区间{f1e['large_patient_transfer']['interval']['ci95']}仍跨0。",
      '已确认的机制：阈值改变判阳性的位置和精确率/召回率取舍，不改变固定概率的排序，所以单模型AUC保持不变。原来从统一0.5改成逐类阈值已得到明显收益，继续细化阈值只是寻找更合适的折中，并不能纠正全部排序错误。',
      f"例如原冠军第1类AUC={champ['class1_auc']:.6f}、AP={champ['class1_ap']:.6f}，当前人群上的区分能力仍有限。降低阈值可以增加命中，也可能引入更多误报；F1非零不等于该类已经识别得好。",
      f"实际反例：稀有类加权曾把默认F1从{champ['f1_at_0_5']:.6f}推到{evidence['rare_weight_epoch2']['default_f1']:.6f}，但患者OOF F1只有{evidence['rare_weight_epoch2']['oof_f1']:.6f}，低于原{champ['patient_oof_f1']:.6f}，AUC也下降。因此不能只比较默认0.5下的F1。",'',
      '## 3. 为什么已测试的数据增强没有建立稳定收益','']
    previous=evidence['previous_augmentation'];st=previous['stages'];aud=previous['evaluation']['audit'];interval=previous['paired_auc_intervals']['audit']
    lines+=['上一轮是从相同ImageNet初值、同一患者划分出发的三组3轮对照，每轮21,291张训练图：', '']
    lines+=table(['上一轮方案','开发集最佳AUC'],[[name,f"{t['development_auc']:.8f}"] for name,t in st.items()])
    lines+=['',f"在2,842次检查/1,658患者的内部保留集上，基线AUC={aud['A_current_affine']['auc']:.8f}，预选光照＋MixUp={aud['C_photometric_mixup']['auc']:.8f}，差值{interval_text(interval)}。这两个内部保留集分数不能与全量冠军官方验证AUC直接混比。",
      '本轮改用全量冠军的Gamma/噪声短程微调，与上一轮不同，不能简单归为“同一个实验再跑一次”。两轮合在一起，只能说明这些具体增强配置尚未给出稳定突破证据。',
      '确定的事实：增强变换的是已有患者图像，不会新增独立阳性患者、病理种类或新标注；Gamma和噪声主要模拟灰度响应与轻微噪声变化。',
      '合理但未证实的解释：主要错误可能来自难病例排序、视角差异、病灶信息或标注边界，而本轮增强未覆盖这些误差；增强也可能太弱，或改变了细微征象。现有总分不足以区分这些解释，不能直接断言“增强破坏病灶”或“标签有错”。']
    res=evidence['resolution_probe']
    lines += [f"另一次固定权重224→320推理，AUC由{res['input224_auc']:.8f}降到{res['input320_auc']:.8f}。这只说明直接改推理尺寸没有成功；没有测试过320重新训练，不能据此否定高分辨率训练。",'',
      '## 4. 统计规模、过拟合及仍未排除的限制','']
    val=evidence['cohort_evidence']['official_val'];train=evidence['cohort_evidence']['official_train'];long=evidence['historical_86epoch_run']
    lines+=[f"官方验证集有{val['studies']}次检查，但仅{val['patients']}位患者；第1类{val['positive_studies_class1to10'][0]}次阳性检查来自{val['positive_patients_class1to10'][0]}位患者。单人最多{val['top_patients_study_totals']['1']}次检查，前10人占{val['top_patients_study_percentage']['10']:.2f}%。重复检查有关联，稀有类阈值和小幅分数变化会受少数患者影响。这支持按患者分折/重采样，但不能单独解释每个失败实验。",'']
    lines+=table(['历史全量训练','训练BCE','验证AUC','F1@0.5'],[[f"第{r['epoch']}轮",f"{r['train_loss']:.8f}",f"{r['study_auc']:.8f}",f"{r['study_f1_at_0_5']:.8f}"] for r in (long['best'],long['last'])])
    require(long['validation_loss_recorded'] is False,'Historical loss assumption changed')
    lines+=['','旧86轮中，训练损失接近零，而验证AUC明显下降，是该训练方案过拟合的强迹象；默认F1反而上升，进一步说明F1与AUC不完全同步。历史没有记录验证损失，不能编造验证loss曲线，也不能把这段历史推广成“所有后续微调都已经过拟合”。',
      '当前仍存在的限制：本轮只有单个训练种子、两轮很低学习率微调；上一轮增强只有三轮子集初筛；224输入、缓存JPEG与验证原图的预处理差异尚未做独立控制实验；验证集已反复用于选择轮次与方案。它们限制结论范围，但尚不能分别被认定为性能停滞的已证实原因。','',
      '## 5. 后续更有价值的验证方向','',
      f"1. **按检查等权或直接检查级损失。** 训练集{train['studies']}次检查中{train['studies_with_multiple_images']}次有多图，最多7张。当前逐图等权BCE使多图检查在训练中获得更多权重，而评分按检查汇总。先做单变量、同预算对照，不能预先保证有效。",
      '2. **复核稀有类错误病例与标注。** 按漏检/误报、视角、病灶大小等归类，并由具备能力的标注者核对；目前没有证据可以宣布标签错误。',
      '3. **预先固定多种子和更独立的验证。** 明确训练预算后再运行，避免围绕同一小验证集无限挑参数。只在赛事允许范围内使用真正未参与选择的患者或外部验证，隐藏测试集不用于调参。',
      '4. 如果资源允许，再对增强重训、较大分辨率训练或视角感知多图汇总做独立实验；直接增加epoch、增强强度或替换损失都不能保证突破。','',
      '## 6. 保存与复核','',
      f"保留原权重：{champ['checkpoint']}；SHA256={champ['sha256']}。原gates与submission均核对未变。环境沿用E:\\比赛数据集\\envs\\cxr-smoke-cu128，没有在C盘安装环境。",'']
    if verification_record['state']=='passed':
        lines += [f"新增最高AUC候选独立保存在：{result['selected_checkpoint']}，对应{LABELS[result['winner_including_original']]}第{verification_record['epoch']}轮；SHA256={result['selected_sha256']}。",
          f"该候选的专属阈值文件：{ROOT/'candidate_gates.json'}，绑定上述权重SHA及同检查概率均值规则；没有替换原冠军的gates。全验证集拟合的最终阈值不能当作独立测试成绩。GPU复核记录为candidate_verification.json，全部验证概率与训练时保存值最大差为{verification_record['max_abs_probability_difference']:.9g}。",'']
    lines+=table(['本轮目录/文件','内容'],[
      ['A_control','原增强控制的每轮权重、预测、history与可续训断点'],['B_gamma','Gamma增强对应产物'],['C_gamma_noise','Gamma＋轻噪声对应产物'],
      ['run_gamma.py / plan.json / preflight.json','可复现实验代码、冻结方案、CPU增强与数据检查'],['historical_evidence.json','历史结果、事实与推断、原JSON路径和SHA'],
      ['总结.md / 六轮明细.csv / summary_checks.json','本报告、实际完成轮次明细、独立复算及保护校验']])
    lines += ['', '报告从每轮保存的Study预测独立重算AUC、F1、AP、混淆计数和患者五折阈值；训练BCE沿用记录，无法从Study均值预测反推。患者bootstrap区间引用runner记录，仅独立核对其点差、次数与范围；未声称本报告重新采样计算过所有区间。',
      '方法来源：[Torchvision adjust_gamma](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.adjust_gamma.html)、[scikit-learn阈值调优说明](https://scikit-learn.org/stable/modules/classification_threshold.html)。这些方法说明不保证本任务会提高。']
    csv_buffer=io.StringIO(newline='');writer=csv.DictWriter(csv_buffer,fieldnames=list(epoch_rows[0]));writer.writeheader();writer.writerows(epoch_rows)
    checks=dict(state='passed',time=datetime.now().astimezone().isoformat(),completed_epochs=actual_total,verified_checks=CHECKS,
        protected_hashes_verified=preserved,historical_source_hashes_verified=True,all_saved_study_predictions_recomputed=True,
        coarse_patient_cv_recomputed=True,bootstrap_intervals_recomputed=False,bootstrap_point_estimates_checked=True,
        historical_control_comparison=control_history_comparison,best_trained_stage=best_name,best_trained_auc=best_auc,
        best_augmentation_stage=aug_name,conclusion=conclusion,source_fingerprints=FINGERPRINTS,
        new_candidate_gpu_reload_verification=verification_record)
    write_atomic(ROOT/'六轮明细.csv',csv_buffer.getvalue(),'utf-8-sig')
    write_atomic(ROOT/'总结.md','\n'.join(lines)+'\n')
    write_atomic(ROOT/'summary_checks.json',json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report='complete',actual_new_epochs=actual_total,best_trained=best_name,best_trained_auc=best_auc,conclusion=conclusion),ensure_ascii=False))

if __name__=='__main__':main()
