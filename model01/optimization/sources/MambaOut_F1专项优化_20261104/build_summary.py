"""Independent CPU metric check and final F1 experiment summary."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TEMP=ROOT/'temp';TEMP.mkdir(exist_ok=True)
for k in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):os.environ[k]=str(TEMP)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):
    with np.load(ROOT/path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def close(a,b):assert np.allclose(a,b,rtol=0,atol=1e-12),(a,b)
def table(head,rows):return ['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]

def main():
    single=read('threshold_search/results.json');ensemble=read('ensemble_thresholds/results.json')
    large=read('large_validation/results.json');average=read('average_probe/results.json');avg_audit=read('average_probe/independent_audit.json')
    protected=single['protected_hashes_after']
    assert all(sha(p)==v for p,v in protected.items())
    checks=[]
    for folder,results in (('threshold_search',single),('ensemble_thresholds',ensemble)):
        for seed in (42,43,44):
            pred=load(f'{folder}/oof_seed_{seed}.npz');y,p=pred['targets'],pred['probabilities']
            assert y.shape==p.shape==(859,10) and len(np.unique(pred['subjects']))==228
            close(roc_auc_score(y,p,average='macro'),results['source_auc'])
            for method in ('coarse','fine','exact','bootstrap_exact_median','bootstrap_fine_median','nested_selection'):
                actual=results['seeds'][str(seed)]['metrics'][method]
                close(f1_score(y,pred[method],average='macro',zero_division=0),actual['macro_f1'])
                close(f1_score(y,pred[method],average=None,zero_division=0),actual['f1_by_class'])
                close(precision_score(y,pred[method],average='macro',zero_division=0),actual['macro_precision'])
                close(recall_score(y,pred[method],average='macro',zero_division=0),actual['macro_recall'])
                checks.append(f'{folder}/seed{seed}/{method}')
    a=load('average_probe/champion_predictions.npz');b=load('average_probe/epoch2_predictions.npz');c=load('average_probe/probability_average_predictions.npz')
    for key in ('targets','subjects','study_ids'):
        assert np.array_equal(a[key],b[key]) and np.array_equal(a[key],c[key])
    np.testing.assert_allclose(c['probabilities'],.5*a['probabilities']+.5*b['probabilities'],rtol=0,atol=1e-8)
    close(roc_auc_score(c['targets'],c['probabilities'],average='macro'),ensemble['source_auc'])
    gate_single=read('threshold_search/candidate_gates.json');gate_ensemble=read('ensemble_thresholds/candidate_gates.json')
    assert gate_single['checkpoint_sha256']==sha(gate_single['checkpoint'])
    for source in gate_ensemble['binding']['source_checkpoints']:assert sha(source['path'])==source['sha256'] and source['weight']==.5
    assert gate_ensemble['binding']['weights']==[.5,.5]
    audit=load('large_validation/audit_decisions.npz')
    for method,metrics in large['audit_metrics'].items():close(f1_score(audit['targets'],audit[method],average='macro',zero_division=0),metrics['macro_f1'])
    base=single['seeds']['42']['metrics']['coarse'];snew=single['seeds']['42']['metrics']['nested_selection'];enew=ensemble['seeds']['42']['metrics']['nested_selection']
    sci=single['seed42_nested_minus_coarse_bootstrap'];eci=ensemble['comparisons']['42']['versus_original_coarse']
    assert sci['ci95'][0]<0<sci['ci95'][1] and eci['ci95'][0]<0<eci['ci95'][1]
    single_method=single['selected_final_strategy'];ensemble_method=ensemble['selected_strategy']
    candidate_manifest={
      'status':'exploratory_candidates_preserved_original_not_replaced','new_training_epochs':0,
      'primary_metric_rule':'Official Macro AUC first, Macro F1 only for equal primary metric',
      'original_protected_hashes_verified':protected,
      'single_model_candidate':{'gate_file':str(ROOT/'threshold_search/candidate_gates.json'),'auc':single['source_auc'],'threshold_selection_procedure_oof_f1':snew['macro_f1'],'fixed_selected_strategy_oof_f1':single['seeds']['42']['metrics'][single_method]['macro_f1'],'ci_vs_original':sci},
      'ensemble_candidate':{'gate_file':str(ROOT/'ensemble_thresholds/candidate_gates.json'),'auc':ensemble['source_auc'],'threshold_selection_procedure_oof_f1':enew['macro_f1'],'fixed_selected_strategy_oof_f1':ensemble['seeds']['42']['metrics'][ensemble_method]['macro_f1'],'ci_vs_original':eci},
      'warnings':['No new independent test; original model/ensemble selection reused the official validation','OOF evaluates thresholds, not the entire historical model-selection process','OOF conditional bootstrap does not refit thresholds; repeated split seeds are sensitivity checks','Never mix a single-model gate file with an ensemble or parameter-average checkpoint','Original submission has not been regenerated or overwritten']}
    (ROOT/'候选方案.json').write_text(json.dumps(candidate_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# F1专项优化结果','',
      '**发现小幅改善候选，但尚无稳定提升证据；原冠军、原阈值与比赛提交文件均完整保留。**',
      '本轮没有新增训练轮次。使用既有预测做阈值优化、两个既有模型的固定50:50概率平均，并额外对50:50参数平均权重做了一次GPU验证。所有新文件在E盘，不上传GitHub。','',
      '官方任务描述中主指标是Macro AUC，仅相等时用Macro F1排序；Gate允许自定义。原文见 [任务描述.docx](C:/Users/asus/Desktop/计算机能力挑战赛/赛题相关资料/任务描述.docx)。因此单模型改阈值最直接：保持概率及AUC不变。','',
      '## 核心比较：同一主划分，患者不跨折','',
      '以下F1均为检查级十类宏平均；新方案用外层5折/内层4折患者嵌套验证，内层只选阈值方法，外层评分。原方案复现原固定粗网格五折。它们都基于同一859次检查、228名患者；不是官方隐藏测试成绩。','']
    rows=[['原冠军＋原粗网格阈值',f"{single['source_auc']:.9f}",f"{base['macro_f1']:.8f}",'基线'],
          ['原冠军＋新阈值选择流程',f"{single['source_auc']:.9f}",f"{snew['macro_f1']:.8f}",f"{100*(snew['macro_f1']-base['macro_f1']):+.4f}个百分点"],
          ['固定双模型概率平均＋原粗网格',f"{ensemble['source_auc']:.9f}",f"{ensemble['seeds']['42']['metrics']['coarse']['macro_f1']:.8f}",'概率平均对照'],
          ['固定双模型概率平均＋新阈值选择流程',f"{ensemble['source_auc']:.9f}",f"{enew['macro_f1']:.8f}",f"{100*(enew['macro_f1']-base['macro_f1']):+.4f}个百分点"]]
    lines+=table(['方案','AUC','患者OOF F1','相对原方案'],rows)
    lines+=['',f"单模型新阈值差值95%患者条件区间：{sci['ci95']}；双模型新流程：{eci['ci95']}。两者均跨0，不能声称显著或稳定提高。",
      f"概率平均AUC差值仅{avg_audit['probability_average_minus_champion_auc']['delta']:+.9f}，95%区间{avg_audit['probability_average_minus_champion_auc']['ci95']}，不能据此宣称主分可靠提高。",
      '新流程最终在完整验证集选择的固定算法是bootstrap_exact_median。其固定算法五折OOF F1分别为'
      f"{single['seeds']['42']['metrics'][single_method]['macro_f1']:.8f}（单模型）和{ensemble['seeds']['42']['metrics'][ensemble_method]['macro_f1']:.8f}（双模型）。"
      '表中的嵌套F1评价的是选择方法的流程，各外折可以选不同方法，不是把某一套最终阈值放到同批标签上的拟合分数。','',
      '## 更换患者划分后是否仍提高','']
    rows=[]
    for seed in ('42','43','44'):
        old=single['seeds'][seed]['metrics']['coarse']['macro_f1'];sn=single['seeds'][seed]['metrics']['nested_selection']['macro_f1'];en=ensemble['seeds'][seed]['metrics']['nested_selection']['macro_f1']
        rows.append([seed,f'{old:.8f}',f'{sn:.8f}',f'{en:.8f}',f'{100*(en-old):+.4f}个百分点'])
    lines+=table(['划分种子','原方案','单模型新流程','双模型新流程','双模型相对原方案'],rows)
    lines+=['','两次改善、一次下降，说明改进依赖患者划分。这三次是同一人群的敏感性检查，不是三次独立测试。第1类只有16名阳性患者；主划分每个留出折仅1、2、4、3、6人，指标波动不可忽略。','',
      '## 具体方法和逐类代价','',
      '固定比较原0.025网格、0.005网格、概率断点精确阈值，以及两种患者bootstrap阈值中位数。每次重采样整位患者、保留其全部检查；在拟合折的固定概率断点上加权计算F1，重采样40次后取各类阈值中位数。没有使用留出折标签或分数定义阈值候选。',
      '双模型是原冠军与之前A_current_regularization第2轮的概率各占50%，再用该组合专属阈值。不是模型参数平均；需要两次模型推理，不能用averaged_model.pth替代。',
      '阈值方法参考 [scikit-learn官方说明](https://scikit-learn.org/stable/modules/classification_threshold.html)。调整阈值改变精确率/召回率的取舍，不能补足全部特征区分能力。','']
    rows=[]
    for idx,label in ((0,'纵隔心影增大'),(2,'肺部实变'),(3,'肺炎')):
        rows.append([label,f"{base['f1_by_class'][idx]:.6f}",f"{snew['f1_by_class'][idx]:.6f}",f"{enew['f1_by_class'][idx]:.6f}",enew['tp_by_class'][idx],enew['fp_by_class'][idx],enew['fn_by_class'][idx]])
    lines+=table(['类别','原OOF F1','单模型新流程','双模型新流程','双模型TP','双模型FP','双模型FN'],rows)
    lines+=['',f"原方案宏精确率/召回率={base['macro_precision']:.6f}/{base['macro_recall']:.6f}；双模型新流程={enew['macro_precision']:.6f}/{enew['macro_recall']:.6f}。整体F1上升不意味着每类、精确率、召回率都同时上升。",
      '之前A组第2轮单独模型的粗网格OOF F1为0.456863，但AUC只有0.801638，低于原冠军，不作为比赛主指标替换方案。参数平均的AUC为0.801973825、OOF F1为0.451287，几乎没有收益。','',
      '## 较大队列复核','',
      '另用上一轮A模型的开发集（1658患者）选择并拟合阈值，在另一组1658患者、2842次检查上比较。模型训练时未见这两组患者；该保留集上一轮已看过，因此是内部复核，不是新盲测。原全量冠军见过这些患者，不能拿它直接做此处对照。']
    lbase=large['audit_metrics']['coarse']['macro_f1'];lnew=large['audit_metrics']['bootstrap_exact_median']['macro_f1'];lci=large['paired_f1_intervals']['bootstrap_exact_median']
    lines += [f"F1从{lbase:.8f}到{lnew:.8f}，仅{100*(lnew-lbase):+.4f}个百分点，95%区间{lci['ci95']}跨0；概率及AUC保持不变。当前证据支持继续研究F1，但不能承诺已有稳定提升。",'',
      '## 结论边界与保存','',
      '原模型/轮次与本轮平均组合曾使用同一官方验证集进行选择，嵌套阈值CV不能撤销这部分偏差。所有置信区间条件于保存的决策，没有重新拟合整个流程，也不包含外部医院或设备差异。没有读取隐藏测试标签。',
      '本轮不继续追加小batch的纯Soft-F1训练：batch12对稀有类别经常没有阳性，直接替换BCE需要另外设计有效批次和对照。普通梯度累积不等于在大批次上统一计算非线性的F1损失。',
      '单模型候选阈值：[threshold_search/candidate_gates.json](threshold_search/candidate_gates.json)。双模型候选：[ensemble_thresholds/candidate_gates.json](ensemble_thresholds/candidate_gates.json)。两份文件均绑定适用权重与SHA，不能互换。最终全验证集拟合内F1（0.4711/0.4720）不作为泛化成绩。',
      '原冠军、gates、submission均通过SHA核验，本轮没有改写比赛提交。候选仍待更强验证；本轮0个新增训练epoch。环境沿用E:\\比赛数据集\\envs\\cxr-smoke-cu128，没有新安装。','',
      '| 子文件夹 | 内容 |','|---|---|',
      '| threshold_search | 原冠军的5种阈值策略、嵌套患者CV、候选Gate和独立核对 |',
      '| average_probe | 固定概率平均与参数平均、一次GPU验证、新参数平均权重和原图预测 |',
      '| ensemble_thresholds | 固定双模型平均的专属阈值、嵌套CV和3种患者划分对照 |',
      '| large_validation | 较大开发/保留患者队列的阈值迁移复核 |','',
      '汇总校验.json记录从保存的OOF决策独立复算的结果；候选方案.json绑定两类候选；实验设计.md记录比较规则。']
    (ROOT/'结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'汇总校验.json').write_text(json.dumps({'passed':True,'checked_method_cohorts':len(checks),'metric_checks':checks,'ensemble_probability_mean_verified':True,'protected_hashes_verified':protected,'new_training_epochs':0,'original_submission_unchanged':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary_written':True,'cohorts_checked':len(checks),'baseline_f1':base['macro_f1'],'single_nested_f1':snew['macro_f1'],'ensemble_nested_f1':enew['macro_f1'],'all_originals_unchanged':True},ensure_ascii=False))
if __name__=='__main__':main()
