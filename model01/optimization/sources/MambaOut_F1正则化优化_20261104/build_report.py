"""Build the final local report after all regularization pilots finish (stdlib only)."""
import csv
import json
import math
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES={'A_current_regularization':'A 原正则化继续微调','B_stronger_regularization':'B 加强Dropout/DropPath','C_stronger_regularization_rare_weight':'C 加强正则化及稀有类阳性加权'}
SHORT={name:name[0] for name in NAMES}
RARE=(1,3,4)


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def link(p,label=None):
 p=Path(p)
 return f'[{label or p.name}]({p.as_posix()})'
def fmt(v,digits=6):return f'{v:.{digits}f}'
def pp(v):return f'{v*100:+.4f}'
def pair(m):return m['study']['macro_auc'],m['study']['macro_f1']
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
def metric_check(m):
 assert m['image']['count']==1422 and m['study']['count']==859
 for scope in ('image','study'):
  a=m[scope]
  assert a['auc_classes']==10 and len(a['auc'])==len(a['f1'])==10
  assert close(sum(a['auc'])/10,a['macro_auc']) and close(sum(a['f1'])/10,a['macro_f1'])


def main():
 if not (ROOT/'results.json').is_file():raise SystemExit('Results not complete. No report written.')
 result=load(ROOT/'results.json')
 status=load(ROOT/'status.json')
 if status.get('state')!='complete':raise SystemExit('Training status is not complete. No report written.')
 plan=load(ROOT/'experiment_plan.json')
 baseline=load(ROOT/'baseline_metrics.json')
 diagnosis=load(ROOT/'diagnosis/diagnosis.json')
 receipt=load(ROOT/'preserved_champion/preservation_receipt.json')
 mapping=list(csv.DictReader((ROOT.parent/'本地跑通/class_mapping.csv').open(encoding='utf-8-sig',newline='')))
 class_names={int(c['class_id']):c['chinese_name'] for c in mapping}
 assert sorted(class_names)==list(range(1,11))
 metric_check(result['initial_validation']);metric_check(result['final_validation'])
 assert pair(result['initial_validation'])==pair(baseline['validation'])
 assert close(result['initial_validation']['study']['macro_auc'],0.8019769550500871)
 assert close(result['initial_validation']['study']['macro_f1'],0.3299175803247271)
 assert close(result['initial_calibration']['oof_macro_f1'],0.451054963682343)
 assert result['preserved_sha256']=='9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585'
 assert set(result['trials'])==set(NAMES)
 rows=[];histories={}
 for name in NAMES:
  trial=result['trials'][name]
  history=load(ROOT/name/'history.json')
  histories[name]=history
  assert [e['epoch'] for e in history]==list(range(1,len(history)+1))
  assert len(history)==trial['epochs_completed'] and len(history)<=2
  for h in history:
   assert h['images']==30344
   metric_check(h['validation'])
   assert close(sum(h['calibration']['oof_f1_by_class_id'])/10,h['calibration']['oof_macro_f1'])
   extra=h['extra']
   assert len(extra['tp'])==len(extra['fp'])==len(extra['fn'])==10
   for c in range(10):
    den=2*extra['tp'][c]+extra['fp'][c]+extra['fn'][c]
    f1=2*extra['tp'][c]/den if den else 0
    assert close(f1,h['validation']['study']['f1'][c])
   weight=ROOT/name/f"epoch_{h['epoch']}.pth"
   assert weight.is_file()
   rows.append(dict(run=name,label=f"{SHORT[name]}第{h['epoch']}轮",epoch=h['epoch'],validation=h['validation'],calibration=h['calibration'],extra=h['extra'],weight=str(weight),train_loss=h['train_loss'],train_seconds=h['train_seconds'],validation_seconds=h['validation_seconds']))
  candidates=[(0,baseline['validation'])]+[(e['epoch'],e['validation']) for e in history]
  epoch,metrics=max(candidates,key=lambda e:pair(e[1]))
  assert trial['best_epoch']==epoch and pair(trial['best_validation'])==pair(metrics)
  metric_check(trial['best_validation'])
 n=len(rows)
 assert n==result['completed_training_epochs'] and n*30344==result['training_image_visits']
 base=dict(run='preserved_champion',label='原最优起点',epoch=0,validation=baseline['validation'],calibration=baseline['calibration'],extra=baseline['extra'],weight=result['preserved_checkpoint'])
 all_rows=[base]+rows
 auc_winner=max(all_rows,key=lambda e:pair(e['validation']))
 default_f1_winner=max(all_rows,key=lambda e:(e['validation']['study']['macro_f1'],e['validation']['study']['macro_auc']))
 oof_f1_winner=max(all_rows,key=lambda e:(e['calibration']['oof_macro_f1'],e['validation']['study']['macro_auc']))
 assert pair(auc_winner['validation'])==pair(result['final_validation'])
 baseline_auc=base['validation']['study']['macro_auc']
 final_auc=result['final_validation']['study']['macro_auc']
 delta=final_auc-baseline_auc
 interval=result['paired_auc_interval'];low,high=interval['ci95']
 assert close(interval['delta'],delta)
 if delta>0:
  decision=f"当前官方开发验证集AUC最高为**{auc_winner['label']}**，AUC从{baseline_auc:.8f}升至{final_auc:.8f}，增加{delta*100:.4f}个百分点。"
  if low<=0<=high:decision+='患者配对95%区间跨0，尚不能认定对新患者有可靠提升。'
  else:decision+='该区间仅描述在已选定权重下的抽样差异，仍受反复选模影响，不能替代独立测试。'
 elif result['winner']!='preserved_champion':decision=f"本次主指标AUC与原最优{baseline_auc:.8f}持平，按默认F1次优排序选择{auc_winner['label']}。"
 else:decision=f"本次主指标排序没有超过原最优AUC {baseline_auc:.8f}，最终候选继续保留原起点。"
 lines=['# MambaOut F1诊断与随机正则化微调结果','',f"整理时间 {datetime.now().astimezone().isoformat(timespec='seconds')}",'',decision,'',f"本次共完成 **{n}个全量微调轮次**，每轮使用30,344张训练图；此前累计110个完整轮次，本次后累计**{110+n}轮**，其中{99+n}轮全量、9轮患者划分实验、2轮2,000图试跑。载入起点、验证复核和阈值校准不增加训练轮数。",'','## 原最优结果的保护','',f"原最优权重来自旧BCE第3轮继续保护性微调的新增第1轮，其开发验证集AUC为{baseline_auc:.10f}、F1@0.5为{base['validation']['study']['macro_f1']:.10f}。",'',f"原最优备份 {link(result['preserved_checkpoint'])}",f"原最优SHA256 `{result['preserved_sha256']}`",'',f"本次主指标候选 {link(result['selected_checkpoint'])}",f"本次候选SHA256 `{result['selected_sha256']}`",'',f"备份清单 {link(ROOT/'preserved_champion/preservation_receipt.json')}，包含{len(receipt)}项源文件与备份对应记录。训练流程结束时再次核对了源文件和备份SHA256。旧最优及此前submission.csv保留，新候选另存。",'','## F1为零的核查','', '旧best3在默认阈值0.5下，类1纵隔心影增大、类3肺部实变、类4肺炎的F1为0。作为本次起点的S7最优模型，只有类1仍为0，类3/4虽已非零，召回仍很低。','', '|权重|类别|阳性检查数|TP@0.5|FP@0.5|FN@0.5|F1@0.5|', '|---|---|---:|---:|---:|---:|---:|']
 for model,label in [('old_best3','旧BCE第3轮'),('current_best','本次原最优起点')]:
  for c in diagnosis['models'][model]['classes']:
   if c['class_id'] in RARE:
    d=c['default_threshold']
    lines.append(f"|{label}|{c['class_id']} {c['class_name']}|{c['positive_studies']}|{d['tp']}|{d['fp']}|{d['fn']}|{d['f1']:.6f}|")
 lines+=['','类1当前最高概率0.211906，24个真实阳性中最高分只有0.129860，统一门槛0.5使其全部被判阴性。旧类3曾仅预测1例阳性且为误报。因此这些零分有明确的混淆矩阵原因。原始标签编码、同次检查概率均值、手算AUC与F1均复核一致，未发现类别错位或指标计算错误。', '', '固定Gate=0.1时，本次起点类1在同一验证集检出4/24阳性并产生20例误报；类1AP仅0.06535。调整门槛有作用，但不能据此认为少数类识别问题已经解决。', '', f"详细诊断与预测来源见 {link(ROOT/'diagnosis/诊断报告.md')}。",'','## 本次实验设置','', '三组从同一份原最优权重重新开始，独立建立优化器，输入224、batch 12、相同seed42、相同数据顺序/增强、AdamW weight_decay=0.05。第一轮主干学习率1e-6、分类头1e-5，第二轮均减半。逐组顺序执行，每组上限2轮。', '', '|分支|Dropout|最大DropPath|损失|阳性加权|', '|---|---:|---:|---|---|']
 for name in NAMES:
  setting=result['trials'][name]['settings']
  weighting='无额外加权' if setting['loss']=='BCE' else '仅训练集阳性比例低于10%的类别；sqrt(负样本/正样本)，上限3'
  lines.append(f"|{NAMES[name]}|{setting['drop_rate']:g}|{setting['drop_path_rate']:g}|{setting['loss']}|{weighting}|")
 lines+=['', '训练时Dropout/DropPath启用，验证与推理时关闭。DropPath按模型各层分配，上表是最大概率；层实例化及启用状态由运行脚本检查并保存在各分支regularization_modules.json。A/B同时改变了Dropout和DropPath，因此只能比较这一组合，无法单独归因给某一项。', '', f"C组实际class_id 1–10阳性权重 `{plan['pos_weight']}`，仅由30,344张训练图的标签频率计算。加权BCE和普通BCE的loss尺度不同，不能直接以训练loss大小比较优劣。",'','## 每个新轮次的结果','', '所有结果均来自同一官方开发验证集：1,422张图、859次检查、228名患者。主指标是逐图sigmoid后按Study取概率均值计算的10类Macro AUC。表中三种F1必须分开解释。','', '- F1@0.5为统一固定阈值；零F1类别也按此口径统计。','- 旧冻结Gate F1对全部新权重使用原模型已确定的同一套阈值，方便固定判定点比较。该阈值原先利用过这份验证集。','- 患者OOF F1每轮使用同样5折患者划分，仅在4折拟合逐类阈值并评估留出折。模型/轮次仍由整份开发验证集选择，因此它属于内部估计。','', '|权重|AUC|F1@0.5|旧冻结Gate F1|患者OOF F1|Macro AP|默认零F1类别|', '|---|---:|---:|---:|---:|---:|---|']
 for r in all_rows:
  m=r['validation']['study'];e=r['extra'];flags=[]
  if r is auc_winner:flags.append('AUC最高')
  if r is default_f1_winner:flags.append('默认F1最高')
  if r is oof_f1_winner:flags.append('OOF F1最高')
  label=r['label']+('（'+ '、'.join(flags)+'）' if flags else '')
  lines.append(f"|{label}|{m['macro_auc']:.6f}|{m['macro_f1']:.6f}|{e['frozen_previous_gate_macro_f1']:.6f}|{r['calibration']['oof_macro_f1']:.6f}|{e['macro_ap']:.6f}|{','.join(map(str,e['zero_f1_classes'])) or '无'}|")
 lines+=['','每组按AUC优先、F1@0.5次优保留权重。若分支best_epoch=0，表示其新训练未在该排序下超过共同起点，以上仍完整列出其实际新轮次的F1与其他结果。','', '|分支|实际完整轮数|按主指标保留的轮|保留AUC|保留F1@0.5|停止原因|','|---|---:|---|---:|---:|---|']
 for name,t in result['trials'].items():
  ep='原起点（0个新增轮）' if t['best_epoch']==0 else f"新增第{t['best_epoch']}轮"
  reason='2轮短测预算完成' if t['stop_reason']=='pilot_epoch_budget' else 'AUC较起点下降超过0.01的保护退出'
  lines.append(f"|{SHORT[name]}|{t['epochs_completed']}|{ep}|{pair(t['best_validation'])[0]:.6f}|{pair(t['best_validation'])[1]:.6f}|{reason}|")
 lines+=['','## 同轮对照','', '差值为B减A或C减B，单位为百分点。B−A比较更强随机正则化的组合；C−B比较在相同较强正则化下加入稀有类阳性加权。每行两组的起点、完整训练轮数和学习率日程相同。','', '|对照|新增轮次|AUC变化|F1@0.5变化|OOF F1变化|类1/3/4平均AP变化|', '|---|---:|---:|---:|---:|---:|']
 for left,right,title in [('A_current_regularization','B_stronger_regularization','B − A'),('B_stronger_regularization','C_stronger_regularization_rare_weight','C − B')]:
  for ep in sorted(set(e['epoch'] for e in histories[left])&set(e['epoch'] for e in histories[right])):
   a=next(e for e in histories[left] if e['epoch']==ep);b=next(e for e in histories[right] if e['epoch']==ep)
   apdiff=sum(b['extra']['average_precision_by_class'][c-1]-a['extra']['average_precision_by_class'][c-1] for c in RARE)/len(RARE)
   lines.append(f"|{title}|{ep}|{pp(pair(b['validation'])[0]-pair(a['validation'])[0])}|{pp(pair(b['validation'])[1]-pair(a['validation'])[1])}|{pp(b['calibration']['oof_macro_f1']-a['calibration']['oof_macro_f1'])}|{pp(apdiff)}|")
 lines+=['','## 三个重点类别逐轮结果','', '|权重|类别|AUC|AP|TP|FP|FN|F1@0.5|旧冻结Gate F1|患者OOF F1|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in all_rows:
  for c in RARE:
   i=c-1;m=r['validation']['study'];e=r['extra']
   lines.append(f"|{r['label']}|{c} {class_names[c]}|{m['auc'][i]:.6f}|{e['average_precision_by_class'][i]:.6f}|{e['tp'][i]}|{e['fp'][i]}|{e['fn'][i]}|{m['f1'][i]:.6f}|{e['frozen_previous_gate_f1_by_class'][i]:.6f}|{r['calibration']['oof_f1_by_class_id'][i]:.6f}|")
 lines+=['','## 候选权重与主指标区间','',f"**AUC候选**为{auc_winner['label']}，AUC={pair(auc_winner['validation'])[0]:.8f}、F1@0.5={pair(auc_winner['validation'])[1]:.8f}，对应 {link(result['selected_checkpoint'])}。",'',f"**默认F1候选**为{default_f1_winner['label']}，F1@0.5={default_f1_winner['validation']['study']['macro_f1']:.8f}、同期AUC={default_f1_winner['validation']['study']['macro_auc']:.8f}，对应 {link(default_f1_winner['weight'])}。",'',f"**内部OOF F1候选**为{oof_f1_winner['label']}，OOF F1={oof_f1_winner['calibration']['oof_macro_f1']:.8f}、同期AUC={oof_f1_winner['validation']['study']['macro_auc']:.8f}，对应 {link(oof_f1_winner['weight'])}。",'', '各指标的最高分可能来自不同权重，不能组合成一个模型的成绩。各新epoch的权重和预测均保留，可追溯到具体分支、轮次与阈值。', '', f"主指标候选减原最优的AUC差值为{delta:+.10f}，即{delta*100:+.4f}个百分点；患者配对bootstrap 95%区间为[{low:+.10f}, {high:+.10f}]，换算为[{low*100:+.4f}, {high*100:+.4f}]个百分点，{interval.get('valid_repeats',interval.get('repeats',400))}个有效重复。",'']
 if result['winner']=='preserved_champion':lines+=['这里差值和区间为0，是因为最终选回原权重，两端比较的是同一份模型。这不构成新正则化或加权配方与原模型无差异的统计证据；新配方必须分别比较其自身预测。','']
 elif low<=0<=high:lines+=['区间跨0或包含0，不能据此宣称可靠提升。','']
 lines+=['该区间只反映已经选定的两份固定权重在这228名患者上的抽样不确定性，未计入多个候选反复筛选的偏差或重新训练的随机性。','', '## 优化局限与使用建议','', '1. 本次每组最多2轮，是从已有最优权重开始的微调短测，不能代表从头训练的表现，也不能说明MambaOut架构已到上限。','2. A/B是Dropout与DropPath的联合修改，未分别隔离两种正则化作用；所有实验只有一个随机种子。','3. 官方验证集已反复用于选轮和选配方。阈值OOF仍有模型选择偏差；没有新增独立患者集或外部中心验证。','4. 少数类别官方阳性检查分别只有24、46、68例，F1对几例判断非常敏感。应同时看AP、TP/FP/FN与召回，避免只看是否从0变非0。','5. 阳性加权会改变分数分布，可能在默认阈值下检出更多病例，也可能增加误报并损伤AUC。旧冻结阈值和重新校准的OOF指标反映不同条件，不能互相替代。','6. 正则化仅在训练时随机生效，最终验证和提交推理使用eval模式。主指标候选经过重新加载权重复核；若AUC优先、默认F1次优的排序未超过起点则保留原权重，其他F1候选单独列为备选。','', '本次未使用测试集进行调参，也没有重新生成或覆盖已交付的测试submission.csv。新候选是否值得替换，应由上述匹配对照及后续独立验证决定。','', '## 逐轮来源与耗时','', '|权重|训练loss|训练秒|验证秒|保存的权重|', '|---|---:|---:|---:|---|']
 for r in rows:lines.append(f"|{r['label']}|{r['train_loss']:.8f}|{r['train_seconds']:.2f}|{r['validation_seconds']:.2f}|{link(r['weight'])}|")
 lines+=['',f"实验计划 {link(ROOT/'experiment_plan.json')}",f"最终结果 {link(ROOT/'results.json')}",f"起点复核 {link(ROOT/'baseline_metrics.json')}",'']
 for name in NAMES:lines.append(f"{SHORT[name]}逐轮来源 {link(ROOT/name/'history.json')}")
 lines+=['']
 comparison_path=ROOT/'candidate_comparison.json'
 verification_path=ROOT/'f1_candidate_verification.json'
 if not comparison_path.is_file() or not verification_path.is_file():
  raise SystemExit('Final candidate comparison and weight verification must be present. Existing report is not overwritten.')
 audit=load(comparison_path);verify=load(verification_path)
 assert audit['verification']['passed'] and verify['verified'] and verify['exact_prediction_reproduction']
 assert audit['verification']['completed_new_epochs']==n
 c2=next(r for r in rows if r['run']=='C_stronger_regularization_rare_weight' and r['epoch']==2)
 assert pair(verify['validation'])==pair(c2['validation'])
 assert close(verify['calibration']['oof_macro_f1'],c2['calibration']['oof_macro_f1'])
 assert Path(verify['checkpoint'])==Path(c2['weight'])
 comp=audit['comparison'];a=comp['study_macro_auc'];f=comp['study_macro_f1_at_0_5']
 assert close(a['baseline'],baseline_auc) and close(a['candidate'],c2['validation']['study']['macro_auc'])
 assert close(f['baseline'],base['validation']['study']['macro_f1']) and close(f['candidate'],c2['validation']['study']['macro_f1'])
 assert close(a['delta'],a['candidate']-a['baseline']) and close(f['delta'],f['candidate']-f['baseline'])
 assert verify['sha256']==audit['source_sha256'][str(Path(c2['weight']))]
 oof_delta=c2['calibration']['oof_macro_f1']-base['calibration']['oof_macro_f1']
 frozen_delta=c2['extra']['frozen_previous_gate_macro_f1']-base['extra']['frozen_previous_gate_macro_f1']
 extra_lines=['## C2默认F1候选的独立复核与配对比较','',
  f"六轮保存的Study预测经独立CPU复算，AUC、F1、TP/FP/FN及固定Gate结果均与history一致。C2权重另经重新加载推理复核，逐图和Study指标、逐Study概率数组及专用阈值均复现；该复核新增训练更新为0。",'',
  f"C2权重SHA256 `{verify['sha256']}`",'',
  '|指标|原最优起点|C2|变化（百分点）|95%患者配对区间（百分点）|',
  '|---|---:|---:|---:|---|',
  f"|Study Macro AUC|{a['baseline']:.8f}|{a['candidate']:.8f}|{a['delta_percentage_points']:+.5f}|[{a['ci95_percentage_points'][0]:+.5f}, {a['ci95_percentage_points'][1]:+.5f}]|",
  f"|Study Macro F1@0.5|{f['baseline']:.8f}|{f['candidate']:.8f}|{f['delta_percentage_points']:+.5f}|[{f['ci95_percentage_points'][0]:+.5f}, {f['ci95_percentage_points'][1]:+.5f}]|",'',
  '以上为C2实际预测减原最优预测，和最终选回同一原权重产生的零差值是不同比较。使用228名患者整簇配对重采样400次，全部重复有效。', '',
  '固定0.5门槛的F1区间高于0，说明对这两份固定预测、这份开发队列而言存在观测到的F1收益；此区间没有计入挑选C2的选择偏差、多个候选比较或训练种子波动，不能据此宣称独立测试改善。AUC区间跨0，也不能据此宣称两种模型等效。', '',
  f"C2的收益集中在统一0.5门槛下。其患者OOF F1为{c2['calibration']['oof_macro_f1']:.8f}，相对起点{oof_delta*100:+.5f}个百分点；沿用原冻结Gate的F1为{c2['extra']['frozen_previous_gate_macro_f1']:.8f}，相对起点{frozen_delta*100:+.5f}个百分点。**此前交付的submission.csv使用自定义逐类Gate，因此不能把默认0.5下的+{f['delta_percentage_points']:.5f}个百分点直接称为比赛提交F1提高。**",'',
  f"C2默认Macro Precision从{base['extra']['macro_precision']:.6f}降至{c2['extra']['macro_precision']:.6f}，Macro Recall从{base['extra']['macro_recall']:.6f}升至{c2['extra']['macro_recall']:.6f}。类3检出的阳性从1例增至9例、误报从1例增至25例；类4检出从1例增至12例、误报从0例增至32例；类1在0.5门槛下仍无阳性预测。",'',
  f"CPU独立核验及配对区间 {link(comparison_path)}",f"C2重新加载权重复核 {link(verification_path)}",'']
 report_text='\n'.join(lines)
 report_text=report_text.replace('## 优化局限与使用建议','\n'.join(extra_lines)+'\n## 优化局限与使用建议')
 (ROOT/'报告.md').write_text(report_text,encoding='utf-8')
 print(json.dumps({'report':str(ROOT/'报告.md'),'completed_new_epochs':n,'cumulative_completed_epochs':110+n,'auc_candidate':auc_winner['label'],'default_f1_candidate':default_f1_winner['label'],'oof_f1_candidate':oof_f1_winner['label'],'delta_auc':delta,'ci95':[low,high]},ensure_ascii=False))

if __name__=='__main__':main()
