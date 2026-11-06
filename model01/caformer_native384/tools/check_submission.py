"""Check submission schema/official IDs/Gates only; cannot recompute model probabilities."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re
from collections import OrderedDict
from pathlib import Path
FIELDS=['Subject_id','Study_id','Predict_class','Probability','Gate']
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('submission','official-test-csv','gates'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output-report',type=Path);a=p.parse_args()
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    if a.output_report:
        inputs=[a.submission,a.official_test_csv,a.gates]+([a.checkpoint] if a.checkpoint else [])
        need(not a.output_report.exists() and a.output_report.resolve() not in [x.resolve() for x in inputs],'Refuse to overwrite an existing report or input file')
    g=json.loads(a.gates.read_text(encoding='utf-8-sig'));need(g['class_ids']==list(range(1,11)),'Wrong Gate class order');gates=g['final_gates_by_class_id']
    need(len(gates)==10 and all(math.isfinite(x) and 0<=x<=1 for x in gates),'Invalid Gates')
    if a.checkpoint:need(sha(a.checkpoint)==g['checkpoint_sha256'],'Checkpoint/Gate binding mismatch')
    need(sha(a.official_test_csv)=='63e107d840cbf27aee7c1c0ced1157aef454baa1141fdc3851e6c3011471529e','Wrong official test metadata SHA')
    with a.official_test_csv.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
    order=list(OrderedDict(((x['Subject_id'],x['Study_id']),None) for x in r))
    need(len(r)==2218 and len(order)==1414 and len({x[0] for x in order})==136,'Wrong official test coverage')
    need(not a.submission.read_bytes().startswith(b'\xef\xbb\xbf'),'Submission must be UTF8 without BOM');n=0
    with a.submission.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);need(reader.fieldnames==FIELDS,'Wrong columns/order')
        for row in reader:
            need(n<14140 and None not in row,'Extra rows/columns');i,c=divmod(n,10)
            need((row['Subject_id'],row['Study_id'])==order[i] and row['Predict_class']==str(c+1),'Wrong Study/class order')
            prob=float(row['Probability']);gate=float(row['Gate'])
            need(math.isfinite(prob) and 0<=prob<=1 and re.fullmatch(r'(?:0\.[0-9]{12}|1\.000000000000)',row['Probability']) is not None,'Invalid twelve-decimal probability')
            need(math.isfinite(gate) and gate==gates[c],'Wrong Gate');n+=1
    need(n==14140,'Missing rows')
    result=dict(state='structure_passed',rows=n,studies=1414,patients=136,submission_sha256=sha(a.submission),official_test_csv_sha256=sha(a.official_test_csv),gates_sha256=sha(a.gates),checkpoint_file_checked=bool(a.checkpoint),expected_checkpoint_sha256=g['checkpoint_sha256'],does_not_recompute_model_predictions=True,test_labels_available=False,test_AUC_or_F1_computed=False)
    if a.output_report:a.output_report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SUBMISSION_STRUCTURE_PASSED',json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
