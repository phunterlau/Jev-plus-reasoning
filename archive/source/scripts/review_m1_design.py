"""Post-hoc mathematical/design review tables from frozen results, on lp.
No inference, fitting, or selection. Original results are immutable.
"""
import hashlib,json,math,os,platform,statistics
from collections import Counter
from pathlib import Path
root=Path(os.environ['JEV_ROOT']);out=root/'results/m1-design-review-v1'
out.mkdir(exist_ok=False)
rows=[];sources={}
for run in ['m0-baseline-v1','m1-pilot-v1']:
 p=root/'results'/run;m=json.loads((p/'manifest.json').read_text())
 assert m['status']=='complete'
 h=hashlib.sha256((p/'predictions.jsonl').read_bytes()).hexdigest()
 assert h==m['artifacts']['predictions.jsonl'];sources[run]=h
 rows.extend(map(json.loads,(p/'predictions.jsonl').read_text().splitlines()))
summary={}
for c in ['A-ref','A','B','C32','C128','C512','D']:
 rs=[r for r in rows if r['condition']==c]
 slices={'all':rs,'deep':[r for r in rs if r['depth'] in [3,5]],'natural':[r for r in rs if r['termination']=='natural_close'],'forced':[r for r in rs if r['termination']=='budget_exhausted']}
 for sl,group in slices.items():
  if not group:continue
  summary[f'{c}/{sl}']={'n':len(group),'accuracy':statistics.mean(r['correct'] for r in group),'candidate_mass_mean':statistics.mean(r['candidate_mass'] for r in group),'candidate_mass_median':statistics.median(r['candidate_mass'] for r in group),'mass_below_0_1':sum(r['candidate_mass']<.1 for r in group),'full_vocab_argmax_outside_slots':sum(r['unrestricted_token_id'] not in [32,33,34] for r in group),'full_vocab_top_tokens':dict(Counter(r['unrestricted_token_text'] for r in group).most_common(8)),'restricted_nll':statistics.mean(-math.log(max(r['probabilities'][r['option_ids'].index(r['gold'])],1e-12)) for r in group),'brier_sum':statistics.mean(sum((p-float(label==r['gold']))**2 for label,p in zip(r['option_ids'],r['probabilities'])) for r in group),'median_latency_ms':1000*statistics.median(r['total_seconds'] for r in group)}
result={'status':'complete','posthoc_review':True,'host':platform.node(),'job':os.environ['JEV_JOB'],'source_predictions_sha256':sources,'uniform_three_class_reference':{'nll':math.log(3),'brier_sum':2/3},'summary':summary,'limits':['Post-hoc diagnostic, not a new test or protocol selection.','Natural/forced groups are selected by model behavior and are not causal comparisons.','Candidate mass is a format diagnostic, not a calibrated estimate of semantic correctness.','No model runs or parameter changes.']}
(out/'review_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':'complete','output':str(out),'host':platform.node()}),flush=True)
