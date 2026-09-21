"""Post-run M1 diagnostic tables; no new model evaluations. Run on lp."""
import hashlib,json,os,platform
from collections import Counter
from pathlib import Path
from jev_reasoning.metrics import metrics
root=Path(os.environ['JEV_ROOT']);out=root/'results/m1-report-v1'
all_rows=[];provenance={}
for run in ['m0-baseline-v1','m1-validation-v1','m1-pilot-v1']:
 folder=root/'results'/run;m=json.loads((folder/'manifest.json').read_text())
 assert m['status']=='complete'
 h=hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest()
 assert h==m['artifacts']['predictions.jsonl']
 provenance[run]={'predictions_sha256':h,'source_sha256':m['source_sha256']}
 for row in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()):
  row['cohort_name']=m['cohort'];all_rows.append(row)
validation=json.loads((root/'results/m1-validation-v1/manifest.json').read_text())
pilot=json.loads((root/'results/m1-pilot-v1/manifest.json').read_text())
assert validation['finished_unix']<pilot['start_unix']
by={}
for cohort in ['pilot','validation']:
 for condition in ['A-ref','A','B','C32','C128','C512','D']:
  rs=[r for r in all_rows if r['cohort_name']==cohort and r['condition']==condition]
  if rs:by[(cohort,condition)]=rs

def describe(rs):
 m=metrics(rs)
 if not rs:return m
 m.update(prediction_counts=dict(Counter(str(r['prediction']) for r in rs)),termination_counts=dict(Counter(r['termination'] for r in rs)),mean_candidate_mass=sum(r['candidate_mass'] for r in rs)/len(rs),input_token_range=[min(r['input_tokens'] for r in rs),max(r['input_tokens'] for r in rs)],mean_input_tokens=sum(r['input_tokens'] for r in rs)/len(rs),processed_tokens=sum(r['processed_tokens'] for r in rs),sampled_reasoning_tokens=sum(r['sampled_reasoning_tokens'] for r in rs))
 return m

slices={}
for (cohort,condition),rs in by.items():
 for name,predicate in [('deep',lambda r:r['depth'] in [3,5]),('shallow',lambda r:r['depth'] in [0,1]),('unknown',lambda r:r['gold']=='Unknown'),('provable',lambda r:r['depth'] is not None),('all',lambda r:True)]:
  group=[r for r in rs if predicate(r)]
  slices[f'{cohort}/{condition}/{name}']=describe(group)
  if condition.startswith('C'):
   for stop in ['natural_close','budget_exhausted','early_eos','nested_think']:
    slices[f'{cohort}/{condition}/{name}/{stop}']=describe([r for r in group if r['termination']==stop])
 if condition=='D':
  for name,predicate in [('nonempty',lambda r:r['oracle_facts']>0),('noop',lambda r:r['oracle_facts']==0)]:
   slices[f'{cohort}/D/oracle_{name}']=describe([r for r in rs if predicate(r)])

transitions={}
base={r['id']:r for r in by[('pilot','A')]}
for c in ['C32','C128','C512','D']:
 rs=by[('pilot',c)]
 for name,predicate in [('all',lambda r:True),('deep',lambda r:r['depth'] in [3,5]),('shallow',lambda r:r['depth'] in [0,1]),('unknown',lambda r:r['gold']=='Unknown'),('oracle_nonempty',lambda r:r.get('oracle_facts',0)>0)]:
  chosen=[r for r in rs if predicate(r)]
  transitions[f'{c}-A/{name}']={'n':len(chosen),'wrong_to_right':sum(not base[r['id']]['correct'] and r['correct'] for r in chosen),'right_to_wrong':sum(base[r['id']]['correct'] and not r['correct'] for r in chosen),'both_right':sum(base[r['id']]['correct'] and r['correct'] for r in chosen),'both_wrong':sum(not base[r['id']]['correct'] and not r['correct'] for r in chosen)}

result={'source_runs':provenance,'host':platform.node(),'job_dir':os.environ['JEV_JOB'],'validation_finished_before_pilot_started':True,'slices':slices,'paired_transitions':transitions,'limitations':['Termination slices are post-treatment selections, not randomized causal comparisons.','Single-seed pilot; unadjusted exploratory interval tables remain in report.json.','No-op oracle and nonempty oracle cases have distinct denominators.']}
with (out/'diagnostics.json').open('x') as f:json.dump(result,f,indent=2)
lines=['# M1 supplementary diagnostics','', 'Computed on lp from complete, checksum-verified results. No additional model runs.','', '## Termination by budget and depth','', '| Condition | Slice | N | Natural close | Forced close | Accuracy |','|---|---|---:|---:|---:|---:|']
for c in ['C32','C128','C512']:
 for name in ['all','deep','shallow','unknown']:
  m=slices[f'pilot/{c}/{name}']
  lines.append(f'| {c} | {name} | {m["n"]} | {m["natural_close_fraction"]:.3f} | {m["forced_close_fraction"]:.3f} | {m["accuracy"]:.3f} |')
lines+=['','Termination groups differ in question difficulty and are not causal treatment comparisons.','', '## Oracle coverage','', '| Slice | N | Accuracy | NLL | Brier sum |','|---|---:|---:|---:|---:|']
for name in ['provable','oracle_nonempty','oracle_noop']:
 m=slices[f'pilot/D/{name}'];lines.append(f'| {name} | {m["n"]} | {m["accuracy"]:.3f} | {m["nll"]:.3f} | {m["brier_sum"]:.3f} |')
with (out/'diagnostics.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'status':'complete','output':str(out/'diagnostics.json'),'validation_precedes_pilot':True}),flush=True)
