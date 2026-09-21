"""Independent completion audit for frozen M0/M1 outputs. Run on lp."""
import hashlib,json,math,os
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from jev_reasoning.prompts import encode,messages,sha
tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",revision="c1899de289a04d12100db370d81485cdf75e47ca",local_files_only=True,trust_remote_code=False)
root=Path(os.environ['JEV_ROOT']);data=root/'data/derived/owa-v2'
cohort_manifest=json.loads((data/'manifest.json').read_text())
cohorts={name:[json.loads(x) for x in (data/f'{name}.jsonl').read_text().splitlines()] for name in ['pilot','validation','smoke']}
for name,rows in cohorts.items():
 assert hashlib.sha256((data/f'{name}.jsonl').read_bytes()).hexdigest()==cohort_manifest['cohorts'][name]['sha256']
 assert len({r['group'] for r in rows})==len(rows)
for a,b in [('pilot','validation'),('pilot','smoke'),('validation','smoke')]:
 assert not {r['group'] for r in cohorts[a]}&{r['group'] for r in cohorts[b]}
all_predictions={};checked={}
for run_id,name,conditions in [('m0-baseline-v1','pilot',['A-ref','A','B']),('m1-validation-v1','validation',['A','C32','C128','C512']),('m1-pilot-v1','pilot',['C32','C128','C512','D'])]:
 folder=root/'results'/run_id;m=json.loads((folder/'manifest.json').read_text())
 assert m['status']=='complete'
 rows={r['id']:r for r in cohorts[name]}
 pred=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
 traces=[json.loads(x) for x in (folder/'traces.jsonl').read_text().splitlines()]
 expected={(r['id'],c) for r in rows.values() for c in conditions if c!='D' or r['gold']!='Unknown'}
 assert len(pred)==len(expected)==m['expected_rows']==m['observed_rows']
 assert {(r['id'],r['condition']) for r in pred}==expected
 assert len(traces)==len(pred) and {(r['id'],r['condition']) for r in traces}==expected
 trace_index={(r['id'],r['condition']):r for r in traces}
 for filename,h in m['artifacts'].items():assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==h
 for r in pred:
  gold=rows[r['id']]
  assert r['gold']==gold['gold'] and r['group']==gold['group'] and r['state_sha256']==gold['state_sha256']
  assert r['depth']==gold['depth'] and r['cohort']==gold['cohort'] and r['family']==gold['family']
  trace=trace_index[(r['id'],r['condition'])]
  assert trace['trace_text']==tokenizer.decode(trace['trace_token_ids'])
  prompt,ids,slots=encode(tokenizer,gold,r['condition'],8192)
  if r['budget']:
   prompt=tokenizer.apply_chat_template(messages(gold),tokenize=False,add_generation_prompt=True,enable_thinking=True)+'<think>\n'
   ids=tokenizer.encode(prompt,add_special_tokens=False)
   tokens=trace['trace_token_ids']
   suffix=([151668] if r['forced_close'] else [])+tokenizer.encode('\n\n',add_special_tokens=False)
   final_ids=ids+tokens+suffix
   assert 151668 not in tokens[:-1]
   assert 151645 not in tokens[:-1] and 151667 not in tokens[:-1]
  else:
   final_ids=ids
   assert trace['trace_token_ids']==([r['unrestricted_token_id']] if r['condition']=='B' else [])
  assert r['prompt_sha256']==sha(prompt) and r['answer_prefix_sha256']==sha(str(final_ids))
  assert r['input_tokens']==len(ids)
  p=r['probabilities'];z=r['option_logits'];assert all(math.isfinite(x) for x in p+z)
  assert len(p)==3 and all(0<=x<=1 for x in p) and abs(sum(p)-1)<1e-5
  weights=[math.exp(x-max(z)) for x in z];computed=[x/sum(weights) for x in weights]
  assert max(abs(x-y) for x,y in zip(p,computed))<1e-5
  best=r['option_ids'][max(range(3),key=lambda i:p[i])]
  if r['condition']!='B':assert r['prediction']==best
  else:
   token=r['unrestricted_token_id'];valid=token in slots
   assert r['readout_valid']==valid
   assert r['prediction']==(r['option_ids'][slots.index(token)] if valid else None)
  assert r['correct']==(r['readout_valid'] and r['prediction']==r['gold'])
  assert r['total_seconds']>0 and 0<=r['candidate_mass']<=1.00001
  assert r['seed']==17 and r['model_revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
  if r['budget']:
   tokens=trace_index[(r['id'],r['condition'])]['trace_token_ids']
   assert len(tokens)==r['sampled_reasoning_tokens']<=r['budget']
   assert r['readout_method']=='full_prefix_recomputation' and r['reprocessed_tokens']>0
   if r['natural_close']:assert tokens[-1]==151668 and not r['forced_close']
   if r['budget_exhausted']:assert len(tokens)==r['budget'] and r['forced_close']
   if not r['readout_valid']:assert r['termination'] in ['early_eos','nested_think']
  if r['condition']=='D':
   assert r['oracle_facts']==len(gold['oracle'])
   q=tuple(gold['query']);n=(*q[:3],'-' if q[3]=='+' else '+')
   assert all(tuple(x['atom']) not in [q,n] for x in gold['oracle'])
  all_predictions[(name,r['id'],r['condition'])]=r
 checked[run_id]={'rows':len(pred),'valid':sum(r['readout_valid'] for r in pred),'condition_counts':dict(Counter(r['condition'] for r in pred))}
noops=0
for row in cohorts['pilot']:
 if row['gold']!='Unknown' and not row['oracle']:
  a=all_predictions[('pilot',row['id'],'A')];d=all_predictions[('pilot',row['id'],'D')]
  assert a['prompt_sha256']==d['prompt_sha256']
  assert max(abs(x-y) for x,y in zip(a['probabilities'],d['probabilities']))<1e-5
  noops+=1
selection=json.loads((root/'results/m1-validation-v1/selected_budget.json').read_text())
scores={c:sum(r['correct'] for (name,uid,condition),r in all_predictions.items() if name=='validation' and condition==c and r['depth'] in [3,5]) for c in ['C32','C128','C512']}
assert selection['selected']==max(scores,key=lambda c:(scores[c],-int(c[1:])))
report=json.loads((root/'results/m1-report-v1/report.json').read_text())
assert report['validation_selection']['selected']==selection['selected']
assert len(report['source_runs'])==3
for cohort in ['pilot','validation']:
 for condition in ['A-ref','A','B','C32','C128','C512','D']:
  records=[r for (n,uid,c),r in all_predictions.items() if n==cohort and c==condition]
  if not records:continue
  summary=report['summary'][f'{cohort}/{condition}/all']
  assert summary['n']==len(records)
  assert abs(summary['accuracy']-sum(r['correct'] for r in records)/len(records))<1e-12
result={'status':'passed','checked':checked,'oracle_noops_identical_to_A':noops,'selected_budget':selection['selected'],'cohort_group_isolation':True,'scope':'Exact frozen M0/M1 cohorts, raw records, probabilities/logits, traces/budgets, oracle roots/noops, artifacts and budget selection. Scientific success is separate.'}
path=root/'results/m1-report-v1/completion_audit.json'
with path.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
