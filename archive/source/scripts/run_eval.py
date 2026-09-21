"""Create-only M0/M1 evaluations, with frozen cohorts and bounded work."""
import argparse,hashlib,json,os,time
from pathlib import Path
from collections import Counter
from jev_reasoning.inference import Scorer
from jev_reasoning.prompts import row_seed
from jev_reasoning.metrics import metrics

p=argparse.ArgumentParser();p.add_argument('--suite',choices=['baseline','validation','m1'],required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
root=Path(os.environ['JEV_ROOT']); data=root/'data/derived/owa-v2'
manifest=json.loads((data/'manifest.json').read_text());cohort='validation' if a.suite=='validation' else 'pilot'
f=data/f'{cohort}.jsonl';digest=hashlib.sha256(f.read_bytes()).hexdigest()
assert digest==manifest['cohorts'][cohort]['sha256']
rows=[json.loads(line) for line in f.read_text().splitlines()]
assert len(rows)==manifest['cohorts'][cohort]['rows'] and len({r['group'] for r in rows})==len(rows)
conditions={'baseline':['A-ref','A','B'],'validation':['A','C32','C128','C512'],'m1':['C32','C128','C512','D']}[a.suite]
expected={(r['id'],c) for r in rows for c in conditions if c!='D' or r['gold']!='Unknown'}
out=root/'results'/a.run_id;out.mkdir(parents=True,exist_ok=False)
scorer=Scorer()
source={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','scripts','experiments'] for f in sorted(Path(folder).rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
run={'run_id':a.run_id,'suite':a.suite,'cohort':cohort,'cohort_sha256':digest,'conditions':conditions,'expected_rows':len(expected),'seed':17,'model':scorer.metadata,'reference_commits':{'SemIf':'ca3ba65f142967030ecb453346e94d6f476a69df','NanoJev':'71a513bb0163b5634467842b523ee0c0ed6fb1c7'},'source_files':source,'source_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),'job_dir':os.environ['JEV_JOB'],'start_unix':time.time(),'max_tokens':8192,'sampling':{'temperature':.6,'top_p':.95,'top_k':20,'min_p':0},'timing':'CUDA synchronized warm batch-1 request wall time; load and parity checks excluded','condition_order':'row-seeded rotation, no shared cross-condition KV cache','status':'running'}
(out/'manifest.json').write_text(json.dumps(run,indent=2))
# Warmup on training-only fixtures, never counted as benchmark samples.
smoke=[json.loads(x) for x in (data/'smoke.jsonl').read_text().splitlines()]
for i in range(3):scorer.score(smoke[i],'A')
all_records=[]
with (out/'predictions.jsonl').open('x') as pred,(out/'traces.jsonl').open('x') as traces:
 for row in rows:
  rotation=row_seed(row['id'],17)%len(conditions);ordered=conditions[rotation:]+conditions[:rotation]
  for condition in ordered:
   if condition=='D' and row['gold']=='Unknown':continue
   r=scorer.score(row,condition)
   trace={k:r.pop(k) for k in ['trace_token_ids','trace_text']}
   trace.update(id=r['id'],condition=condition,seed=17)
   traces.write(json.dumps(trace)+'\n');traces.flush()
   r.update(run_id=a.run_id,model_revision=scorer.metadata['revision'],cohort_sha256=digest,source_sha256=run['source_sha256'])
   pred.write(json.dumps(r,allow_nan=False)+'\n');pred.flush();all_records.append(r)
   if len(all_records)%20==0:print(json.dumps({'done':len(all_records),'expected':len(expected),'last_id':row['id'],'last_condition':condition,'elapsed_seconds':time.time()-run['start_unix']}),flush=True)
observed={(r['id'],r['condition']) for r in all_records}
assert observed==expected and len(observed)==len(all_records)
summary={c:metrics([r for r in all_records if r['condition']==c]) for c in conditions}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
if a.suite=='validation':
 scored={c:metrics([r for r in all_records if r['condition']==c and r['depth'] in [3,5]]) for c in conditions if c!='A'}
 winner=max(scored,key=lambda c:(scored[c]['accuracy'],-int(c[1:])))
 (out/'selected_budget.json').write_text(json.dumps({'selected':winner,'rule':'highest validation provable depth-3/5 accuracy; lower budget on exact tie','metrics':scored,'cohort_sha256':digest},indent=2))
run.update(status='complete',finished_unix=time.time(),observed_rows=len(all_records),artifacts={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['predictions.jsonl','traces.jsonl','summary.json']})
tmp=out/'manifest.tmp';tmp.write_text(json.dumps(run,indent=2));tmp.replace(out/'manifest.json')
print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)
