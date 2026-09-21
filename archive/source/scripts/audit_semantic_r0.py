"""Independent R0 artifact and semantic audit; run on lp without inference."""
import hashlib,itertools,json,math,os
from pathlib import Path
from transformers import AutoTokenizer
from jev_reasoning.proofwriter import closure,label,canonical_hash
from jev_reasoning.semantic_microcases import make_cases,parse_semantic
from jev_reasoning.prompts import LABELS,encode
root=Path(os.environ['JEV_ROOT']);folder=root/'results/r0-semantic-v2'
m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete'
for n,h in m['artifacts'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h
rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()]
assert rows==json.loads(json.dumps(make_cases()))
old=set()
for name in ['pilot','validation','smoke']:
 old.update(json.loads(x)['group'] for x in (root/f'data/derived/owa-v2/{name}.jsonl').read_text().splitlines())
assert not old&{r['group'] for r in rows}
for r in rows:
 facts={k:tuple(v) for k,v in r['facts'].items()};rules={k:(tuple(tuple(p) for p in v[0]),tuple(v[1])) for k,v in r['rules'].items()}
 assert canonical_hash(facts,rules)==r['group']
 entities={a[0] for a in facts.values()}|{r['query'][0]};ground=[]
 for rid,(ps,h) in rules.items():
  for e in entities:ground.append((rid,tuple((e,*p[1:]) for p in ps),(e,*h[1:])))
 depths,_=closure(facts,ground);assert label(tuple(r['query']),depths)==r['gold']
t=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision=m['model']['revision'],local_files_only=True)
preds=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
expected={(r['id'],c,tuple(o)) for r in rows for c in ['original','json_cue','plain_cue'] for o in itertools.permutations(LABELS)}|{(r['id'],'plain_semantic',()) for r in rows}
assert len(preds)==len(expected)==1368
assert {(r['id'],r['condition'],tuple(r['order'] or [])) for r in preds}==expected
index={r['id']:r for r in rows}
for r in preds:
 source=index[r['id']];assert all(r[k]==source[k] for k in ['gold','group','nominal_depth','query_polarity'])
 assert r['correct']==(r['readout_valid'] and r['prediction']==r['gold'])
 if r['condition']=='original':
  prompt,ids,slots=encode(t,source,'A',8192,order=r['order'])
  assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
  assert r['answer_prefix_sha256']==hashlib.sha256(str(ids).encode()).hexdigest()
 else:
  assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
  assert t.encode(r['prompt'],add_special_tokens=False)==r['input_ids']
 if r['condition']=='plain_semantic':
  tokens=r['generated_ids'];assert 0<len(tokens)<=32
  eos=[151645,151643];assert not any(x in eos for x in tokens[:-1])
  assert r['eos_reached']==(tokens[-1] in eos)
  assert t.decode(tokens,skip_special_tokens=True)==r['natural_output']
  prediction=parse_semantic(r['natural_output']) if r['eos_reached'] else None
  assert r['prediction']==prediction and r['readout_valid']==(prediction is not None)
 else:
  z=r['option_logits'];ws=[math.exp(x-max(z)) for x in z];ps=[x/sum(ws) for x in ws]
  assert max(abs(a-b) for a,b in zip(ps,r['probabilities']))<1e-5
  assert r['prediction']==r['order'][max(range(3),key=lambda i:ps[i])]
summary=json.loads((folder/'summary.json').read_text())
for c,s in summary.items():
 rs=[r for r in preds if r['condition']==c];assert s['n']==len(rs)
 assert s['accuracy']==sum(r['correct'] for r in rs)/len(rs)
 assert s['invalid']==sum(not r['readout_valid'] for r in rs)
 passes=[]
 for d,g,p in itertools.product([0,1],LABELS,['+','-']):
  xs=[r for r in rs if r['nominal_depth']==d and r['gold']==g and r['query_polarity']==p];a=sum(r['correct'] for r in xs)/len(xs)
  assert s['cells'][f'{d}/{g}/{p}']=={'n':len(xs),'accuracy':a};passes.append(a>=.9)
 if c!='plain_semantic':
  agreement=sum(len({r['prediction'] for r in rs if r['id']==row['id']})==1 for row in rows)/72
  assert agreement==s['all_order_agreement'];passes.append(agreement>=.9)
  for o in itertools.permutations(LABELS):
   xs=[r for r in rs if r['order']==list(o)];a=sum(r['correct'] for r in xs)/len(xs)
   assert a==s['order_accuracy'][str(o)];passes.append(a>=.9)
 else:passes.append(s['invalid']==0)
 assert all(passes)==s['gate_pass']
result={'status':'passed','cases':72,'records':1368,'all_gold_labels_independently_grounded':True,'old_cohorts_disjoint':True,'hashes_and_metrics_reconciled':True,'scope':'Artifact integrity and procedural correctness, not semantic sanity-gate success.'}
with (folder/'completion_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
