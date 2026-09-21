"""Frozen semantic-interface diagnostic. Execute only on lp, create-only outputs."""
import hashlib,itertools,json,os,time,random
from collections import Counter
from pathlib import Path
import torch
from jev_reasoning.inference import Scorer
from jev_reasoning.prompts import LABELS,MATCHED_SYSTEM,DESCRIPTIONS,messages,sha
from jev_reasoning.semantic_microcases import make_cases,parse_semantic
root=Path(os.environ['JEV_ROOT']);out=root/'results/r0-semantic-v2';out.mkdir(exist_ok=False)
cfg=json.loads(Path('experiments/semantic_interface/r0_v2.json').read_text())
rows=make_cases();old=set()
for name in ['pilot','validation','smoke']:
 old.update(json.loads(x)['group'] for x in (root/f'data/derived/owa-v2/{name}.jsonl').read_text().splitlines())
assert not old & {r['group'] for r in rows}
(out/'cases.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
source={str(p):sha(p.read_text()) for folder in ['src','scripts','experiments'] for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json'] and '__pycache__' not in p.parts}
sc=Scorer();run={'status':'running','start_unix':time.time(),'config':cfg,'model':sc.metadata,'source_files':source,'job':os.environ['JEV_JOB'],'cohort_isolation':True,'expected_records':1368}
(out/'manifest.json').write_text(json.dumps(run,indent=2))
for r in rows[:3]:sc.score(r,'A')

def plain(row,order=None):
 criterion='Use only the supplied facts and rules, in their stated direction. An unstated fact is not automatically false. Classify the statement.'
 content=row['state']+'\n\n'+criterion+'\nStatement: '+row['question']
 if order is not None:
  content+='\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {label}: {DESCRIPTIONS[label]}' for i,label in enumerate(order))
  system=MATCHED_SYSTEM
 else:
  content+='\nDefinitions:\n'+'\n'.join(f'{label}: {DESCRIPTIONS[label]}' for label in LABELS)
  system='Classify the supplied statement using only the supplied evidence. Respond with exactly True, False, or Unknown. Do not explain or reason.'
 return [{'role':'system','content':system},{'role':'user','content':content}]

@torch.inference_mode()
def evaluate(row,condition,order):
 if condition=='original':
  r=sc.score(row,'A',order=order);r['condition']=condition;r['order']=order
  r['natural_output']=None;return r
 ms=messages(row,order=order) if condition=='json_cue' else plain(row,order if condition=='plain_cue' else None)
 prompt=sc.tokenizer.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 if condition!='plain_semantic':prompt+='Answer:\n'
 ids=sc.tokenizer.encode(prompt,add_special_tokens=False);assert len(ids)+32<8192
 sc.sync();start=time.perf_counter();output=sc.forward(ids)
 base={'id':row['id'],'condition':condition,'order':order,'gold':row['gold'],'prompt':prompt,'prompt_sha256':sha(prompt),'input_ids':ids,'input_tokens':len(ids),'nominal_depth':row['nominal_depth'],'query_polarity':row['query_polarity']}
 if condition=='plain_semantic':
  tokens=[];ended=False;eos=set(sc.model.generation_config.eos_token_id)
  for step in range(32):
   token=int(output.logits[0,-1].argmax());tokens.append(token)
   if token in eos:ended=True;break
   if step<31:output=sc.forward([token],output.past_key_values)
  text=sc.tokenizer.decode(tokens,skip_special_tokens=True);pred=parse_semantic(text) if ended else None
  base.update(prediction=pred,readout_valid=pred is not None,natural_output=text,generated_ids=tokens,eos_reached=ended,probabilities=None,option_ids=None,candidate_mass=None)
 else:
  slots=[sc.tokenizer.encode(chr(65+i),add_special_tokens=False)[0] for i in range(3)]
  for i,t in enumerate(slots):assert sc.tokenizer.encode(prompt+chr(65+i),add_special_tokens=False)==ids+[t]
  logits=output.logits[0,-1].float();assert torch.isfinite(logits).all()
  ps=torch.softmax(logits[slots],0);pred=order[int(ps.argmax())]
  base.update(prediction=pred,readout_valid=True,probabilities=ps.tolist(),option_ids=order,option_logits=logits[slots].tolist(),candidate_mass=float(torch.exp(torch.logsumexp(logits[slots],0)-torch.logsumexp(logits,0))),natural_output=None,unrestricted_token_id=int(logits.argmax()))
 sc.sync();base.update(correct=base['readout_valid'] and base['prediction']==row['gold'],total_seconds=time.perf_counter()-start)
 return base

jobs=[(r,c,list(order)) for r in rows for c in cfg['typed_conditions'] for order in itertools.permutations(LABELS)]+[(r,'plain_semantic',None) for r in rows]
random.Random(17).shuffle(jobs);records=[]
with (out/'predictions.jsonl').open('x') as f:
 for row,c,order in jobs:
  r=evaluate(row,c,order);r.update(nominal_depth=row['nominal_depth'],query_polarity=row['query_polarity'],group=row['group'])
  f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();records.append(r)
  if len(records)%200==0:print(json.dumps({'done':len(records),'expected':len(jobs)}),flush=True)
assert len(records)==1368 and len({(r['id'],r['condition'],str(r['order'])) for r in records})==1368
summary={}
for c in cfg['typed_conditions']+['plain_semantic']:
 rs=[r for r in records if r['condition']==c]
 cells={f'{d}/{g}/{p}':[r for r in rs if r['nominal_depth']==d and r['gold']==g and r['query_polarity']==p] for d,g,p in itertools.product([0,1],LABELS,['+','-'])}
 acc=lambda xs:sum(r['correct'] for r in xs)/len(xs)
 order_acc={str(order):acc([r for r in rs if r['order']==list(order)]) for order in itertools.permutations(LABELS)} if c!='plain_semantic' else {}
 agreement=sum(len({r['prediction'] for r in rs if r['id']==row['id']})==1 for row in rows)/72 if order_acc else None
 cellstats={k:{'n':len(v),'accuracy':acc(v)} for k,v in cells.items()}
 gate=all(v['accuracy']>=.9 for v in cellstats.values()) and (all(x>=.9 for x in order_acc.values()) and agreement>=.9 if order_acc else all(r['readout_valid'] for r in rs))
 summary[c]={'n':len(rs),'accuracy':acc(rs),'invalid':sum(not r['readout_valid'] for r in rs),'predictions':dict(Counter(str(r['prediction']) for r in rs)),'cells':cellstats,'order_accuracy':order_acc,'all_order_agreement':agreement,'gate_pass':gate,'mean_candidate_mass':sum(r['candidate_mass'] for r in rs)/len(rs) if order_acc else None}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# R0 semantic-interface diagnostic','', '| Condition | N | Accuracy | Order agreement | Sanity gate |','|---|---:|---:|---:|---|']
for c,s in summary.items():lines.append(f'| {c} | {s["n"]} | {s["accuracy"]:.4f} | {s["all_order_agreement"]} | {s["gate_pass"]} |')
lines+=['','72 unique microcases; option permutations are repeated measurements, not independent questions.','Natural semantic generation has no option-order gate and cannot alone qualify typed scoring.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
run.update(status='complete',finished_unix=time.time(),observed_records=len(records),artifacts={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['cases.jsonl','predictions.jsonl','summary.json','report.md']})
tmp=out/'manifest.tmp';tmp.write_text(json.dumps(run,indent=2));tmp.replace(out/'manifest.json')
print(json.dumps({'status':'complete','summary':{c:{k:v for k,v in s.items() if k in ['accuracy','gate_pass','all_order_agreement','invalid']} for c,s in summary.items()}}),flush=True)
