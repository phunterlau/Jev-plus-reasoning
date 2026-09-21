"""GPU acceptance checks; all measurements performed on lp."""
import json,os,time
from pathlib import Path
import torch
from jev_reasoning.inference import Scorer
from jev_reasoning.prompts import encode,LABELS
root=Path(os.environ['JEV_ROOT']);out=Path(os.environ['JEV_JOB'])
rows=[json.loads(x) for x in (root/'data/derived/owa-v2/smoke.jsonl').read_text().splitlines()]
s=Scorer();records=[];checks={}
for row in rows:
 for c in ['A','A-ref','D']:
  encode(s.tokenizer,row,c,8192)
# Warmup excludes load and first-kernel effects from acceptance timing.
for _ in range(3):s.score(rows[0],'A')
a=s.score(rows[0],'A');c0=s.score(rows[0],'C0')
assert a['prompt_sha256']==c0['prompt_sha256'] and a['probabilities']==c0['probabilities']
checks['A_C0_identical']=True
pilot=[json.loads(x) for x in (root/'data/derived/owa-v2/pilot.jsonl').read_text().splitlines()]
# Select longest pilot input by length only, without inspecting model outcomes.
for row in [rows[0],max(pilot,key=lambda r:len(r['state']))]:
 for condition in ['A-ref','A','B','D','C32','C128','C512']:
  r=s.score(row,condition,verify_cache=condition.startswith('C'))
  assert sum(r['probabilities'])>0.999 and sum(r['probabilities'])<1.001
  records.append(r)
  print(json.dumps({k:r[k] for k in ['id','condition','prediction','correct','sampled_reasoning_tokens','termination','total_seconds','cache_probability_max_difference','peak_allocated_bytes']}),flush=True)
# Exercise natural close, exhaustion, early EOS and malformed thinking through real cached forwards.
original=s.sample
for name,token,condition,expected in [('natural',s.close,'C32','natural_close'),('cap',s.tokenizer.encode('x',add_special_tokens=False)[0],'C1','budget_exhausted'),('eos',s.eos,'C32','early_eos'),('nested',s.open,'C32','nested_think')]:
 s.sample=lambda logits,generator,t=token:t
 r=s.score(rows[0],condition,verify_cache=True)
 assert r['termination']==expected
 assert r['readout_valid']==(name in ['natural','cap'])
 checks[name]={'termination':r['termination'],'cache_max_difference':r['cache_probability_max_difference']}
s.sample=original
r16=s.score(rows[0],'A');s.model.float();r32=s.score(rows[0],'A')
fp32_cache=s.score(max(pilot,key=lambda r:len(r['state'])),'C128',verify_cache=True)
assert fp32_cache['cache_probability_max_difference']<0.0001
checks['fp32_cache_parity']={'difference':fp32_cache['cache_probability_max_difference'],'threshold':0.0001}
s.model.half()
checks['fp16_vs_fp32']={'max_probability_difference':max(abs(x-y) for x,y in zip(r16['probabilities'],r32['probabilities'])),'same_argmax':r16['prediction']==r32['prediction']}
checks['max_tokens_overflow_rejected']=False
try:encode(s.tokenizer,rows[0],'A',1)
except ValueError:checks['max_tokens_overflow_rejected']=True
assert checks['max_tokens_overflow_rejected']
for letter,slot in zip('ABC',[32,33,34]):
 assert s.tokenizer.encode('</think>\n\n'+letter,add_special_tokens=False)==[s.close]+s.separator+[slot]
checks['post_thinking_slot_boundary']=True
(out/'smoke.json').write_text(json.dumps({'metadata':s.metadata,'checks':checks,'records':records},indent=2))
print('SMOKE_PASS',json.dumps(checks),flush=True)
