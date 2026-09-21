"""Reconcile frozen R1 outcomes without new inference; run on lp."""
import hashlib,itertools,json,math,os,re,statistics
from pathlib import Path
from transformers import AutoTokenizer
from jev_reasoning.proofwriter import closure,label
root=Path(os.environ['JEV_ROOT']);p=root/'results/r1-mode-capacity-v1'
m=json.loads((p/'manifest.json').read_text());assert m['status']=='complete'
for n,h in m['artifacts'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
rows=[json.loads(x) for x in (p/'cases.jsonl').read_text().splitlines()];index={r['id']:r for r in rows};assert len(index)==36
for r in rows:
 facts={k:tuple(v) for k,v in r['facts'].items()};ground=[]
 for rid,(ps,h) in r['rules'].items():
  for e in {a[0] for a in facts.values()}:ground.append((rid,tuple((e,*a[1:]) for a in ps),(e,*h[1:])))
 ds,_=closure(facts,ground);assert label(tuple(r['query']),ds)==r['gold']
rs=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()];s=json.loads((p/'summary.json').read_text())
sizes=list(m['models']);expected={(r['id'],size,mode) for r in rows for size in sizes for mode in ['nonthinking','thinking']}
assert len(rs)==len(expected)==m['records']
assert {(r['id'],r['size'],r['mode']) for r in rs}==expected
assert ('large' in sizes)==(not all(s['small/'+mode]['gate_pass'] for mode in ['nonthinking','thinking']))
for size in sizes:
 cfg=m['models'][size];tok=AutoTokenizer.from_pretrained(cfg['repo'],revision=cfg['revision'],local_files_only=True)
 for r in [r for r in rs if r['size']==size]:
  assert all(r[k]==index[r['id']][k] for k in ['gold','group','nominal_depth','query_polarity'])
  assert tok.encode(r['prompt'],add_special_tokens=False)==r['prompt_ids']
  assert tok.decode(r['trace_ids'])==r['trace_text']
  assert len(r['trace_ids'])<=2048 and len(r['generated_answer_ids'])<=32
  if r['mode']=='thinking':assert r['closed']==(bool(r['trace_ids']) and r['trace_ids'][-1]==151668)
  else:assert not r['trace_ids'] and r['closed']
  if not r['closed']:assert r['native'] is None and r['cue'] is None and not r['correct'] and r['prediction'] is None
  assert tok.decode(r['generated_answer_ids'],skip_special_tokens=True)==r['generated_answer']
  match=re.fullmatch(r'\s*([ABC])[.!]?\s*',r['generated_answer'])
  tokens=r['generated_answer_ids']
  parsed=['True','False','Unknown'][ord(match.group(1))-65] if match and tokens and tokens[-1] in [151645,151643] else None
  assert r['prediction']==parsed
  assert r['correct']==(r['prediction']==r['gold'])
  for part in ['native','cue']:
   a=r[part]
   if a is None:continue
   z=a['option_logits'];w=[math.exp(x-max(z)) for x in z];ps=[x/sum(w) for x in w]
   assert max(abs(x-y) for x,y in zip(ps,a['probabilities']))<1e-5
   assert a['prediction']==['True','False','Unknown'][max(range(3),key=lambda i:ps[i])]
summary_lines=['# R1 reference and mode/capacity results','', '| Model | Mode | N | Generated accuracy | Native typed | Cue typed | Closure | Gate |','|---|---|---:|---:|---:|---:|---:|---|']
for key,a in s.items():
 size,mode=key.split('/');group=[r for r in rs if r['size']==size and r['mode']==mode]
 assert a['n']==len(group)==36
 assert a['generated_accuracy']==sum(r['correct'] for r in group)/36
 assert a['closure_fraction']==sum(r['closed'] for r in group)/36
 for part in ['native','cue']:assert a[part+'_accuracy']==sum(r[part] is not None and r[part]['prediction']==r['gold'] for r in group)/36
 gate=a['closure_fraction']>=.9
 for d,g,pol in itertools.product([0,1],['True','False','Unknown'],['+','-']):
  items=[r for r in group if (r['nominal_depth'],r['gold'],r['query_polarity'])==(d,g,pol)];cell=a['cells'][f'{d}/{g}/{pol}'];assert len(items)==cell['n']==3
  assert cell['generated_accuracy']==sum(r['correct'] for r in items)/3
  assert cell['cue_accuracy']==sum(r['cue'] is not None and r['cue']['prediction']==g for r in items)/3
  gate=gate and cell['generated_accuracy']>=.9 and cell['cue_accuracy']>=.9
 assert gate==a['gate_pass']
 summary_lines.append(f'| {size} | {mode} | 36 | {a["generated_accuracy"]:.4f} | {a["native_accuracy"]:.4f} | {a["cue_accuracy"]:.4f} | {a["closure_fraction"]:.4f} | {a["gate_pass"]} |')
parity=json.loads((p/'reference_parity.json').read_text());assert len(parity)==3 and all(r['token_ids_identical'] and r['max_logit_difference']<=.001 for r in parity)
with (p/'completion_audit.json').open('x') as f:json.dump({'status':'passed','records':len(rs),'cases':36,'large_trigger_verified':True,'scope':'Artifact and procedure integrity; semantic gate success is separate.'},f,indent=2)
with (p/'report.md').open('x') as f:f.write('\n'.join(summary_lines)+'\n')
print('R1_AUDIT_PASSED',len(rs),flush=True)

# Descriptive post-hoc diagnostics do not change the frozen strict answer parser.
from jev_reasoning.prompts import LABELS,DESCRIPTIONS
extra={}
for size in sizes:
 for mode in ['nonthinking','thinking']:
  xs=[r for r in rs if r['size']==size and r['mode']==mode]
  copies={f'{chr(65+i)}. {g}: {DESCRIPTIONS[g]}':g for i,g in enumerate(LABELS)}
  extra[size+'/'+mode]={'strict_invalid':sum(r['prediction'] is None for r in xs),'exact_option_copy_count':sum(r['generated_answer'].strip() in copies for r in xs),'exact_option_copy_correct_count':sum(copies.get(r['generated_answer'].strip())==r['gold'] for r in xs),'median_seconds':statistics.median(r['total_seconds'] for r in xs),'mean_thinking_tokens':statistics.mean(len(r['trace_ids']) for r in xs),'max_thinking_tokens':max(len(r['trace_ids']) for r in xs),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in xs)}
 for part in ['native','cue']:
  off={r['id']:r for r in rs if r['size']==size and r['mode']=='nonthinking'}
  on={r['id']:r for r in rs if r['size']==size and r['mode']=='thinking'}
  correct=lambda r:r[part] is not None and r[part]['prediction']==r['gold']
  extra[size+'/'+part+'_paired']={'repairs':sum(correct(on[k]) and not correct(off[k]) for k in on),'regressions':sum(not correct(on[k]) and correct(off[k]) for k in on),'n':len(on)}
with (p/'diagnostics.json').open('x') as f:json.dump({'posthoc':True,'description':'Exact option-copy counts are descriptive only, not a replacement parser or revised gate.','summary':extra},f,indent=2)
