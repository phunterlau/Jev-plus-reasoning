"""Reconcile completed suites and report paired effects on lp."""
import argparse,csv,hashlib,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np
from jev_reasoning.metrics import metrics

p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(os.environ['JEV_ROOT']);out=root/'results'/a.out;out.mkdir(parents=True,exist_ok=False)
all_records=[];manifests=[]
for name in a.runs:
 run=root/'results'/name;m=json.loads((run/'manifest.json').read_text())
 if m['status']!='complete':raise RuntimeError(f'{name} is incomplete')
 for filename,h in m['artifacts'].items():
  if hashlib.sha256((run/filename).read_bytes()).hexdigest()!=h:raise ValueError('Artifact hash mismatch')
 records=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
 assert len(records)==m['expected_rows']==m['observed_rows']
 data=[json.loads(x) for x in (root/'data/derived/owa-v2'/f'{m["cohort"]}.jsonl').read_text().splitlines()]
 assert hashlib.sha256((root/'data/derived/owa-v2'/f'{m["cohort"]}.jsonl').read_bytes()).hexdigest()==m['cohort_sha256']
 expected={(r['id'],c) for r in data for c in m['conditions'] if c!='D' or r['gold']!='Unknown'}
 assert {(r['id'],r['condition']) for r in records}==expected
 for r in records:
  assert np.isfinite(r['probabilities']).all() and abs(sum(r['probabilities'])-1)<1e-5
  assert r['model_revision']==m['model']['revision']
  r['cohort_name']=m['cohort']
 all_records+=records;manifests.append(m)
# Same scorer, prompt, checkpoint, precision and GPU across compared runs.
for m in manifests[1:]:
 for filename in ['src/jev_reasoning/inference.py','src/jev_reasoning/prompts.py']:
  assert m['source_files'][filename]==manifests[0]['source_files'][filename]
 for key in ['revision','dtype','gpu','torch','transformers']:
  assert m['model'][key]==manifests[0]['model'][key]
by=defaultdict(list)
for r in all_records:by[(r['cohort_name'],r['condition'])].append(r)
summary={};tables=[]
for condition,filename in [('A-ref','proofwriter_direct_qwen3_06b.jsonl'),('A','proofwriter_matched_direct_qwen3_06b.jsonl')]:
 if ('pilot',condition) in by:
  (out/filename).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in by[('pilot',condition)]))
for (cohort,c),rs in sorted(by.items()):
 slices={'all':rs,'deep':[r for r in rs if r['depth'] in [3,5]],'shallow':[r for r in rs if r['depth'] in [0,1]],'unknown':[r for r in rs if r['gold']=='Unknown']}
 for d in [0,1,2,3,5]:slices[f'depth_{d}']=[r for r in rs if r['depth']==d]
 for label in ['True','False','Unknown']:slices[f'class_{label}']=[r for r in rs if r['gold']==label]
 for family in sorted({r['family'] for r in rs}):slices[f'family_{family}']=[r for r in rs if r['family']==family]
 for sl,items in slices.items():
  met=metrics(items);met['prediction_counts']={label:sum(r['prediction']==label for r in items) for label in ['True','False','Unknown',None]};summary[f'{cohort}/{c}/{sl}']=met
  tables.append(dict(cohort=cohort,condition=c,slice=sl,**met))

rng=np.random.default_rng(17)
def paired(left,right,predicate):
 l={r['id']:r for r in left if predicate(r)};r={x['id']:x for x in right if predicate(x)}
 ids=sorted(l.keys()&r.keys())
 if not ids:return {'n':0}
 groups=defaultdict(list)
 for uid in ids:
  assert l[uid]['group']==r[uid]['group'] and l[uid]['gold']==r[uid]['gold']
  groups[l[uid]['group']].append(int(l[uid]['correct'])-int(r[uid]['correct']))
 vals=list(groups.values());sums=np.array([sum(x) for x in vals]);counts=np.array([len(x) for x in vals]);ix=rng.integers(0,len(vals),(2000,len(vals)))
 boot=sums[ix].sum(1)/counts[ix].sum(1)
 return {'n':len(ids),'theory_groups':len(groups),'difference':float(sums.sum()/counts.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_replicates':2000},boot
comparisons={}
for cohort in ['pilot','validation']:
 base=by.get((cohort,'A'),[])
 if not base:continue
 for c in ['A-ref','B','C32','C128','C512','D']:
  rs=by.get((cohort,c),[])
  if not rs:continue
  boots={}
  for sl,fn in [('all',lambda r:True),('deep',lambda r:r['depth'] in [3,5]),('shallow',lambda r:r['depth'] in [0,1]),('unknown',lambda r:r['gold']=='Unknown')]:
   value=paired(rs,base,fn)
   if isinstance(value,tuple):comparisons[f'{cohort}/{c}-A/{sl}'],boots[sl]=value
  if 'deep' in boots and 'shallow' in boots:
   interaction=boots['deep']-boots['shallow']
   comparisons[f'{cohort}/{c}-A/deep_minus_shallow']={'difference':comparisons[f'{cohort}/{c}-A/deep']['difference']-comparisons[f'{cohort}/{c}-A/shallow']['difference'],'ci95':np.quantile(interaction,[.025,.975]).tolist(),'bootstrap_replicates':2000}
 # Oracle coverage is explicit; compare only rows with actual added intermediate facts.
 oracle=[r for r in by.get((cohort,'D'),[]) if r['oracle_facts']>0]
 if oracle:
  eligible={r['id'] for r in oracle}
  for c in ['A','C32','C128','C512']:
   rs=by.get((cohort,c),[])
   if not rs:continue
   value=paired(oracle,rs,lambda r:r['id'] in eligible)
   if isinstance(value,tuple):comparisons[f'{cohort}/D-{c}/oracle_nonempty']=value[0]

selection=None
for m in manifests:
 if m['suite']=='validation':selection=json.loads((root/'results'/m['run_id']/'selected_budget.json').read_text())
report={'source_runs':a.runs,'summary':summary,'paired_comparisons':comparisons,'validation_selection':selection,'limitations':['Single decoding seed pilot; not confirmation.','Required proof depth excludes Unknown.','Theory bootstrap does not measure decoding-seed reliability.','Model is an open Qwen proxy, not proprietary Jev.','All reasoned final readouts use full-prefix recomputation.'],'completion_scope':'only the listed complete suites; M1 requires baseline, validation and m1 suites'}
(out/'report.json').write_text(json.dumps(report,indent=2))
keys=sorted(set().union(*(r.keys() for r in tables)))
with (out/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(tables)
lines=['# Evaluation report','',f'Source runs: {", ".join(a.runs)}. All runs executed on lp.','', '| Cohort | Condition | N | Accuracy | Deep accuracy | NLL | Brier | ECE | p50 ms | p95 ms |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for (cohort,c),rs in sorted(by.items()):
 m=summary[f'{cohort}/{c}/all'];deep=summary[f'{cohort}/{c}/deep'].get('accuracy',float('nan'))
 lines.append(f'| {cohort} | {c} | {m["n"]} | {m["accuracy"]:.3f} | {deep:.3f} | {m["nll"]:.3f} | {m["brier_sum"]:.3f} | {m["ece_10_equal_bins"]:.3f} | {1000*m["latency_p50_seconds"]:.1f} | {1000*m["latency_p95_seconds"]:.1f} |')
lines+=['','## Paired deep effects','', '| Comparison | N | Gain, pp | 95% interval, pp |','|---|---:|---:|---|']
for key,m in comparisons.items():
 if key.endswith('/deep'):lines.append(f'| {key} | {m["n"]} | {100*m["difference"]:.2f} | [{100*m["ci95"][0]:.2f}, {100*m["ci95"][1]:.2f}] |')
lines+=['','These are single-seed pilot measurements, not a completed confirmation or a finding about Jev internals.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig,ax=plt.subplots(figsize=(7,4))
for c in ['A','C32','C128','C512','D']:
 if ('pilot',c) not in by:continue
 depths=[0,1,2,3,5];ys=[summary[f'pilot/{c}/depth_{d}'].get('accuracy',float('nan')) for d in depths]
 ax.plot(depths,ys,marker='o',label=c)
ax.set(xlabel='Minimum proof depth (provable questions only)',ylabel='Accuracy',ylim=(0,1),xticks=[0,1,2,3,5],title='Qwen3-0.6B: frozen ProofWriter pilot');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'accuracy_by_depth.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for (cohort,c),rs in sorted(by.items()):
 if cohort!='pilot' or c=='D':continue
 m=summary[f'{cohort}/{c}/all'];x=m['latency_p50_seconds']*1000;y=m['accuracy'];ax.scatter(x,y);ax.annotate(c,(x,y),xytext=(4,4),textcoords='offset points')
ax.set(xlabel='Median warm request latency, ms (log scale)',ylabel='Accuracy',xscale='log',ylim=(0,1),title='Accuracy versus latency on the same 400 questions');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'accuracy_latency.png');plt.close(fig)
print(json.dumps({'output':str(out),'source_runs':a.runs,'rows':len(all_records),'selection':selection['selected'] if selection else None}),flush=True)
