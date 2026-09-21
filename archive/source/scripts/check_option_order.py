"""Fixed validation-only option-order diagnostic; no prompt optimization."""
import hashlib,itertools,json,os
from collections import Counter
from pathlib import Path
from jev_reasoning.inference import Scorer
from jev_reasoning.prompts import LABELS,row_seed
root=Path(os.environ['JEV_ROOT']);out=root/'results/m0-option-order-v1';out.mkdir(parents=True,exist_ok=False)
rows=[json.loads(x) for x in (root/'data/derived/owa-v2/validation.jsonl').read_text().splitlines()]
rows=sorted(rows,key=lambda r:row_seed(r['id'],17))[:60]
s=Scorer();s.score(rows[0],'A');all_records=[]
with (out/'predictions.jsonl').open('x') as f:
 for row in rows:
  for order in itertools.permutations(LABELS):
   r=s.score(row,'A',order=list(order));r.pop('trace_token_ids');r.pop('trace_text')
   f.write(json.dumps(r)+'\n');all_records.append(r)
by={r['id']:[x for x in all_records if x['id']==r['id']] for r in rows}
summary={'rows':len(rows),'evaluations':len(all_records),'semantic_prediction_counts':dict(Counter(r['prediction'] for r in all_records)),'position_prediction_counts':dict(Counter(r['option_ids'].index(r['prediction']) if r['prediction'] else -1 for r in all_records)),'all_six_semantic_agreement':sum(len({r['prediction'] for r in rs})==1 for rs in by.values())/len(by),'mean_semantic_total_variation_vs_canonical':sum(.5*sum(abs(r['probabilities'][r['option_ids'].index(l)]-rs[0]['probabilities'][rs[0]['option_ids'].index(l)]) for l in LABELS) for rs in by.values() for r in rs[1:])/(len(by)*5),'accuracy':sum(r['correct'] for r in all_records)/len(all_records),'ids':[r['id'] for r in rows],'model':s.metadata,'purpose':'validation-only diagnostic, no prompt or budget selection based on this output'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
