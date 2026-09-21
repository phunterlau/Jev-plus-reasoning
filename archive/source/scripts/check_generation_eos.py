"""Audit all model-declared EOS tokens against completed reasoning traces on lp."""
import json,os
from pathlib import Path
from transformers import GenerationConfig
root=Path(os.environ['JEV_ROOT'])
cfg=GenerationConfig.from_pretrained('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_files_only=True)
eos=cfg.eos_token_id if isinstance(cfg.eos_token_id,list) else [cfg.eos_token_id]
counts={str(t):0 for t in eos};affected=[];total=0
for name in ['m1-validation-v1','m1-pilot-v1']:
 folder=root/'results'/name
 assert json.loads((folder/'manifest.json').read_text())['status']=='complete'
 preds={(r['id'],r['condition']):r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())}
 for trace in map(json.loads,(folder/'traces.jsonl').read_text().splitlines()):
  key=trace['id'],trace['condition'];pred=preds[key]
  if not pred['budget']:continue
  total+=1;tokens=trace['trace_token_ids']
  for token in eos:counts[str(token)]+=tokens.count(token)
  encountered=[t for t in tokens if t in eos]
  if encountered:
   ok=encountered==[151645] and tokens[-1]==151645 and not pred['readout_valid'] and pred['termination']=='early_eos'
   if not ok:affected.append({'run':name,'id':trace['id'],'condition':trace['condition'],'tokens':encountered,'recorded_termination':pred['termination']})
result={'status':'passed' if not affected else 'failed','declared_eos_token_ids':eos,'reasoned_rows':total,'observed_eos_counts':counts,'incorrectly_handled_rows':affected,'scope':'Both pinned-model EOS tokens checked; the scorer implements tokenizer EOS 151645 explicitly. No unhandled EOS may occur in accepted completed results.'}
out=root/'results/m1-report-v1/generation_eos_audit.json'
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
if affected:raise RuntimeError('Unhandled declared EOS occurred; do not mark M1 complete.')
