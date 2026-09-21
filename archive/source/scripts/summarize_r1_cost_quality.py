"""Additional 0.6B R1 cost/probability metrics from immutable outputs, on lp."""
import hashlib,json,os
from pathlib import Path
from jev_reasoning.metrics import metrics
p=Path(os.environ['JEV_ROOT'])/'results/r1-mode-capacity-v1'
m=json.loads((p/'manifest.json').read_text());assert m['status']=='complete'
h=hashlib.sha256((p/'predictions.jsonl').read_bytes()).hexdigest();assert h==m['artifacts']['predictions.jsonl']
rs=list(map(json.loads,(p/'predictions.jsonl').read_text().splitlines()));out={}
for mode in ['nonthinking','thinking']:
 xs=[r for r in rs if r['size']=='small' and r['mode']==mode]
 assert len(xs)==36 and all(r['cue'] is not None for r in xs)
 adapted=[dict(gold=r['gold'],prediction=r['cue']['prediction'],correct=r['cue']['prediction']==r['gold'],readout_valid=True,probabilities=r['cue']['probabilities'],option_ids=['True','False','Unknown'],total_seconds=r['total_seconds'],sampled_reasoning_tokens=len(r['trace_ids']),natural_close=r['mode']=='thinking' and r['closed'],forced_close=False,peak_allocated_bytes=r['peak_allocated_bytes']) for r in xs]
 out[mode]=metrics(adapted)
result={'source_predictions_sha256':h,'job':os.environ['JEV_JOB'],'scope':'0.6B answer-cued typed probabilities on all 36 matched cases per mode; diagnostic whole-request latency includes two readouts and native answer generation. No new inference.','metrics':out}
with (p/'small_cost_quality.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
