"""Verify our reference condition against unmodified pinned SemIf functions on lp."""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path('tests').resolve()))
from reference_semif import direct
from jev_reasoning.inference import Scorer
from jev_reasoning.prompts import encode,messages,LABELS
root=Path(os.environ['JEV_ROOT']);out=Path(os.environ['JEV_JOB'])
provenance=json.loads(Path('tests/reference_semif/PROVENANCE.json').read_text())
for filename,h in provenance['files'].items():
 assert hashlib.sha256(Path('tests/reference_semif',filename).read_bytes()).hexdigest()==h
rows=[json.loads(x) for x in (root/'data/derived/owa-v2/smoke.jsonl').read_text().splitlines()][::3]
scorer=Scorer();results=[]
for row in rows:
 payload=json.loads(messages(row,reference=True)[1]['content'])
 reference_row={'id':row['id'],'state':payload['evidence'],'question':payload['criterion'],'options':[{'id':label,'description':option['description']} for label,option in zip(LABELS,payload['options'])]}
 _,our_ids,our_slots=encode(scorer.tokenizer,row,'A-ref',8192)
 their_ids,their_slots,_=direct.encode_prompt(scorer.tokenizer,reference_row,8192)
 assert our_ids==their_ids and our_slots==their_slots
 ours=scorer.score(row,'A-ref')
 theirs=direct.score(scorer.model,scorer.tokenizer,reference_row,scorer.metadata,max_tokens=8192)
 delta=max(abs(a-b) for a,b in zip(ours['probabilities'],theirs['probabilities']))
 logit_delta=max(abs(a-b) for a,b in zip(ours['option_logits'],theirs['option_logits']))
 assert delta<1e-6 and logit_delta<1e-6
 results.append({'id':row['id'],'probability_max_difference':delta,'logit_max_difference':logit_delta,'input_tokens_identical':True,'slot_ids_identical':True})
result={'status':'passed','source':provenance,'model':scorer.metadata,'rows':results,'scope':'Same input token IDs, slots, logits and conditional probabilities on eight training-source fixtures. FP16 loader differs intentionally from upstream BF16; no latency equivalence claim.'}
(out/'semif_parity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
