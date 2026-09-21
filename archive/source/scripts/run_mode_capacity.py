"""R1 reference path and crossed mode/capacity diagnostic, only on lp."""
import gc,hashlib,itertools,json,os,re,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,StoppingCriteriaList
from huggingface_hub import snapshot_download
from jev_reasoning.inference import Scorer
from jev_reasoning.prompts import LABELS,DESCRIPTIONS,sha
from jev_reasoning.semantic_microcases import make_cases
from jev_reasoning.proofwriter import canonical_hash,closure,label
root=Path(os.environ['JEV_ROOT']);out=root/'results/r1-mode-capacity-v1';out.mkdir(exist_ok=False)
cfg=json.loads(Path('experiments/mode_capacity/r1_v1.json').read_text());started=time.time()
manifest={'status':'running','config':cfg,'job':os.environ['JEV_JOB'],'started_unix':started,'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['src','scripts','experiments'] for p in Path(directory).rglob('*') if p.is_file() and p.suffix in ['.py','.json']},'models':{}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sync():torch.cuda.synchronize()
def free():gc.collect();torch.cuda.empty_cache()
def load(repo,rev):
 tok=AutoTokenizer.from_pretrained(repo,revision=rev,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
 model.requires_grad_(False);return tok,model

def generate(model,ids,n,**kw):
 x=torch.tensor([ids],device='cuda');o=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=n,pad_token_id=model.config.eos_token_id[0] if isinstance(model.config.eos_token_id,list) else model.config.eos_token_id,**kw)
 return o[0,len(ids):].tolist()

# Reference parity uses previously recorded prompts/outputs, with a fresh stock loader.
legacy=Scorer();old=[json.loads(x) for x in (root/'results/r0-semantic-v2/predictions.jsonl').read_text().splitlines()]
fixtures=[sorted([r for r in old if r['condition']=='plain_semantic' and r['gold']==g],key=lambda r:r['id'])[0] for g in LABELS]
logits=[]
with torch.inference_mode():
 for r in fixtures:logits.append(legacy.forward(r['input_ids']).logits[0,-1].float().cpu())
del legacy;free();tok,model=load(cfg['small_model'],cfg['small_revision']);parity=[]
with torch.inference_mode():
 for r,z in zip(fixtures,logits):
  x=torch.tensor([r['input_ids']],device='cuda');stock=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
  gen=generate(model,r['input_ids'],32,do_sample=False)
  parity.append({'id':r['id'],'gold':r['gold'],'max_logit_difference':float((stock-z).abs().max()),'token_ids_identical':gen==r['generated_ids'],'generated_ids':gen,'text':tok.decode(gen,skip_special_tokens=True)})
(out/'reference_parity.json').write_text(json.dumps(parity,indent=2))
assert all(r['token_ids_identical'] and r['max_logit_difference']<=.001 for r in parity),'Reference discrepancy: stop before mode experiment'
del model;free();print('REFERENCE_PARITY_PASSED',flush=True)

# Rename a fixed subset without changing its logic; all rows retain checked gold.
rows=[]
for r in make_cases():
 if r['variant']>2:continue
 oldname=r['query'][0];newname=[a+' '+b for a in ['Nell','Owen','Pia','Quinn','Rae','Seth'] for b in ['Aspen','Beech','Cypress','Fir','Hazel','Linden']][len(rows)];mapping={oldname:newname,'Morgan Oak':'Tessa Pine'}
 def rename(a):return (mapping.get(a[0],a[0]),*a[1:])
 r['state']=r['state'].replace(oldname,newname).replace('Morgan Oak','Tessa Pine');r['question']=r['question'].replace(oldname,newname)
 r['query']=rename(r['query']);r['facts']={k:rename(a) for k,a in r['facts'].items()}
 r['group']=canonical_hash(r['facts'],r['rules']);r['state_sha256']=sha(r['state']);r['id']='r1-'+r['id'];rows.append(r)
oldgroups=set()
for folder,files in [('data/derived/owa-v2',['pilot.jsonl','validation.jsonl','smoke.jsonl']),('results/r0-semantic-v2',['cases.jsonl'])]:
 for f in files:oldgroups.update(json.loads(x)['group'] for x in (root/folder/f).read_text().splitlines())
assert len(rows)==36 and len({r['group'] for r in rows})==36 and not oldgroups&{r['group'] for r in rows}
for r in rows:
 ground=[]
 for rid,(ps,h) in r['rules'].items():
  for e in {a[0] for a in r['facts'].values()}:
   ground.append((rid,tuple((e,*p[1:]) for p in ps),(e,*h[1:])))
 ds,_=closure(r['facts'],ground);assert label(r['query'],ds)==r['gold']
(out/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
class Close(StoppingCriteria):
 def __init__(self,token):self.token=token
 def __call__(self,input_ids,scores,**kw):return input_ids[0,-1].item()==self.token

def prompt(row,tok,thinking):
 content=row['state']+'\nUse only the supplied facts and rules, in their stated direction. An unstated fact is not automatically false.\nStatement: '+row['question']+'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {g}: {DESCRIPTIONS[g]}' for i,g in enumerate(LABELS))
 ms=[{'role':'system','content':'Classify the statement using the supplied evidence. You may reason inside the thinking block. Outside it, respond with only the uppercase letter of the correct option.'},{'role':'user','content':content}]
 return tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)

def score_prefix(model,ids,slots):
 o=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1).logits[0,-1].float();ps=o[slots].softmax(0)
 return {'probabilities':ps.tolist(),'option_logits':o[slots].tolist(),'prediction':LABELS[int(ps.argmax())],'candidate_mass':float(torch.exp(torch.logsumexp(o[slots],0)-torch.logsumexp(o,0)))}

records=[];summaries={}
with (out/'predictions.jsonl').open('x') as f:
 for size in ['small','large']:
  if size=='large' and all(s['gate_pass'] for s in summaries.values()):
   manifest['large_decision']='skipped: both small modes passed';break
  repo=cfg[size+'_model'];rev=cfg[size+'_revision']
  if size=='large':
   manifest['large_decision']='triggered: at least one small mode failed'
   print('LARGE_MODEL_TRIGGERED',flush=True)
   snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
  free();tok,model=load(repo,rev);eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
  close=tok.encode('</think>',add_special_tokens=False)[0];slots=[tok.encode(x,add_special_tokens=False)[0] for x in ['A','B','C']]
  manifest['models'][size]={'repo':repo,'revision':rev,'gpu':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters()),'load_allocated_bytes':torch.cuda.memory_allocated()}
  # Engineering smoke before cohort: literal fixture, not used for model selection.
  warm=tok.encode('A short test.',add_special_tokens=False)
  with torch.inference_mode():generate(model,warm,4,do_sample=False)
  for i,row in enumerate(rows):
   modes=['nonthinking','thinking'] if i%2==0 else ['thinking','nonthinking']
   for mode in modes:
    text=prompt(row,tok,mode=='thinking');ids=tok.encode(text,add_special_tokens=False);assert len(ids)+2048+40<4096
    sync();t0=time.perf_counter();torch.cuda.reset_peak_memory_stats();torch.manual_seed(int(sha(row['id']+'17')[:8],16))
    trace=[];closed=mode=='nonthinking'
    with torch.inference_mode():
     if mode=='thinking':
      trace=generate(model,ids,2048,do_sample=True,temperature=.6,top_p=.95,top_k=20,min_p=0.,stopping_criteria=StoppingCriteriaList([Close(close)]))
      closed=bool(trace) and trace[-1]==close
     native=cue=None;answer=[];answertext='';pred=None
     if closed:
      prefix=ids+trace+ (tok.encode('\n\n',add_special_tokens=False) if trace else [])
      cueids=prefix+tok.encode('Answer:\n',add_special_tokens=False)
      for seq in [prefix,cueids]:
       for slot,ch in zip(slots,['A','B','C']):
        decoded=tok.decode(seq)
        assert tok.encode(decoded+ch,add_special_tokens=False)==tok.encode(decoded,add_special_tokens=False)+[slot]
      native=score_prefix(model,prefix,slots);cue=score_prefix(model,cueids,slots)
      answer=generate(model,prefix,32,do_sample=False);answertext=tok.decode(answer,skip_special_tokens=True)
      match=re.fullmatch(r'\s*([ABC])[.!]?\s*',answertext)
      pred=LABELS[ord(match.group(1))-65] if match and answer[-1] in eos else None
    sync()
    r={'id':row['id'],'group':row['group'],'gold':row['gold'],'nominal_depth':row['nominal_depth'],'query_polarity':row['query_polarity'],'size':size,'mode':mode,'prompt':text,'prompt_ids':ids,'trace_ids':trace,'trace_text':tok.decode(trace),'closed':closed,'generated_answer_ids':answer,'generated_answer':answertext,'prediction':pred,'correct':pred==row['gold'],'native':native,'cue':cue,'total_seconds':time.perf_counter()-t0,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    f.write(json.dumps(r)+'\n');f.flush();records.append(r)
   if (i+1)%6==0:print(json.dumps({'model':size,'cases_done':i+1,'cases_total':36,'elapsed_seconds':time.time()-started}),flush=True)
  for mode in ['nonthinking','thinking']:
   rs=[r for r in records if r['size']==size and r['mode']==mode];cells={}
   for d,g,p in itertools.product([0,1],LABELS,['+','-']):
    group=[r for r in rs if (r['nominal_depth'],r['gold'],r['query_polarity'])==(d,g,p)]
    cells[f'{d}/{g}/{p}']={'n':len(group),'generated_accuracy':sum(r['correct'] for r in group)/len(group),'cue_accuracy':sum(r['cue'] is not None and r['cue']['prediction']==g for r in group)/len(group)}
   summary={'n':len(rs),'generated_accuracy':sum(r['correct'] for r in rs)/len(rs),'native_accuracy':sum(r['native'] is not None and r['native']['prediction']==r['gold'] for r in rs)/len(rs),'cue_accuracy':sum(r['cue'] is not None and r['cue']['prediction']==r['gold'] for r in rs)/len(rs),'closure_fraction':sum(r['closed'] for r in rs)/len(rs),'cells':cells}
   summary['gate_pass']=summary['closure_fraction']>=.9 and all(v['generated_accuracy']>=.9 and v['cue_accuracy']>=.9 for v in cells.values());summaries[size+'/'+mode]=summary
  (out/'summary.json').write_text(json.dumps(summaries,indent=2));print(json.dumps({'finished_model':size,'summary':summaries}),flush=True)
  del model;free()
assert len(records) in [72,144]
manifest.update(status='complete',finished_unix=time.time(),records=len(records),artifacts={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['reference_parity.json','cases.jsonl','predictions.jsonl','summary.json']})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('R1_COMPLETE',flush=True)
