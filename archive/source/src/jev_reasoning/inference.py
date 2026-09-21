"""Single-GPU frozen FP16 inference with bounded thinking and typed readout."""
import inspect,time,platform,subprocess,importlib.metadata
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from .prompts import LABELS,VERSION,encode,messages,sha,row_seed

MODEL='Qwen/Qwen3-0.6B'
REVISION='c1899de289a04d12100db370d81485cdf75e47ca'

class Scorer:
 def __init__(self,max_tokens=8192):
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one CUDA GPU')
  self.max_tokens=max_tokens;self.device=torch.device('cuda:0')
  torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.allow_tf32=False
  start=time.perf_counter()
  self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,trust_remote_code=False)
  self.model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to(self.device).eval()
  self.model.requires_grad_(False)
  self.forward_options={}
  params=inspect.signature(self.model.forward).parameters
  if 'logits_to_keep' in params:self.forward_options['logits_to_keep']=1
  elif 'num_logits_to_keep' in params:self.forward_options['num_logits_to_keep']=1
  else:raise RuntimeError('Last-logit-only support required to bound memory')
  self.close=self.tokenizer.encode('</think>',add_special_tokens=False)
  self.open=self.tokenizer.encode('<think>',add_special_tokens=False)
  if len(self.close)!=1 or len(self.open)!=1:raise ValueError('Reasoning delimiters must be single tokens')
  self.close=self.close[0];self.open=self.open[0]
  self.separator=self.tokenizer.encode('\n\n',add_special_tokens=False)
  self.eos=self.tokenizer.eos_token_id
  torch.cuda.synchronize()
  self.metadata={'model':MODEL,'revision':REVISION,'dtype':'float16','score_dtype':'float32','gpu':torch.cuda.get_device_name(),'compute_capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'cuda_runtime':torch.version.cuda,'python':platform.python_version(),'host':platform.node(),'load_seconds':time.perf_counter()-start,'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'attention':'sdpa','batch_size':1,'forward_options':self.forward_options,'tf32':False}
 def forward(self,ids,cache=None):
  return self.model(input_ids=torch.tensor([ids],dtype=torch.long,device=self.device),past_key_values=cache,use_cache=True,return_dict=True,**self.forward_options)
 def sync(self):torch.cuda.synchronize(self.device)
 def sample(self,logits,generator):
  # Exact top-k then nucleus sampling, temperature 0.6, min-p=0.
  values,indices=torch.topk(logits.float()/0.6,20)
  probs=torch.softmax(values,dim=-1)
  cumulative=probs.cumsum(-1)
  remove=(cumulative-probs)>=0.95
  probs=probs.masked_fill(remove,0);probs=probs/probs.sum()
  return indices[torch.multinomial(probs,1,generator=generator)].item()
 @torch.inference_mode()
 def score(self,row,condition,seed=17,verify_cache=False,order=None):
  order=order or LABELS;budget=int(condition[1:]) if condition.startswith('C') else 0
  self.sync();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  direct_prompt,direct_ids,slots=encode(self.tokenizer,row,condition,self.max_tokens,order)
  generated=[];natural=False;forced=False;reason=None;cache_diff=None
  if budget:
   prompt=self.tokenizer.apply_chat_template(messages(row,order=order),tokenize=False,add_generation_prompt=True,enable_thinking=True)+'<think>\n'
   ids=self.tokenizer.encode(prompt,add_special_tokens=False)
  else:prompt=direct_prompt;ids=direct_ids
  if len(ids)+budget+len(self.separator)+1>self.max_tokens:raise ValueError('Full context exceeds max_tokens')
  tokenization_seconds=time.perf_counter()-started
  self.sync();t=time.perf_counter()
  out=self.forward(ids);self.sync();prefill_seconds=time.perf_counter()-t
  generator=torch.Generator(device=self.device).manual_seed(row_seed(row['id'],seed))
  decode_seconds=0.;readout_seconds=0.;valid=True;suffix=[]
  cached_probs=None
  if budget:
   t=time.perf_counter()
   for step in range(budget):
    token=self.sample(out.logits[0,-1],generator);generated.append(token)
    if token==self.close:natural=True;reason='natural_close';break
    if token==self.eos:valid=False;reason='early_eos';break
    if token==self.open:valid=False;reason='nested_think';break
    if step+1<budget:out=self.forward([token],out.past_key_values)
   self.sync();decode_seconds=time.perf_counter()-t
   forced=not natural
   if reason is None:reason='budget_exhausted'
   suffix=([self.close] if forced else [])+self.separator
   # Last sampled token has not yet been consumed by the cache.
   continuation=[generated[-1]]+suffix
   final_ids=ids+generated+suffix
   if verify_cache:
    cached_out=self.forward(continuation,out.past_key_values)
    cached_probs=torch.softmax(cached_out.logits[0,-1].float()[slots],-1)
    del cached_out
   del out
   self.sync();t=time.perf_counter()
   out=self.forward(final_ids);self.sync();readout_seconds=time.perf_counter()-t
  else:final_ids=ids
  vocab=out.logits[0,-1].float()
  if not torch.isfinite(vocab).all():raise FloatingPointError('Nonfinite vocabulary logits')
  selected=vocab[slots];probs=torch.softmax(selected,dim=-1)
  mass=torch.exp(torch.logsumexp(selected,dim=-1)-torch.logsumexp(vocab,dim=-1)).item()
  pred=order[probs.argmax().item()]
  free_token=int(vocab.argmax().item())
  if condition=='B':
   generated=[free_token]
   valid=free_token in slots
   pred=order[slots.index(free_token)] if valid else None
  self.sync();total=time.perf_counter()-started
  if verify_cache and budget:
   cache_diff=float((probs-cached_probs).abs().max().item())
  return {'id':row['id'],'group':row['group'],'gold':row['gold'],'depth':row['depth'],'cohort':row['cohort'],'family':row['family'],'condition':condition,'seed':seed,'row_seed':row_seed(row['id'],seed),'option_ids':order,'option_logits':selected.cpu().tolist(),'probabilities':probs.cpu().tolist(),'prediction':pred,'correct':valid and pred==row['gold'],'readout_valid':valid,'candidate_mass':mass,'unrestricted_token_id':free_token,'unrestricted_token_text':self.tokenizer.decode([free_token]),'prompt_version':VERSION,'prompt_sha256':sha(prompt),'answer_prefix_sha256':sha(str(final_ids)),'state_sha256':row['state_sha256'],'input_tokens':len(ids),'budget':budget,'sampled_reasoning_tokens':len(generated) if budget else 0,'reasoning_content_tokens':sum(x not in [self.open,self.close,self.eos] for x in generated) if budget else 0,'forced_scaffold_tokens':len(suffix),'natural_close':natural,'forced_close':forced,'termination':reason or 'no_reasoning','budget_exhausted':reason=='budget_exhausted','oracle_facts':len(row['oracle']) if condition=='D' else 0,'tokenization_seconds':tokenization_seconds,'prefill_seconds':prefill_seconds,'decode_seconds':decode_seconds,'readout_seconds':readout_seconds,'total_seconds':total,'readout_method':'full_prefix_recomputation' if budget else 'direct_full_prefix','processed_tokens':(len(ids)+len(generated)-1+len(final_ids)) if budget else len(final_ids),'reprocessed_tokens':len(final_ids) if budget else 0,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cache_probability_max_difference':cache_diff,'trace_token_ids':generated,'trace_text':self.tokenizer.decode(generated),'probability_status':'conditional option scores, uncalibrated decision confidence'}
