import json,os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoConfig
repo='Qwen/Qwen3-0.6B';rev='c1899de289a04d12100db370d81485cdf75e47ca'
path=snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE','README.md'])
t=AutoTokenizer.from_pretrained(path,local_files_only=True)
c=AutoConfig.from_pretrained(path,local_files_only=True)
r={'model':repo,'revision':rev,'snapshot':path,'config_class':type(c).__name__,'chat_direct':t.apply_chat_template([{'role':'user','content':'Choose A, B or C.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False),'slots':{x:t.encode(x,add_special_tokens=False) for x in ['A','B','C','<think>','</think>','\n\n']}}
Path(os.environ['JEV_JOB'],'model.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
