"""Bounded exact-word counting illustration; no training or prompt search."""
import hashlib, json, os, subprocess, time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODEL = 'Qwen/Qwen3-0.6B'
REV = 'c1899de289a04d12100db370d81485cdf75e47ca'
WORD = 'rasperry'
OPTIONS = list(range(9))
assert WORD.lower().count('r') == 3 and len(WORD) == 8
out = Path(os.environ['JEV_ROOT']) / 'results/counting-rasperry-v1'
out.mkdir(exist_ok=False)
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, local_files_only=True, torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
model.requires_grad_(False)
assert all(p.device == torch.device('cuda:0') for p in model.parameters())
slots = [tok.encode(chr(65+i), add_special_tokens=False) for i in OPTIONS]
assert all(len(s) == 1 for s in slots)
slots = [s[0] for s in slots]
close = tok.encode('</think>', add_special_tokens=False)
assert len(close) == 1
class Close(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == close[0]

def telemetry():
    s = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    assert any(line.split(',')[0].strip() == str(os.getpid()) for line in s.splitlines())
    return s

messages = [
    {'role': 'system', 'content': 'Choose the correct count. You may reason inside the thinking block. Outside it, respond with only the uppercase letter of the correct option.'},
    {'role': 'user', 'content': 'How many R are in the exact word "rasperry"? Count case-insensitively (R and r are the same letter). Do not correct the spelling.\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {i}' for i in OPTIONS)}
]
# Warm up CUDA before measurement.
with torch.inference_mode():
    z = model(input_ids=torch.tensor([[slots[0]]], device='cuda:0')).logits
    assert z.device == torch.device('cuda:0')
torch.cuda.synchronize()
evidence = [telemetry()]
records = []
for mode, seed in [('nonthinking', 17), ('thinking', 17), ('thinking', 18), ('thinking', 19)]:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=mode=='thinking')
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], device='cuda:0')
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    trace = []
    with torch.inference_mode():
        if mode == 'thinking':
            y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=2048, do_sample=True, temperature=.6, top_p=.95, top_k=20, min_p=0., stopping_criteria=StoppingCriteriaList([Close()]))
            assert y.device == torch.device('cuda:0')
            trace = y[0, len(ids):].tolist()
        torch.cuda.synchronize(); think_seconds = time.perf_counter()-start
        closed = mode == 'nonthinking' or bool(trace) and trace[-1] == close[0]
        probs = None; prediction = None; candidate_mass = None
        if closed:
            prefix = ids + trace + (tok.encode('\n\n', add_special_tokens=False) if trace else []) + tok.encode('Answer:\n', add_special_tokens=False)
            for i, slot in enumerate(slots):
                decoded = tok.decode(prefix)
                assert tok.encode(decoded+chr(65+i), add_special_tokens=False) == tok.encode(decoded, add_special_tokens=False)+[slot]
            z = model(input_ids=torch.tensor([prefix], device='cuda:0'), use_cache=False, logits_to_keep=1).logits[0,-1].float()
            assert z.device == torch.device('cuda:0')
            probs = z[slots].softmax(0).tolist()
            prediction = OPTIONS[max(range(len(probs)), key=probs.__getitem__)]
            candidate_mass = float((z[slots].logsumexp(0)-z.logsumexp(0)).exp())
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    r = dict(mode=mode, seed=seed, prompt=prompt, prompt_ids=ids, trace_ids=trace, trace_text=tok.decode(trace), natural_close=closed if mode=='thinking' else None, options=OPTIONS, probabilities=probs, prediction=prediction, gold=3, correct=prediction==3, candidate_mass=candidate_mass, thinking_seconds=think_seconds if mode=='thinking' else 0, total_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    records.append(r)
    with (out/'predictions.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    evidence.append(telemetry())
    print(json.dumps({k:r[k] for k in ['mode','seed','prediction','correct','total_seconds']}), flush=True)
manifest = dict(status='complete', model=MODEL, revision=REV, word=WORD, gold=3, cap=2048, seeds=[17,18,19], messages=messages, gpu=torch.cuda.get_device_name(), pid=os.getpid(), torch=torch.__version__, transformers=transformers.__version__, telemetry=evidence, job=os.environ['JEV_JOB'], predictions_sha256=hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest(), timing_scope='thinking plus one answer-cued full-prefix typed readout and boundary checks; no native answer generation; excludes loading and prompt tokenization')
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
