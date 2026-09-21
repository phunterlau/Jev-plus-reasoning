"""Portable Qwen typed-choice comparison. Run inference on a CUDA GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def validate_cases(cases):
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Need nonempty cases with unique IDs')
    for c in cases:
        if not 2 <= len(c['options']) <= 26:
            raise ValueError('Need 2..26 options')
        values = [o['value'] for o in c['options']]
        if len(set(values)) != len(values) or c['gold'] not in values:
            raise ValueError('Options must be unique and contain gold')
        if not c['question'] or not c['system']:
            raise ValueError('Missing prompt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='experiments/showcase/cases.json')
    parser.add_argument('--case', action='append', help='Case ID; repeat or omit for all')
    parser.add_argument('--output', required=True, help='New directory; existing results are never overwritten')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 18, 19])
    parser.add_argument('--cap', type=int, default=2048)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    validate_cases(cases)
    if args.case:
        if set(args.case) - {c['id'] for c in cases}:
            raise ValueError('Unknown case ID')
        cases = [c for c in cases if c['id'] in args.case]
    if args.cap < 1 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Positive cap and distinct seeds required')
    if args.dry_run:
        print(json.dumps({'cases': [c['id'] for c in cases], 'runs': len(cases)*(1+len(args.seeds)), 'cap': args.cap}))
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'model': MODEL, 'revision': REVISION,
                'cases': cases, 'seeds': args.seeds, 'cap': args.cap,
                'sampling': {'temperature': .6, 'top_p': .95, 'top_k': 20, 'min_p': 0.},
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'started_unix': time.time(), 'pid': os.getpid(),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'timing_scope': 'thinking plus one full-prefix answer-cued readout and boundary checks; excludes loading and initial prompt tokenization; no generated final answer'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=args.local_files_only, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False)
    assert all(p.device == torch.device('cuda:0') for p in model.parameters())
    close = tok.encode('</think>', add_special_tokens=False)
    assert len(close) == 1

    class Close(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids[0, -1].item() == close[0]

    def telemetry():
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        matching = [line for line in raw.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        assert matching, 'Inference PID absent from GPU telemetry'
        return {'time_unix': time.time(), 'matching_process': matching}

    with torch.inference_mode():
        warm = model(input_ids=torch.tensor([tok.encode('Warmup', add_special_tokens=False)], device='cuda:0')).logits
        assert warm.device == torch.device('cuda:0')
    del warm
    torch.cuda.synchronize()
    manifest.update(gpu=torch.cuda.get_device_name(), dtype='float16', telemetry=[telemetry()])
    count = 0
    with (out/'predictions.jsonl').open('x') as f:
        for case in cases:
            values = [o['value'] for o in case['options']]
            slots = [tok.encode(chr(65+i), add_special_tokens=False) for i in range(len(values))]
            assert all(len(s) == 1 for s in slots)
            slots = [s[0] for s in slots]
            content = case['question']+'\nOptions:\n'+'\n'.join(f"{chr(65+i)}. {o['description']}" for i, o in enumerate(case['options']))
            messages = [{'role': 'system', 'content': case['system']}, {'role': 'user', 'content': content}]
            for mode, seed in [('nonthinking', args.seeds[0])] + [('thinking', s) for s in args.seeds]:
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=mode=='thinking')
                ids = tok.encode(prompt, add_special_tokens=False)
                assert len(ids)+args.cap+16 <= 4096, 'Context overflow; no truncation'
                x = torch.tensor([ids], device='cuda:0')
                torch.manual_seed(seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started = time.perf_counter()
                trace = []
                probabilities = prediction = mass = None
                with torch.inference_mode():
                    if mode == 'thinking':
                        generated = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                            max_new_tokens=args.cap, do_sample=True, **manifest['sampling'],
                            stopping_criteria=StoppingCriteriaList([Close()]))
                        assert generated.device == torch.device('cuda:0')
                        trace = generated[0, len(ids):].tolist()
                    torch.cuda.synchronize(); thinking_s = time.perf_counter()-started if mode=='thinking' else 0.
                    closed = mode=='nonthinking' or bool(trace) and trace[-1]==close[0]
                    if closed:
                        prefix = ids+trace+(tok.encode('\n\n', add_special_tokens=False) if trace else [])+tok.encode('Answer:\n', add_special_tokens=False)
                        decoded = tok.decode(prefix)
                        for i, slot in enumerate(slots):
                            assert tok.encode(decoded+chr(65+i), add_special_tokens=False)==tok.encode(decoded, add_special_tokens=False)+[slot]
                        z = model(input_ids=torch.tensor([prefix], device='cuda:0'), use_cache=False, logits_to_keep=1).logits[0,-1].float()
                        assert z.device == torch.device('cuda:0')
                        probabilities = z[slots].softmax(0).tolist()
                        prediction = values[max(range(len(values)), key=probabilities.__getitem__)]
                        mass = float((z[slots].logsumexp(0)-z.logsumexp(0)).exp())
                torch.cuda.synchronize(); elapsed = time.perf_counter()-started
                record = dict(id=case['id'], mode=mode, seed=seed, prompt=prompt, prompt_ids=ids,
                    trace_ids=trace, trace_text=tok.decode(trace), natural_close=closed if mode=='thinking' else None,
                    options=values, probabilities=probabilities, prediction=prediction, gold=case['gold'],
                    correct=prediction==case['gold'], candidate_mass=mass, thinking_seconds=thinking_s,
                    total_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                f.write(json.dumps(record)+'\n'); f.flush(); count += 1
                manifest['telemetry'].append(telemetry())
                print(json.dumps({k: record[k] for k in ['id','mode','seed','prediction','correct','total_seconds']}), flush=True)
    manifest.update(status='complete', finished_unix=time.time(), records=count,
        predictions_sha256=hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
