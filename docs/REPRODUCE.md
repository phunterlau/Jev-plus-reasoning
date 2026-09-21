# Reproduce the showcase

## Inspect without inference

From this repository root:

```bash
python3 scripts/verify_evidence.py
python3 scripts/run_showcase.py --output results/local-preview --dry-run
```

These use only the Python standard library. Dry run validates the three case
definitions and prints the planned 12 records; it loads no weights, creates no
output directory, and performs no inference. The verifier checks imported hashes,
completion metadata, all bundled prediction files, and recalculates reported
R1 0.6B metrics. It does not audit omitted M1 raw data.

## CUDA environment

The measured environment used Python 3.12, CUDA 12.4 PyTorch wheels, and the pinned
packages in requirements.txt. Use a CUDA-capable machine with sufficient GPU memory.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/run_showcase.py --case rae-short --output results/local-rae
```

The default model/revision are constants in the runner. The first invocation can
download weights. Reuse a verified existing environment/cache if available;
`--local-files-only` prevents downloads. Do not copy model weights into Git.

Select `--case rasperry` or `--case strawberry`, or omit --case to run all three.
Each selected case produces one direct row and three thinking rows. Multiple
--case flags are supported. Defaults are seeds 17/18/19 and cap 2048. Changed
seeds, cap, prompt, software, or hardware constitute a new experiment; do not
replace the archived original results. Bitwise sampling reproducibility across
hardware/software is not guaranteed.

## Output contract

Each JSONL row includes case ID, mode, seed, exact prompt and token IDs, complete
thinking trace, natural completion status, ordered option values/probabilities,
argmax decision, gold label, correctness, candidate probability mass, elapsed
seconds, and peak allocated GPU bytes. Non-closing traces are retained as failures
with null prediction/probabilities. The manifest becomes complete only after all
planned rows are written and includes the prediction SHA-256.

This portable runner matches the individual-example inference procedure. It does
not reproduce R1's extra native readout and generated-answer timing, nor its
reference-parity experiment. Those historical artifacts and source are preserved
separately; they must not be relabeled as outputs of this new runner.
