# Provenance and reference projects

## Open-source reference

The baseline design comes from [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf),
previously named OpenJev, inspected at commit
`ca3ba65f142967030ecb453346e94d6f476a69df`. It recreates a typed decision interface
with open models and explicitly does not reproduce proprietary Jev. We adapted
the direct option-scoring idea and reference prompt in the original research
code. Its MIT attribution is included in the root THIRD_PARTY_NOTICES.md.

[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev), commit
`71a513bb0163b5634467842b523ee0c0ed6fb1c7`, was inspected during planning only.
No NanoJev implementation code is used in this prototype. Neither reference
clone was modified, and neither clone nor its model weights is bundled here.

## Model and hardware

- Model: Qwen/Qwen3-0.6B, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Official model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Inference: FP16, SDPA, PyTorch 2.6.0+cu124, Transformers 4.51.3,
  batch size one, TF32 disabled, frozen weights.
- Hardware: NVIDIA GeForce RTX 2070 with Max-Q Design, 8 GiB.
- New example runs assert CUDA parameter/output placement and record inference
  PID-matched NVIDIA telemetry. Local work is preparation and evidence review.
- R1 also contains a bounded Qwen3-1.7B control; see its original manifest. The
  README's main comparison is exclusively 0.6B.

Qwen model weights retain their upstream license and are downloaded separately.
The project is independently MIT licensed. No affiliation or endorsement by
TypeSafe or Qwen is claimed.

## Source import

This package was assembled from the existing `jev-reasoning` research workspace
on 2026-09-21. Existing result files and archived source were copied unchanged.
`provenance/import_manifest.json` records SHA-256 hashes of those imported files
without bundling local planning notes, GPU runbooks, or machine-specific job
receipts. Result manifests retain their original execution metadata and hashes;
historical paths in those records are not runtime dependencies.

`results/r1-mode-capacity-v1/` includes all 144 rows (36 cases, two modes, two
model sizes), not just successful examples. Its `small_cost_quality.json` is a
posthoc summary of the 72 small-model rows; it does not revise the original gate.
The two counting folders include all four rows per word. `rae-short-v1` adds the
new exact-prompt run, including its successful direct baseline.

`archive/source/` preserves the research implementation and configs. Historical
R1 reconstruction needs upstream R0/ProofWriter artifacts not all included here;
it should not be mistaken for the self-contained runnable showcase. The compact
M1 archive contains reports, not a fresh audit of its omitted raw predictions.
Use `scripts/verify_evidence.py` to validate bundled hashes and recompute R1 metrics
and per-example summaries from the included predictions.

## Motivation source

TypeSafe's official [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
were checked on 2026-09-21. They motivate the experimental categories, not evidence
about our prototype. The vendor's version scope matters: this project does not
claim results for all current or future Jev versions.
