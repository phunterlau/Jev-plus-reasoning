# Example results and interpretation

All measurements below are Qwen3-0.6B, not proprietary Jev. The new examples use
FP16 and the same three thinking seeds on an RTX 2070 Max-Q. A run means a mode/seed applied to
one question; it is not an independent question. All requested attempts are shown.
Probabilities are normalized over the declared options, not calibrated confidence.

## Exact short Rae question

| Mode | Seed | Answer | Correct | P(gold) | P(chosen) | Thinking tokens | Latency |
|---|---:|---|---|---:|---:|---:|---:|
| nonthinking | 17 | No | True | 57.7251% | 57.7251% | 0 | 0.0685 s |
| thinking | 17 | No | True | 99.9953% | 99.9953% | 193 | 5.7197 s |
| thinking | 18 | No | True | 99.9948% | 99.9948% | 208 | 6.0843 s |
| thinking | 19 | No | True | 99.9981% | 99.9981% | 163 | 4.7703 s |

[All records and full traces](../results/rae-short-v1/predictions.jsonl) · [Manifest](../results/rae-short-v1/manifest.json). All three thinking traces closed naturally.

The exact user question is: “Rae is round. Anyone round is not blue. Is Rae
blue?” The correct answer is No; options are Yes/No/Unknown. Both modes succeed.
The thinking traces apply round -> not blue, then map that conclusion to No.
Direct mode already assigns the correct label the largest probability (57.73%),
with 42.23% on Unknown. Thinking increases the correct label probability, but this
is not an accuracy repair or proof of necessity. The short and historical Rae
prompts differ in several ways; the cause of their differing direct performance
is not isolated by this comparison.

## R count in rasperry

| Mode | Seed | Answer | Correct | P(gold) | P(chosen) | Thinking tokens | Latency |
|---|---:|---|---|---:|---:|---:|---:|
| nonthinking | 17 | 5 | False | 4.1888% | 37.3347% | 0 | 0.0928 s |
| thinking | 17 | 2 | False | 0.1776% | 99.4985% | 305 | 9.1289 s |
| thinking | 18 | 3 | True | 99.9744% | 99.9744% | 352 | 10.3868 s |
| thinking | 19 | 2 | False | 0.0362% | 99.9153% | 320 | 9.1384 s |

[All records and full traces](../results/counting-rasperry-v1/predictions.jsonl) · [Manifest](../results/counting-rasperry-v1/manifest.json). All three thinking traces closed naturally.

Correct count: 3, at positions 1, 6, 7 in r a s p e r r y. Seed 17 names
positions 1 and 6 but misses 7. Seed 19 counts the ending pair but misses position
1. Seed 18 reconciles all three positions and succeeds. A posthoc majority vote
would be wrong. One successful seed is a selected possibility result, not reliable
counting. The two failed traces are over 99% confident in count 2.

## R count in strawberry

| Mode | Seed | Answer | Correct | P(gold) | P(chosen) | Thinking tokens | Latency |
|---|---:|---|---|---:|---:|---:|---:|
| nonthinking | 17 | 7 | False | 1.8436% | 34.7860% | 0 | 0.0670 s |
| thinking | 17 | 2 | False | 0.0102% | 99.5904% | 259 | 7.6006 s |
| thinking | 18 | 2 | False | 0.0296% | 98.5252% | 217 | 6.3335 s |
| thinking | 19 | 2 | False | 0.0215% | 99.3107% | 229 | 6.6681 s |

[All records and full traces](../results/counting-strawberry-v1/predictions.jsonl) · [Manifest](../results/counting-strawberry-v1/manifest.json). All three thinking traces closed naturally.

Correct count: 3, at positions 3, 8, 9 in s t r a w b e r r y. Seed 17
initially scrambles the spelling and later lists it correctly, but still counts
two. Seed 18 misses the early R. Seed 19 names positions 3 and 9 but misses 8.
All three return 2 with high option confidence. Reasoning does not repair this
example under the tested configuration. Deterministic counting would return 3.

## Original R1 Rae repair

The original selected case is `r1-r0-048` with `size=small`. It includes the awake
fact, the longer name Rae Aspen, and True/False/Unknown descriptions. The direct
answer is True, with P(False)=0.0055428678. Thinking returns False, with
P(False)=0.9999020100, after 277 tokens. Diagnostic times are 0.475272 and
8.541692 seconds. The trace explicitly applies the supplied rule and rejects the
positive blue statement. This is an observed repair of elementary inference and
explicit negation, while the exact short prompt is a both-correct control.

[Original full prompt](rae-original-prompt.txt) and [thinking trace](rae-original-thinking.txt)
are extracted from the bundled original JSONL without rewriting the prompt.
This selected case came from 36 elementary cases; its actual model seed was
computed from case ID plus base seed 17, not directly set to 17.

## Aggregate evidence and unresolved goals

On all 36 R1 elementary cases, small-model cued accuracy improves 12/36 -> 35/36,
with 23 paired repairs and no regressions. All 36 small-model thinking traces close.
Native-prefix typed accuracy is 12/36 -> 32/36; adding the answer cue matters.
Strict native generated-answer accuracy is only 0/36 -> 4/36 because most answers
copy the full option. The frozen combined gate remains failed; no parser was
relaxed after seeing the results.

The 1.7B thinking control also reaches 35/36 cued accuracy, with median diagnostic
latency 11.397 seconds and mean 468.97 tokens; one trace hits the cap and has no
readout. It offers no accuracy advantage on this small cohort. Its absent
probability vector is not filled in for a probability-metric comparison.

For 0.6B, R1 median diagnostic time rises 0.476 -> 7.168 seconds (about 15x).
This includes two readouts plus native answer generation. New individual-example
latencies include one readout without final generation, so the numbers are not
interchangeable. R1 probability metrics are available in
[small_cost_quality.json](../results/r1-mode-capacity-v1/small_cost_quality.json)
and recomputed by the verifier.

Earlier M1's validation-selected C32 reasoning did not recover depth-3/5 accuracy:
38.35% versus direct 50.38%, a paired change of -12.03 percentage points on 133
pilot questions. Its reported interval uses theory-group bootstrap resampling and does not
establish decoding-seed reliability. The larger C512
budget also did not recover deep accuracy (41.35%). See the
[archived report](../archive/m1/report.md). Those older small-budget protocols are
different from R1 and the current showcase; neither positive nor negative results
should be silently transferred between them.

No arithmetic/date, adversarial, long-context, or broad consistency repair has
been demonstrated. The justified conclusion is limited to some elementary
inference gains in an open-model typed interface, alongside clear failure cases.
