# Method and experimental scope

## Question

Can adding reasoning help some of the weaknesses identified for Jev while
retaining a bounded decision interface? The operational experiment is on an
open Qwen proxy. No Jev API result or proprietary model improvement is measured.
The intervention is the complete thinking procedure: chat-template mode,
generated scratchpad, and subsequent conditioning. It does not isolate “more
compute” from all other differences in token context or decoding policy.

## Baseline and intervention

The baseline follows SemIf's direct next-token option scoring. It maps declared
options to A/B/C (or up to I for counting), runs a frozen Qwen3-0.6B model, and
normalizes those slot logits. The interface is choice-like rather than free text.
Full-vocabulary probability mass on the candidate slots is retained separately as
`candidate_mass`; the normalized option scores condition on the restricted set.

Thinking mode uses Qwen's existing `enable_thinking=True` chat template and
samples up to 2048 tokens, stopping on its natural thinking delimiter. The entire
prompt and scratchpad, followed by an answer cue, are recomputed for typed scoring.
A capped or otherwise unclosed trace has null probabilities/prediction and counts
as incorrect. No forced closing or generated-answer repair is accepted.

This is test-time inference only. There is no training, new head, parameter update,
latent recurrence, adaptive compute policy, or reconstruction of TypeSafe RLCD.
The reference clones were read-only. SemIf's optimization paths are not used here.

## Exact examples

The portable runner's JSON declares the system message, exact task, options, and
gold label. Gold is used for evaluation only, never appended to the model prompt.
The Rae rule supports explicit negation, so its correct answer is No. Both words
have three R's under the declared case-insensitive interpretation; `rasperry`
is preserved exactly without spelling correction.

Individual examples use one deterministic direct readout and thinking seeds
17/18/19 with temperature .6, top-p .95, top-k 20, min-p 0. There is no native
answer generation; typed decisions are argmax over the declared options.
The runner uses an empty thinking block for direct mode and a generated block for
thinking mode. A seed is recorded for the direct row for uniform provenance,
but no sampling occurs in that condition.

The original R1 example has different names, an extra fact, and True/False/Unknown
option descriptions. Its seed is deterministically derived from the case ID and
base seed 17; it is not the literal torch seed 17 used in the newer examples.
The new short prompt was tested separately instead of attaching historical metrics
to a paraphrase. We did not search alternative prompts after seeing its direct
success.

## Timing and probability metrics

- CUDA is synchronized at measurement boundaries; loading and initial prompt
  tokenization are excluded. Boundary checks and final option scoring are included.
- R1 also includes an additional native-prefix readout and native answer generation
  up to 32 tokens. Its latency is a diagnostic request cost, not serving-only cost.
- The short Rae and counting runs include one answer-cued readout. They still
  include diagnostic boundary checks; these are not production benchmarks.
- GPU memory is maximum PyTorch allocated bytes, not total process/device memory.
- NLL is mean negative natural logarithm of the gold option probability.
- Brier is mean SUM of squared errors across the three R1 labels (not divided by 3).
- ECE uses ten equal-width confidence bins. With 36 templated cases it is
  descriptive and cannot demonstrate broad calibration.
- Macro F1 treats the three labels equally. R1's 23 repairs/zero regressions are
  paired on the same 36 unique cases, not inferred from independent accuracies.

## Gates and limits

R1's combined frozen gate required both strict generated and cued typed outputs
to reach 90% accuracy in every depth/label/polarity cell, with at least 90% natural
thinking completion. It failed, despite high aggregate typed accuracy. Native
option-text copies are not silently reclassified as letter-only successes.

The current objective permits selected demonstrations. The original Rae repair
is selected from 36 elementary cases; the full cohort is included. The requested
short Rae and both counting examples are retained even when no gain occurs.
No benchmark population claim follows from these examples. Multiple seeds on one
word measure variation on that word, not independent task coverage. Option order,
paraphrases, adversarial content, long contexts, arithmetic, and dates are not
validated by this showcase.

Earlier M1 explored depth-3/5 ProofWriter examples with smaller token budgets and
a different readout procedure. Its negative results remain part of the evidence;
R1's elementary improvement does not overturn them.
