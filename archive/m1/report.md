# Evaluation report

Source runs: m0-baseline-v1, m1-validation-v1, m1-pilot-v1. All runs executed on lp.

| Cohort | Condition | N | Accuracy | Deep accuracy | NLL | Brier | ECE | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot | A | 400 | 0.335 | 0.504 | 5.979 | 1.323 | 0.663 | 33.3 | 45.8 |
| pilot | A-ref | 400 | 0.338 | 0.504 | 5.464 | 1.310 | 0.658 | 32.7 | 37.1 |
| pilot | B | 400 | 0.335 | 0.504 | 5.979 | 1.323 | 0.663 | 33.4 | 45.3 |
| pilot | C128 | 400 | 0.367 | 0.308 | 2.143 | 0.946 | 0.415 | 3579.6 | 3647.4 |
| pilot | C32 | 400 | 0.372 | 0.383 | 1.857 | 0.891 | 0.343 | 916.5 | 941.5 |
| pilot | C512 | 400 | 0.525 | 0.414 | 2.366 | 0.764 | 0.333 | 14075.2 | 14453.6 |
| pilot | D | 267 | 0.502 | 0.504 | 4.831 | 0.996 | 0.498 | 36.6 | 48.3 |
| validation | A | 120 | 0.333 | 0.500 | 5.939 | 1.325 | 0.664 | 36.2 | 46.4 |
| validation | C128 | 120 | 0.375 | 0.375 | 2.251 | 0.972 | 0.403 | 3591.9 | 3674.9 |
| validation | C32 | 120 | 0.417 | 0.400 | 1.761 | 0.832 | 0.292 | 918.3 | 947.6 |
| validation | C512 | 120 | 0.492 | 0.375 | 2.759 | 0.828 | 0.384 | 14225.1 | 14532.0 |

## Paired deep effects

| Comparison | N | Gain, pp | 95% interval, pp |
|---|---:|---:|---|
| pilot/A-ref-A/deep | 133 | 0.00 | [0.00, 0.00] |
| pilot/B-A/deep | 133 | 0.00 | [0.00, 0.00] |
| pilot/C32-A/deep | 133 | -12.03 | [-19.55, -4.51] |
| pilot/C128-A/deep | 133 | -19.55 | [-27.07, -12.78] |
| pilot/C512-A/deep | 133 | -9.02 | [-16.54, -1.50] |
| pilot/D-A/deep | 133 | 0.00 | [0.00, 0.00] |
| validation/C32-A/deep | 40 | -10.00 | [-22.50, 2.50] |
| validation/C128-A/deep | 40 | -12.50 | [-22.50, -2.50] |
| validation/C512-A/deep | 40 | -12.50 | [-25.00, -2.50] |

These are single-seed pilot measurements, not a completed confirmation or a finding about Jev internals.
