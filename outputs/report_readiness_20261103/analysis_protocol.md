# Exploratory follow-up protocol — 3 November 2026

The protocol was recorded before the follow-up comparisons were calculated. The comparisons are exploratory because the original test results had already been inspected.

## Inputs and outcomes

The six-model archive (SHA-256 `70ea712be64e8625bb0638635e09036c091e9122ff56cf98e350dbe69749bfe4`) contains 184 traces and 3,312 saved responses used in the analysis. Joint correctness requires the labelled agent and exact zero-indexed step. Invalid responses contribute a no-attribution vote. Three-way ties retain the earliest response.

Use the original 110/74 split for direct primary reproduction and fixed-split ablations. Calibrators and rules for those ablations are fitted/selected on the 110 validation cases only. Use the existing 22-cutoff grid including the all-strong fallback, 95% strong-only validation target and original tie-breaking conventions. Cost includes every cheap review plus each escalated strong review. Local cost is recorded generation elapsed time at US$3.29/hour; GPT uses saved request usage prices. No loading, idle, failed-attempt or batching savings are imputed.

Terminology clarification (no methodological change): an invalid response supplies a `(None,None)` no-attribution vote. This vote is counted, not discarded, and can become the selected output through a tie or majority; an invalid selected output is incorrect. The original fixed split shares 25 canonical questions across its partitions. Clustering fixed-test resamples does not remove this fitting/evaluation dependence.

## Analyses

1. Reproduce the eight corrected primary policies, counts, costs and routing decisions. Preserve original test counts as primary results.
2. Fixed split: compare isotonic token confidence with raw token confidence using the same cutoff grid; also allow raw confidence to select among the original three signals. Include a constant validation-correctness-rate confidence baseline for Brier/ECE diagnostics. Report test discrimination separately from calibration.
3. Random controls: (a) post-hoc expected-test-cost-matched independent random escalation, with probability derived from the observed cascade's resource budget only; (b) validation-budget-matched probability frozen and applied to test. These are distinct controls. Use 50,000 seeded random assignments for conditional accuracy/cost ranges, not confidence intervals for new populations. Use 10,000 underlying-question cluster bootstrap draws for the fixed-policy difference versus the expected-cost-matched control, recomputing its resource probability in each draw. No correctness labels are used to choose random escalation probabilities.
4. Robustness: canonical question groups join shared nonempty question_ID or exact case-folded/whitespace-normalised question text. Use ten repetitions of five approximately source-balanced group folds, seeds 20261103 + repeat. Each case appears once in outer test per repetition. Compare three predeclared methods on the same folds: shared-fit isotonic (fit and select on outer training); cross-fitted isotonic (three inner question-group folds generate out-of-fold calibration scores for cutoff selection, then refit calibration on all outer training before test); raw confidence (select on outer training). Each method selects among the original three signals. Never use outer test correctness for fitting or selection. Report across-repeat ranges as split sensitivity, not 1,840 independent samples or a fresh confirmatory test. Inner calibrators have fewer cases than final calibrators; cross-fitting is not guaranteed to improve results.
5. Error review: tabulate all agent/step error categories, completed invalid responses and output-cap flags. Inspect the six existing label/speaker discrepancies, high-end Llama calibration errors, capped invalid Qwen cases and representative agent-correct/step-wrong records. Do not change benchmark labels or rerun answers. Distinguish observed patterns from causal hypotheses.

## Interpretation boundaries

Passing validation is guaranteed to have a feasible candidate because always-strong routing matches the strong baseline; savings and test retention are not guaranteed. Additional selection after viewing these outcomes remains exploratory. Question grouping reduces known dependence, but does not establish absence of pretraining contamination. Fixed-policy intervals exclude uncertainty from collecting new model responses. Publication claims must distinguish saved-output routing estimates from live deployment and billing measurements.
