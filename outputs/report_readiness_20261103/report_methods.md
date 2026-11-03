# Report methods facts — 3 November 2026

- Research scope: retrospective agent-and-exact-step attribution on 184 recorded Who&When conversations; correct task answer and failure status supplied to judges.
- Six models; eight cheap→strong pairings; three recorded responses per model/case (3,312 responses).
- Aggregation: joint agent/step modal vote, earliest response on ties. Invalid responses contribute a counted no-attribution vote. Invalid winning votes are incorrect.
- Primary split: 110 validation / 74 test, recorded original seed 42. Trace records are disjoint; 25 test records share underlying question IDs with validation. Canonical follow-up groups: 139.
- Three cheap confidence signals: vote consistency, mean self-reported confidence /100, mean generated-token likelihood score. Token likelihood includes generated explanation text; it is not directly the probability that the chosen attribution is correct.
- Isotonic fitting and signal/cutoff selection share the original validation set. Grid: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.000001]. Cutoff selects individual routes; the 95% requirement concerns whole-set correctness relative to strong-only.
- Select the lowest rounded reference cost among qualifying candidates; within a signal, tie-break by higher accuracy then lower cutoff; equal-cost signal winners retain original signal ordering. The all-strong fallback guarantees a feasible validation candidate, not savings or generalisation.
- All eight primary policies use calibrated token confidence. Original selected cutoffs are retained in primary_results.csv.
- Primary cost: frozen public H100 PCIe reference US$3.29/hour × saved local generation elapsed time; GPT independently repriced using saved input/output/cache token usage. Loading, idle time, downloads, failures, taxes and live scheduling effects excluded.
- Fixed-split ablations retain original fitting/test boundaries. Same numerical grid on raw and calibrated scales is an operational comparison, not an isolated ranking effect.
- Random diagnostics: expected-test-cost match and separate validation-frozen budget probability; 50,000 seeded assignments each. Assignment ranges and 10,000-draw fixed-policy question-cluster bootstrap intervals have different meanings. Original cross-split question dependence remains.
- Grouped follow-up: 10 repetitions × 5 outer question-group folds; 3 inner calibration folds for cross-fitting. Greedy approximate source balancing. Seed base 20261103. Same 184 cases reused; ranges are split sensitivity, not new independent experiments.
- All follow-up analyses are exploratory because original test results were already inspected. Original primary outputs and benchmark labels are preserved.

## Sources

- Lambda GPU instance pricing, accessed 1 Nov 2026: https://lambda.ai/pricing (H100 PCIe 80GB US$3.29/GPU-hour; taxes additional).
- Runpod central Pods pricing, accessed 1 Nov 2026: https://www.runpod.io/pricing (H100 PCIe US$2.89/hour; sensitivity reference in earlier cost pack).
- OpenAI GPT-5.6 Sol model pricing, accessed 1 Nov 2026: https://developers.openai.com/api/docs/models/gpt-5.6-sol (US$4/M input, US$0.40/M cached input, US$20/M output, published long-context/cache-write modifiers).
- Cawley, G.C. and Talbot, N.L.C. (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation, JMLR 11:2079–2107: https://www.jmlr.org/papers/v11/cawley10a.html.
- scikit-learn, probability calibration: https://scikit-learn.org/stable/modules/calibration.html.
- scikit-learn, grouped cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data.

Methodology references support the general evaluation principles; this offline analysis uses independently tested NumPy implementations, not scikit-learn runtime calls.
