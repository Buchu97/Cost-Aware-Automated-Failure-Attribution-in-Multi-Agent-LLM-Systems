# Cost-Aware Automated Failure Attribution

This repository contains the inference code and results for FOSE7901 Placement 2.
The study tests whether a cheaper judge can review a multi-agent failure trace first
and defer to a stronger judge while retaining at least 95% of strong-only joint accuracy.

## Results

- [Original test comparison](outputs/report_readiness_20261103/primary_results.csv) contains all eight cascades and their strong-only baselines.
- [Standalone results](outputs/report_readiness_20261103/standalone_baselines.csv) give joint accuracy and estimated generation cost for each judge.
- [Random routing control](outputs/report_readiness_20261103/random_routing_controls.csv) compares escalation at the same expected cost.
- [Question-grouped evaluation](outputs/report_readiness_20261103/grouped_cv_summary.csv) gives the exploratory results when related questions stay in the same fold.
- [Figures](figures/) contains the four figures used in the report. The [cost correction](outputs/cost_review_20261101/cost_correction.md) explains the generation-cost estimates used in the paper.

The original split has 110 validation and 74 test traces. The random-routing and
question-grouped comparisons are exploratory follow-ups on the saved outputs.

## Code and records

The `src/`, `scripts/` and `qwen27_extension/` directories hold the inference code
from the run archive. The [experiment manifest](provenance/experiment_manifest.json)
and [split](provenance/data_split.json) record the models, settings and trace IDs.
The offline analysis is in `analysis/report_readiness_20261103/` and uses the
sourced cost model in `outputs/cost_review_20261101/`. The
[analysis protocol](outputs/report_readiness_20261103/analysis_protocol.md)
describes the fitting, selection and evaluation procedures.

The [Who&When repository](https://github.com/mingyin1/Agents_Failure_Attribution)
released by [Zhang et al. (2025)](https://proceedings.mlr.press/v267/zhang25cq.html),
provides the benchmark and its original evaluation code. This project repository
contains derived outcomes and trace IDs. The full source export, including
benchmark traces and raw model responses, is a separate placement record.

## Offline reproduction

The analysis expects the original run export at
`records/six-model-analysis-20261024-145231.zip` and verifies its SHA-256.

```text
70ea712be64e8625bb0638635e09036c091e9122ff56cf98e350dbe69749bfe4
```

Commands for Python 3.12 from the repository root

```sh
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s analysis/report_readiness_20261103 -p test_engine.py -v
python analysis/report_readiness_20261103/run_analysis.py
```

This reruns the analysis on saved responses and writes CSVs under
`outputs/report_readiness_20261103/`. It runs on a CPU without model weights or
API calls. The local `records/` folder is excluded from Git.
