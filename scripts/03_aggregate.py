"""Aggregate multi-run raw results into majority-vote predictions.

Usage:
    python scripts/03_aggregate.py
    python scripts/03_aggregate.py --models llama-8b qwen-7b

Reads per-run JSON files from results/raw/{model}/ and writes
per-trace aggregated JSON files to results/aggregated/{model}/.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.aggregator import aggregate_runs
from src.run_records import load_manifest, existing_runs, signature, save_json
from src.data_loader import load_traces
from src.prompt_builder import build_attribution_prompt
from src.analysis_io import validate_manifest_traces
from src.cost_tracker import compute_usage_cost


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate multi-run results.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(config.MODELS.keys()),
        help=f"Which models to aggregate. Default from config: {config.ACTIVE_MODELS}.",
    )
    args = parser.parse_args()
    config.print_config_banner("STAGE 03: MULTI-RUN AGGREGATION")
    manifest = load_manifest()
    traces = validate_manifest_traces(manifest, load_traces(config.DATA_DIR))

    validity_report = {'manifest_signature': signature(manifest), 'models': {}}
    for model_alias in (args.models or manifest['models']):
        if model_alias not in manifest['models']:
            raise ValueError(f'{model_alias} is absent from the inference manifest.')
        raw_dir = config.RAW_RESULTS_DIR / model_alias
        agg_dir = config.AGGREGATED_RESULTS_DIR / model_alias
        agg_dir.mkdir(parents=True, exist_ok=True)

        if not raw_dir.exists():
            raise ValueError(f'No raw results for {model_alias}; complete stage 2 first.')

        print(f"\n[aggregate] Aggregating {model_alias} ...")
        invalid_responses = invalid_attributions = 0
        for trace in tqdm(traces, desc=model_alias):
            trace_id = trace['trace_id']
            expected = signature({'manifest': signature(manifest), 'model': model_alias,
                                  'prompt': build_attribution_prompt(trace)})
            if len(existing_runs(raw_dir, trace_id, manifest['num_runs'], expected,
                                 trace=trace, model_alias=model_alias)) != manifest['num_runs']:
                raise ValueError(f'Incomplete or stale runs for {model_alias}/{trace_id}; rerun stage 2.')
            runs = [json.loads((raw_dir / f'{Path(trace_id).stem}_run{i}.json').read_text(encoding='utf-8'))
                    for i in range(manifest['num_runs'])]
            settings = manifest['model_configs'][model_alias]
            for run in runs:
                # Recompute each individual request against the frozen price table.
                run['usage_cost_usd'] = compute_usage_cost(settings, run)
            agg = aggregate_runs(runs)
            invalid_responses += agg['invalid_response_count']
            invalid_attributions += int(not agg['attribution_valid'])
            agg["trace_id"] = trace_id
            agg["model_alias"] = model_alias
            agg['manifest_signature'] = signature(manifest)

            out_path = agg_dir / trace_id
            save_json(out_path, agg)

        print(f"[aggregate] Wrote {len(traces)} aggregated files to {agg_dir}")
        completed = len(traces) * manifest['num_runs']
        validity_report['models'][model_alias] = {
            'completed_responses': completed, 'valid_responses': completed - invalid_responses,
            'invalid_responses': invalid_responses, 'invalid_response_rate': invalid_responses / completed,
            'traces': len(traces), 'invalid_final_attributions': invalid_attributions}
        print(f'[aggregate] {model_alias}: {completed} responses; {invalid_responses} invalid responses; '
              f'{invalid_attributions} invalid final attributions.')

    save_json(config.ANALYSIS_DIR / 'response_validity.json', validity_report)

    print("\n[aggregate] All done!")


if __name__ == "__main__":
    main()
