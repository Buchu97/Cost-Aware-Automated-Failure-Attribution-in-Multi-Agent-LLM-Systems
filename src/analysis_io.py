"""Validate the provenance and contents of frozen research artifacts."""
import json
import hashlib
from pathlib import Path
from src.run_records import signature, valid_prediction, valid_resources, finite_number, valid_api_usage

CONFIDENCE_SIGNALS = ('cheap_consistency', 'cheap_self_conf', 'cheap_logprob_conf')


def validate_manifest_traces(manifest: dict, traces: list[dict]) -> list[dict]:
    ids = manifest.get('trace_ids', [])
    by_id = {t['trace_id']: t for t in traces}
    if not ids or len(ids) != len(set(ids)) or len(by_id) != len(traces):
        raise ValueError('Manifest or dataset has empty/duplicate trace IDs.')
    for tid in ids:
        if tid not in by_id or signature(by_id[tid]) != manifest.get('trace_hashes', {}).get(tid):
            raise ValueError(f'Dataset changed since inference: {tid}; preserve results and use a new experiment.')
    # This frozen order also makes the split reproducible after sampling.
    return [by_id[tid] for tid in ids]


def manifest_pairs(manifest: dict) -> list[tuple[str, str]]:
    pairs = [tuple(pair) for pair in manifest.get('cascade_pairs', [])]
    if len(pairs) != len(set(pairs)) or any(len(p) != 2 or any(a not in manifest['models'] for a in p) for p in pairs):
        raise ValueError('Invalid cascade pairs in manifest.')
    return pairs


def load_aggregated(model_alias: str, manifest: dict, traces: list[dict], root: Path = None) -> dict:
    if root is None:
        import config
        root = config.AGGREGATED_RESULTS_DIR
    if model_alias not in manifest['models']:
        raise ValueError(f'Model absent from manifest: {model_alias}')
    results = {}
    for trace in traces:
        tid = trace['trace_id']
        path = Path(root) / model_alias / tid
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Missing or corrupt aggregate: {path}; rerun stage 3.') from exc
        valid = (data.get('manifest_signature') == signature(manifest)
                 and data.get('trace_id') == tid and data.get('model_alias') == model_alias
                 and type(data.get('n_runs')) is int and data['n_runs'] == manifest['num_runs']
                 and finite_number(data.get('parse_success_rate'), maximum=1)
                 and type(data.get('invalid_response_count')) is int
                 and 0 <= data['invalid_response_count'] <= manifest['num_runs']
                 and data['parse_success_rate'] == round(
                     1 - data['invalid_response_count'] / manifest['num_runs'], 4)
                 and type(data.get('attribution_valid')) is bool
                 and ((data['attribution_valid'] and valid_prediction(
                     data.get('majority_agent'), data.get('majority_step'), trace))
                     or (not data['attribution_valid'] and data.get('majority_agent') is None
                         and data.get('majority_step') is None and data['invalid_response_count'] > 0))
                 and valid_resources(data, 'total_elapsed')
                 and (manifest['model_configs'][model_alias].get('provider') != 'openai'
                      or valid_api_usage(data, aggregated=True))
                 and all(finite_number(data.get(k), maximum=1) for k in
                         ('consistency_confidence', 'agent_agreement', 'step_agreement', 'joint_agreement'))
                 and all(data.get(k) is None or finite_number(data[k], maximum=1) for k in
                         ('mean_self_confidence', 'mean_logprob_confidence')))
        if not valid:
            raise ValueError(f'Stale or invalid aggregate: {path}; rerun stage 3.')
        results[tid] = data
    return results


def calibration_input_signature(cal: dict, aggregates: dict) -> str:
    return signature({a: {tid: aggregates[a][tid] for tid in cal['val_trace_ids']}
                      for a in (cal['cheap_model'], cal['strong_model'])})


def validate_calibration(records: list[dict], manifest: dict, traces: list[dict], aggregates=None) -> None:
    from src.data_loader import split_val_test
    selected = validate_manifest_traces(manifest, traces)
    val, test = split_val_test(selected, manifest['val_ratio'], manifest['seed'])
    expected_pairs = {f'{c}_to_{s}': (c, s) for c, s in manifest_pairs(manifest)}
    if not isinstance(records, list) or len(records) != len(expected_pairs):
        raise ValueError('Incomplete calibration results; rerun stage 4.')
    seen = set()
    for cal in records:
        pair = cal.get('pair')
        if (pair not in expected_pairs or pair in seen
                or (cal.get('cheap_model'), cal.get('strong_model')) != expected_pairs[pair]
                or cal.get('manifest_signature') != signature(manifest)
                or cal.get('val_trace_ids') != [t['trace_id'] for t in val]
                or cal.get('test_trace_ids') != [t['trace_id'] for t in test]
                or cal.get('confidence_signal') not in CONFIDENCE_SIGNALS
                or not finite_number(cal.get('threshold'), maximum=1.000001)
                or cal['threshold'] not in manifest['threshold_grid']):
            raise ValueError('Calibration settings/split do not match the manifest; rerun stage 4.')
        seen.add(pair)
        if aggregates is not None and cal.get('validation_aggregate_signature') != calibration_input_signature(cal, aggregates):
            raise ValueError('Validation aggregates changed since calibration; rerun stage 4.')
        knots = cal.get('calibrator', {})
        x, y = knots.get('x', []), knots.get('y', [])
        if (not x or len(x) != len(y)
                or not all(finite_number(v, maximum=1) for v in x + y)
                or any(b <= a for a, b in zip(x, x[1:]))
                or any(b < a for a, b in zip(y, y[1:]))):
            raise ValueError('Invalid saved isotonic calibration knots; rerun stage 4.')
        accuracy = cal.get('strong_validation_accuracy')
        if (not finite_number(accuracy, maximum=1)
                or cal.get('minimum_validation_accuracy') != accuracy * manifest['min_accuracy_retention']):
            raise ValueError('Calibration accuracy target does not match the manifest.')


def evaluation_provenance(manifest: dict, calibration: list[dict], aggregates: dict) -> dict:
    """Fingerprint plot inputs, tying figures to the evaluated artifact contents."""
    import config
    paths = [config.TABLES_DIR / 'evaluation_results.csv'] + [
        config.ANALYSIS_DIR / f"sweep_{c['pair']}_{c['confidence_signal']}.csv" for c in calibration]
    try:
        digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    except OSError as exc:
        raise ValueError('Missing evaluation/sweep CSV; complete stages 4 and 5 first.') from exc
    return {'manifest_signature': signature(manifest), 'calibration_signature': signature(calibration),
            'aggregate_signatures': {a: signature(data) for a, data in aggregates.items()}, 'csv_sha256': digests}
