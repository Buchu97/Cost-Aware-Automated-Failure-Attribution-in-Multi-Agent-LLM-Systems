"""Atomic JSON records, stable settings signatures and experiment manifests."""
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile

SCHEMA_VERSION = 2


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def signature(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def finite_number(value, minimum=0, maximum=None) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum))


def valid_prediction(agent, step, trace=None) -> bool:
    if not isinstance(agent, str) or not agent.strip() or isinstance(step, bool):
        return False
    if not (isinstance(step, int) or isinstance(step, str) and step.isdecimal()):
        return False
    if int(step) < 0:
        return False
    if trace is not None:
        from src.agent_names import allowed_agents
        return agent in allowed_agents(trace) and int(step) < len(trace['messages'])
    return True


def valid_resources(data, elapsed_key='elapsed_seconds') -> bool:
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    return (all(type(data.get(k)) is int and data[k] >= 0 for k in keys)
            and data['prompt_tokens'] > 0 and data['completion_tokens'] > 0
            and data['total_tokens'] == data['prompt_tokens'] + data['completion_tokens']
            and finite_number(data.get(elapsed_key)))


def valid_api_usage(data, *, require_cost=True, aggregated=False) -> bool:
    keys = ('cached_prompt_tokens', 'cache_write_prompt_tokens', 'reasoning_tokens')
    if not (data.get('provider') == 'openai'
            and all(type(data.get(k)) is int and data[k] >= 0 for k in keys)
            and data['cached_prompt_tokens'] + data['cache_write_prompt_tokens'] <= data.get('prompt_tokens', -1)
            and data['reasoning_tokens'] <= data.get('completion_tokens', -1)
            and (not require_cost or finite_number(data.get('usage_cost_usd')))):
        return False
    return aggregated or (data.get('requested_model') == 'gpt-5.6-sol'
            and isinstance(data.get('response_model'), str)
            and (data['response_model'] == 'gpt-5.6-sol' or data['response_model'].startswith('gpt-5.6-sol-'))
            and isinstance(data.get('response_id'), str) and bool(data['response_id'])
            and data.get('service_tier') == 'default')


def completed_response_record(data) -> bool:
    """Separate a received invalid model answer from a partial/service failure."""
    if data.get('parse_success') is True:
        return (data.get('provider') != 'openai'
                or data.get('response_status', 'completed') == 'completed')
    if data.get('parse_success') is not False:
        return False
    if (not isinstance(data.get('generated_text'), str)
            or 'predicted_agent' not in data or 'predicted_step' not in data
            or not (data['predicted_agent'] is None or isinstance(data['predicted_agent'], str))
            or not (data['predicted_step'] is None or type(data['predicted_step']) in (str, int))):
        return False
    # Output-limit exhaustion is an observed failed generation with known usage.
    # API failed/cancelled states remain infrastructure failures, not abstentions.
    return (data.get('provider') != 'openai'
            or data.get('response_status') in ('completed', 'incomplete'))


def existing_runs(model_dir: Path, trace_id: str, n_runs: int, run_signature: str,
                  *, trace=None, model_alias=None) -> set[int]:
    found = set()
    for index in range(n_runs):
        path = Path(model_dir) / f"{Path(trace_id).stem}_run{index}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("schema_version") == SCHEMA_VERSION and data.get("trace_id") == trace_id
                    and type(data.get('run_index')) is int and data['run_index'] == index
                    and data.get("run_signature") == run_signature
                    and completed_response_record(data)
                    and (model_alias is None or data.get('model_alias') == model_alias)
                    and (not data['parse_success'] or valid_prediction(
                        data.get('predicted_agent'), data.get('predicted_step'), trace))
                    and valid_resources(data)
                    and (model_alias != 'gpt-5.6-sol' or valid_api_usage(data))
                    and all(key in data and (data[key] is None or finite_number(data[key], maximum=maximum))
                            for key, maximum in [('self_reported_confidence', 100), ('logprob_confidence', 1)])):
                found.add(index)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return found


def ensure_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(f"Experiment settings changed. Choose a new EXPERIMENT_RESULTS_DIR; preserve {path}.")
    else:
        save_json(path, manifest)


def load_manifest() -> dict:
    import config
    path = config.RESULTS_DIR / "experiment_manifest.json"
    if not path.exists():
        raise ValueError("No experiment manifest. Complete stage 2 first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Legacy inference records cannot be used; rerun stage 2 into a new results directory.")
    required = {'models', 'model_configs', 'trace_ids', 'trace_hashes', 'num_runs', 'seed', 'val_ratio',
                'cascade_pairs', 'threshold_grid', 'min_accuracy_retention'}
    if not required.issubset(data):
        raise ValueError('Incomplete experiment manifest; rerun stage 2 into a new results directory.')
    if (not data['models'] or len(data['models']) != len(set(data['models']))
            or type(data['num_runs']) is not int or data['num_runs'] < 1
            or not finite_number(data['val_ratio']) or not 0 < data['val_ratio'] < 1
            or not finite_number(data['min_accuracy_retention'], maximum=1)
            or not data['threshold_grid']
            or not all(finite_number(t, maximum=1.000001) for t in data['threshold_grid'])
            or 1.000001 not in data['threshold_grid']):
        raise ValueError('Invalid frozen experiment settings in manifest.')
    for alias in data['models']:
        rates = data['model_configs'].get(alias, {})
        if not all(finite_number(rates.get(k)) for k in
                   ('theoretical_cost_per_1m_input', 'theoretical_cost_per_1m_output')):
            raise ValueError(f'Invalid frozen cost rates: {alias}')
        if rates.get('provider') == 'openai' and not all(finite_number(rates.get(k)) for k in
                ('theoretical_cost_per_1m_cached_input', 'cache_write_multiplier',
                 'long_context_threshold', 'long_context_input_multiplier', 'long_context_output_multiplier')):
            raise ValueError(f'Invalid frozen API pricing: {alias}')
    return data
