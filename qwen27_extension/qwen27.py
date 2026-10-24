"""Run Qwen3.8-27B BF16 and combine verified, previously completed model results."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

ALIAS = 'qwen-27b'
MODEL_ID = 'Qwen/Qwen3.8-27B'
EXPECTED_HASHES = {
    'config.py': '24B2A779A277FD3A8B025EEACF1D22056730F6FC98B1754E5644647875141DD9',
    'src/model_loader.py': 'D9B64C15ACCBB18BBCDB10AD3437AFA7A6C83514651D3F14930A6E563AB877EA',
    'src/inference.py': '957291D7C52536DE8265EF9C036F5B36473A6F260EF671B284C2E1C466A3FC80',
    'scripts/02_run_inference.py': '84F872746365F2645D3FCDDF83B35805B54833CF1A5EACFA919793197D9880D3',
}
SHARED = ('trace_ids', 'trace_hashes', 'prompt_hashes', 'num_runs', 'temperature',
          'seed', 'val_ratio', 'top_p', 'top_k', 'threshold_grid', 'min_accuracy_retention')


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def frozen_write(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Settings changed; preserve the existing file: {path}')
    else:
        write_json(path, value)


def safe_destination(destination, original):
    destination, original = Path(destination).resolve(), Path(original).resolve()
    if (destination == original or destination.is_relative_to(original)
            or original.is_relative_to(destination)):
        raise ValueError('New outputs must be separate from the completed results directory.')
    return destination


def snapshot(root):
    return {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(root).rglob('*')) if path.is_file()}


def assert_snapshot(root, expected):
    if snapshot(root) != expected:
        raise ValueError('The completed source results changed. Stop and inspect concurrent processes.')


def new_manifest(original, revision, versions):
    settings = {
        'provider': 'huggingface', 'hf_id': MODEL_ID, 'revision': revision,
        'quantisation': None, 'dtype': 'bfloat16', 'context_window': 262144,
        'max_new_tokens': 512, 'enable_thinking': False, 'preserve_thinking': False,
        'architecture': 'Qwen3_5ForConditionalGeneration', 'attention': 'sdpa',
        'attention_policy': 'efficient_sdpa_explicit_gqa',
        'cost_basis': 'illustrative API-equivalent local cost; same frozen rate as qwen-72b; not a market quote or GPU bill',
        'cost_reference': 'source_manifest:model_configs:qwen-72b',
    }
    for key in ('theoretical_cost_per_1m_input', 'theoretical_cost_per_1m_output'):
        settings[key] = original['model_configs']['qwen-72b'][key]
    result = {key: copy.deepcopy(original[key]) for key in SHARED}
    result.update(schema_version=2, extension_version=2, profile='qwen27_bf16',
                  models=[ALIAS], model_configs={ALIAS: settings}, cascade_pairs=[],
                  resolved_revisions={ALIAS: revision}, runtime_versions=versions,
                  source_manifest_signature=signature(original))
    return result


def combined_manifest(original, new):
    if new.get('source_manifest_signature') != signature(original):
        raise ValueError('Qwen results belong to a different original experiment.')
    for key in SHARED:
        if original[key] != new[key]:
            raise ValueError(f'Original and Qwen experiments differ: {key}')
    result = copy.deepcopy(original)
    result['profile'] = 'full_run_six_models'
    result['models'] = list(original['models']) + [ALIAS]
    result['model_configs'][ALIAS] = copy.deepcopy(new['model_configs'][ALIAS])
    result['resolved_revisions'][ALIAS] = new['resolved_revisions'][ALIAS]
    result['cascade_pairs'] += [['llama-8b', ALIAS], ['qwen-7b', ALIAS]]
    result['source_manifests'] = {'original': signature(original), 'qwen': signature(new)}
    result['aggregation_basis'] = 'Verified source aggregates copied with source content digests; raw responses remain in their original directories.'
    return result


def rebased_aggregate(original, combined_signature):
    result = copy.deepcopy(original)
    result['source_manifest_signature'] = original['manifest_signature']
    result['source_aggregate_signature'] = signature(original)
    result['manifest_signature'] = combined_signature
    return result


class NonThinkingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.update(enable_thinking=False, preserve_thinking=False)
        return self.tokenizer.apply_chat_template(*args, **kwargs)


@contextmanager
def exclusive_run(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'qwen27.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def project_context(root, baseline):
    for name, expected in EXPECTED_HASHES.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().upper() != expected:
            raise ValueError(f'Unexpected project version: {name}. No source files were changed.')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    os.environ['EXPERIMENT_PROFILE'] = 'full_run'
    os.environ['EXPERIMENT_RESULTS_DIR'] = str(baseline)
    import config
    from src.analysis_io import validate_manifest_traces
    from src.data_loader import load_traces, get_split_indices
    from src.prompt_builder import build_attribution_prompt
    original = read_json(baseline / 'experiment_manifest.json')
    expected_models = {'llama-8b', 'qwen-7b', 'llama-70b', 'qwen-72b', 'gpt-5.6-sol'}
    if (set(original['models']) != expected_models or len(original['trace_ids']) != 184
            or original['num_runs'] != 3 or original.get('schema_version') != 2):
        raise ValueError('Expected the completed five-model, 184-trace, three-run experiment.')
    traces = validate_manifest_traces(original, load_traces(config.DATA_DIR))
    prompts = {t['trace_id']: build_attribution_prompt(t) for t in traces}
    if {tid: signature(prompt) for tid, prompt in prompts.items()} != original['prompt_hashes']:
        raise ValueError('Task prompts changed since the completed experiment.')
    splits = get_split_indices(traces, original['val_ratio'], original['seed'])
    previous = read_json(baseline / 'analysis/calibration_results.json')
    for record in previous:
        if (record['val_trace_ids'] != splits['val'] or record['test_trace_ids'] != splits['test']):
            raise ValueError('Validation/test membership differs from the completed analysis.')
    if len(splits['val']) != 110 or len(splits['test']) != 74:
        raise ValueError('Expected 110 validation traces and 74 test traces.')
    config.TOP_P, config.TOP_K = original['top_p'], original['top_k']
    return config, original, traces, prompts, splits


def verify_original_aggregates(original, traces, baseline):
    from src.analysis_io import load_aggregated
    return {alias: load_aggregated(alias, original, traces, baseline / 'aggregated')
            for alias in original['models']}


def runtime_versions():
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'accelerate', 'huggingface-hub', 'tokenizers')}
    if versions['transformers'] != '5.20.1':
        raise ValueError('Use the isolated environment with transformers==5.20.1; do not upgrade the original .venv.')
    if not versions['torch'].startswith('2.7.1'):
        raise ValueError('The isolated environment must retain torch 2.7.1 CUDA 12.6.')
    from transformers import Qwen3_5ForConditionalGeneration  # architecture check before weight downloads
    return versions


def prepare(original, destination):
    versions = runtime_versions()
    manifest_path = destination / 'experiment_manifest.json'
    if manifest_path.exists():
        saved = read_json(manifest_path)
        manifest = new_manifest(original, saved['resolved_revisions'][ALIAS], versions)
    elif (destination.parent / 'qwen27-bf16/experiment_manifest.json').exists():
        previous = read_json(destination.parent / 'qwen27-bf16/experiment_manifest.json')
        if previous.get('source_manifest_signature') != signature(original):
            raise ValueError('The previous Qwen attempt belongs to a different baseline.')
        revision = previous['resolved_revisions'][ALIAS]
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError('The previous Qwen checkpoint revision is invalid.')
        manifest = new_manifest(original, revision, versions)
        print('[checkpoint] Reusing the previous immutable Qwen revision and weight cache.', flush=True)
    else:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(MODEL_ID).sha
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError('Could not resolve an immutable model revision.')
        manifest = new_manifest(original, revision, versions)
    frozen_write(manifest_path, manifest)
    return manifest


def rendered_inputs(manifest, traces, prompts, cache):
    from transformers import AutoTokenizer
    settings = manifest['model_configs'][ALIAS]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=settings['revision'],
                                             cache_dir=str(cache), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = NonThinkingTokenizer(tokenizer)
    lengths = {}
    for trace in traces:
        tid = trace['trace_id']
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompts[tid]}],
                                                tokenize=False, add_generation_prompt=True)
        lengths[tid] = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
        if lengths[tid] + settings['max_new_tokens'] > settings['context_window']:
            raise ValueError(f'{tid} exceeds the model context. No input was truncated.')
    print(f'[context] {len(lengths)} complete traces checked; largest input={max(lengths.values()):,} tokens.', flush=True)
    return tokenizer, lengths


def response_signature(manifest, prompt):
    return signature({'manifest': signature(manifest), 'model': ALIAS, 'prompt': prompt})


def run_seed(manifest, trace_id, index, *, pilot=False):
    fields = {'seed': manifest['seed'], 'model': ALIAS, 'trace': trace_id, 'run': index}
    if pilot:
        fields['pilot'] = True
    return int(signature(fields)[:8], 16) % (2**31)


def completed_indices(raw, trace, prompt, manifest):
    from src.run_records import existing_runs
    return existing_runs(raw, trace['trace_id'], manifest['num_runs'],
                         response_signature(manifest, prompt), trace=trace, model_alias=ALIAS)


def expand_gqa_heads(query, key, value):
    heads, kv_heads = query.shape[-3], key.shape[-3]
    if kv_heads < 1 or kv_heads != value.shape[-3] or heads % kv_heads:
        raise ValueError('Invalid grouped-query head counts.')
    if heads != kv_heads:
        key = key.repeat_interleave(heads // kv_heads, dim=-3)
        value = value.repeat_interleave(heads // kv_heads, dim=-3)
    return key, value


@contextmanager
def qwen_attention():
    import torch.nn.functional as functional
    from torch.nn.attention import sdpa_kernel, SDPBackend
    original = functional.scaled_dot_product_attention

    def attention(query, key, value, *args, **kwargs):
        if query.device.type != 'cuda':
            return original(query, key, value, *args, **kwargs)
        key, value = expand_gqa_heads(query, key, value)
        kwargs['enable_gqa'] = False
        backends = [SDPBackend.EFFICIENT_ATTENTION]
        if query.shape[-2] == 1:
            backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends, set_priority=True):
            return original(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = attention
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


def execute_inference(config, manifest, selected, prompts, tokenizer, output, cache, *, pilot=False):
    if not config.PHYSICAL_GPU_UUID:
        raise RuntimeError('Cannot verify the physical GPU 0 UUID. Restore nvidia-smi access before model loading; ordinal fallback is not permitted.')
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from src.runtime import validate_gpu
    from src.inference import run_single_inference, save_run_result
    import src.inference as inference_engine
    from src.agent_names import canonical_agent
    from src.run_records import valid_prediction
    from src.progress import heartbeat
    validate_gpu(torch, config.PHYSICAL_GPU_UUID)
    raw = output / ('pilot/raw' if pilot else 'raw') / ALIAS
    n = 1 if pilot else manifest['num_runs']
    expected_manifest = dict(manifest, num_runs=n, run_phase='pilot') if pilot else manifest
    total, finished = len(selected) * n, 0
    for trace in selected:
        finished += len(completed_indices(raw, trace, prompts[trace['trace_id']], expected_manifest))
    if finished == total:
        print(f'[resume] {finished}/{total} received responses verified; no model load needed.', flush=True)
        return
    free, _ = torch.cuda.mem_get_info(0)
    if free < 65 * 1024**3:
        raise RuntimeError(f'Only {free / 1024**3:.1f} GiB free on physical GPU 0; need at least 65 GiB for the BF16 pilot. No CPU offload or quantisation will be substituted.')
    settings = manifest['model_configs'][ALIAS]
    print(f'[load] {MODEL_ID} | BF16 | physical GPU 0 | thinking disabled', flush=True)
    model = None
    try:
        with heartbeat('Load Qwen-27B BF16 onto physical GPU 0'):
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL_ID, revision=settings['revision'], cache_dir=str(cache),
                device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa',
                low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False)
        model.eval()
        model.generation_config.disable_compile = True
        if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
            raise RuntimeError('Expected BF16 weights, not a quantised model.')
        if any(parameter.device.type != 'cuda' or parameter.device.index != 0 for parameter in model.parameters()):
            raise RuntimeError('All model parameters must remain on physical GPU 0.')
        dtype_counts = {}
        for parameter in model.parameters():
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
        if not dtype_counts.get('torch.bfloat16'):
            raise RuntimeError('No BF16 model parameters were loaded.')
        weight_memory = round(model.get_memory_footprint() / 1024**3, 3)
        invalid = 0
        for trace in selected:
            tid = trace['trace_id']
            done = completed_indices(raw, trace, prompts[tid], expected_manifest)
            for index in range(n):
                path = raw / f'{Path(tid).stem}_run{index}.json'
                if index in done:
                    continue
                if path.exists():
                    raise ValueError(f'Incomplete or stale response already exists; inspect before retrying: {path}')
                seed = run_seed(manifest, tid, index, pilot=pilot)
                torch.cuda.reset_peak_memory_stats(0)
                with heartbeat(f'{ALIAS}/{tid} sample {index + 1}/{n}'):
                    previous_attention = inference_engine.h100_attention
                    inference_engine.h100_attention = qwen_attention
                    try:
                        result = run_single_inference(model, tokenizer, prompts[tid],
                            max_new_tokens=settings['max_new_tokens'], temperature=manifest['temperature'],
                            seed=seed, context_window=settings['context_window'])
                    finally:
                        inference_engine.h100_attention = previous_attention
                result['predicted_agent'] = canonical_agent(result.get('predicted_agent'), trace)
                result['parse_success'] = bool(result.get('parse_success') and valid_prediction(
                    result['predicted_agent'], result.get('predicted_step'), trace))
                result.update(model_alias=ALIAS, trace_id=tid, run_index=index, schema_version=2,
                    provider='huggingface', model_revision=settings['revision'], dtype='bfloat16',
                    enable_thinking=False, quantisation=None,
                    attention_policy='efficient_sdpa_explicit_gqa',
                    peak_memory_allocated_gib=round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
                    peak_memory_reserved_gib=round(torch.cuda.max_memory_reserved(0) / 1024**3, 3),
                    model_weight_memory_gib=weight_memory, parameter_dtype_counts=dtype_counts,
                    run_signature=response_signature(expected_manifest, prompts[tid]))
                save_run_result(result, output / ('pilot/raw' if pilot else 'raw'), ALIAS, tid, index)
                invalid += int(not result['parse_success'])
                finished += 1
                if pilot or finished % 10 == 0 or finished == total:
                    print(f'[progress] {ALIAS} {finished}/{total} received | invalid this session={invalid} | peak={result["peak_memory_allocated_gib"]:.1f} GiB | {result["elapsed_seconds"]:.1f}s', flush=True)
    finally:
        model = None
        from src.model_loader import release_cuda_memory
        release_cuda_memory()


def verify_new_raw(manifest, traces, prompts, destination):
    verified = {}
    for trace in traces:
        tid = trace['trace_id']
        indices = completed_indices(destination / 'raw' / ALIAS, trace, prompts[tid], manifest)
        if indices != set(range(manifest['num_runs'])):
            raise ValueError(f'Qwen inference is incomplete for {tid}. Resume the new run; leave original results unchanged.')
        verified[tid] = [read_json(destination / 'raw' / ALIAS / f'{Path(tid).stem}_run{i}.json')
                         for i in range(manifest['num_runs'])]
    return verified


def analyse(config, original, traces, prompts, baseline, destination, combined):
    from src.aggregator import aggregate_runs
    from src.cost_tracker import compute_usage_cost
    from src.analysis_io import load_aggregated
    new = read_json(destination / 'experiment_manifest.json')
    full = combined_manifest(original, new)
    originals = verify_original_aggregates(original, traces, baseline)
    raw = verify_new_raw(new, traces, prompts, destination)
    frozen_write(combined / 'experiment_manifest.json', full)
    for alias, records in originals.items():
        for tid, record in records.items():
            frozen_write(combined / 'aggregated' / alias / tid, rebased_aggregate(record, signature(full)))
    for tid, records in raw.items():
        priced = [dict(record, usage_cost_usd=compute_usage_cost(new['model_configs'][ALIAS], record))
                  for record in records]
        aggregate = aggregate_runs(priced)
        aggregate.update(trace_id=tid, model_alias=ALIAS, manifest_signature=signature(new))
        frozen_write(destination / 'aggregated' / ALIAS / tid, aggregate)
        frozen_write(combined / 'aggregated' / ALIAS / tid, rebased_aggregate(aggregate, signature(full)))
    # Validate copied aggregates under the combined manifest before producing any comparisons.
    for alias in full['models']:
        load_aggregated(alias, full, traces, combined / 'aggregated')
    config.PROFILE, config.ACTIVE_MODELS, config.MODELS = full['profile'], full['models'], full['model_configs']
    config.RESULTS_DIR, config.RAW_RESULTS_DIR = combined, combined / 'raw'
    config.AGGREGATED_RESULTS_DIR, config.ANALYSIS_DIR = combined / 'aggregated', combined / 'analysis'
    config.TABLES_DIR, config.FIGURES_DIR = config.ANALYSIS_DIR / 'tables', config.ANALYSIS_DIR / 'figures'
    config.REPORTS_DIR = config.ANALYSIS_DIR / 'reports'
    config.ALL_CASCADE_PAIRS = config.CASCADE_PAIRS = [tuple(pair) for pair in full['cascade_pairs']]
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for script in ('04_calibrate_cascade.py', '05_evaluate.py', '06_generate_figures.py'):
        sys.argv = [str(config.PROJECT_ROOT / 'scripts' / script)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    report = config.REPORTS_DIR / 'summary_report.md'
    if report.exists():
        text = report.read_text(encoding='utf-8')
        old = 'Qwen uses a fixed factor-4 YaRN extension with original context 32768 and target context 131072, where Qwen is part of this experiment.'
        replacement = ('The original Qwen2.5-7B and Qwen2.5-72B runs used factor-4 YaRN '
                       '(32768 to 131072). Qwen3.8-27B used its native 262144-token context, '
                       'BF16 weights, and thinking disabled. Its dollar costs reuse the original '
                       'qwen-72b illustrative token rates; these are not a model-specific market quote '
                       'or measured GPU charges. Original predictions were reused without new inference.')
        report.write_text(text.replace(old, replacement), encoding='utf-8')
    write_json(config.ANALYSIS_DIR / 'model_extension_provenance.json', {
        'original_directory': str(baseline), 'qwen_directory': str(destination),
        'source_manifests': full['source_manifests'],
        'original_models_rerun': False, 'paid_api_calls': 0,
        'qwen_invalid_responses': sum(not r['parse_success'] for rows in raw.values() for r in rows),
        'cost_assumption': new['model_configs'][ALIAS]['cost_basis'],
    })
    print(f'[complete] Six-model analysis saved separately: {combined}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit', 'check', 'pilot', 'run', 'analyse'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = root / 'results/full-run-all-5-models'
    destination = safe_destination(root / 'results/qwen27-bf16-gqa', baseline)
    combined = safe_destination(root / 'results/full-run-six-models-qwen27-gqa', baseline)
    before = snapshot(baseline)
    if not before:
        raise ValueError('Completed baseline results are missing; do not start a replacement experiment.')
    config, original, traces, prompts, splits = project_context(root, baseline)
    frozen_write(destination / 'original_results_sha256.json', before)
    verify_original_aggregates(original, traces, baseline)
    print('[baseline] Five completed models verified; 184 traces, 110 validation / 74 test, 3 runs.', flush=True)
    try:
        if args.mode == 'audit':
            print('[audit] Source hashes, original prompts, split and saved aggregates verified. No inference.', flush=True)
            return
        with exclusive_run(destination):
            if args.mode == 'analyse':
                analyse(config, original, traces, prompts, baseline, destination, combined)
                return
            manifest = prepare(original, destination)
            if args.mode == 'run':
                marker = read_json(destination / 'pilot_pass.json')
                if marker.get('manifest_signature') != signature(manifest):
                    raise ValueError('The successful validation pilot does not match these settings.')
                if all(len(completed_indices(destination / 'raw' / ALIAS, trace,
                           prompts[trace['trace_id']], manifest)) == 3 for trace in traces):
                    print('[complete] All 552 Qwen responses already present; no model load or new inference.', flush=True)
                    return
            cache = root / 'model_cache/qwen27-bf16'
            tokenizer, lengths = rendered_inputs(manifest, traces, prompts, cache)
            frozen_write(destination / 'token_lengths.json', lengths)
            if args.mode == 'check':
                print('[check] Architecture, immutable revision and full input lengths checked. Weights not loaded.', flush=True)
                return
            if args.mode == 'pilot':
                by_id = {t['trace_id']: t for t in traces}
                ordered = sorted(splits['val'], key=lambda tid: lengths[tid])
                chosen = [by_id[ordered[i]] for i in (0, len(ordered) // 2, -1)]
                print('[pilot] Short, median and longest validation inputs; no test predictions used.', flush=True)
                execute_inference(config, manifest, chosen, prompts, tokenizer, destination, cache, pilot=True)
                for trace in chosen:
                    record = read_json(destination / 'pilot/raw' / ALIAS / f'{Path(trace["trace_id"]).stem}_run0.json')
                    if not record['parse_success']:
                        raise RuntimeError('Pilot completed with invalid agent/step output. Inspect the saved response before the full run.')
                write_json(destination / 'pilot_pass.json', {
                    'manifest_signature': signature(manifest), 'trace_ids': [t['trace_id'] for t in chosen],
                    'criterion': 'GPU execution and parsable in-range attribution; not correctness against ground truth',
                })
                print('[pilot] Execution and output-format checks passed. Pilot responses are separate from the 552 formal responses.', flush=True)
            else:
                execute_inference(config, manifest, traces, prompts, tokenizer, destination, cache)
                raw = verify_new_raw(manifest, traces, prompts, destination)
                invalid = sum(not r['parse_success'] for rows in raw.values() for r in rows)
                print(f'[complete] Qwen 552/552 received responses; {invalid} invalid answers retained. No original model was rerun.', flush=True)
    finally:
        assert_snapshot(baseline, before)
        print('[preserved] Original results and API ledger are unchanged.', flush=True)


if __name__ == '__main__':
    main()
