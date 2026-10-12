"""Sequential local inference on physical H100 GPU 0 and OpenAI API inference."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # GPU mask established before any torch/transformers import
from src.data_loader import load_traces, sample_traces
from src.prompt_builder import build_attribution_prompt
from src.model_loader import ModelSession, load_tokenizer, delete_model_from_disk
from src.inference import run_single_inference, save_run_result
from src.run_records import SCHEMA_VERSION, signature, existing_runs, ensure_manifest
from src.agent_names import canonical_agent
from src.run_records import valid_prediction, completed_response_record
from src.openai_inference import check_api_environment, run_openai_inference, update_ledger_result
from src.progress import InferenceProgress, heartbeat


def saved_invalid_count(raw_dir, trace_runs):
    return sum(json.loads((raw_dir / f'{Path(tid).stem}_run{i}.json').read_text(
        encoding='utf-8'))['parse_success'] is False
        for tid, indices in trace_runs.items() for i in indices)


def run_api_model(alias, settings, pending, prompts, signatures, num_runs, budget, progress=None):
    raw_dir = config.RAW_RESULTS_DIR / alias
    if progress is None:
        saved = {t['trace_id']: existing_runs(raw_dir, t['trace_id'], num_runs, signatures[t['trace_id']],
                                            trace=t, model_alias=alias) for t in pending}
        progress = InferenceProgress(alias, len(pending) * num_runs, sum(map(len, saved.values())),
                                     saved_invalid=saved_invalid_count(raw_dir, saved))
    for trace in pending:
        tid = trace['trace_id']
        done = existing_runs(raw_dir, tid, num_runs, signatures[tid], trace=trace, model_alias=alias)
        for index in range(num_runs):
            if index in done:
                continue
            progress.begin(tid, index, num_runs)
            try:
                with heartbeat(f'{alias}/{tid} sample {index + 1}/{num_runs}: waiting for API response'):
                    result = run_openai_inference(prompts[tid], settings, config.RESULTS_DIR / 'api_ledger', budget,
                        request_metadata={'model_alias': alias, 'trace_id': tid, 'run_index': index,
                                          'run_signature': signatures[tid]})
            except Exception as exc:
                progress.fail(str(exc))
                progress.summary()
                raise
            result['predicted_agent'] = canonical_agent(result.get('predicted_agent'), trace)
            result['parse_success'] = bool(result['parse_success'] and valid_prediction(
                result['predicted_agent'], result.get('predicted_step'), trace))
            result.update(run_index=index, model_alias=alias, trace_id=tid,
                          schema_version=SCHEMA_VERSION, run_signature=signatures[tid])
            update_ledger_result(config.RESULTS_DIR / 'api_ledger', result)
            save_run_result(result, config.RAW_RESULTS_DIR, alias, tid, index)
            if not completed_response_record(result):
                progress.fail('API service failure or partial response; usage and raw response retained.')
                progress.summary()
                raise RuntimeError(f'{alias}/{tid} run {index}: API service failure or partial response; '
                                   'inspect the saved response and ledger before retrying.')
            progress.finish(result)
    progress.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(config.MODELS), default=config.ACTIVE_MODELS)
    parser.add_argument('--fraction', '--sample-fraction', type=float, default=config.SAMPLE_FRACTION)
    parser.add_argument('--num-traces', type=int)
    parser.add_argument('--num-runs', type=int, default=config.NUM_RUNS)
    parser.add_argument('--temperature', type=float, default=config.TEMPERATURE)
    parser.add_argument('--keep-weights', action='store_true')
    args = parser.parse_args()
    if args.num_runs < 1 or args.temperature <= 0 or not 0 < args.fraction <= 1:
        parser.error('Runs and temperature must be positive; fraction must be in (0, 1].')
    if args.num_traces is not None and args.num_traces < 1:
        parser.error('--num-traces must be positive.')
    if len(args.models) != len(set(args.models)):
        parser.error('Model aliases must be unique.')
    local_models = [a for a in args.models if config.MODELS[a].get('provider') != 'openai']
    api_budget = check_api_environment() if len(local_models) != len(args.models) else None
    if local_models:
        if not config.PHYSICAL_GPU_UUID:
            raise RuntimeError('Cannot identify physical GPU 0. Run scripts/00_preflight.py first.')
        import torch
        from src.runtime import validate_gpu
        validate_gpu(torch, config.PHYSICAL_GPU_UUID)
    config.print_config_banner('STAGE 02: MODEL INFERENCE')
    traces = sample_traces(load_traces(config.DATA_DIR), args.fraction, config.RANDOM_SEED)
    if args.num_traces:
        traces = traces[:args.num_traces]
    prompts = {t['trace_id']: build_attribution_prompt(t) for t in traces}
    manifest = {'schema_version': SCHEMA_VERSION, 'profile': config.PROFILE,
                'models': args.models, 'model_configs': {a: config.MODELS[a] for a in args.models},
                'trace_ids': [t['trace_id'] for t in traces],
                'trace_hashes': {t['trace_id']: signature(t) for t in traces},
                'prompt_hashes': {tid: signature(prompt) for tid, prompt in prompts.items()},
                'num_runs': args.num_runs, 'temperature': args.temperature,
                'seed': config.RANDOM_SEED, 'val_ratio': config.VAL_RATIO,
                'top_p': config.TOP_P, 'top_k': config.TOP_K,
                'min_accuracy_retention': config.MIN_ACCURACY_RETENTION,
                'cascade_pairs': [[c, s] for c, s in config.ALL_CASCADE_PAIRS if c in args.models and s in args.models],
                'threshold_grid': [round(config.THRESHOLD_MIN + i * config.THRESHOLD_STEP, 6)
                                   for i in range(1 + int((config.THRESHOLD_MAX - config.THRESHOLD_MIN) / config.THRESHOLD_STEP))
                                   if config.THRESHOLD_MIN + i * config.THRESHOLD_STEP < config.THRESHOLD_MAX]
                                   + [config.ALL_STRONG_THRESHOLD]}
    manifest_path = config.RESULTS_DIR / 'experiment_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['resolved_revisions'] = old.get('resolved_revisions', {})
    else:
        if local_models:
            from huggingface_hub import HfApi
        with heartbeat('Resolve immutable model revisions (Hub metadata only)'):
            manifest['resolved_revisions'] = {a: (config.MODELS[a]['api_model']
                if config.MODELS[a].get('provider') == 'openai' else HfApi().model_info(config.MODELS[a]['hf_id'],
                revision=config.MODELS[a]['revision']).sha) for a in args.models}
    ensure_manifest(manifest_path, manifest)
    print(f'[plan] {len(args.models)} models x {len(traces)} traces x {args.num_runs} samples '
          f'= {len(args.models) * len(traces) * args.num_runs} calls before resume skips.', flush=True)
    for model_number, alias in enumerate(args.models, 1):
        settings = dict(config.MODELS[alias], revision=manifest['resolved_revisions'][alias])
        cache = config.MODEL_CACHE_DIR / alias
        raw_dir = config.RAW_RESULTS_DIR / alias
        raw_dir.mkdir(parents=True, exist_ok=True)
        signatures = {tid: signature({'manifest': signature(manifest), 'model': alias, 'prompt': prompt})
                      for tid, prompt in prompts.items()}
        verified = {t['trace_id']: existing_runs(raw_dir, t['trace_id'], args.num_runs,
                    signatures[t['trace_id']], trace=t, model_alias=alias) for t in traces}
        pending = [t for t in traces if len(verified[t['trace_id']]) != args.num_runs]
        print(f'\n[model {model_number}/{len(args.models)}] {alias}', flush=True)
        progress = InferenceProgress(alias, len(traces) * args.num_runs, sum(map(len, verified.values())),
                                     saved_invalid=saved_invalid_count(raw_dir, verified))
        if not pending:
            print(f'[inference] {alias}: all saved runs verified; skipping.', flush=True)
            progress.summary()
            if alias in local_models and config.AUTO_DELETE_WEIGHTS and not args.keep_weights:
                delete_model_from_disk(cache)
            continue
        if settings.get('provider') == 'openai':
            run_api_model(alias, settings, pending, prompts, signatures, args.num_runs, api_budget, progress)
            continue
        # Fail on context overflow before downloading this model's huge weights.
        with heartbeat(f'{alias}: load tokenizer and check complete prompt lengths'):
            tokenizer = load_tokenizer(settings, cache)
            for trace in pending:
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompts[trace['trace_id']]}],
                                                           tokenize=False, add_generation_prompt=True)
                length = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
                print(f"[context] {alias}/{trace['trace_id']}: input={length}, output cap={settings['max_new_tokens']}, "
                      f"context={settings['context_window']}", flush=True)
                if length + settings['max_new_tokens'] > settings['context_window']:
                    raise ValueError(f"{alias}/{trace['trace_id']}: {length} tokens exceed model context; no truncation performed.")
        del tokenizer
        failures = []
        with ModelSession(settings, cache) as session:
            for trace in pending:
                tid = trace['trace_id']
                done = existing_runs(raw_dir, tid, args.num_runs, signatures[tid], trace=trace, model_alias=alias)
                for index in range(args.num_runs):
                    if index in done:
                        continue
                    run_seed = int(signature({'seed': config.RANDOM_SEED, 'model': alias,
                                              'trace': tid, 'run': index})[:8], 16) % (2**31)
                    progress.begin(tid, index, args.num_runs)
                    try:
                        with heartbeat(f'{alias}/{tid} sample {index + 1}/{args.num_runs}: generating on GPU 0'):
                            result = run_single_inference(session.model, session.tokenizer, prompts[tid],
                                max_new_tokens=settings['max_new_tokens'], temperature=args.temperature,
                                seed=run_seed, context_window=settings['context_window'])
                        result['predicted_agent'] = canonical_agent(result.get('predicted_agent'), trace)
                        result['parse_success'] = bool(result['parse_success'] and valid_prediction(
                            result['predicted_agent'], result.get('predicted_step'), trace))
                        result.update(run_index=index, model_alias=alias, trace_id=tid,
                                      schema_version=SCHEMA_VERSION, run_signature=signatures[tid])
                        save_run_result(result, config.RAW_RESULTS_DIR, alias, tid, index)
                        progress.finish(result)
                    except Exception as exc:
                        failures.append(f'{tid} run {index}: {exc}')
                        progress.fail(failures[-1])
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
        progress.summary()
        if failures:
            raise RuntimeError(f'{alias}: {len(failures)} failed runs; weights preserved for retry. First failures: {failures[:5]}')
        if config.AUTO_DELETE_WEIGHTS and not args.keep_weights:
            delete_model_from_disk(cache)
    print('[inference] All requested responses completed and records verified. '
          'Invalid attributions are retained for evaluation; see the model summaries.')


if __name__ == '__main__':
    main()
