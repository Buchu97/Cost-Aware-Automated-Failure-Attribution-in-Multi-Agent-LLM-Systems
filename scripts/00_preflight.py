"""Check Windows server dependencies, physical H100 GPU 0 and actual NF4 kernels."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from src.runtime import gpu_zero_info, validate_gpu, required_download_bytes
from src.run_records import save_json, signature, existing_runs, load_manifest, valid_prediction


def disk_budget() -> int:
    """Allow interrupted caches and fully completed inference to resume."""
    from src.data_loader import load_traces
    from src.prompt_builder import build_attribution_prompt
    from src.analysis_io import validate_manifest_traces
    manifest = None
    if (config.RESULTS_DIR / 'experiment_manifest.json').exists():
        manifest = load_manifest()
        traces = validate_manifest_traces(manifest, load_traces(config.DATA_DIR))
    budgets, cached = [], []
    for alias in config.ACTIVE_MODELS:
        if config.MODELS[alias].get('provider') == 'openai':
            continue
        complete = False
        if manifest and alias in manifest['models']:
            complete = all(len(existing_runs(config.RAW_RESULTS_DIR / alias, t['trace_id'], manifest['num_runs'],
                signature({'manifest': signature(manifest), 'model': alias, 'prompt': build_attribution_prompt(t)}),
                trace=t, model_alias=alias)) == manifest['num_runs'] for t in traces)
        budget = 0 if complete else (180 if config.MODELS[alias]['quantisation'] == '4bit' else 25) * 1024**3
        cache = config.MODEL_CACHE_DIR / alias
        # HF blobs include resumable .incomplete files. Skip snapshot symlinks
        # to avoid counting the same shards twice on Windows with symlink support.
        size = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file() and not p.is_symlink())
        budgets.append(budget)
        cached.append(size)
    return required_download_bytes(budgets, cached, purge=config.AUTO_DELETE_WEIGHTS)


def check_dataset() -> dict:
    counts = {}
    for source, expected in [('Algorithm-Generated', 126), ('Hand-Crafted', 58)]:
        files = list((config.DATA_DIR / 'Who&When' / source).glob('*.json'))
        if len(files) != expected:
            raise ValueError(f'Dataset incomplete: {source} has {len(files)} files, expected {expected}.')
        for path in files:
            data = json.loads(path.read_text(encoding='utf-8'))
            messages = data.get('messages') or data.get('history')
            if not messages or not data.get('mistake_agent') or 'mistake_step' not in data:
                raise ValueError(f'Invalid trace: {path}')
            if not 0 <= int(data['mistake_step']) < len(messages):
                raise ValueError(f'Invalid annotated step: {path}')
            trace = {'messages': messages, 'is_handcrafted': source == 'Hand-Crafted'}
            if not valid_prediction(data['mistake_agent'], str(data['mistake_step']), trace):
                raise ValueError(f'Annotated agent is absent from the conversation: {path}')
        counts[source] = len(files)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only', action='store_true', help='Dataset/configuration checks only, without GPU dependencies.')
    parser.add_argument('--check-models', action='store_true', help='Check Hub access, tokenizers and complete-trace context sizes; no weight download.')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10+ required; Windows setup uses Python 3.11 or 3.12.')
    report = {'python': sys.version, 'platform': platform.platform(), 'profile': config.PROFILE,
              'dataset': check_dataset(), 'results_dir': str(config.RESULTS_DIR)}
    print('[preflight] Dataset: 184 valid traces; namespaced trace IDs.')
    if args.cpu_only:
        print('[preflight] GPU checks skipped (--cpu-only).')
    else:
        api_models = [a for a in config.ACTIVE_MODELS if config.MODELS[a].get('provider') == 'openai']
        if api_models:
            from src.openai_inference import check_api_environment
            report['api_budget_usd'] = check_api_environment()
            report['api_models'] = api_models
            print('[preflight] OpenAI key and budget configured; account/model access is checked by the first request. No API request sent.')
        local_models = [a for a in config.ACTIVE_MODELS if a not in api_models]
        if not local_models:
            if args.report:
                save_json(args.report, report)
            print('[preflight] API-only profile; no CUDA or checkpoint checks needed.')
            return
        info = gpu_zero_info()
        if not info or info['uuid'] != config.PHYSICAL_GPU_UUID:
            raise RuntimeError('Cannot resolve nvidia-smi physical GPU 0. Check the driver and nvidia-smi PATH.')
        versions = {}
        for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'numpy', 'pandas',
                     'sklearn', 'matplotlib', 'seaborn', 'huggingface_hub']:
            module = importlib.import_module(name)
            versions[name] = getattr(module, '__version__', 'unknown')
        import torch
        import bitsandbytes as bnb
        report.update(gpu=validate_gpu(torch, info['uuid']), physical_gpu=info, versions=versions)
        # A real small CUDA computation detects CPU wheels, broken drivers and
        # Windows bitsandbytes binaries before any 70B checkpoint download.
        with torch.inference_mode():
            layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                                      quant_type='nf4').to('cuda:0')
            x = torch.randn(2, 64, dtype=torch.bfloat16, device='cuda:0')
            y = layer(x)
            if not torch.isfinite(y).all().item():
                raise RuntimeError('NF4 CUDA smoke test produced non-finite values.')
            q = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16, device='cuda:0')
            attention = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            if not torch.isfinite(attention).all().item():
                raise RuntimeError('BF16 SDPA CUDA smoke test produced non-finite values.')
            del q, attention
            del layer, x, y
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        print(f"[preflight] Physical GPU 0: {info['name']} / {info['uuid']}; one visible CUDA device; NF4 and SDPA kernels passed.")
        required_gib = disk_budget() / 1024**3
        free_gib = shutil.disk_usage(ROOT).free / 1024**3
        report['disk_free_gib'] = free_gib
        report['additional_disk_budget_gib'] = required_gib
        print(f'[preflight] Additional checkpoint disk budget: {required_gib:.1f} GiB; free: {free_gib:.1f} GiB.')
        if free_gib < required_gib:
            raise RuntimeError(f'Need about {required_gib:.1f} GiB additional space for raw checkpoint shards; only {free_gib:.1f} GiB available.')
        if args.check_models:
            from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
            from src.model_loader import load_tokenizer
            from src.data_loader import load_traces, sample_traces
            from src.prompt_builder import build_attribution_prompt
            traces = sample_traces(load_traces(config.DATA_DIR), config.SAMPLE_FRACTION, config.RANDOM_SEED)
            report['models'] = {}
            for alias in config.ACTIVE_MODELS:
                settings = dict(config.MODELS[alias])
                if settings.get('provider') == 'openai':
                    continue
                model_info = HfApi().model_info(settings['hf_id'], revision=settings['revision'])
                settings['revision'] = model_info.sha
                shard = next(f.rfilename for f in model_info.siblings if f.rfilename.endswith('.safetensors'))
                get_hf_file_metadata(hf_hub_url(settings['hf_id'], shard, revision=model_info.sha))
                tokenizer = load_tokenizer(settings, config.MODEL_CACHE_DIR / alias)
                sizes = {}
                for trace in traces:
                    text = tokenizer.apply_chat_template([{'role': 'user', 'content': build_attribution_prompt(trace)}],
                                                        tokenize=False, add_generation_prompt=True)
                    sizes[trace['trace_id']] = len(tokenizer(text, add_special_tokens=False)['input_ids'])
                overlong = [tid for tid, count in sizes.items() if count + settings['max_new_tokens'] > settings['context_window']]
                if overlong:
                    raise ValueError(f'{alias}: complete traces exceed context: {overlong}')
                report['models'][alias] = {'revision': model_info.sha, 'max_input_tokens': max(sizes.values()), 'trace_tokens': sizes}
                print(f"[preflight] {alias}: weight access verified; largest input {max(sizes.values())} tokens.")
                del tokenizer
    if args.report:
        save_json(args.report, report)
    print('[preflight] Requested checks passed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[preflight] FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
