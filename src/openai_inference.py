"""Stateless Responses API inference with a persistent, conservative budget ledger.

Uses the standard library HTTPS client; no CUDA or additional SDK dependency.
Requests never retry automatically. Ledger reservations survive process crashes.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from src.cost_tracker import compute_usage_cost
from src.response_parser import parse_attribution_response
from src.run_records import save_json, finite_number


class APIHTTPError(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f'OpenAI HTTP {status_code}')


@contextmanager
def ledger_lock(directory: Path):
    """OS lock releases automatically after crashes; lock file contains no state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.request.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another API request is active for this results directory; use one runner.') from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_api_environment() -> float:
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('Set OPENAI_API_KEY in the server .env file or shell environment.')
    try:
        budget = float(os.environ.get('OPENAI_BUDGET_USD', '0'))
    except ValueError:
        budget = 0
    if not finite_number(budget) or budget <= 0:
        raise ValueError('Set positive OPENAI_BUDGET_USD in .env or use run_server.ps1 -ApiBudgetUsd.')
    return budget


def post_response(payload: dict) -> dict:
    """Send exactly one request without logging credentials or response errors."""
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        raise ValueError('OPENAI_API_KEY is missing.')
    request = Request('https://api.openai.com/v1/responses',
                      data=json.dumps(payload).encode('utf-8'),
                      headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                      method='POST')
    try:
        with urlopen(request, timeout=600) as stream:
            return json.load(stream)
    except HTTPError as exc:
        raise APIHTTPError(exc.code) from None


def reserved_request_cost(prompt: str, settings: dict) -> float:
    # A conservative byte-based estimate for text BPE input plus envelope margin.
    # It is deliberately larger than a typical token count; not an exact quote.
    bound = len(prompt.encode('utf-8')) + 1024
    if bound + settings['max_new_tokens'] > settings['context_window']:
        raise ValueError('Conservative API context estimate exceeds the limit; no truncation performed.')
    return compute_usage_cost(settings, {'prompt_tokens': bound,
        'cache_write_prompt_tokens': bound, 'completion_tokens': settings['max_new_tokens']})


def run_openai_inference(prompt: str, settings: dict, ledger_dir: Path, budget_usd: float,
                         *, post=None, request_metadata=None) -> dict:
    with ledger_lock(ledger_dir):
        return _run_openai_inference(prompt, settings, ledger_dir, budget_usd,
                                     post=post, request_metadata=request_metadata)


def update_ledger_result(ledger_dir: Path, result: dict) -> None:
    """Retain canonical/semantic prediction validation for recovery or retry."""
    with ledger_lock(ledger_dir):
        path = Path(ledger_dir) / f"{result['ledger_attempt_id']}.json"
        entry = json.loads(path.read_text(encoding='utf-8'))
        entry['result'] = result
        save_json(path, entry)


def _run_openai_inference(prompt: str, settings: dict, ledger_dir: Path, budget_usd: float,
                          *, post=None, request_metadata=None) -> dict:
    if not finite_number(budget_usd) or budget_usd <= 0:
        raise ValueError('API budget must be positive.')
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for path in ledger_dir.glob('*.json'):
        entry = json.loads(path.read_text(encoding='utf-8'))
        if not finite_number(entry.get('accounted_cost_usd')):
            raise ValueError(f'Invalid API budget ledger: {path}')
        previous = entry.get('result', {})
        if (request_metadata is not None and entry.get('request_metadata') == request_metadata
                and entry.get('status') == 'response_received'):
            from src.run_records import valid_resources, valid_api_usage, completed_response_record
            if (not isinstance(previous, dict) or not valid_resources(previous) or not valid_api_usage(previous)
                    or not completed_response_record(previous)):
                raise ValueError(f'Invalid saved API response in {path}; inspect the ledger before retrying.')
            print('[api] Recovering received response from ledger; no new API request.', flush=True)
            return previous
        spent += entry['accounted_cost_usd']
    reservation = reserved_request_cost(prompt, settings)
    if spent + reservation > budget_usd:
        raise ValueError(f'API budget would be exceeded: accounted ${spent:.4f}, '
                         f'next reservation ${reservation:.4f}, limit ${budget_usd:.4f}.')
    attempt = ledger_dir / f'{uuid.uuid4().hex}.json'
    entry = {'status': 'reserved_or_unknown', 'accounted_cost_usd': reservation,
             'request_metadata': request_metadata, 'requested_model': settings['api_model']}
    save_json(attempt, entry)
    print(f'[api budget] accounted=${spent:.4f} | current reservation=${reservation:.4f} | '
          f'ceiling=${budget_usd:.4f}', flush=True)
    payload = {'model': settings['api_model'], 'input': prompt,
               'reasoning': {'effort': settings['reasoning_effort']},
               'max_output_tokens': settings['max_new_tokens'],
               'service_tier': settings['service_tier'], 'truncation': 'disabled', 'store': False}
    start = time.perf_counter()
    try:
        response = (post or post_response)(payload)
    except Exception as exc:
        # Do not copy server messages/credentials into terminal output.
        detail = f'HTTP {exc.status_code}' if isinstance(exc, APIHTTPError) else type(exc).__name__
        entry.update(failure_type=type(exc).__name__)
        if isinstance(exc, APIHTTPError):
            entry['http_status'] = exc.status_code
        save_json(attempt, entry)
        raise RuntimeError(f'OpenAI request failed ({detail}); reservation retained in {attempt}.') from None
    elapsed = time.perf_counter() - start
    usage = response.get('usage') or {}
    inp = usage.get('input_tokens_details') or {}
    out = usage.get('output_tokens_details') or {}
    text = '\n'.join(part['text'] for item in response.get('output', []) if item.get('type') == 'message'
                     for part in item.get('content', []) if part.get('type') == 'output_text')
    parsed = parse_attribution_response(text)
    result = {'generated_text': text, 'predicted_agent': parsed['agent_name'],
              'predicted_step': parsed['step_number'], 'self_reported_confidence': parsed['confidence'],
              'reason': parsed['reason'], 'logprob_confidence': None, 'seed': None,
              'prompt_tokens': usage.get('input_tokens'), 'completion_tokens': usage.get('output_tokens'),
              'total_tokens': usage.get('total_tokens'), 'cached_prompt_tokens': inp.get('cached_tokens', 0),
              'cache_write_prompt_tokens': inp.get('cache_write_tokens', 0),
              'reasoning_tokens': out.get('reasoning_tokens', 0), 'elapsed_seconds': elapsed,
              'provider': 'openai', 'requested_model': settings['api_model'],
              'response_model': response.get('model'), 'response_id': response.get('id'),
              'response_status': response.get('status'), 'service_tier': response.get('service_tier'),
              'reasoning_effort': settings['reasoning_effort'], 'model_revision': response.get('model'),
              'sampling_policy': settings['sampling_policy'],
              'ledger_attempt_id': attempt.stem,
              'parse_success': bool(response.get('status') == 'completed' and parsed['agent_name']
                                    and parsed['step_number'] is not None)}
    from src.run_records import valid_resources, valid_api_usage
    # Validate usage before replacing a conservative reservation with a price.
    if not valid_resources(result) or not valid_api_usage(result, require_cost=False):
        raise ValueError(f'Invalid API usage/model/tier; reservation retained in {attempt}.')
    result['usage_cost_usd'] = compute_usage_cost(settings, result)
    entry.update(status='response_received', accounted_cost_usd=result['usage_cost_usd'],
                 response_id=result['response_id'], response_status=result['response_status'],
                 usage=usage, response_model=result['response_model'], result=result)
    save_json(attempt, entry)
    return result
