"""Human-readable progress with live heartbeats for blocking model/API operations."""
from contextlib import contextmanager
import threading
import time


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


@contextmanager
def heartbeat(label: str, interval: float = 20):
    """Elapsed-time liveness only: a wait heartbeat does not claim completed work."""
    stopped = threading.Event()
    start = time.perf_counter()
    print(f'[working] {label}', flush=True)
    def pulse():
        while not stopped.wait(interval):
            print(f'[heartbeat] {label}: still running; elapsed {duration(time.perf_counter() - start)}', flush=True)
    worker = threading.Thread(target=pulse, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


class InferenceProgress:
    """Count calls, including verified resume records; estimate only within one model."""
    def __init__(self, alias: str, total: int, verified: int = 0, saved_invalid: int = 0):
        self.alias, self.total, self.completed = alias, total, verified
        self.verified = verified - saved_invalid
        self.saved_invalid = saved_invalid
        self.invalid_responses = self.failed_attempts = 0
        self.attempted = self.invalid = 0
        self.started = time.perf_counter()
        self.call_seconds = []
        print(f'[model] {alias}: {verified}/{total} completed; {self.verified} valid, '
              f'{saved_invalid} invalid responses saved; {total - verified} calls pending.', flush=True)

    def begin(self, trace_id: str, index: int, num_runs: int):
        self.call_start = time.perf_counter()
        print(f'[call] {self.alias} | {self.completed}/{self.total} completed | {trace_id} | sample {index + 1}/{num_runs}', flush=True)

    def finish(self, result: dict):
        elapsed = time.perf_counter() - self.call_start
        self.call_seconds.append(elapsed)
        self.attempted += 1
        valid = bool(result.get('parse_success'))
        self.completed += 1
        self.verified += int(valid)
        self.invalid += int(not valid)
        self.invalid_responses += int(not valid)
        status = 'OK' if valid else 'INVALID'
        details = (f"agent={result.get('predicted_agent')} step={result.get('predicted_step')} | "
                   f"input={result.get('prompt_tokens')} output={result.get('completion_tokens')} tokens | "
                   f"inference={result.get('elapsed_seconds', elapsed):.1f}s")
        usage = result.get('usage', {})
        # API results also carry a normalized reasoning count.
        reasoning = result.get('reasoning_tokens', usage.get('reasoning_tokens'))
        if reasoning is not None:
            details += f' | reasoning={reasoning}'
        if result.get('usage_cost_usd') is not None:
            details += f" | usage price=${result['usage_cost_usd']:.6f}"
        remaining = self.total - self.completed
        eta = duration(sum(self.call_seconds) / len(self.call_seconds) * remaining)
        print(f'[{status}] {self.alias} {self.completed}/{self.total} completed '
              f'({100 * self.completed / self.total:.1f}%) | {details}', flush=True)
        print(f'[progress] elapsed={duration(time.perf_counter() - self.started)} | rough same-model ETA={eta} '
              '(completed-call average; trace lengths vary; excludes model loading)', flush=True)

    def fail(self, message: str):
        self.attempted += 1
        self.invalid += 1
        self.failed_attempts += 1
        print(f'[ERROR] {self.alias} {self.completed}/{self.total} completed | {message}', flush=True)

    def summary(self):
        print(f'[model summary] {self.alias}: {self.completed}/{self.total} completed; '
              f'{self.verified} valid, {self.saved_invalid + self.invalid_responses} invalid responses; '
              f'{self.attempted} attempts this session, {self.failed_attempts} infrastructure failures; '
              f'elapsed {duration(time.perf_counter() - self.started)}', flush=True)
