"""Aggregate multiple inference runs into a single prediction per trace.

Handles run-to-run variability by:
  1. Majority-voting across runs for the final prediction
  2. Computing consistency-based confidence (fraction of runs agreeing)
  3. Averaging self-reported and logprob confidence scores
"""

from collections import Counter
from typing import Optional


def aggregate_runs(run_results: list[dict]) -> dict:
    """Aggregate N run results into a single prediction with confidence signals.

    Args:
        run_results: List of dicts from inference.run_n_times().

    Returns:
        Dict with majority-vote prediction and multiple confidence signals.
    """
    n = len(run_results)
    if n == 0:
        return _empty_aggregation()

    # Every scheduled response gets a vote. An invalid response contributes an
    # abstention, including no step vote, without modifying its raw record.
    joint_preds = [
        (r["predicted_agent"], str(r["predicted_step"])) if r.get('parse_success')
        else (None, None) for r in run_results
    ]
    agent_counter = Counter(agent for agent, _ in joint_preds)
    step_counter = Counter(step for _, step in joint_preds)
    joint_counter = Counter(joint_preds)
    majority_joint, joint_count = joint_counter.most_common(1)[0]
    majority_agent, majority_step = majority_joint
    agent_agreement = agent_counter[majority_agent] / n
    step_agreement = step_counter[majority_step] / n
    joint_agreement = joint_count / n

    # ── Confidence signals ──
    self_confs = [
        r["self_reported_confidence"]
        for r in run_results
        if r["self_reported_confidence"] is not None
    ]
    logprob_confs = [
        r["logprob_confidence"]
        for r in run_results
        if r["logprob_confidence"] is not None
    ]

    mean_self = sum(self_confs) / len(self_confs) if self_confs else None
    mean_logprob = sum(logprob_confs) / len(logprob_confs) if logprob_confs else None

    # Normalise self-reported to 0-1 range for fair comparison
    normalised_self = mean_self / 100.0 if mean_self is not None else None

    # ── Token / timing totals ──
    total_tokens = sum(r.get("total_tokens", 0) for r in run_results)
    total_elapsed = sum(r.get("elapsed_seconds", 0.0) for r in run_results)
    parse_success_rate = sum(1 for r in run_results if r.get("parse_success")) / n

    return {
        "majority_agent": majority_agent,
        "majority_step": majority_step,
        "agent_agreement": round(agent_agreement, 4),
        "step_agreement": round(step_agreement, 4),
        "joint_agreement": round(joint_agreement, 4),
        # Confidence signals (all on 0-1 scale)
        "consistency_confidence": round(joint_agreement, 4),
        "mean_self_confidence": round(normalised_self, 4) if normalised_self is not None else None,
        "mean_logprob_confidence": round(mean_logprob, 6) if mean_logprob is not None else None,
        # Resource usage (summed across all N runs)
        "total_tokens": total_tokens,
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in run_results),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in run_results),
        "provider": run_results[0].get('provider', 'huggingface'),
        "cached_prompt_tokens": sum(r.get('cached_prompt_tokens', 0) for r in run_results),
        "cache_write_prompt_tokens": sum(r.get('cache_write_prompt_tokens', 0) for r in run_results),
        "reasoning_tokens": sum(r.get('reasoning_tokens', 0) for r in run_results),
        "usage_cost_usd": sum(r.get('usage_cost_usd', 0) for r in run_results),
        "total_elapsed": round(total_elapsed, 3),
        "n_runs": n,
        "parse_success_rate": round(parse_success_rate, 4),
        "invalid_response_count": sum(not bool(r.get('parse_success')) for r in run_results),
        "attribution_valid": majority_agent is not None and majority_step is not None,
    }


def _empty_aggregation() -> dict:
    """Return an empty aggregation dict when no runs are available."""
    return {
        "majority_agent": None,
        "majority_step": None,
        "agent_agreement": 0.0,
        "step_agreement": 0.0,
        "joint_agreement": 0.0,
        "consistency_confidence": 0.0,
        "mean_self_confidence": None,
        "mean_logprob_confidence": None,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "provider": None,
        "cached_prompt_tokens": 0,
        "cache_write_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "usage_cost_usd": 0.0,
        "total_elapsed": 0.0,
        "n_runs": 0,
        "parse_success_rate": 0.0,
        "invalid_response_count": 0,
        "attribution_valid": False,
    }
