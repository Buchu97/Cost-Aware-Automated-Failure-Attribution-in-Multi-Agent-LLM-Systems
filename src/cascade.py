"""Confidence-guided cascade evaluation.

Evaluates the three experimental conditions from the placement plan:
  1. Cheap-only baseline
  2. Strong-only baseline
  3. Confidence-guided cascade (cheap model + escalation to strong)
"""

import pandas as pd
from src.metrics import compute_all_metrics


def build_cascade_dataframe(
    traces: list[dict],
    cheap_aggregated: dict[str, dict],
    strong_aggregated: dict[str, dict],
    cheap_alias: str,
    strong_alias: str,
    model_configs: dict | None = None,
) -> pd.DataFrame:
    """Build a unified DataFrame for cascade evaluation.

    Args:
        traces: List of trace dicts (with gt_agent, gt_step, trace_id).
        cheap_aggregated: {trace_id: aggregated_dict} for the cheap model.
        strong_aggregated: {trace_id: aggregated_dict} for the strong model.
        cheap_alias: Model alias for cost computation (e.g. 'llama-8b').
        strong_alias: Model alias for cost computation (e.g. 'llama-70b').

    Returns:
        DataFrame with columns for both models' predictions, confidence,
        tokens, cost, and ground truth.
    """
    from src.cost_tracker import aggregate_cost

    rows: list[dict] = []
    for trace in traces:
        tid = trace["trace_id"]
        cheap = cheap_aggregated.get(tid, {})
        strong = strong_aggregated.get(tid, {})

        # Compute per-trace theoretical costs
        cheap_prompt_tok = cheap.get("prompt_tokens", 0)
        cheap_comp_tok = cheap.get("completion_tokens", 0)
        strong_prompt_tok = strong.get("prompt_tokens", 0)
        strong_comp_tok = strong.get("completion_tokens", 0)

        rows.append(
            {
                "trace_id": tid,
                "source_type": trace["source_type"],
                "gt_agent": trace["gt_agent"],
                "gt_step": trace["gt_step"],
                # Cheap model
                "cheap_agent": cheap.get("majority_agent"),
                "cheap_step": cheap.get("majority_step"),
                "cheap_consistency": cheap.get("consistency_confidence", 0.0),
                "cheap_self_conf": cheap.get("mean_self_confidence"),
                "cheap_logprob_conf": cheap.get("mean_logprob_confidence"),
                "cheap_tokens": cheap.get("total_tokens", 0),
                "cheap_elapsed": cheap.get("total_elapsed", 0.0),
                "cheap_cost": aggregate_cost(
                    cheap_alias, cheap, model_configs
                ),
                # Strong model
                "strong_agent": strong.get("majority_agent"),
                "strong_step": strong.get("majority_step"),
                "strong_tokens": strong.get("total_tokens", 0),
                "strong_elapsed": strong.get("total_elapsed", 0.0),
                "strong_cost": aggregate_cost(
                    strong_alias, strong, model_configs
                ),
            }
        )

    df = pd.DataFrame(rows)

    # Correctness columns
    df["is_agent_correct_cheap"] = df["cheap_agent"] == df["gt_agent"]
    df["is_step_correct_cheap"] = df["cheap_step"].astype(str) == df["gt_step"].astype(str)
    df["is_joint_correct_cheap"] = df["is_agent_correct_cheap"] & df["is_step_correct_cheap"]

    df["is_agent_correct_strong"] = df["strong_agent"] == df["gt_agent"]
    df["is_step_correct_strong"] = df["strong_step"].astype(str) == df["gt_step"].astype(str)
    df["is_joint_correct_strong"] = df["is_agent_correct_strong"] & df["is_step_correct_strong"]

    return df


def evaluate_three_conditions(
    test_df: pd.DataFrame,
    threshold: float,
    confidence_col: str = "cheap_consistency",
) -> dict[str, dict]:
    """Evaluate all three experimental conditions on the test set.

    Returns:
        Dict mapping condition name to its metrics dict.
    """
    results: dict[str, dict] = {}
    n = len(test_df)

    # ── Condition 1: Cheap-only ──
    cheap_df = test_df.rename(
        columns={"cheap_agent": "predicted_agent", "cheap_step": "predicted_step",
                 "cheap_tokens": "total_tokens", "cheap_cost": "theoretical_cost",
                 "cheap_elapsed": "elapsed_seconds"}
    )
    results["cheap_only"] = compute_all_metrics(cheap_df)

    # ── Condition 2: Strong-only ──
    strong_df = test_df.rename(
        columns={"strong_agent": "predicted_agent", "strong_step": "predicted_step",
                 "strong_tokens": "total_tokens", "strong_cost": "theoretical_cost",
                 "strong_elapsed": "elapsed_seconds"}
    )
    results["strong_only"] = compute_all_metrics(strong_df)

    # ── Condition 3: Cascade ──
    keep_mask = test_df[confidence_col] >= threshold

    cascade_df = test_df.copy()
    cascade_df["predicted_agent"] = test_df["cheap_agent"].where(
        keep_mask, test_df["strong_agent"]
    )
    cascade_df["predicted_step"] = test_df["cheap_step"].where(
        keep_mask, test_df["strong_step"]
    )
    # Tokens: always pay cheap + pay strong where escalated
    cascade_df["total_tokens"] = (
        test_df["cheap_tokens"] + (~keep_mask).astype(int) * test_df["strong_tokens"]
    )
    cascade_df["theoretical_cost"] = (
        test_df["cheap_cost"] + (~keep_mask).astype(int) * test_df["strong_cost"]
    )
    cascade_df["escalated"] = (~keep_mask).astype(int)
    if "cheap_elapsed" in test_df and "strong_elapsed" in test_df:
        cascade_df["elapsed_seconds"] = test_df["cheap_elapsed"] + (~keep_mask).astype(int) * test_df["strong_elapsed"]

    results["cascade"] = compute_all_metrics(
        cascade_df, escalation_col="escalated"
    )
    results["cascade"]["threshold"] = threshold

    return results
