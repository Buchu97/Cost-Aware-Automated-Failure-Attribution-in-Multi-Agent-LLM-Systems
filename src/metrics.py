"""Evaluation metrics for failure attribution.

Metrics from the placement plan:
  - Agent accuracy: predicted_agent == gt_agent
  - Step accuracy: predicted_step == gt_step
  - Joint accuracy: both match simultaneously
  - Total tokens
  - Total cost (theoretical)
  - Escalation rate (cascade only)
"""

import pandas as pd


def agent_accuracy(df: pd.DataFrame, pred_col: str = "predicted_agent",
                   gt_col: str = "gt_agent") -> float:
    """Fraction of traces where the predicted agent matches ground truth."""
    return (df[pred_col] == df[gt_col]).mean()


def step_accuracy(df: pd.DataFrame, pred_col: str = "predicted_step",
                  gt_col: str = "gt_step") -> float:
    """Fraction of traces where the predicted step matches ground truth."""
    return (df[pred_col].astype(str) == df[gt_col].astype(str)).mean()


def joint_accuracy(df: pd.DataFrame,
                   agent_pred: str = "predicted_agent",
                   step_pred: str = "predicted_step",
                   agent_gt: str = "gt_agent",
                   step_gt: str = "gt_step") -> float:
    """Fraction where BOTH agent AND step match ground truth."""
    agent_match = df[agent_pred] == df[agent_gt]
    step_match = df[step_pred].astype(str) == df[step_gt].astype(str)
    return (agent_match & step_match).mean()


def compute_all_metrics(df: pd.DataFrame,
                        agent_pred: str = "predicted_agent",
                        step_pred: str = "predicted_step",
                        tokens_col: str = "total_tokens",
                        cost_col: str = "theoretical_cost",
                        escalation_col: str | None = None) -> dict:
    """Compute the full set of evaluation metrics.

    Returns:
        Dict with keys: agent_accuracy, step_accuracy, joint_accuracy,
        total_tokens, total_cost, escalation_rate.
    """
    metrics = {
        "agent_accuracy": agent_accuracy(df, agent_pred, "gt_agent"),
        "step_accuracy": step_accuracy(df, step_pred, "gt_step"),
        "joint_accuracy": joint_accuracy(df, agent_pred, step_pred,
                                         "gt_agent", "gt_step"),
        "total_tokens": int(df[tokens_col].sum()) if tokens_col in df.columns else 0,
        "total_cost": float(df[cost_col].sum()) if cost_col in df.columns else 0.0,
        "total_elapsed_seconds": float(df["elapsed_seconds"].sum()) if "elapsed_seconds" in df else 0.0,
        "escalation_rate": (
            float(df[escalation_col].mean())
            if escalation_col and escalation_col in df.columns
            else 0.0
        ),
    }
    return metrics
