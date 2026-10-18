"""Confidence calibration and threshold selection.

Fits a calibration model (isotonic regression) on the validation set so that
raw confidence scores map to actual probabilities of being correct.  Then
sweeps thresholds to find the best accuracy-cost trade-off for the cascade.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def calibrate(
    val_df: pd.DataFrame,
    confidence_col: str,
    correct_col: str = "is_joint_correct",
) -> IsotonicRegression:
    """Fit isotonic regression: raw confidence -> P(correct).

    Args:
        val_df: Validation DataFrame with confidence scores and correctness labels.
        confidence_col: Column name for the raw confidence score.
        correct_col: Column name for the binary correctness label.

    Returns:
        Fitted IsotonicRegression model.
    """
    X = val_df[confidence_col].values.astype(float)
    y = val_df[correct_col].astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)
    return iso


def apply_calibration(
    df: pd.DataFrame,
    confidence_col: str,
    calibrator: IsotonicRegression,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Apply a fitted calibrator to transform raw confidence into calibrated."""
    if output_col is None:
        output_col = f"{confidence_col}_calibrated"
    df = df.copy()
    df[output_col] = calibrator.predict(df[confidence_col].values.astype(float))
    return df


def serialize_calibrator(calibrator: IsotonicRegression) -> dict:
    """Save fitted knots as portable JSON; test data never refits this model."""
    return {"x": calibrator.X_thresholds_.tolist(), "y": calibrator.y_thresholds_.tolist()}


def apply_saved_calibration(df: pd.DataFrame, confidence_col: str, saved: dict,
                            output_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    values = pd.to_numeric(df[confidence_col], errors="coerce")
    output_col = output_col or f"{confidence_col}_calibrated"
    df[output_col] = np.interp(values, saved["x"], saved["y"])
    # Missing confidence must always escalate, including at tau=0.
    df.loc[values.isna(), output_col] = -1.0
    return df


def sweep_thresholds(
    val_df: pd.DataFrame,
    confidence_col: str,
    cheap_correct_col: str = "is_joint_correct_cheap",
    strong_correct_col: str = "is_joint_correct_strong",
    cheap_tokens_col: str = "cheap_tokens",
    strong_tokens_col: str = "strong_tokens",
    cheap_cost_col: str = "cheap_cost",
    strong_cost_col: str = "strong_cost",
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep confidence thresholds on validation data.

    For each threshold t:
      - Traces with cheap confidence >= t  -> keep cheap prediction
      - Traces with cheap confidence < t   -> escalate to strong model

    Returns:
        DataFrame with columns: threshold, cascade_joint_accuracy,
        cascade_agent_accuracy, escalation_rate, cascade_tokens, cascade_cost.
    """
    if thresholds is None:
        thresholds = np.r_[np.arange(0.0, 1.01, 0.05), 1.000001]

    results: list[dict] = []
    n = len(val_df)

    for t in thresholds:
        keep_mask = val_df[confidence_col] >= t
        escalation_rate = float((~keep_mask).sum()) / n

        # Cascade correctness
        correct_kept = val_df.loc[keep_mask, cheap_correct_col].sum() if keep_mask.any() else 0
        correct_escalated = (
            val_df.loc[~keep_mask, strong_correct_col].sum()
            if (~keep_mask).any()
            else 0
        )
        cascade_accuracy = (correct_kept + correct_escalated) / n

        # Cascade tokens (always pay for cheap; add strong where escalated)
        cascade_tokens = int(
            val_df[cheap_tokens_col].sum()
            + val_df.loc[~keep_mask, strong_tokens_col].sum()
        )

        # Cascade cost
        cascade_cost = float(
            val_df[cheap_cost_col].sum()
            + val_df.loc[~keep_mask, strong_cost_col].sum()
        )

        results.append(
            {
                "threshold": round(float(t), 6),
                "cascade_joint_accuracy": float(cascade_accuracy),
                "escalation_rate": float(escalation_rate),
                "cascade_tokens": cascade_tokens,
                "cascade_cost": round(cascade_cost, 6),
            }
        )

    return pd.DataFrame(results)


def select_best_threshold(
    sweep_df: pd.DataFrame,
    strong_accuracy: float,
    accuracy_col: str = "cascade_joint_accuracy",
    cost_col: str = "cascade_cost",
    min_accuracy_fraction: float = 0.95,
) -> dict:
    """Select the threshold that minimises cost while retaining near-strong accuracy.

    Among thresholds retaining the required fraction of STRONG-ONLY validation
    accuracy, choose lowest cost. Never weaken the target to the sweep maximum.

    Returns:
        Dict with 'threshold' and associated metrics.
    """
    if sweep_df.empty or not 0 <= strong_accuracy <= 1:
        raise ValueError("Need a nonempty sweep and strong_accuracy in [0, 1].")
    cutoff = strong_accuracy * min_accuracy_fraction
    eligible = sweep_df[sweep_df[accuracy_col] >= cutoff]

    if eligible.empty:
        raise ValueError("No threshold meets the strong-only accuracy target.")
    else:
        best_row = eligible.sort_values([cost_col, accuracy_col, "threshold"],
                                       ascending=[True, False, True]).iloc[0]

    return best_row.to_dict()
