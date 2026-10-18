"""Calibrate confidence scores and select cascade thresholds.

Usage:
    python scripts/04_calibrate_cascade.py

For each cascade pair (cheap -> strong):
  1. Build a unified DataFrame from aggregated results
  2. Split into val / test (using fixed seed)
  3. Calibrate each confidence signal on the validation set
  4. Sweep thresholds and select the best one
  5. Save calibration results to results/analysis/
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_traces, split_val_test
from src.cascade import build_cascade_dataframe
from src.confidence import calibrate, apply_saved_calibration, serialize_calibrator, sweep_thresholds, select_best_threshold
from src.run_records import load_manifest, save_json, signature
from src.analysis_io import load_aggregated, validate_manifest_traces, manifest_pairs, calibration_input_signature


def main() -> None:
    config.print_config_banner("STAGE 04: CALIBRATION & THRESHOLD SELECTION")

    # ── Load traces and split ──
    print("[calibrate] Loading traces ...")
    traces = load_traces(config.DATA_DIR)
    manifest = load_manifest()
    traces = validate_manifest_traces(manifest, traces)
    val_traces, test_traces = split_val_test(traces, manifest['val_ratio'], manifest['seed'])
    print(f"[calibrate] Val: {len(val_traces)} traces, Test: {len(test_traces)} traces")

    # Save the split for reproducibility
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    split_info = {
        "val_trace_ids": [t["trace_id"] for t in val_traces],
        "test_trace_ids": [t["trace_id"] for t in test_traces],
        "val_ratio": manifest['val_ratio'],
        "seed": manifest['seed'],
        "manifest_signature": signature(manifest),
    }
    save_json(config.ANALYSIS_DIR / 'data_split.json', split_info)

    # ── Process each cascade pair ──
    confidence_signals = ["cheap_consistency", "cheap_self_conf", "cheap_logprob_conf"]
    all_pair_results: list[dict] = []

    pairs = manifest_pairs(manifest)
    if not pairs:
        print("[calibrate] Inference-only profile; no complete cascade pair. Skipping calibration.")
        return
    for cheap_alias, strong_alias in pairs:
        pair_name = f"{cheap_alias}_to_{strong_alias}"
        print(f"\n{'='*60}")
        print(f"[calibrate] Cascade pair: {cheap_alias} -> {strong_alias}")
        print(f"{'='*60}")

        # Load aggregated results
        cheap_agg = load_aggregated(cheap_alias, manifest, traces)
        strong_agg = load_aggregated(strong_alias, manifest, traces)

        if not cheap_agg or not strong_agg:
            raise ValueError(f"Missing aggregated data for {pair_name}.")

        # Build cascade DataFrames for val and test
        val_df = build_cascade_dataframe(val_traces, cheap_agg, strong_agg, cheap_alias, strong_alias, manifest['model_configs'])
        test_df = build_cascade_dataframe(test_traces, cheap_agg, strong_agg, cheap_alias, strong_alias, manifest['model_configs'])

        # load_aggregated verified every expected completed record. A null
        # attribution here is an observed model abstention, not missing data.

        if len(val_df) == 0:
            print(f"[calibrate] No evaluated validation traces found for {pair_name}. Skipping.")
            continue

        print(f"[calibrate] Using {len(val_df)} val traces, {len(test_df)} test traces.")

        # Add joint correctness column for calibration target
        val_df["is_joint_correct"] = val_df["is_joint_correct_cheap"]

        # Try each confidence signal
        best_overall = None
        best_overall_cost = float("inf")
        strong_accuracy = float(val_df["is_joint_correct_strong"].mean())

        for conf_col in confidence_signals:
            if conf_col not in val_df.columns or val_df[conf_col].isna().all():
                print(f"[calibrate]   Skipping {conf_col} — no data.")
                continue

            # Drop NaN rows for this signal
            valid_val = val_df.dropna(subset=[conf_col])
            if len(valid_val) < 2:
                print(f"[calibrate]   Skipping {conf_col} — too few valid values ({len(valid_val)}).")
                continue

            print(f"[calibrate]   Sweeping thresholds for {conf_col} ...")
            saved_calibrator = serialize_calibrator(calibrate(valid_val, conf_col))
            calibrated_val = apply_saved_calibration(val_df, conf_col, saved_calibrator)
            calibrated_col = f"{conf_col}_calibrated"
            sweep_df = sweep_thresholds(
                calibrated_val, calibrated_col,
                thresholds=np.array(manifest['threshold_grid']),
            )

            best = select_best_threshold(sweep_df, strong_accuracy=strong_accuracy,
                                        min_accuracy_fraction=manifest['min_accuracy_retention'])
            print(f"[calibrate]   Best threshold: {best['threshold']:.2f} "
                  f"-> accuracy={best['cascade_joint_accuracy']:.3f}, "
                  f"escalation={best['escalation_rate']:.3f}")

            # Save sweep results
            sweep_path = config.ANALYSIS_DIR / f"sweep_{pair_name}_{conf_col}.csv"
            sweep_df['manifest_signature'] = signature(manifest)
            sweep_df.to_csv(sweep_path, index=False)

            if best["cascade_cost"] < best_overall_cost:
                best_overall = {
                    "confidence_signal": conf_col,
                    "calibrator": saved_calibrator,
                    "strong_validation_accuracy": strong_accuracy,
                    "minimum_validation_accuracy": strong_accuracy * manifest['min_accuracy_retention'],
                    "manifest_signature": signature(manifest),
                    "val_trace_ids": split_info["val_trace_ids"],
                    "test_trace_ids": split_info["test_trace_ids"],
                    **best,
                }
                best_overall['validation_aggregate_signature'] = calibration_input_signature(
                    dict(best_overall, cheap_model=cheap_alias, strong_model=strong_alias),
                    {cheap_alias: cheap_agg, strong_alias: strong_agg})
                best_overall_cost = best["cascade_cost"]

        if best_overall:
            all_pair_results.append({
                "pair": pair_name,
                "cheap_model": cheap_alias,
                "strong_model": strong_alias,
                **best_overall,
            })
            print(f"[calibrate]   >> Best signal: {best_overall['confidence_signal']} "
                  f"at threshold {best_overall['threshold']:.2f}")

    # ── Save all pair results ──
    results_path = config.ANALYSIS_DIR / "calibration_results.json"
    if not all_pair_results:
        raise ValueError("No calibration result could be produced; check inference confidence fields.")
    save_json(results_path, all_pair_results)
    print(f"\n[calibrate] Saved calibration results to {results_path}")
    print("[calibrate] All done!")


if __name__ == "__main__":
    main()
