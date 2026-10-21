"""Final evaluation on the test set.

Usage:
    python scripts/05_evaluate.py

Loads the calibration results (thresholds from step 04), evaluates all three
experimental conditions on the held-out test set, and prints a comparison table.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_traces
from src.cascade import build_cascade_dataframe, evaluate_three_conditions
from src.confidence import apply_saved_calibration
from src.run_records import load_manifest, signature, save_json
from src.analysis_io import load_aggregated, validate_manifest_traces, validate_calibration, manifest_pairs, evaluation_provenance


def format_dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without requiring external dependencies."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        rows = [[str(item) for item in row] for row in df.values]
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        data_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows]
        return "\n".join([header_line, sep_line] + data_lines)


def main() -> None:
    config.print_config_banner("STAGE 05: TEST SET EVALUATION")
    manifest = load_manifest()
    if not manifest_pairs(manifest):
        print("[evaluate] Inference-only profile; skipping cascade evaluation.")
        return

    # ── Load calibration results ──
    cal_path = config.ANALYSIS_DIR / "calibration_results.json"
    if not cal_path.exists():
        print("[evaluate] ERROR: calibration_results.json not found. Run 04_calibrate_cascade.py first.")
        sys.exit(1)

    with open(cal_path, "r", encoding="utf-8") as f:
        calibration_results = json.load(f)
    if not calibration_results:
        raise ValueError('Calibration results are empty; rerun stage 4.')

    # ── Load traces and split ──
    traces = validate_manifest_traces(manifest, load_traces(config.DATA_DIR))
    traces_by_id = {t["trace_id"]: t for t in traces}
    aggregates = {a: load_aggregated(a, manifest, traces) for a in manifest['models']}
    validate_calibration(calibration_results, manifest, traces, aggregates)

    # ── Evaluate each cascade pair ──
    all_results: list[dict] = []

    for cal in calibration_results:
        if cal.get("manifest_signature") != signature(manifest):
            raise ValueError("Calibration does not match inference manifest; rerun stage 4.")
        test_traces = [traces_by_id[tid] for tid in cal["test_trace_ids"]]
        cheap_alias = cal["cheap_model"]
        strong_alias = cal["strong_model"]
        threshold = cal["threshold"]
        conf_signal = cal["confidence_signal"]
        pair_name = cal["pair"]

        print(f"\n{'='*60}")
        print(f"[evaluate] {pair_name}: threshold={threshold:.2f}, signal={conf_signal}")
        print(f"{'='*60}")

        cheap_agg = aggregates[cheap_alias]
        strong_agg = aggregates[strong_alias]

        if not cheap_agg or not strong_agg:
            raise ValueError(f'Missing aggregated data for {pair_name}.')

        test_df = build_cascade_dataframe(
            test_traces, cheap_agg, strong_agg, cheap_alias, strong_alias, manifest['model_configs']
        )
        # Provenance checks distinguish missing data from observed abstentions;
        # abstentions stay in the accuracy denominator and compare incorrect.

        if len(test_df) == 0:
            print(f"[evaluate] No evaluated test traces found for {pair_name}. Skipping.")
            continue

        test_df = apply_saved_calibration(test_df, conf_signal, cal["calibrator"])
        conditions = evaluate_three_conditions(test_df, threshold, f"{conf_signal}_calibrated")

        for cond_name, metrics in conditions.items():
            row = {
                "pair": pair_name,
                "cheap_model": cheap_alias,
                "strong_model": strong_alias,
                "condition": cond_name,
                "threshold": threshold if cond_name == "cascade" else None,
                "confidence_signal": conf_signal if cond_name == "cascade" else None,
                "manifest_signature": signature(manifest),
                **metrics,
            }
            all_results.append(row)

            print(f"\n  [{cond_name}]")
            print(f"    Agent accuracy:  {metrics['agent_accuracy']:.3f}")
            print(f"    Step accuracy:   {metrics['step_accuracy']:.3f}")
            print(f"    Joint accuracy:  {metrics['joint_accuracy']:.3f}")
            print(f"    Total tokens:    {metrics['total_tokens']:,}")
            print(f"    Total cost:      ${metrics['total_cost']:.4f}")
            print(f"    Escalation rate: {metrics['escalation_rate']:.3f}")

    # ── Ensure output directories exist ──
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        raise ValueError('No evaluation results generated.')

    # ── Calculate Comparative Metrics (Savings & Retention) ──
    comparison_rows: list[dict] = []
    for pair_name, group in results_df.groupby("pair"):
        cheap_row = group[group["condition"] == "cheap_only"].iloc[0] if (group["condition"] == "cheap_only").any() else None
        strong_row = group[group["condition"] == "strong_only"].iloc[0] if (group["condition"] == "strong_only").any() else None
        cascade_row = group[group["condition"] == "cascade"].iloc[0] if (group["condition"] == "cascade").any() else None

        for _, r in group.iterrows():
            row_dict = r.to_dict()
            if r["condition"] == "cascade" and strong_row is not None:
                strong_cost = strong_row["total_cost"]
                strong_acc = strong_row["joint_accuracy"]

                cost_savings = ((strong_cost - r["total_cost"]) / strong_cost * 100) if strong_cost > 0 else 0.0
                acc_retention = (r["joint_accuracy"] / strong_acc * 100) if strong_acc > 0 else 100.0

                row_dict["cost_savings_vs_strong_pct"] = round(cost_savings, 1)
                row_dict["accuracy_retention_pct"] = round(acc_retention, 1)
            else:
                row_dict["cost_savings_vs_strong_pct"] = 0.0 if r["condition"] == "strong_only" else None
                row_dict["accuracy_retention_pct"] = 100.0 if r["condition"] == "strong_only" else None

            comparison_rows.append(row_dict)

    comp_df = pd.DataFrame(comparison_rows)

    # ── Export CSV ──
    csv_path = config.TABLES_DIR / "evaluation_results.csv"
    comp_df.to_csv(csv_path, index=False)
    # Also save in root analysis dir for backward compatibility
    comp_df.to_csv(config.ANALYSIS_DIR / "evaluation_results.csv", index=False)
    print(f"\n[evaluate] Saved CSV table to: {csv_path}")

    # ── Export Markdown Table (.md) for direct copy-paste ──
    md_table_path = config.TABLES_DIR / "evaluation_results.md"
    cols_display = [
        "pair", "condition", "agent_accuracy", "step_accuracy", "joint_accuracy",
        "total_tokens", "total_cost", "escalation_rate", "cost_savings_vs_strong_pct"
    ]
    cols_exist = [c for c in cols_display if c in comp_df.columns]

    pretty_df = comp_df[cols_exist].copy()
    pretty_df["agent_accuracy"] = (pretty_df["agent_accuracy"] * 100).round(1).astype(str) + "%"
    pretty_df["step_accuracy"] = (pretty_df["step_accuracy"] * 100).round(1).astype(str) + "%"
    pretty_df["joint_accuracy"] = (pretty_df["joint_accuracy"] * 100).round(1).astype(str) + "%"
    pretty_df["total_cost"] = "$" + pretty_df["total_cost"].round(4).astype(str)
    pretty_df["escalation_rate"] = (pretty_df["escalation_rate"] * 100).round(1).astype(str) + "%"
    if "cost_savings_vs_strong_pct" in pretty_df.columns:
        pretty_df["cost_savings_vs_strong_pct"] = pretty_df["cost_savings_vs_strong_pct"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "-"
        )

    pretty_df.columns = [
        "Cascade Pair", "Condition", "Agent Acc", "Step Acc", "Joint Acc",
        "Tokens", "Cost", "Escalation", "Cost Savings"
    ]
    markdown_table_str = format_dataframe_to_markdown(pretty_df)

    if config.EXPORT_MARKDOWN:
        with open(md_table_path, "w", encoding="utf-8") as f:
            f.write("# Evaluation Results Table (Ready to Copy-Paste)\n\n")
            f.write(markdown_table_str)
            f.write("\n")
        print(f"[evaluate] Saved Markdown table to: {md_table_path}")

    # ── Export LaTeX Table (.tex) for paper inclusion ──
    tex_table_path = config.TABLES_DIR / "evaluation_results.tex"
    if config.EXPORT_LATEX:
        latex_code = pretty_df.to_latex(index=False, caption="Failure Attribution Evaluation across Baseline and Cascade Conditions", label="tab:failure_attribution_results")
        with open(tex_table_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
        print(f"[evaluate] Saved LaTeX table to: {tex_table_path}")

    # ── Export Executive Summary Report (.md) ──
    report_path = config.REPORTS_DIR / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Placement Plan 2: Research Evaluation Report\n")
        f.write(f"**Experiment Configuration:** Profile: {manifest['profile']}, Traces: {len(manifest['trace_ids'])}, Runs per Trace: {manifest['num_runs']}, Local temperature: {manifest['temperature']}\n\n")
        f.write("Local dollar costs are illustrative API-equivalent estimates. OpenAI costs use returned token usage and manifest-frozen published rates, including cache reads/writes and per-request long-context multipliers; these estimates are not billing invoices. Output tokens already include hidden reasoning. Runtime and tokens in comparison tables cover all completed saved responses, including invalid attributions. Local timing measures generation; API timing measures request round trips. Cascades route saved model outputs, so their simulated latency and cache savings may differ in deployment. Failed local infrastructure attempts are excluded; all API attempts retain prices or conservative reservations in api_ledger.\n\n")
        f.write("Invalid model responses are retained without resampling or correction. Each contributes an abstention vote with no agent or step. Confidence averages retain available reported values from invalid responses; missing confidence values remain unavailable. Resource totals include all completed responses. If the modal vote is an abstention, the final attribution is incorrect for agent, step and joint accuracy. Per-model invalid response counts are saved in response_validity.json. Long prompts may use 1024-token prefill chunks without truncation; new local records include the attention policy and chunk size. Backend changes may affect floating-point numerics and timing.\n\n")
        f.write("Qwen uses a fixed factor-4 YaRN extension with original context 32768 and target context 131072, where Qwen is part of this experiment. Local models use BF16 computation, with NF4 weights for the 70B/72B models. These settings are recorded in the manifest.\n\n")
        for alias in manifest['models']:
            settings = manifest['model_configs'][alias]
            if settings.get('provider') == 'openai':
                f.write(f"OpenAI baseline: `{settings['api_model']}`, reasoning `{settings['reasoning_effort']}`, max output {settings['max_new_tokens']} tokens including reasoning, service tier `{settings['service_tier']}`. API default sampling uses no supplied seed, temperature, top-p or top-k; local sampling settings do not apply. No tools are enabled. The requested ID and returned model ID are recorded; API aliases can change over time.\n\n")
        f.write("## 1. Summary Comparison Table\n\n")
        f.write(markdown_table_str)
        f.write("\n\n")
        f.write("## 2. Key Research Findings\n\n")
        for pair_name, group in comp_df.groupby("pair"):
            casc = group[group["condition"] == "cascade"]
            strg = group[group["condition"] == "strong_only"]
            chp = group[group["condition"] == "cheap_only"]
            if not casc.empty and not strg.empty:
                c_row = casc.iloc[0]
                s_row = strg.iloc[0]
                ch_row = chp.iloc[0] if not chp.empty else None
                f.write(f"### Pair: `{pair_name}`\n")
                f.write(f"- **Joint Attribution Accuracy:** Cheap: {ch_row['joint_accuracy']*100:.1f}%, Cascade: {c_row['joint_accuracy']*100:.1f}%, Strong: {s_row['joint_accuracy']*100:.1f}%\n")
                f.write(f"- **Inference Cost:** Cheap: ${ch_row['total_cost']:.4f}, Cascade: ${c_row['total_cost']:.4f}, Strong: ${s_row['total_cost']:.4f}\n")
                f.write(f"- **Cost Reduction:** Achieved **{c_row.get('cost_savings_vs_strong_pct', 0.0):.1f}% cost savings** compared to running the strong model on every trace.\n")
                f.write(f"- **Accuracy Retention:** Maintained **{c_row.get('accuracy_retention_pct', 100.0):.1f}%** of the strong model's accuracy.\n")
                f.write(f"- **Escalation Rate:** Only **{c_row['escalation_rate']*100:.1f}%** of failure traces required escalation to the stronger model.\n\n")
    print(f"[evaluate] Saved Executive Report to: {report_path}")
    save_json(config.ANALYSIS_DIR / 'evaluation_provenance.json',
              evaluation_provenance(manifest, calibration_results, aggregates))

    # ── Print Console Table ──
    print(f"\n{'='*85}")
    print("FINAL COMPARISON TABLE (Test Set)")
    print(f"{'='*85}")
    print(markdown_table_str)
    print("\n[evaluate] All done!")


if __name__ == "__main__":
    main()
