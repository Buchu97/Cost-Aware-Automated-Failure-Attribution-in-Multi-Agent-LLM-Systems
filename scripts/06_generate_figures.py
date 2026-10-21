"""Generate publication-ready figures for the research report.

Usage:
    python scripts/06_generate_figures.py

Reads analysis outputs from results/analysis/ and generates:
  1. threshold_sweeps.png: Accuracy and escalation rate vs threshold across cascade pairs
  2. cost_vs_accuracy_tradeoff.png: Cheap, Strong, and Cascade comparison
  3. run_consistency_distribution.png: Run-to-run consistency distributions
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.run_records import load_manifest
from src.data_loader import load_traces
from src.analysis_io import load_aggregated, validate_manifest_traces, manifest_pairs, validate_calibration, evaluation_provenance

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

FIGURES_DIR = config.FIGURES_DIR


def save_plot(fig, base_name: str) -> None:
    """Save plot in both high-res PNG and PDF vector format."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(config, "SAVE_PNG", True):
        png_path = FIGURES_DIR / f"{base_name}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"[figures] Saved PNG: {png_path}")
    if getattr(config, "SAVE_PDF", True):
        pdf_path = FIGURES_DIR / f"{base_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[figures] Saved PDF: {pdf_path}")
    plt.close(fig)


def plot_threshold_sweeps(sweeps: dict[str, pd.DataFrame]) -> None:
    """Plot accuracy and escalation rate curves across swept thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, df in sweeps.items():
        label = name.replace('_to_', ' -> ')

        axes[0].plot(df["threshold"], df["cascade_joint_accuracy"] * 100, marker="o", markersize=3, label=label)
        axes[1].plot(df["threshold"], df["escalation_rate"] * 100, marker="s", markersize=3, label=label)

    axes[0].set_title("Cascade Joint Accuracy vs Confidence Threshold")
    axes[0].set_xlabel("Confidence Threshold (tau)")
    axes[0].set_ylabel("Joint Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="best", frameon=True)

    axes[1].set_title("Escalation Rate vs Confidence Threshold")
    axes[1].set_xlabel("Confidence Threshold (tau)")
    axes[1].set_ylabel("Escalation to Strong Model (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="best", frameon=True)

    plt.tight_layout()
    save_plot(fig, "threshold_sweeps")


def plot_cost_accuracy_comparison(df: pd.DataFrame) -> None:
    """Plot cost/accuracy comparison across the three conditions."""

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {"cheap_only": "#4c72b0", "strong_only": "#c44e52", "cascade": "#55a868"}
    markers = {"cheap_only": "o", "strong_only": "s", "cascade": "D"}

    # Draw trajectories connecting cheap -> cascade -> strong for each pair
    for pair_name, group in df.groupby("pair"):
        chp = group[group["condition"] == "cheap_only"]
        cas = group[group["condition"] == "cascade"]
        stg = group[group["condition"] == "strong_only"]

        if not chp.empty and not cas.empty and not stg.empty:
            pts_x = [chp.iloc[0]["total_cost"], cas.iloc[0]["total_cost"], stg.iloc[0]["total_cost"]]
            pts_y = [chp.iloc[0]["joint_accuracy"] * 100, cas.iloc[0]["joint_accuracy"] * 100, stg.iloc[0]["joint_accuracy"] * 100]
            ax.plot(pts_x, pts_y, linestyle="--", color="gray", alpha=0.5, zorder=1)

    for cond, group in df.groupby("condition"):
        ax.scatter(
            group["total_cost"],
            group["joint_accuracy"] * 100,
            label=cond.replace("_", " ").title(),
            color=palette.get(cond, "#8172b3"),
            marker=markers.get(cond, "o"),
            s=130,
            alpha=0.95,
            edgecolors="black",
            zorder=2,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["pair"].replace("_to_", " -> "),
                (row["total_cost"], row["joint_accuracy"] * 100),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    ax.set_title("Cost vs Joint Attribution Accuracy (Test Set)")
    ax.set_xlabel("Theoretical Inference Cost (USD)")
    ax.set_ylabel("Joint Accuracy (Agent & Step Correct %)")
    ax.set_ylim(0, 105)
    ax.legend(title="Experimental Condition", loc="best", frameon=True)

    plt.tight_layout()
    save_plot(fig, "cost_vs_accuracy_tradeoff")


def plot_run_consistency(aggregates: dict) -> None:
    """Plot consistency distribution across models from aggregated results."""
    data = []
    for model_alias, records in aggregates.items():
        for res in records.values():
            data.append({
                "model": model_alias,
                "joint_agreement": res.get("joint_agreement", 0.0) * 100,
                "agent_agreement": res.get("agent_agreement", 0.0) * 100,
                "step_agreement": res.get("step_agreement", 0.0) * 100,
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="model", y="joint_agreement", hue="model", data=df, ax=ax, palette="Blues", legend=False)
    ax.set_title("Run-to-Run Prediction Consistency Across Models")
    ax.set_xlabel("Evaluated Model")
    ax.set_ylabel("Joint Agreement Across Runs (%)")
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    save_plot(fig, "run_consistency_distribution")


def main() -> None:
    config.print_config_banner("STAGE 06: FIGURE & PLOT GENERATION")
    manifest = load_manifest()
    traces = validate_manifest_traces(manifest, load_traces(config.DATA_DIR))
    aggregates = {a: load_aggregated(a, manifest, traces) for a in manifest['models']}
    if manifest_pairs(manifest):
        try:
            calibration = json.loads((config.ANALYSIS_DIR / 'calibration_results.json').read_text(encoding='utf-8'))
            provenance = json.loads((config.ANALYSIS_DIR / 'evaluation_provenance.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('Missing or corrupt analysis; complete stages 4 and 5 before generating figures.') from exc
        validate_calibration(calibration, manifest, traces, aggregates)
        if provenance != evaluation_provenance(manifest, calibration, aggregates):
            raise ValueError('Figure inputs changed since evaluation; rerun stages 4 and 5.')
        evaluation = pd.read_csv(config.TABLES_DIR / 'evaluation_results.csv')
        sweeps = {c['pair']: pd.read_csv(config.ANALYSIS_DIR / f"sweep_{c['pair']}_{c['confidence_signal']}.csv")
                  for c in calibration}
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figures] Generating publication figures into {config.FIGURES_DIR} ...")
    if manifest_pairs(manifest):
        plot_threshold_sweeps(sweeps)
        plot_cost_accuracy_comparison(evaluation)
    plot_run_consistency(aggregates)
    print("[figures] Completed figure generation.")


if __name__ == "__main__":
    main()
