"""Load and split the Who&When benchmark dataset."""

import json
import glob
import os
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split


def load_traces(data_dir: str | Path) -> list[dict]:
    """Load all 184 Who&When JSON traces from Algorithm-Generated and Hand-Crafted dirs.

    Each trace is a dict with keys:
        trace_id, source_type, is_handcrafted, question_id, question,
        ground_truth, messages, gt_agent, gt_step, gt_reason
    """
    data_dir = Path(data_dir)
    traces: list[dict] = []

    for source_type in ["Algorithm-Generated", "Hand-Crafted"]:
        source_dir = data_dir / "Who&When" / source_type
        is_handcrafted = source_type == "Hand-Crafted"

        json_files = sorted(
            glob.glob(str(source_dir / "*.json")),
            key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))) or "0"),
        )

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # The dataset may use 'messages' or 'history' for the conversation
            messages = data.get("messages") or data.get("history", [])

            traces.append(
                {
                    "trace_id": f"{source_type}__{os.path.basename(json_file)}",
                    "source_type": source_type,
                    "is_handcrafted": is_handcrafted,
                    "question_id": data.get("question_ID", ""),
                    "question": data.get("question", ""),
                    "ground_truth": data.get("ground_truth", ""),
                    "messages": messages,
                    "gt_agent": data.get("mistake_agent", ""),
                    "gt_step": str(data.get("mistake_step", "")),
                    "gt_reason": data.get("mistake_reason", ""),
                }
            )

    return traces


def split_val_test(
    traces: list[dict],
    val_ratio: float = 0.6,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified split by source_type into validation and test sets."""
    labels = [t["source_type"] for t in traces]
    val_traces, test_traces = train_test_split(
        traces, train_size=val_ratio, random_state=seed, stratify=labels
    )
    return val_traces, test_traces


def get_split_indices(
    traces: list[dict],
    val_ratio: float = 0.6,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Return dict mapping 'val'/'test' to lists of trace_ids. Useful for reproducibility."""
    val_traces, test_traces = split_val_test(traces, val_ratio, seed)
    return {
        "val": [t["trace_id"] for t in val_traces],
        "test": [t["trace_id"] for t in test_traces],
    }


def sample_traces(
    traces: list[dict],
    fraction: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """Sample a proportional subset of traces stratified by source_type.

    Args:
        traces: List of trace dicts.
        fraction: Fraction of traces to keep (e.g., 1.0 = 100%, 0.1 = 10% test run, 0.5 = 50%).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Sampled subset of traces.
    """
    if fraction >= 1.0:
        return traces
    if fraction <= 0.0:
        raise ValueError(f"Fraction must be > 0.0, got {fraction}")

    total_target = max(2, int(round(len(traces) * fraction)))
    if total_target >= len(traces):
        return traces

    labels = [t["source_type"] for t in traces]
    sampled, _ = train_test_split(
        traces,
        train_size=total_target,
        random_state=seed,
        stratify=labels,
    )
    return sampled

