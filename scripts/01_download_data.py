"""Download the Who&When dataset from GitHub.

Usage:
    python scripts/01_download_data.py

This script clones the Agents_Failure_Attribution repo and copies
only the Who&When/ directory into data/.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPO_URL = "https://github.com/ag2ai/Agents_Failure_Attribution.git"
TEMP_DIR = PROJECT_ROOT / "_temp_clone"
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    target_dir = DATA_DIR / "Who&When"

    if target_dir.exists():
        print(f"[download] Data directory already exists: {target_dir}")
        print("[download] Delete it manually if you want a fresh download.")
        # Count files
        algo = list((target_dir / "Algorithm-Generated").glob("*.json"))
        hand = list((target_dir / "Hand-Crafted").glob("*.json"))
        print(f"[download] Found {len(algo)} Algorithm-Generated + {len(hand)} Hand-Crafted = {len(algo)+len(hand)} traces.")
        return

    print(f"[download] Cloning {REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(TEMP_DIR)],
        check=True,
    )

    # Copy the Who&When directory
    source = TEMP_DIR / "Who&When"
    if not source.exists():
        print(f"[download] ERROR: {source} not found in cloned repo!")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(source), str(target_dir))

    # Clean up
    shutil.rmtree(str(TEMP_DIR), ignore_errors=True)

    # Verify
    algo = list((target_dir / "Algorithm-Generated").glob("*.json"))
    hand = list((target_dir / "Hand-Crafted").glob("*.json"))
    total = len(algo) + len(hand)
    print(f"[download] Done! {len(algo)} Algorithm-Generated + {len(hand)} Hand-Crafted = {total} traces.")

    if total != 184:
        print(f"[download] WARNING: Expected 184 traces, got {total}.")


if __name__ == "__main__":
    main()
