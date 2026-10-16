#!/usr/bin/env python3
"""Post-processing orchestrator: ablations + analysis + report + presentation.

Runs after the main grid (tag part4) is complete. Each ablation reuses the
resumable runner with its own tag so it can be interrupted and restarted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"WARNING: {cmd} exited {r.returncode}", flush=True)


def main() -> None:
    # Ablation 1: cosine-normalized kernels (h=2 only, wl + rw-inverse)
    run([PY, "-u", "scripts/run_experiments.py", "--folds", "10", "--repetitions", "5",
         "--depths", "2", "--methods", "wl", "rw-inverse", "--normalize",
         "--tag", "ablation_normalize"])
    # Ablation 2: degree initial labeling (h=2, wl + rw-inverse)
    run([PY, "-u", "scripts/run_experiments.py", "--folds", "10", "--repetitions", "5",
         "--depths", "2", "--methods", "wl", "rw-inverse", "--initial-labeling", "degree",
         "--tag", "ablation_degree"])
    # Rebuild aggregated views from fold-level ground truth
    run([PY, "scripts/finalize_results.py", "part4"])
    # Statistics, figures, report, presentation
    run([PY, "scripts/analyze_results.py"])
    run([PY, "scripts/make_report.py"])
    run([PY, "scripts/make_presentation.py"])
    print("POSTPROCESS_DONE", flush=True)


if __name__ == "__main__":
    main()
