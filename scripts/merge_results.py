#!/usr/bin/env python3
"""Merge fold-level CSVs from split runs into the primary results directory.

Usage: python scripts/merge_results.py [extra_outdir ...]
Dedupes on (variant, depth, method, rep, fold) keeping the first occurrence,
writes the merged fold CSV to results/fold_wise_results_<tag>.csv, then rebuilds
aggregated views via finalize_results.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

TAG = "part4"
KEYS = ["variant", "depth", "method", "rep", "fold"]


def main() -> None:
    base = pd.read_csv(Path("results") / f"fold_wise_results_{TAG}.csv")
    parts = [base]
    for extra in sys.argv[1:]:
        p = Path(extra) / f"fold_wise_results_{TAG}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
            print(f"loaded {p}")
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(subset=KEYS, keep="first")
    merged.to_csv(Path("results") / f"fold_wise_results_{TAG}.csv", index=False)
    n_cells = merged.groupby(["variant", "depth", "method"]).ngroups
    print(f"merged: {len(merged)} fold rows, {n_cells} cells")

    # Merge confusion matrices too (primary file wins on key conflicts)
    cm_path = Path("results") / f"confusion_matrices_{TAG}.json"
    confusion = json.loads(cm_path.read_text(encoding="utf-8")) if cm_path.exists() else {}
    for extra in sys.argv[1:]:
        p = Path(extra) / f"confusion_matrices_{TAG}.json"
        if p.exists():
            for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                confusion.setdefault(k, v)
    cm_path.write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "scripts/finalize_results.py", TAG], check=True)


if __name__ == "__main__":
    main()
