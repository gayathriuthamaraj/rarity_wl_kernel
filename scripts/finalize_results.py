#!/usr/bin/env python3
"""Rebuild the aggregated per-cell views from the authoritative fold-level CSV.

The fold-level file (fold_wise_results_<tag>.csv) is the ground truth of what
has been computed. This script regenerates:
  - per_fold_results_<tag>.csv       (one row per cell with means/medians)
  - aggregated_results_<tag>.json    (protocol + dataset info + aggregated rows)
so it can never drift from the fold-level data, regardless of resume history.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "part4"
    results = Path("results")
    folds = pd.read_csv(results / f"fold_wise_results_{tag}.csv")
    folds["kernel_total_s"] = folds["kernel_fit_s"] + folds["kernel_transform_s"]

    rows = []
    for (variant, depth, method), g in folds.groupby(["variant", "depth", "method"]):
        rows.append({
            "variant": variant, "depth": int(depth), "method": method,
            "kernel": "standard_wl" if method == "wl" else "rarity_weighted",
            "weighting": "none" if method == "wl" else method.replace("rw-", "", 1),
            "n_folds_total": len(g),
            "accuracy_mean": g["accuracy"].mean(), "accuracy_std": g["accuracy"].std(),
            "accuracy_median": g["accuracy"].median(),
            "accuracy_min": g["accuracy"].min(), "accuracy_max": g["accuracy"].max(),
            "macro_f1_mean": g["macro_f1"].mean(), "macro_f1_std": g["macro_f1"].std(),
            "macro_f1_median": g["macro_f1"].median(),
            "n_features_mean": g["n_features"].mean(),
            "n_features_max": int(g["n_features"].max()),
            "kernel_fit_s_median": g["kernel_fit_s"].median(),
            "kernel_transform_s_median": g["kernel_transform_s"].median(),
            "svc_fit_s_median": g["svc_fit_s"].median(),
            "svc_predict_s_median": g["svc_predict_s"].median(),
            "kernel_total_s_median": g["kernel_total_s"].median(),
        })
    df = pd.DataFrame(rows).sort_values(["variant", "depth", "method"])
    df.to_csv(results / f"per_fold_results_{tag}.csv", index=False)

    agg_path = results / f"aggregated_results_{tag}.json"
    agg = json.loads(agg_path.read_text(encoding="utf-8")) if agg_path.exists() else {}
    agg["aggregated"] = df.to_dict(orient="records")
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Sanity: method naming — runner stores 'none' for the standard kernel rows
    std = df[df.weighting == "none"]
    print(f"Rebuilt aggregated views: {len(df)} cells "
          f"({df.variant.nunique()} variants, {std.variant.nunique()} with standard WL rows)")


if __name__ == "__main__":
    main()
