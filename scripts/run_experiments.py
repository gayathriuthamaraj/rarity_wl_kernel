#!/usr/bin/env python3
"""Part 4 experiment runner: standard WL subtree kernel vs rarity-weighted WL kernel.

Protocol (leakage-safe)
-----------------------
- Repeated stratified K-fold cross-validation (folds x repetitions, seeds 42..).
- For every fold the kernel is (re)fit on the TRAINING split only:
  WL refinement, feature vocabulary, and rarity weights never see test graphs.
- Identical data splits and classifier protocol (SVC, kernel='precomputed', C=1.0)
  for both kernels, so observed differences come from the weighting alone.
- Accuracy, macro-F1, and wall-clock runtime (kernel fit / Gram assembly /
  Gram transform / classifier fit / predict) are recorded per fold.

Outputs
-------
- results/per_fold_results.csv      (one row per fold x method x dataset x depth)
- results/aggregated_results.json   (means, medians, stds, CIs, feature counts)
- results/confusion_matrices.json   (aggregated confusion counts per config)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egonets_wl.data.loader import load_dataset  # noqa: E402
from egonets_wl.kernels.wl_subtree import WLSubtreeKernel  # noqa: E402
from egonets_wl.kernels.rarity_weighted_wl import RarityWeightedWLKernel  # noqa: E402


def make_kernel(method: str, n_iter: int, normalize: bool, initial_labeling: str = "uniform"):
    """Instantiate one of the compared kernels; 'rw-uniform' ≡ standard WL."""
    if method == "wl":
        return WLSubtreeKernel(n_iter=n_iter, normalize=normalize, initial_labeling=initial_labeling)
    weighting = method.split("-", 1)[1]
    return RarityWeightedWLKernel(
        n_iter=n_iter, normalize=normalize, weighting=weighting,
        initial_labeling=initial_labeling,
    )


def record_fold(rows_fold, variant, depth, method, rep, fold_id, out):
    """Append one per-fold record for paired statistical testing later."""
    rows_fold.append({
        "variant": variant, "depth": depth, "method": method,
        "rep": rep, "fold": fold_id,
        "accuracy": out["accuracy"], "macro_f1": out["macro_f1"],
        "n_features": out["n_features"], **out["timings"],
    })


def run_one_fold(kernel, graphs_train, y_train, graphs_test, y_test):
    """Fit kernel on train only, build Gram matrices, classify, and time each stage."""
    timings = {}
    t0 = time.perf_counter()
    K_train = kernel.fit_transform(graphs_train)
    timings["kernel_fit_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    K_test = kernel.transform(graphs_test)
    timings["kernel_transform_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)
    timings["svc_fit_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(K_test)
    timings["svc_predict_s"] = time.perf_counter() - t0

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_features": int(len(kernel.feature_names_)),
        "confusion": confusion_matrix(y_test, y_pred, labels=clf.classes_),
        "classes": clf.classes_,
        "timings": timings,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["EGO-1", "EGO-2", "EGO-3", "EGO-4"])
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wl", "rw-uniform", "rw-inverse", "rw-sqrt_inverse", "rw-idf", "rw-binary"],
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--tag", default="part4")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply cosine kernel normalization (ablation flag)")
    parser.add_argument("--initial-labeling", default="uniform",
                        choices=["uniform", "degree"],
                        help="Initial node label convention (ablation flag)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in args.methods if m != "rw-uniform"]
    methods = sorted(set(methods))
    rows = []
    rows_fold = []
    confusion_store = {}
    dataset_info = {}

    # ---- Resume support: skip cells whose fold rows already exist on disk ----
    skip: set = set()
    folds_path = outdir / f"fold_wise_results_{args.tag}.csv"
    if folds_path.exists():
        prior = pd.read_csv(folds_path)
        need = args.folds * args.repetitions
        done_cells = prior.groupby(["variant", "depth", "method"]).size()
        skip = set()
        for (variant, depth, method), n in done_cells.items():
            if variant in args.variants and depth in args.depths and method in methods and n == need:
                skip.add((str(variant), int(depth), str(method)))
        rows_fold = prior.to_dict("records")
        rows = prior.to_dict("records")  # keep all prior aggregated rows; new cells append
        cm_path = outdir / f"confusion_matrices_{args.tag}.json"
        if cm_path.exists():
            confusion_store = json.loads(cm_path.read_text(encoding="utf-8"))
        if skip:
            print(f"Resuming: skipping {len(skip)} completed cells: {sorted(skip)}")

    for variant in args.variants:
        graphs = load_dataset(variant, raw_dir=args.raw_dir)
        labels = np.array([g.graph["label"] for g in graphs])
        n_nodes = int(np.mean([g.number_of_nodes() for g in graphs]))
        n_edges = int(np.mean([g.number_of_edges() for g in graphs]))
        dataset_info[variant] = {
            "n_graphs": len(graphs),
            "label_counts": dict(Counter(labels.tolist())),
            "mean_nodes": n_nodes,
            "mean_edges": n_edges,
        }
        print(f"\n=== {variant}: {len(graphs)} graphs, labels {dict(Counter(labels.tolist()))} ===")

        for depth in args.depths:
            for method in methods:
                if (variant, depth, method) in skip:
                    print(f"-- skip done: {variant} h={depth} {method}")
                    continue
                scheme = method.split("-", 1)[1] if method.startswith("rw-") else "none"
                fold_acc, fold_f1, feature_counts = [], [], []
                agg_cm = None
                agg_times = {}
                for rep in range(args.repetitions):
                    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                          random_state=args.seed + rep)
                    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(graphs, labels)):
                        g_train = [graphs[i] for i in tr_idx]
                        g_test = [graphs[i] for i in te_idx]
                        out = run_one_fold(make_kernel(method, depth, args.normalize,
                                                       args.initial_labeling),
                                           g_train, labels[tr_idx], g_test, labels[te_idx])
                        record_fold(rows_fold, variant, depth, method, rep, fold_i, out)
                        fold_acc.append(out["accuracy"])
                        fold_f1.append(out["macro_f1"])
                        feature_counts.append(out["n_features"])
                        for k, v in out["timings"].items():
                            agg_times.setdefault(k, []).append(v)
                        cm = out["confusion"]
                        agg_cm = cm.astype(float) if agg_cm is None else agg_cm + cm

                acc = np.array(fold_acc)
                f1s = np.array(fold_f1)
                row = {
                    "variant": variant,
                    "depth": depth,
                    "method": scheme,
                    "kernel": "rarity_weighted" if method.startswith("rw-") else "standard_wl",
                    "weighting": scheme,
                    "n_folds_total": len(acc),
                    "accuracy_mean": acc.mean(),
                    "accuracy_std": acc.std(),
                    "accuracy_median": float(np.median(acc)),
                    "accuracy_min": acc.min(),
                    "accuracy_max": acc.max(),
                    "macro_f1_mean": f1s.mean(),
                    "macro_f1_std": f1s.std(),
                    "macro_f1_median": float(np.median(f1s)),
                    "n_features_mean": float(np.mean(feature_counts)),
                    "n_features_max": int(np.max(feature_counts)),
                    "kernel_fit_s_median": float(np.median(agg_times["kernel_fit_s"])),
                    "kernel_transform_s_median": float(np.median(agg_times["kernel_transform_s"])),
                    "svc_fit_s_median": float(np.median(agg_times["svc_fit_s"])),
                    "svc_predict_s_median": float(np.median(agg_times["svc_predict_s"])),
                    "kernel_total_s_median": float(np.median(
                        np.array(agg_times["kernel_fit_s"]) + np.array(agg_times["kernel_transform_s"]))),
                }
                rows.append(row)
                confusion_store[f"{variant}|d{depth}|{scheme}"] = {
                    "confusion": agg_cm.astype(int).tolist(),
                    "classes": [int(c) for c in out["classes"]],
                }
                print(f"  h={depth} {method:>16s}: acc={row['accuracy_mean']:.4f}"
                      f" +/- {row['accuracy_std']:.4f}  macroF1={row['macro_f1_mean']:.4f}"
                      f"  feats={row['n_features_mean']:.0f}  kern={row['kernel_total_s_median']:.3f}s")
                # ---- Incremental checkpoint after every cell (atomic replace) ----
                import os as _os
                for _df, _name in [
                    (pd.DataFrame(rows), f"per_fold_results_{args.tag}.csv"),
                    (pd.DataFrame(rows_fold), f"fold_wise_results_{args.tag}.csv"),
                ]:
                    _tmp = str(outdir / _name) + ".tmp"
                    _df.to_csv(_tmp, index=False)
                    _os.replace(_tmp, outdir / _name)
                _tmpj = str(outdir / f"confusion_matrices_{args.tag}.json") + ".tmp"
                Path(_tmpj).write_text(json.dumps(confusion_store, indent=2), encoding="utf-8")
                _os.replace(_tmpj, outdir / f"confusion_matrices_{args.tag}.json")
                print(f"     checkpoint saved ({len(rows)} cells done)")

    df = pd.DataFrame(rows)
    per_fold_path = outdir / f"per_fold_results_{args.tag}.csv"
    df.to_csv(per_fold_path, index=False)

    df_folds = pd.DataFrame(rows_fold)
    df_folds.to_csv(outdir / f"fold_wise_results_{args.tag}.csv", index=False)

    summary = {
        "protocol": {
            "folds": args.folds,
            "repetitions": args.repetitions,
            "seeds": [args.seed + r for r in range(args.repetitions)],
            "classifier": "SVC(kernel='precomputed', C=1.0)",
            "normalize": args.normalize,
            "initial_labeling": args.initial_labeling,
            "feature_extraction": "WL subtree kernel, uniform initial labels (documented convention)",
            "leakage_control": "Kernel fit + rarity weights recomputed on eachfold's training split only",
        },
        "dataset_info": dataset_info,
        "aggregated": df.to_dict(orient="records"),
    }
    summary_path = outdir / f"aggregated_results_{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / f"confusion_matrices_{args.tag}.json").write_text(
        json.dumps(confusion_store, indent=2), encoding="utf-8")

    print(f"\nSaved: {per_fold_path}\nSaved: {summary_path}\nSaved: confusion_matrices_{args.tag}.json")


if __name__ == "__main__":
    main()
