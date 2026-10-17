#!/usr/bin/env python3
"""High-Performance Comprehensive Egonet Classification Experiment across all 3 WL Kernel Families:
1. WL Subtree Kernel
2. WL Edge Kernel
3. WL Shortest-Path Kernel

Evaluates standard unweighted baseline vs 4 rarity formulas:
- inverse: w_j = 1 / (total_count_j + eps)
- sqrt_inverse: w_j = 1 / sqrt(total_count_j + eps)
- idf: w_j = log((N_train + 1) / (df_j + 1))
- binary: w_j = 1 / (df_j + eps)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egonets_wl.data.loader import load_tu_dataset
from egonets_wl.kernels.wl_subtree import WLSubtreeKernel
from egonets_wl.kernels.wl_edge import WLEdgeKernel
from egonets_wl.kernels.wl_shortest_path import WLShortestPathKernel
from egonets_wl.kernels.base import normalize_kernel


def compute_weights(X_train, scheme: str) -> np.ndarray:
    """Compute feature weights strictly from training fold count matrix."""
    eps = 1e-12
    N_train = X_train.shape[0]
    if sp.issparse(X_train):
        total_counts = np.asarray(X_train.sum(axis=0)).ravel()
        doc_freqs = np.asarray((X_train > 0).sum(axis=0)).ravel()
    else:
        total_counts = np.sum(X_train, axis=0)
        doc_freqs = np.sum(X_train > 0, axis=0)

    if scheme == "standard":
        return np.ones_like(total_counts, dtype=float)
    elif scheme == "inverse":
        return 1.0 / (total_counts + eps)
    elif scheme == "sqrt_inverse":
        return 1.0 / np.sqrt(total_counts + eps)
    elif scheme == "idf":
        return np.log((N_train + 1.0) / (doc_freqs + 1.0))
    elif scheme == "binary":
        return 1.0 / (doc_freqs + eps)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def evaluate_svm(K_train: np.ndarray, y_train: np.ndarray, K_test: np.ndarray, y_test: np.ndarray, all_classes: np.ndarray):
    """Fit SVC on precomputed kernel and evaluate standard classification metrics."""
    clf = SVC(kernel="precomputed", C=1.0, decision_function_shape="ovr")
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    y_scores = clf.decision_function(K_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    try:
        y_test_bin = label_binarize(y_test, classes=all_classes)
        if y_test_bin.shape[1] == 1:
            auc = float(roc_auc_score(y_test, y_scores))
        else:
            auc = float(roc_auc_score(y_test_bin, y_scores, average="macro", multi_class="ovr"))
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "roc_auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Run complete egonet WL kernels comparison.")
    parser.add_argument("--variants", nargs="+", default=["EGO-1", "EGO-2"])
    parser.add_argument("--families", nargs="+", default=["subtree", "edge", "shortest_path"])
    parser.add_argument("--weightings", nargs="+", default=["standard", "inverse", "sqrt_inverse", "idf", "binary"])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fold_rows = []

    for variant in args.variants:
        data_path = Path("data/raw") / variant
        if not data_path.exists():
            print(f"Skipping {variant}: path does not exist.", flush=True)
            continue

        print(f"\n================ Loading {variant} ================", flush=True)
        graphs = load_tu_dataset(data_path, prefix=variant)
        labels = np.array([g.graph["label"] for g in graphs])
        all_classes = np.unique(labels)
        print(f"Loaded {len(graphs)} graphs with {len(all_classes)} classes: {all_classes}", flush=True)

        for family in args.families:
            for depth in args.depths:
                print(f"\n--- {variant} | Family: {family:13s} | Depth: {depth} ---", flush=True)
                
                # Run repetitions and folds
                for rep in range(args.repetitions):
                    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42 + rep * 100)
                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(graphs, labels)):
                        g_train = [graphs[i] for i in train_idx]
                        y_train = labels[train_idx]
                        g_test = [graphs[i] for i in test_idx]
                        y_test = labels[test_idx]

                        # Instantiate base kernel and extract raw counts ONCE for this fold
                        if family == "subtree":
                            base_kernel = WLSubtreeKernel(n_iter=depth, normalize=False)
                        elif family == "edge":
                            base_kernel = WLEdgeKernel(n_iter=depth, normalize=False)
                        elif family == "shortest_path":
                            base_kernel = WLShortestPathKernel(n_iter=depth, normalize=False)

                        t0 = time.perf_counter()
                        X_train, _ = base_kernel.extract_features(g_train, fit=True)
                        base_kernel.vocabulary_ = getattr(base_kernel, "vocabulary_", {})
                        base_kernel.is_fitted_ = True
                        base_kernel.X_train_features_ = X_train
                        X_test, _ = base_kernel.extract_features(g_test, fit=False)
                        extract_time = time.perf_counter() - t0

                        n_feats = X_train.shape[1]

                        # Evaluate all 5 weightings
                        for weighting in args.weightings:
                            w = compute_weights(X_train, weighting)
                            sqrt_w = np.sqrt(w)

                            if sp.issparse(X_train):
                                X_tr_w = X_train.multiply(sqrt_w)
                                X_te_w = X_test.multiply(sqrt_w)
                                K_train = (X_tr_w @ X_tr_w.T).toarray()
                                K_test = (X_te_w @ X_tr_w.T).toarray()
                            else:
                                X_tr_w = X_train * sqrt_w[None, :]
                                X_te_w = X_test * sqrt_w[None, :]
                                K_train = np.dot(X_tr_w, X_tr_w.T)
                                K_test = np.dot(X_te_w, X_tr_w.T)

                            if args.normalize:
                                diag_tr = np.diag(K_train)
                                diag_te = np.asarray(X_te_w.multiply(X_te_w).sum(axis=1)).ravel() if sp.issparse(X_train) else np.sum(X_te_w ** 2, axis=1)
                                K_train = normalize_kernel(K_train, diag_tr)
                                K_test = normalize_kernel(K_test, diag_tr, diag_test=diag_te)

                            metrics = evaluate_svm(K_train, y_train, K_test, y_test, all_classes)

                            row = {
                                "dataset": variant,
                                "kernel_family": family,
                                "depth": depth,
                                "weighting": weighting,
                                "repetition": rep,
                                "fold": fold_idx,
                                "n_features": n_feats,
                                "extract_time_s": extract_time,
                                **metrics,
                            }
                            fold_rows.append(row)

                # Report summary for this (family, depth) block
                sub_df = pd.DataFrame([r for r in fold_rows if r["dataset"] == variant and r["kernel_family"] == family and r["depth"] == depth])
                for wgt in args.weightings:
                    w_group = sub_df[sub_df["weighting"] == wgt]
                    print(f"  [{wgt:12s}] Acc: {w_group['accuracy'].mean()*100:5.2f}% ± {w_group['accuracy'].std()*100:4.2f}% | F1: {w_group['macro_f1'].mean()*100:5.2f}% | AUC: {w_group['roc_auc'].mean():.3f}", flush=True)

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(outdir / "egonet_all_kernels_folds.csv", index=False)

    # Compute aggregated summary table
    summary_cols = ["accuracy", "precision", "recall", "macro_f1", "roc_auc", "n_features"]
    grouped = df_folds.groupby(["dataset", "kernel_family", "depth", "weighting"])
    
    summary_records = []
    for (dset, fam, dep, wgt), group in grouped:
        rec = {
            "dataset": dset,
            "kernel_family": fam,
            "depth": dep,
            "weighting": wgt,
            "acc_mean": group["accuracy"].mean(),
            "acc_std": group["accuracy"].std(),
            "prec_mean": group["precision"].mean(),
            "prec_std": group["precision"].std(),
            "rec_mean": group["recall"].mean(),
            "rec_std": group["recall"].std(),
            "f1_mean": group["macro_f1"].mean(),
            "f1_std": group["macro_f1"].std(),
            "auc_mean": group["roc_auc"].mean(),
            "auc_std": group["roc_auc"].std(),
            "features_mean": group["n_features"].mean(),
        }
        summary_records.append(rec)

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(outdir / "egonet_all_kernels_summary.csv", index=False)
    
    with open(outdir / "egonet_all_kernels_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print(f"\n================ Study Complete ================", flush=True)
    print(f"Aggregated summary saved to: {outdir / 'egonet_all_kernels_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
