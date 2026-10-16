#!/usr/bin/env python3
"""Part 4 analysis: significance tests, comparison tables, and publication figures.

Inputs (produced by scripts/run_experiments.py):
    results/fold_wise_results_part4.csv   one row per (variant, depth, method, rep, fold)
    results/per_fold_results_part4.csv    aggregated per (variant, depth, method)
    results/confusion_matrices_part4.json aggregated confusion matrices

Outputs:
    results/stats_summary_part4.json      paired Wilcoxon + bootstrap CIs per comparison
    results/figures/*.png                 figures used in the report and slides
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RESULTS = Path("results")
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

PRIMARY_DEPTH = 2  # a priori primary refinement depth (main table)

METHOD_LABELS = {
    "wl": "Standard WL",
    "rw-inverse": "RW-1/count",
    "rw-sqrt_inverse": "RW-1/sqrt(count)",
    "rw-idf": "RW-IDF",
    "rw-binary": "RW-binary",
}
COLORS = {"wl": "#4472c4", "rw-inverse": "#c00000", "rw-sqrt_inverse": "#ed7d31",
          "rw-idf": "#2e7d32", "rw-binary": "#6a1b9a"}


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    stats = np.array([statistic(rng.choice(values, size=len(values), replace=True))
                      for _ in range(n_boot)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    folds = pd.read_csv(RESULTS / "fold_wise_results_part4.csv")
    agg = pd.read_csv(RESULTS / "per_fold_results_part4.csv")
    with open(RESULTS / "confusion_matrices_part4.json", encoding="utf-8") as fh:
        confusions = json.load(fh)

    variants = sorted(folds["variant"].unique())
    depths = sorted(folds["depth"].unique())
    methods = ["wl", "rw-inverse", "rw-sqrt_inverse", "rw-idf", "rw-binary"]

    # ------------------------------------------------------------------
    # 1. Paired significance testing: each RW variant vs standard WL,
    #    paired across identical folds (method+variant+depth, rep+fold ids).
    # ------------------------------------------------------------------
    stats_rows = []
    for variant in variants:
        for depth in depths:
            for method in methods[1:]:
                base = folds[(folds.variant == variant) & (folds.depth == depth)
                             & (folds.method == "wl")].astype(
                    {"rep": int, "fold": int}).set_index(["rep", "fold"])
                cand = folds[(folds.variant == variant) & (folds.depth == depth)
                             & (folds.method == method)].astype(
                    {"rep": int, "fold": int}).set_index(["rep", "fold"]) \
                    if method.replace("rw-", "rw-") in set(folds.method) else None
                if cand is None or len(cand) == 0:
                    continue
                common = base.index.intersection(cand.index)
                base_acc = base.loc[common, "accuracy"].values
                cand_acc = cand.loc[common, "accuracy"].values
                diff = cand_acc - base_acc
                nonzero = diff[diff != 0]
                if len(nonzero) >= 5:
                    stat, p = wilcoxon(diff)
                else:
                    stat, p = np.nan, np.nan
                lo, hi = bootstrap_ci(diff)
                cand_f1 = cand.loc[common, "macro_f1"].values
                base_f1 = base.loc[common, "macro_f1"].values
                stats_rows.append({
                    "variant": variant, "depth": int(depth),
                    "rw_method": method,
                    "n_pairs": int(len(common)),
                    "wl_acc_mean": float(base_acc.mean()),
                    f"rw_acc_mean": float(cand_acc.mean()),
                    "delta_accuracy_mean": float(diff.mean()),
                    "delta_accuracy_ci95": [lo, hi],
                    "delta_macro_f1_mean": float((cand_f1 - base_f1).mean()),
                    "wilcoxon_w": None if np.isnan(stat) else float(stat),
                    "p_value": None if np.isnan(p) else float(p),
                    "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                })
    stats = {"primary_depth": PRIMARY_DEPTH, "comparisons": stats_rows}
    (RESULTS / "stats_summary_part4.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    # Best depth per variant for potential per-variant winner analysis
    print("== Paired deltas at primary depth", PRIMARY_DEPTH, "==")
    for r in stats_rows:
        if r["depth"] != PRIMARY_DEPTH:
            continue
        sig = ""
        if r["p_value"] is not None:
            sig = f" p={r['p_value']:.4f}{'*' if r['p_value'] < 0.05 else ''}"
        print(f"{r['variant']} {r['rw_method']:>14s}: "
              f"acc {r['delta_accuracy_mean']:+.4f} [{r['delta_accuracy_ci95'][0]:+.4f}, "
              f"{r['delta_accuracy_ci95'][1]:+.4f}]{sig}")

    # ------------------------------------------------------------------
    # 2. Figure: accuracy vs depth per variant (wl overlay + best RW)
    # ------------------------------------------------------------------
    for meth in methods[1:]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=False)
        for ax, variant in zip(axes, variants):
            sub = agg[(agg.variant == variant)]
            for m in ["wl", meth]:
                d = sub[sub.weighting.isin(
                    {"wl": ["none"], meth: [meth.split("-", 1)[1]]}[m])]
                d = d.sort_values("depth")
                ax.plot(d["depth"], d["accuracy_mean"], marker="o",
                        color=COLORS[m], label=METHOD_LABELS[m])
                ax.fill_between(d["depth"], d["accuracy_mean"] - d["accuracy_std"],
                                d["accuracy_mean"] + d["accuracy_std"],
                                color=COLORS[m], alpha=0.15)
            ax.set_title(variant)
            ax.set_xlabel("WL refinement depth h")
            ax.set_xticks(depths)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Test accuracy (mean +/- sd)")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"Accuracy vs refinement depth: {METHOD_LABELS[meth]}", y=1.04)
        fig.tight_layout()
        fig.savefig(FIGS / f"accuracy_vs_depth_{meth}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Runtime figure at primary depth: kernel vs classifier time
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    sub = agg[agg.depth == PRIMARY_DEPTH]
    x = np.arange(len(sub))
    labels = [f"{r.variant}\n{METHOD_LABELS.get(m, m)}"
              for r in sub.itertuples() for m in [r.method]]
    top = (sub["kernel_fit_s_median"] + sub["kernel_transform_s_median"]).values
    bot = (sub["svc_fit_s_median"] + sub["svc_predict_s_median"]).values
    ax.bar(x, top, label="Kernel (fit + transform)", color="#4472c4")
    ax.bar(x, bot, bottom=top, label="SVC fit + predict", color="#ed7d31")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.variant[:6]}\n{r.method.replace('rw-','RW-')}"
                        for r in sub.itertuples()], rotation=30, ha="right")
    ax.set_ylabel("Wall-clock seconds per fold (median)")
    ax.set_yscale("log")
    ax.set_title(f"Runtime decomposition, h={PRIMARY_DEPTH}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / f"runtime_h{PRIMARY_DEPTH}.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Confusion matrices at primary depth across all folds
    # ------------------------------------------------------------------
    for key, entry in confusions.items():
        variant, depth_tag, method_tag = key.split("|")
        if depth_tag != f"d{PRIMARY_DEPTH}" or method_tag == "wl":
            continue
        cm = np.array(entry["confusion"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        classes = entry["classes"]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, data, title in [
            (axes[0], cm, "counts"),
            (axes[1], cm_norm, "row-normalized"),
        ]:
            im = ax.imshow(data, cmap="Blues")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(title)
            thresh = data.max() / 2
            for i in range(len(classes)):
                for j in range(len(classes)):
                    v = data[i, j]
                    ax.text(j, i, f"{v:.0f}" if title == "counts" else f"{v:.2f}",
                            ha="center", va="center", color="white" if v > thresh else "black",
                            fontsize=8)
        fig.suptitle(f"{variant} h={PRIMARY_DEPTH}: {METHOD_LABELS.get(method_tag, method_tag)}",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / f"confusion_{variant}_{method_tag}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Feature-count explosion figure (shared across all methods)
    # ------------------------------------------------------------------
    feats = agg[agg.weighting == "none"][[
        "variant", "depth", "n_features_mean", "n_features_max"]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for variant in variants:
        d = feats[feats.variant == variant].sort_values("depth")
        ax.plot(d["depth"], d["n_features_mean"], marker="s", label=variant)
    ax.set_xlabel("WL refinement depth h")
    ax.set_ylabel("Number of distinct features (mean over folds)")
    ax.set_title("Feature-space growth (standard WL; identical for all weightings)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "feature_growth.png", dpi=200)
    plt.close(fig)

    print("\nFigures written to", FIGS)


if __name__ == "__main__":
    sys.exit(main())
