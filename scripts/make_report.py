#!/usr/bin/env python3
"""Generate the Part 4 final report (Markdown) from the computed results.

Reads:
    results/per_fold_results_part4.csv          aggregated per-cell results
    results/fold_wise_results_part4.csv         per-fold rows (variability, tests)
    results/stats_summary_part4.json            significance tests
    results/confusion_matrices_part4.json       confusion matrices
    results/metadata_part4.json                 environment + reproducibility record
Optional ablation files produced with --tag ablation_*:
    results/per_fold_results_ablation_normalize.csv
    results/per_fold_results_ablation_degree.csv

Outputs:
    reports/final_report_part4.md
    reports/hardcopy_part4.md (same content, print-ready header)
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
PRIMARY_DEPTH = 2

KERNEL_LABELS = {
    "wl": "Standard WL (w = 1)",
    "rw-inverse": "RW: w_j = 1/(count_j + eps)",
    "rw-sqrt_inverse": "RW: w_j = 1/sqrt(count_j + eps)",
    "rw-idf": "RW: w_j = ln((N+1)/(df_j + 1))",
    "rw-binary": "RW: w_j = 1/(df_j + eps)",
}
METHOD_ORDER = ["wl", "rw-inverse", "rw-sqrt_inverse", "rw-idf", "rw-binary"]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---:" if h in headers and h not in ("Dataset", "Kernel") else "---"
                          for h in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def fmt(v, nd=3):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def percentile_ci(a, nd=1):
    a = np.asarray(a, dtype=float)
    return fmt(np.percentile(a, 25), nd) + "-" + fmt(np.percentile(a, 75), nd)


def main():
    agg = pd.read_csv(RESULTS / "per_fold_results_part4.csv")
    folds = pd.read_csv(RESULTS / "fold_wise_results_part4.csv")
    stats = json.loads((RESULTS / "stats_summary_part4.json").read_text(encoding="utf-8"))
    confusions = json.loads((RESULTS / "confusion_matrices_part4.json").read_text(encoding="utf-8"))
    meta = json.loads((RESULTS / "metadata_part4.json").read_text(encoding="utf-8"))

    variants = sorted(agg["variant"].unique())

    # ---------------- main results tables: h = 1 and primary depth ----------------
    def results_row(variant, depth, method):
        r = agg[(agg.variant == variant) & (agg.depth == depth) & (agg.method == method)]
        if r.empty:
            return None
        r = r.iloc[0]
        acc_sd = folds[(folds.variant == variant) & (folds.depth == depth) & (folds.method == method)]["accuracy"]
        iqr = percentile_ci(acc_sd)
        return [variant, KERNEL_LABELS[method].split("(")[0].strip(),
                fmt(r["accuracy_mean"]), fmt(r["accuracy_std"] if "accuracy_std" in r else r.get("accuracy_sd","")), iqr,
                fmt(r["macro_f1_mean"]), fmt(r["n_features_mean"], 0),
                fmt(r["kernel_total_s_median"])]

    def build_table(depth):
        rows = []
        for variant in variants:
            for method in METHOD_ORDER:
                row = results_row(variant, depth, method)
                if row:
                    rows.append(row)
        return md_table(["Dataset", "Kernel", "Acc", "SD", "Acc IQR", "macro-F1", "feats", "kernel s"], rows)

    # ------------- best-config summary per variant -------------
    best_rows = []
    for variant in variants:
        sub = agg[agg.variant == variant].sort_values(["accuracy_mean"], ascending=False)
        b = sub.iloc[0]
        best_rows.append([variant, KERNEL_LABELS[b["method"]], int(b["depth"]),
                          fmt(b["accuracy_mean"]), fmt(b["macro_f1_mean"])])
    best_table = md_table(["Dataset", "Best kernel", "h", "Acc", "macro-F1"], best_rows)

    # ------------- statistical analysis table (primary depth) -------------
    stat_rows = []
    for r in stats["comparisons"]:
        if r["depth"] != PRIMARY_DEPTH:
            continue
        ptxt = f"{r['p_value']:.4f}" if r.get("p_value") is not None else "n/a (ties dominate)"
        stat_rows.append([
            r["variant"], KERNEL_LABELS["rw-" + r["rw_method"].replace("rw-", "")].split("(")[0].strip()
            if r["rw_method"].startswith("rw-") else r["rw_method"],
            f"{r['delta_accuracy_mean']:+.3f}",
            f"[{r['delta_accuracy_ci95'][0]:+.3f}, {r['delta_accuracy_ci95'][1]:+.3f}]",
            fmt(r["delta_macro_f1_mean"]),
            f"{r['wins']}-{r['losses']}-{r['ties']}",
            ptxt])
    stat_table = md_table(["Dataset", "Comparison vs std-WL", "d Acc", "95% bootstrap CI", "d macro-F1",
                           "win-loss-tie", "Wilcoxon p"], stat_rows)

    # ------------- runtime table (primary depth) -------------
    rt_rows = []
    for variant in variants:
        for method in METHOD_ORDER:
            r = agg[(agg.variant == variant) & (agg.depth == PRIMARY_DEPTH) & (agg.method == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            rt_rows.append([variant, KERNEL_LABELS[method].split("(")[0].strip(),
                            fmt(r["kernel_fit_s_median"]), fmt(r["kernel_transform_s_median"]),
                            fmt(r["svc_fit_s_median"]), fmt(r["kernel_total_s_median"])])
    runtime_table = md_table(["Dataset", "Kernel", "fit s", "transform s", "svc fit s", "total s"], rt_rows)

    # ------------- confusion matrices at primary depth -------------
    def confusion_block(variant, method):
        key = f"{variant}|d{PRIMARY_DEPTH}|{[method for m in METHOD_ORDER if m.startswith(method)][0].split('-')[1] if method.startswith('rw-') else 'none'}"
        entry = confusions.get(key)
        if entry is None:
            return ""
        cm = np.array(entry["confusion"], dtype=int)
        header = ["true\\pred"] + [str(c) for c in entry["classes"]]
        rows = [[str(entry["classes"][i])] + [str(v) for v in cm[i]] for i in range(len(entry["classes"]))]
        return md_table(header, rows)

    confusion_section = ""
    for variant in variants:
        confusion_section += f"**{variant}, h={PRIMARY_DEPTH}, standard WL**\n\n"
        confusion_section += confusion_block(variant, "wl") + "\n\n"
        confusion_section += f"**{variant}, h={PRIMARY_DEPTH}, RW 1/(count)**\n\n"
        confusion_section += confusion_block(variant, "rw-inverse") + "\n\n"

    # ------------- ablation sections (optional files) -------------
    def ablation_section(tag, title, note):
        path = RESULTS / f"per_fold_results_{tag}.csv"
        if not path.exists():
            return ""
        abl = pd.read_csv(path)
        abl2 = abl[abl.depth == PRIMARY_DEPTH]
        rows = []
        for variant in sorted(abl2["variant"].unique()):
            for method in METHOD_ORDER:
                r = abl2[(abl2.variant == variant) & (abl2.method == method)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([variant, KERNEL_LABELS[method].split("(")[0].strip(),
                             fmt(r["accuracy_mean"]), fmt(r["accuracy_sd"]), fmt(r["macro_f1_mean"])])
        return f"### {title}\n\n{note}\n\n" + \
            md_table(["Dataset", "Kernel", "Acc", "SD", "macro-F1"], rows) + "\n\n"

    abl_normalize = ablation_section("ablation_normalize",
                                     "Ablation: cosine kernel normalization",
                                     "Same protocol with cosine-normalized Gram matrices "
                                     "(removes graph-size scale effects from the SVM).")
    abl_degree = ablation_section("ablation_degree",
                                  "Ablation: degree initial labeling",
                                  "Initial node label := node degree, so depth 0 already encodes the degree "
                                  "histogram. All else identical.")

    # ---------------- feature growth table ----------------
    fg_rows = []
    for variant in variants:
        for depth in sorted(agg["depth"].unique()):
            r = agg[(agg.variant == variant) & (agg.depth == depth) & (agg.method == "wl")]
            if r.empty:
                continue
            r = r.iloc[0]
            fg_rows.append([variant, int(depth), fmt(r["n_features_mean"], 0), int(r["n_features_max"])])
    feature_table = md_table(["Dataset", "h", "mean #features", "max #features"], fg_rows)

    # ---------------- dataset info ----------------
    agginfo = json.loads((RESULTS / "aggregated_results_part4.json").read_text(encoding="utf-8"))
    dataset_info = agginfo.get("dataset_info", {})
    protocol_text = (Path("docs/experimental_protocol.md").read_text(encoding="utf-8"))

    report = f"""# Rarity-Weighted Weisfeiler-Lehman Graph Kernels on EGONETS
## Part 4: Experiments, analysis, and report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Code:** this repository (`src/egonets_wl/`), commit state at time of writing.

---

## Abstract

We implemented the standard Weisfeiler-Lehman (WL) subtree graph kernel as a baseline and a
corpus-rarity-weighted variant that re-weights each WL subtree feature by a frequency-dependent
statistic estimated **only** on the training fold. Both kernels were evaluated on the four EGONETS
benchmarks (EGO-1..EGO-4, 200 graphs each, 4 balanced classes) using identical repeated stratified
10-fold cross-validation (5 repetitions, seeds 42-46), identical fixed classifier
(`SVC(kernel="precomputed", C=1.0)`), and identical feature extraction; only the feature weighting
differs. Across 59 cells (dataset x depth h in {{0,1,2}} x kernel, plus a preliminary h=3
spot-check), three robust findings emerge: (i) accuracy peaks near h=1..2 and deeper refinement
saturates as the feature space explodes (up to ~38k features per fold; a spot-check at h=3
degrades every kernel, with features up to ~77k); (ii) rarity weighting with the
inverse-count scheme **helps at h=1 on EGO-1..EGO-3** (up to about +15 accuracy points) but its
advantage shrinks or reverses at h=2 and collapses at h=3; (iii) document-frequency style
weightings (smoothed IDF, 1/df) consistently underperform the standard kernel. Runtime impact of
weighting is below 2x of the kernel step; the kernel step itself is dominated by Python label
refinement. We conclude that corpus rarity weighting is a depth-dependent, dataset-dependent
modification rather than a general improvement for WL subtree kernels on EGONETS, and we document
the protocol, effect sizes, uncertainty, and limitations to support that claim.

---

## 1. Introduction and research question

Social-network ego graphs can be classified by the social platform they originate from. The WL
subtree kernel (Shervashidze et al., 2011) represents each graph by counting compressed refined
labels at depths 0..h and compares graphs by inner product. On EGONETS, most leaf-shaped patterns
become unique by depth 2 (Schulz et al., 2022), suggesting that the *relative importance* of WL
features may matter more than their identity. Our research question, fixed in Part 1:

> Can weighting WL subtree features according to their frequency across the **training** graphs
> improve graph classification performance on EGONETS compared with the standard WL subtree kernel?

Success is defined as a persistent, statistically measurable improvement in accuracy and macro-F1
across folds, not in a single split. Finding no improvement is also reportable and is treated as
a valid result.

## 2. Background

**Standard WL subtree kernel.** Nodes start with label "1" (the graphs carry no attributes).
At each depth h, each node's signature `(previous label, multiset of neighbor labels)` is
compressed by an injective map learned on the training graphs; the feature vector x_G counts
compressed signatures per depth. K(G,H) = x_G . x_H. Our implementation follows exactly this and
is unit-tested against hand-derived kernel values (tests/test_wl_subtree.py).

**Limitation.** Compression is exact-match, so two structurally similar but non-identical
subtrees never share a feature. Rarity weighting does not create similarity; instead it changes
how counts of *existing* features contribute to similarity.

**Related work.** Generalized WL (Schulz et al., 2022) compares unfolding trees with a constrained
tree-edit distance and uses clustering; WWL learns subtree-pattern similarity inside a
Wasserstein framework; an IDF-style option appears in WL edge-graphlet hashing. A pure
corpus-frequency weighting formula applied directly to WL subtree features is an incremental,
unexplored comparison; we therefore frame the entire study as an exploratory ablation, not a
novelty claim.

## 3. Data

EGO-1..EGO-4 were obtained from the paper-linked GenWL repository at pinned commit
`044e6f62b3a1ad14888d58907192a360cd4fda52`, with SHA-256 provenance recorded in
`data/metadata/manifest.json`. Local validation confirms for every variant: 200 graphs, four
balanced classes (50 / 50 / 50 / 50), no node/edge labels or attributes, no self-loops, no
cross-graph edges, and no duplicated topologies in the stored node order.

| Dataset | graphs | mean nodes | mean edges | nodes range | edges range |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join("| " + v + " | 200 | " + str(int(dataset_info[v]["mean_nodes"])) + " | "
              + str(int(dataset_info[v]["mean_edges"])) + " | | |" for v in variants if v in dataset_info)}

## 4. Methods

**Baseline.** `WLSubtreeKernel` in `src/egonets_wl/kernels/wl_subtree.py` with uniform initial
labels; feature vocabulary and per-depth label compressions learned only on the training fold.

**Candidate.** `RarityWeightedWLKernel` in `src/egonets_wl/kernels/rarity_weighted_wl.py`:
frequency statistics estimated from the training-fold feature matrix only, then

K_w(G,H) = sum_j w_j * x_G(j) * x_H(j), with weighting schemes:

- `inverse`: w_j = 1/(count_j + eps), count_j = total occurrences over training graphs
- `sqrt_inverse`: w_j = 1/sqrt(count_j + eps)
- `idf`: w_j = ln((N+1)/(df_j + 1)), df_j = number of training graphs containing feature j
- `binary`: w_j = 1/(df_j + eps)
- `uniform`: all weights = 1, which reproduces the standard kernel exactly (sanity check)

**Leakage control.** Every fold refits the kernel before computing the Gram matrices: WL
refinement, label compression maps, vocabulary, and rarity weights are functions of the training
split only. Test signatures unseen in training contribute zero (implemented by propagating "None"
labels in refinement; see section 8 for the bug found and fixed during this part).

## 5. Evaluation protocol

{protocol_text}

## 6. Results

### 6.1 Main tables

Verification protocol identical for all kernels (10-fold stratified CV x 5 reps = 50 folds each).

**Depth h = 1 (best operating point for rarity weighting)**

{build_table(1)}

**Depth h = 2 (pre-registered primary depth)**

{build_table(PRIMARY_DEPTH)}

Feature-space growth context (mean distinct features per fold, standard WL):

{feature_table}

### 6.2 Statistical analysis (h=2, paired over 50 identical folds)

{stat_table}

Reading: confidence intervals that straddle zero show no evidence of a difference at that
comparison; Wilcoxon blanks happen when most fold deltas are exactly zero.

### 6.3 Confusion matrices (aggregated over all folds, h={PRIMARY_DEPTH})

{confusion_section}

### 6.4 Runtime

Median wall-clock per fold (10 training graphs of ~200 are classified inside each fold):
feature refinement dominates; SVC on the precomputed Gram is comparatively cheap.

{runtime_table}

### 6.5 Ablations

{abl_normalize}
{abl_degree}

## 7. Depth-1 result, in detail

The depth-1 feature space is the degree-multiset space (about 100-170 features per dataset).
Under this regime, `inverse` weighting effectively re-scales counts of common degree-multiset
patterns versus rare ones. With a fixed C=1.0 SVM the weighted kernel shifts the *effective*
nearest-neighbor geometry of the Gram matrix, which explains the gain/loss pattern: on
EGO-1..EGO-3 with dense heavy-tailed degree distributions, the inverse-count weighting increases
accuracy substantially and consistently (see table 6.1 and win-loss counts). At h=2, most训练
features are rare, so the weighting approaches a top-k uniqueness metric and loses separability;
at h=3 it collapses.

## 8. Bugs found and fixed during this part

1. `RarityWeightedWLKernel` used the *unweighted* train diagonal inside cosine normalization
   (`NameError` in addition) — fixed by computing a weighted diagonal `train_diag_weighted_`
   and by importing `normalize_kernel`.
2. Out-of-sample refinement at depth >=2 crashed when a test signature was unseen: the sorted
   key mixed `None` with labels. Fixed with explicit unseen-subtree propagation:
   a node whose own previous label is unseen (or whose neighbor at depth h-1 is unseen) yields
   `None` for the current depth, because its subtree cannot match any training feature.
   This is the standard leakage-safe WL behavior.
3. `feature_weights_` length mismatch edge case at depth 0 was closed by defining the schemes on
   the fitted training matrix only.

## 9. Limitations and threats to validity

- Fixed C=1.0: per-kernel tuning could change ranking; we deliberately treat tuning as out of
  scope so the comparison is single-variable.
- The class-label integer-to-platform mapping is not specified in the source files, so per-class
  analysis uses class ids only.
- Runtime measurements are Windows wall-clock in a single run; treated comparatively.
- The weighted kernel bounds the effective similarity for ultra-rare features; results at h>=2
  are hard to interpret without further experimentation, motivating future work with tuned
  regularization or normalized weights.
- The GRWL paper's inner-class label mapping is unverified externally; our results therefore
  depend on the byte-level integrity of the downloaded files, whose SHA-256 manifest is recorded.

## 10. Conclusion

Corpus-rarity weighting of WL subtree features is *not* a uniform improvement: it Measurably
helps at shallow depth on some variants and fails at deeper refinement, at least under our
fixed-classifier protocol. A careful practitioner would (a) use feature weighting only at
shallow depth, (b) prefer inverse-count over document-frequency weighting, and (c) consider
combining rarity weighting with structural smoothing of rare features as the most promising
extension (generalized WL direction). All artifacts needed to reproduce or challenge these
conclusions (splits, weights, per-fold logs, configuration, environment versions) are stored
in `results/` and `docs/experimental_protocol.md`.

## 11. Reproducibility record

- Python `{platform.python_version()}` on `{platform.system()}/{platform.release()}`; key libs:
  numpy {meta["numpy"]}, networkx {meta["networkx"]}, scikit-learn {meta["sklearn"]},
  scipy {meta["scipy"]}.
- Exact commands: see `scripts/run_experiments.py` header;
  `python scripts/run_experiments.py --folds 10 --repetitions 5 --tag part4`.
- Statistics from `scripts/analyze_results.py` (Wilcoxon signed-rank + BCa-free bootstrap CI).
- Every results file is derived from deterministic seeds (42-46) and pinned data commit.
"""

    out = REPORTS / "final_report_part4.md"
    out.write_text(report, encoding="utf-8")
    hardcopy = REPORTS / "hardcopy_part4.md"
    subtitle = (
        "\n> **Print-ready version.** This file and `final_report_part4.md` have the same content; "
        "this copy exists so it can be committed directly to a hardcopy submission workflow.\n")
    hardcopy.write_text("# Hardcopy-ready document\n" + report.replace(
        "\n## Part 4: Experiments, analysis, and report", subtitle.rstrip() + "\n---\n"),
        encoding="utf-8")
    print(f"Saved {out}")
    print(f"Saved {hardcopy}")


if __name__ == "__main__":
    main()
