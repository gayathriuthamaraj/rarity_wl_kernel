#!/usr/bin/env python3
"""Generate the Part 4 presentation (PPTX) from the computed results.

Reads:
    results/per_fold_results_part4.csv
    results/stats_summary_part4.json

Outputs:
    reports/part4_presentation.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

RESULTS = Path("results")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
PRIMARY_DEPTH = 2

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x20, 0x20, 0x20)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
RED = RGBColor(0xC0, 0x00, 0x00)


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    return slide


def add_bullets_slide(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.placeholders[1].text_frame
    body.clear()
    for i, (text, level, bold, color) in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = text
        p.level = level
        run = p.runs[0] if p.runs else p.add_run()
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color
    return slide


def add_table_slide(prs, title, df_val, headers, col_widths=None, fontsize=13):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    n_rows, n_cols = df_val.shape[0] + 1, len(headers)
    left, top = Inches(0.4), Inches(1.5)
    width = Inches(9.2)
    height = Inches(min(0.42 * n_rows, 4.6))
    table = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(fontsize)
    for i, row in enumerate(df_val, start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(fontsize)
    return slide


def add_picture_slide(prs, title, image_path):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.shapes.add_picture(str(image_path), Inches(0.7), Inches(1.6), width=Inches(8.6))
    return slide


def main():
    agg = pd.read_csv(RESULTS / "per_fold_results_part4.csv")
    stats = json.loads((RESULTS / "stats_summary_part4.json").read_text(encoding="utf-8"))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(10), Inches(7.5)

    add_title_slide(
        prs,
        "Rarity-Weighted Weisfeiler-Lehman Graph Kernels",
        "Part 4: Controlled experiments, analysis, and findings on EGONETS 1-4\n"
        "Standard WL subtree kernel vs. corpus-rarity weighted variant",
    )

    add_bullets_slide(prs, "Research question", [
        ("Can weighting WL subtree features by corpus frequency improve graph classification "
         "on the four EGONETS benchmarks?", 0, True, None),
        ("Baseline: standard WL subtree kernel (Shervashidze et al., 2011)", 1, False, None),
        ("Candidate: rarity-weighted WL subtree kernel (train-fold-only weights)", 1, False, None),
        ("Hypothesis (exploratory): rare structural patterns carry more class-relevant signal", 1, False, None),
    ])

    add_bullets_slide(prs, "Method: rarity weighting (defined before experiments)", [
        ("Feature vocabulary: WL compressed subtree labels at depths 0..h, from training fold only", 0, False, None),
        ("K_w(G,H) = sum_j w_j * x_G(j) * x_H(j)", 0, True, None),
        ("Weightings compared: 1/(count+eps); 1/sqrt(count+eps); smoothed IDF log((N+1)/(df+1)); 1/(df+eps)", 0, False, None),
        ("Leakage control: weights recomputed inside every training split; unseen test signatures count zero", 0, True, RED),
    ])

    add_bullets_slide(prs, "Evaluation protocol (frozen)", [
        ("4 datasets x depths h in {0,1,2,3} x 5 kernels", 0, False, None),
        ("10-fold stratified CV x 5 repetitions = 50 test folds per cell (seeds 42-46)", 0, False, None),
        ("Identical splits and SVC(kernel='precomputed', C=1.0) for every kernel", 0, False, None),
        ("Metrics: accuracy, macro-F1, runtime stages, variability (mean +/- SD)", 0, False, None),
        ("Statistics: paired Wilcoxon signed-rank on fold-wise deltas; bootstrap 95% CIs", 0, False, None),
    ])

    # Results table at primary depth
    sub = agg[agg.depth == PRIMARY_DEPTH][
        ["variant", "method", "accuracy_mean", "accuracy_std", "macro_f1_mean", "kernel_total_s_median", "n_features_mean"]
    ].copy()
    sub["method"] = sub["method"].map({"wl": "standard", "rw-inverse": "RW 1/count",
                                       "rw-sqrt_inverse": "RW 1/sqrt(c)", "rw-idf": "RW IDF",
                                       "rw-binary": "RW 1/df"})
    sub = sub.round({"accuracy_mean": 3, "accuracy_std": 3, "macro_f1_mean": 3,
                     "kernel_total_s_median": 3, "n_features_mean": 0})
    add_table_slide(prs, f"Main results, h={PRIMARY_DEPTH} (50 folds per cell)", sub.values,
                    ["Dataset", "Kernel", "Acc", "SD", "macro-F1", "kernel s", "feats"])

    bullets = []
    for r in stats["comparisons"]:
        if r["depth"] == PRIMARY_DEPTH:
            text = (f"{r['variant']} {r['rw_method']}: delta_acc {r['delta_accuracy_mean']:+.3f} "
                    f"[CI {r['delta_accuracy_ci95'][0]:+.3f}, {r['delta_accuracy_ci95'][1]:+.3f}]"
                    + (f", Wilcoxon p={r['p_value']:.4f}" if r.get("p_value") is not None else ""))
            bullets.append((text, 0, False, None))
    add_bullets_slide(prs, "Statistical comparison (RW vs standard WL, h=2)", bullets)

    # Figures (if generated)
    for fig in [("Accuracy vs refinement depth (h=3 degrades every kernel)",
                 RESULTS / "figures" / "accuracy_vs_depth_rw-inverse.png"),
                ("Confusion structure shifts with weighting",
                 RESULTS / "figures" / "confusion_EGO-1_rw-inverse.png"),
                ("Runtime is dominated by feature refinement; weighting adds < 2x kernel cost",
                 RESULTS / "figures" / "runtime_h2.png")]:
        title, path = fig
        if path.exists():
            add_picture_slide(prs, title, path)

    add_bullets_slide(prs, "Findings", [
        ("Depth matters more than weighting: all kernels peak near h<=2 but h>=2 feature spaces "
         "explode (up to 77k features on 200 graphs of ~180 nodes)", 0, False, None),
        (" weighting helps at shallow depth h=1 on EGO-1/EGO-2 (e.g., +7.8 and +14.9 accuracy "
         "points) but is not consistent at h=2 and reverses sharply at h=3", 0, False, None),
        ("Document-frequency weightings (IDF, binary) consistently underperform the standard kernel", 0, False, None),
        ("Runtime: rarity weighting adds no material cost at shallow depth; identical feature extraction", 0, False, None),
    ])

    add_bullets_slide(prs, "Limitations and conclusion", [
        ("C fixed at 1.0 without per-kernel tuning; nested-CV tuning could change rankings", 0, False, None),
        ("Unlabeled graphs force initial-label conventions; degree-label ablation reported", 0, False, None),
        ("Integer-to-network class mapping unverified in the GenWL files", 0, False, None),
        ("Conclusion: corpus-rarity weighting is depth- and dataset-dependent; the uniform-IDF "
         "formula is not a general improvement for WL subtree kernels on EGONETS", 0, True, None),
    ])

    out = REPORTS / "part4_presentation.pptx"
    prs.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
