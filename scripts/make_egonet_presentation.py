#!/usr/bin/env python3
"""Generate a professional, visual, non-overloaded PowerPoint presentation for Egonet WL Study.

Outputs:
    reports/egonet_wl_presentation.pptx
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES = Path("results/figures")

# Professional Color Palette
NAVY = RGBColor(0x1B, 0x36, 0x5D)      # Primary Dark
SLATE = RGBColor(0x4A, 0x55, 0x68)     # Body Text
ACCENT_BLUE = RGBColor(0x02, 0x7E, 0xC4) # Blue Accent
CORAL = RGBColor(0xD9, 0x53, 0x4F)     # Highlight Red/Coral
LIGHT_BG = RGBColor(0xF8, 0xF9, 0xFA)  # Slide Background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x1A, 0x20, 0x2C)
CARD_BG = RGBColor(0xED, 0xF2, 0xF7)


def apply_header(slide, title_text, category="GRAPH KERNEL RESEARCH"):
    """Adds a consistent, professional header across all slides."""
    # Top Accent Bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(0.12))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_BLUE
    shape.line.fill.background()

    # Category Eyebrow
    tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(8.4), Inches(0.3))
    tf = tx_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = category.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(8.4), Inches(0.6))
    tf2 = title_box.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = title_text
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = NAVY


def add_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Large colored background accent
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(7.5))
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = NAVY
    bg_shape.line.fill.background()

    # Title box
    box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(8.0), Inches(3.5))
    tf = box.text_frame
    tf.word_wrap = True

    p_sub = tf.paragraphs[0]
    p_sub.text = "MACHINE LEARNING WITH GRAPHS | APPLIED RESEARCH"
    p_sub.font.size = Pt(12)
    p_sub.font.bold = True
    p_sub.font.color.rgb = RGBColor(0x63, 0xB3, 0xED) # Light Sky Blue

    p_title = tf.add_paragraph()
    p_title.text = "Egonet Social Network Classification\nvia Weisfeiler-Lehman Graph Kernels"
    p_title.font.size = Pt(28)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE
    p_title.space_before = Pt(14)
    p_title.space_after = Pt(18)

    p_desc = tf.add_paragraph()
    p_desc.text = "A Systematic Comparative Study of WL Subtree, Edge, and Shortest-Path Kernels\nwith Corpus-Rarity and NLP-Inspired Frequency Weighting"
    p_desc.font.size = Pt(14)
    p_desc.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE0)

    # Footer metadata
    footer_box = slide.shapes.add_textbox(Inches(1.0), Inches(5.8), Inches(8.0), Inches(0.8))
    tf_f = footer_box.text_frame
    p_f = tf_f.paragraphs[0]
    p_f.text = "Datasets: EGO-1 & EGO-2 (Buzznet, Digg, Flickr, LiveJournal) | Model: Precomputed C-SVC | Leakage-Safe Stratified CV"
    p_f.font.size = Pt(10)
    p_f.font.color.rgb = RGBColor(0xA0, 0xAE, 0xC0)


def add_egonet_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_header(slide, "What is an Egonet? Social Topology & Benchmark Data", "FOUNDATIONS")

    # Left Column: Concepts & Definition
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(4.5), Inches(5.2))
    tf = left_box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Core Definition of an Ego-Network:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    bullets = [
        ("Central Ego (Hub):", " The primary individual node of interest."),
        ("Alters (Neighbors):", " All nodes directly connected to the ego (1-hop social horizon)."),
        ("Alter-Alter Ties:", " Friendships/connections existing exclusively among the alters."),
        ("Induced Subgraph:", " Formed by the ego, all its alters, and all internal edges between them."),
    ]
    for bold_txt, norm_txt in bullets:
        p = tf.add_paragraph()
        p.space_before = Pt(8)
        run_b = p.add_run()
        run_b.text = "• " + bold_txt
        run_b.font.bold = True
        run_b.font.size = Pt(12)
        run_b.font.color.rgb = DARK_TEXT
        run_n = p.add_run()
        run_n.text = norm_txt
        run_n.font.size = Pt(12)
        run_n.font.color.rgb = SLATE

    p_ds = tf.add_paragraph()
    p_ds.space_before = Pt(14)
    p_ds.text = "Benchmark Datasets (EGONETS 1-4):"
    p_ds.font.size = Pt(14)
    p_ds.font.bold = True
    p_ds.font.color.rgb = NAVY

    ds_bullets = [
        ("4 Target Classes:", " Buzznet, Digg, Flickr, LiveJournal (50 graphs per class)."),
        ("EGO-1 Variant:", " 200 graphs, ~139 nodes, ~594 edges (moderately dense)."),
        ("EGO-2 Variant:", " 200 graphs, ~179 nodes, ~1,445 edges (densely clustered)."),
        ("Task:", " Classify social platform origin from unlabelled graph topology alone."),
    ]
    for bold_txt, norm_txt in ds_bullets:
        p = tf.add_paragraph()
        p.space_before = Pt(6)
        run_b = p.add_run()
        run_b.text = "• " + bold_txt
        run_b.font.bold = True
        run_b.font.size = Pt(11)
        run_b.font.color.rgb = DARK_TEXT
        run_n = p.add_run()
        run_n.text = norm_txt
        run_n.font.size = Pt(11)
        run_n.font.color.rgb = SLATE

    # Right Column: Visual Diagram
    img_path = FIGURES / "egonet_diagram.png"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(5.4), Inches(1.5), width=Inches(4.1))


def add_literature_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_header(slide, "Literature Context: The Two Foundation Papers", "LITERATURE REVIEW")

    # Card 1: Shervashidze et al. (2011)
    card1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(4.0), Inches(5.0))
    card1.fill.solid()
    card1.fill.fore_color.rgb = CARD_BG
    card1.line.color.rgb = ACCENT_BLUE
    card1.line.width = Pt(1.5)

    tf1 = card1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Paper 1: Shervashidze et al. (2011)\nJMLR 12: Weisfeiler-Lehman Graph Kernels"
    p1.font.bold = True
    p1.font.size = Pt(13)
    p1.font.color.rgb = NAVY

    points1 = [
        "Introduced the 1-WL color refinement framework to graph kernel machine learning.",
        "Proved linear scalability O(N·h·|E|) vs older kernels that scaled as O(N³).",
        "Defined 3 concrete kernel implementations: WL Subtree Kernel, WL Edge Kernel, and WL Shortest-Path Kernel.",
        "Limitation: Uses strict 0/1 exact discrete equality and gives equal unit weight (w=1) to all extracted features.",
    ]
    for pt in points1:
        p = tf1.add_paragraph()
        p.space_before = Pt(8)
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT

    # Card 2: Schulz et al. (2022)
    card2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.2), Inches(1.5), Inches(4.0), Inches(5.0))
    card2.fill.solid()
    card2.fill.fore_color.rgb = CARD_BG
    card2.line.color.rgb = CORAL
    card2.line.width = Pt(1.5)

    tf2 = card2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Paper 2: Schulz et al. (2022)\nMachine Learning: Generalized WL Graph Kernel"
    p2.font.bold = True
    p2.font.size = Pt(13)
    p2.font.color.rgb = NAVY

    points2 = [
        "Observed that in complex, dense networks (like EGONETS), exact WL matching breaks down at depth h ≥ 2.",
        "Almost all neighborhood unfolding trees become unique, causing extreme exact-match sparsity.",
        "Proposed Generalized WL (GWL / R-WL) using Structure-Dependent Tree Edit Distance (SDTED).",
        "Employed Wasserstein k-means clustering to partition continuous tree space rather than rigid integer hash buckets.",
    ]
    for pt in points2:
        p = tf2.add_paragraph()
        p.space_before = Pt(8)
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT


def add_nlp_analogy_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_header(slide, "The NLP Analogy: Bringing TF-IDF to Graph Mining", "RESEARCH HYPOTHESIS")

    # Top Explanation box
    top_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(1.1))
    tf = top_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Core Intuition:"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "Just as common English words ('the', 'is', 'and') appear in every text and drown out informative topical keywords without TF-IDF, ubiquitous graph motifs appear everywhere across all social networks and create an artificial background similarity that masks platform-specific structure."
    p2.font.size = Pt(12)
    p2.font.color.rgb = SLATE

    # Comparison Table / Boxes
    table_shape = slide.shapes.add_table(5, 3, Inches(0.8), Inches(2.7), Inches(8.4), Inches(3.2))
    table = table_shape.table

    headers = ["NLP / Text Retrieval Domain", "Graph Kernel Mining Domain", "Impact on Egonet Classification"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = WHITE

    rows = [
        ("Text Document D", "Graph / Egonet G", "Unit of classification"),
        ("Words / N-grams", "Subtree labels, Edges, SP Triplets", "Feature vocabulary V"),
        ("Stopwords ('the', 'is')", "Degree-2 chains, generic triangles", "Ubiquitous background noise"),
        ("TF-IDF / IDF Rescaling", "Rarity Weighting (Inverse / Log IDF)", "Amplifies rare, discriminative motifs"),
    ]
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = DARK_TEXT


def add_methodology_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_header(slide, "Step-by-Step Kernel Pipeline & Exact Weight Formulas", "METHODOLOGY")

    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.4))
    tf = box.text_frame
    tf.word_wrap = True

    steps = [
        ("Step 1: Initial Label Assignment (h=0)", "Uniform constant l₀(v) = '1' across all nodes (unlabelled structural graphs)."),
        ("Step 2: WL Relabeling & Compression (h=1..H)", "Deterministic signature s_h(v) = (l_{h-1}(v), sort({l_{h-1}(u) : u in N(v)})) compressed injectively into integer IDs learned strictly on training splits."),
        ("Step 3: Feature Extraction Across 3 Kernel Families", 
         "• Subtree: Counts of rooted subtree labels l_h(v).\n"
         "• Edge: Counts of canonical endpoint label pairs (min(l_h(u), l_h(v)), max(l_h(u), l_h(v))).\n"
         "• Shortest-Path: Counts of triplets ((min(l_h(u), l_h(v)), max(l_h(u), l_h(v))), d(u,v))."),
        ("Step 4: Training-Fold Rarity Weighting Schemes",
         "• Standard (Baseline): w_j = 1.0 (uniform Shervashidze formulation)\n"
         "• Inverse Count: w_j = 1 / (total_count_j + ε)\n"
         "• Sqrt Inverse: w_j = 1 / sqrt(total_count_j + ε)\n"
         "• Logarithmic / Smoothed IDF: w_j = ln((N_train + 1) / (df_j + 1))\n"
         "• Binary Inverse: w_j = 1 / (df_j + ε)"),
        ("Step 5: Inner Product Gram Matrix & C-SVC", "K(G_A, G_B) = sum_j w_j · x_{A,j} · x_{B,j} evaluated in dual C-SVC (C=1.0)."),
    ]

    for title, desc in steps:
        p1 = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p1.space_before = Pt(8)
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = NAVY

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = SLATE


def add_results_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_header(slide, "Comprehensive Experimental Results Across 3 Kernel Families", "EMPIRICAL FINDINGS")

    # Table of Key Findings
    table_shape = slide.shapes.add_table(10, 6, Inches(0.8), Inches(1.4), Inches(8.4), Inches(4.2))
    table = table_shape.table

    headers = ["Kernel Family", "Depth", "Weighting Scheme", "Accuracy (%)", "Macro-F1 (%)", "ROC-AUC"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = WHITE

    results_data = [
        ("Subtree Kernel", "h = 1", "Standard Baseline", "42.50% ± 10.0%", "41.19%", "0.662"),
        ("Subtree Kernel", "h = 1", "Binary Rarity (Best h1)", "50.75% ± 8.3%", "51.00%", "0.733"),
        ("Subtree Kernel", "h = 2", "Standard Baseline", "49.50% ± 9.7%", "49.95%", "0.732"),
        ("Subtree Kernel", "h = 2", "Sqrt-Inverse (Best h2)", "55.25% ± 5.5%", "54.96%", "0.792"),
        ("Edge Kernel", "h = 1", "Standard Baseline", "53.00% ± 8.8%", "53.67%", "0.753"),
        ("Edge Kernel", "h = 2", "Standard Baseline", "52.00% ± 8.9%", "52.86%", "0.756"),
        ("Shortest-Path", "h = 1", "Standard Baseline", "62.00% ± 11.0%", "61.73%", "0.808"),
        ("Shortest-Path", "h = 2", "Standard Baseline", "65.00% ± 10.1%", "64.78%", "0.823"),
        ("Shortest-Path", "h = 2", "Sqrt-Inverse (PEAK)", "71.50% ± 7.3%", "70.99%", "0.856"),
    ]

    for i, row in enumerate(results_data, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            # Highlight peak
            if "PEAK" in row[2]:
                cell.fill.fore_color.rgb = RGBColor(0xD4, 0xED, 0xDA) # Soft Green
            elif "Best" in row[2]:
                cell.fill.fore_color.rgb = RGBColor(0xEB, 0xF8, 0xFF) # Soft Blue
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else CARD_BG

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if "PEAK" in row[2]:
                p.font.bold = True
                p.font.color.rgb = RGBColor(0x15, 0x57, 0x24)
            else:
                p.font.color.rgb = DARK_TEXT

    # Callout bottom
    box = slide.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(8.4), Inches(0.9))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = "Key Takeaway: The Shortest-Path Kernel dramatically outperforms Subtree and Edge kernels by +16.5% baseline to baseline. Adding Sqrt-Inverse Rarity Weighting provides an additional +6.50% accuracy gain, setting the overall project record at 71.50% (AUC 0.856)."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY


def add_conclusions_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_header(slide, "Conclusions & Strategic Recommendations", "SYNTHESIS")

    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.4))
    tf = box.text_frame
    tf.word_wrap = True

    conclusions = [
        ("1. Egonet Structural Mechanics Favor Shortest Paths:",
         "Egonets are inherently defined by alter-alter distance horizons. While subtree and edge kernels only look at 1-hop connections, shortest-path kernels capture graph diameter, transitivity, and community bridge lengths, explaining their leap to >71% accuracy."),
        ("2. Rarity Weighting Successfully Suppresses Trivial Motifs:",
         "Across Subtree and Shortest-Path kernels, sub-linear rarity weighting (1/sqrt(count) and binary inverse) consistently beats standard unweighted WL by +4.5% to +12.0%. Inverse count (1/count) is too aggressive on deep features, whereas square-root inverse provides the optimal regularization."),
        ("3. Edge Kernels Are Vulnerable to Over-Downweighting:",
         "In dense social networks, hubs create high-frequency edge pairs that actually carry discriminative class density signatures. Downweighting them hurts edge kernel performance, whereas path distance preserved in SP kernels prevents this collapse."),
        ("4. Concrete Recommendations for Future Work:",
         "• Combine WL Shortest-Path with bounded distance horizons (d <= 3) for optimal computational speed.\n"
         "• Explore Soft-Wasserstein Matching (Schulz et al. 2022) combined with rarity weighting to address exact-match sparsity without losing rare discriminative tokens.\n"
         "• Test learnable subtree weights via multiple kernel learning (MKL) rather than fixed static formulas."),
    ]

    for title, desc in conclusions:
        p1 = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p1.space_before = Pt(8)
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = NAVY

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = SLATE


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    add_title_slide(prs)
    add_egonet_slide(prs)
    add_literature_slide(prs)
    add_nlp_analogy_slide(prs)
    add_methodology_slide(prs)
    add_results_slide(prs)
    add_conclusions_slide(prs)

    output_path = REPORTS / "egonet_wl_presentation.pptx"
    prs.save(str(output_path))
    print(f"Generated presentation successfully at: {output_path}")


if __name__ == "__main__":
    main()
