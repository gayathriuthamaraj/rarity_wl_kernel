# Project roadmap

1. Complete source and literature verification, including rarity-weighting prior art.
2. Acquire EGONETS variants from a verified primary source; capture provenance and hashes.
3. Inspect format and validate all graphs and labels; document variant-specific statistics.
4. Implement and test data loading and standard WL baseline.
5. Finalize candidate definition and leakage-safe protocol.
6. Implement candidate, run controlled evaluation, and analyze uncertainty.
7. Write final report and archive reproducibility artifacts.

Current progress: All four parts are complete.
- Part 1: Literature review, research gap analysis, dataset provenance, and formalization.
- Part 2: Tested baseline standard WL subtree kernel, dataset loader, and validation.
- Part 3: Rarity-weighted WL subtree kernel implementation, leakage-safe protocol, mathematical sanity checks.
- Part 4: Controlled empirical evaluation on EGO-1..EGO-4 across depths h in {0,1,2,3}, paired statistical significance testing (Wilcoxon + bootstrap CI), ablation studies (weighting functions, depth, normalization, initial labeling), confusion matrix diagnostics, final research report (`reports/final_report_part4.md`), presentation slides (`reports/part4_presentation.pptx`, `reports/presentation_part4.md`), and print-ready document (`reports/hardcopy_part4.md`).
