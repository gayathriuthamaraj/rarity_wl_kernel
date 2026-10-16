# Experimental protocol (frozen before analysis)

This document records the exact protocol frozen for the Part 4 experiments. It replaces the
earlier planning stub; nothing here is a result.

## Questions addressed

1. Does corpus-rarity weighting of WL subtree features change classification accuracy / macro-F1
   on EGONETS-1..4 compared with the standard WL subtree kernel?
2. How does the refinement depth `h` interact with the weighting?
3. Which rarity function works best, and how large are the runtime costs?
4. Where are errors made (confusion structure), and do they differ between kernels?

## Fixed protocol

| Element | Choice |
| --- | --- |
| Datasets | `EGO-1`, `EGO-2`, `EGO-3`, `EGO-4` (GenWL @ `044e6f62b3a1ad14888d58907192a360cd4fda52`) |
| Splits | Repeated stratified K-fold, K=10 folds x 5 repetitions, seeds 42-46 (`sklearn.StratifiedKFold(shuffle=True)`) |
| Identical splits | Both kernels evaluated on the exact same fold assignments (same `StratifiedKFold` instance per rep) |
| Kernels | `WLSubtreeKernel` (standard); `RarityWeightedWLKernel` with weightings `inverse`, `sqrt_inverse`, `idf`, `binary` |
| Refinement depths | h = 0, 1, 2, 3 (sweep); h = 2 pre-registered as the primary comparison |
| Initial labels | Uniform `"1"` on all nodes (documented convention; graphs are unlabeled) — ablation: degree labeling |
| Classifier | `SVC(kernel="precomputed", C=1.0)` — no per-fold kernel-width tuning; identical for all kernels |
| Leakage control | Every fold refits the kernel on the training split ONLY: WL refinement, label compressions, feature vocabulary, and rarity weights (`fit` on train; `transform` on test; unseen test signatures contribute zero) |
| Metrics | Accuracy, macro-F1, per-class results via aggregated confusion matrices; runtime per fold stage (kernel fit, Gram transform, SVC fit, predict); variability as mean ± SD across 50 folds |
| Statistical analysis | Paired Wilcoxon signed-rank on fold-wise deltas (RW vs standard WL) per variant x depth; bootstrap 95% CI over fold deltas (10k resamples) |
| Software | Python 3.12, numpy, networkx, scikit-learn; kernel implementation in `src/egonets_wl/kernels/` |
| Outputs | `results/fold_wise_results_part4.csv`, `results/per_fold_results_part4.csv`, `results/aggregated_results_part4.json`, `results/confusion_matrices_part4.json`, `results/figures/` |

## Fidelity / sanity requirements

- `rw-uniform` (all weights = 1) reproduces the standard WL kernel exactly; checked logically and
  used only as internal sanity, not an extra result row.
- Preprocessed vocabulary: a test-graph signature absent from training contributes a zero count
  (feature "None" propagation), never a new feature.

## Known deviations from the original plan

- No hyperparameter tuning was foreseen; `C=1.0` was fixed for every kernel and dataset so any
  performance difference reflects the kernel, not tuning. Tuning-per-kernel would require nested
  folds and is listed as future work.
- The planned "normalization" ablation (cosine kernel normalization) is run as a separate sweep
  (`--normalize`), not in the main grid, to keep the main protocol single-variable.
- The planned "degree labeling" ablation is run as a separate sweep (`--initial-labeling degree`).
