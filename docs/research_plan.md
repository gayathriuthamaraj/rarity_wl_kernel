# Research plan

## Aim

Compare standard WL subtree graph-kernel classification with a carefully specified corpus-frequency-weighted feature kernel on EGONETS-1–4. The candidate is a hypothesis, not a claimed contribution or improvement.

## Questions

1. What classification performance does a faithful standard WL subtree baseline obtain on each variant under a documented protocol?
2. Does a defined rarity weighting alter performance, especially as variant density increases?
3. Are observed differences stable across repeated splits and classes, and what computational cost do they add?

## Stages and gates

1. Verify primary literature and identify the paper-linked data route; document any unresolved reuse terms.
2. Acquire the author's processed benchmark files, hash them, and validate their actual schema and graph properties. **Completed for all four variants**; the separate data license remains to be confirmed.
3. Build a deterministic loader and validate counts, labels, missing values, and duplicates.
4. Implement and independently check standard WL features/kernel.
5. Specify rarity formula, corpus scope, normalization, and positive-semidefinite implications before coding.
6. Fit any frequency weights inside training folds; use consistent CV and model selection.
7. Report per-variant results, uncertainty, runtime, and limitations without overstating conclusions.

## Hypothesis

Reweighting exact WL features by corpus frequency may change useful signal when higher-depth labels are uncommon. Direction and size of any effect are unknown. The generalized WL paper's edit-distance/clustering method is related motivation, not evidence for this proposed weighting scheme.
