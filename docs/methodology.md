# Methodology (provisional; no kernels implemented)

## Data unit

Each example is one source-labeled ego-network graph. The local files use the TU graph-kernel text layout: global endpoint pairs, node-to-graph indicators, and integer graph labels. They represent unlabeled, unattributed, undirected graphs. The integer label-to-source-network mapping is not supplied by the inspected benchmark files or GenWL loader and remains unresolved. The dataset-specific license also remains unresolved; see [dataset documentation](dataset_documentation.md).

## Baseline

The planned baseline is the standard 1-WL subtree feature map: count each exact refined node label at each included depth and use an inner product across the concatenated count coordinates. It is not implemented. Any later implementation should be checked for symmetric Gram matrices and PSD behavior; numerical eigenvalue checks are diagnostics, not a proof of correctness.

At depth 0, labels are the initial node labels; each refinement combines a node's current label with the multiset of neighbor labels. The exact label count at a given depth is one feature coordinate. Greater depth describes a wider neighborhood and can make exact features sparser. This baseline is distinct from generalized WL: Schulz et al. compare unfolding trees using a structure- and depth-preserving edit distance, cluster similar trees, and count cluster assignments. Their method changes matching; the candidate below leaves exact matching intact. Nonnegative depth coefficients in the WL framework weight whole refinement levels, not individual features by corpus rarity.

## Rarity-weighted candidate

The candidate remains **unimplemented and unvalidated**. For a training fold of `N_train` graphs and a depth-specific WL feature `s`, define

`df(s) = number of training graphs G with c_G(s) > 0`

`a(s) = log((N_train + 1)/(df(s) + 1)) + 1`

where `c_G(s)` is the within-graph count. The candidate vector in the project brief is `phi_a(G)[s] = a(s)c_G(s)`. Its inner product is

`K_a(G,H) = sum_s a(s)^2 c_G(s)c_H(s)`.

Therefore, multiplying counts by IDF and then taking a dot product squares IDF in each kernel contribution. If the intended contribution is weighted by IDF once, use the conceptual feature coordinate `sqrt(a(s))c_G(s)` instead. This is an unresolved specification choice, not an implementation recommendation silently applied. Record the choice explicitly before any coding.

For fixed nonnegative coordinate weights shared by all graphs in a fold, either feature-map inner product is PSD. The smoothed weight is 1 for features present in every training graph; it is finite for all `df`, and a feature unseen in training has the maximum formula value but contributes zero to training-training entries. For held-out graphs, use only the weights learned from that fold's training portion; do not recompute document frequencies using validation/test graphs. The statistic depends only on feature presence and uses no class labels, so it is unsupervised. Any held-out-only feature should receive a predeclared handling based solely on training information.

## Normalization and ablations

Raw count dot products can give larger graphs and graphs with more feature occurrences greater norms. Kernel cosine normalization `K(G,H)/sqrt(K(G,G)K(H,H))` is a reasonable controlled comparison, with explicit handling for zero-norm graphs. It changes the kernel and must be held constant across the IDF and baseline comparison. Per-depth scaling is a separate design choice from per-feature IDF.

The minimum useful ablation separates: (1) standard WL counts, (2) standard WL with graph-level normalization, (3) IDF-weighted counts without graph normalization, and (4) IDF-weighted counts with the same normalization. Compare both once-weighted and squared-IDF conventions only if selected in advance. Vary WL depth separately. Keep splits, classifier, tuning budget, and random seeds paired. Calculate `df` within each training fold to avoid evaluation-distribution leakage; use nested or otherwise fold-local model selection.

This weighting question is exploratory rather than a novelty claim. Published related work includes learned subtree-pattern importance in a Wasserstein WL kernel, IDF-scaled pairwise scores in VeRNAl's edge-labeled WL graphlet matching, and SYS2VEC's TF-IDF weighting of node content before WL hashing and graph embedding. These methods use different feature units and objectives; their existence means a broad claim of new WL feature weighting or first use of IDF with WL is not supportable. See [research-gap analysis](research_gap_analysis.md) and [literature review](literature_review.md).

## Evaluation plan

Only after literature review, label semantics, data reuse terms, and the weight convention are settled, a later study may compare methods using a frozen stratified evaluation protocol. Report accuracy, macro-F1, per-class outcomes, runtime, and uncertainty. Do not interpret planned metrics or existing paper numbers as results for this repository. The generalized WL method is a distinct published method based on unfolding-tree edit distance and clustering; it is not the candidate implemented here.
