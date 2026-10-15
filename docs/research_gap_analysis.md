# Research-gap analysis

## Original motivation

The candidate motivation is that exact WL subtree matches can become sparse at greater refinement depth, especially on dense or structurally varied graphs. A training-corpus document-frequency statistic might change the influence of common versus rare exact patterns while preserving the WL feature identities. This is a hypothesis, not a demonstrated EGONETS result.

## Related work comparison

| Approach | Feature/similarity mechanism | How it relates |
|---|---|---|
| Standard WL subtree kernel (Shervashidze et al., 2011) | Inner product of exact WL-label count vectors, summed across depths | Baseline. No pattern-specific corpus frequency weighting in the standard definition. |
| Generalized WL / R-WL (Schulz et al., 2022) | Structure- and depth-preserving edit distances between unfolding trees; cluster and count similar labels | Addresses sparse exact matches by changing pattern matching. Not IDF. Also allows nonnegative weights across depths, not per-pattern corpus weights. |
| Learned subtree-pattern importance in WWL (Nguyen et al., 2021) | Paper abstract says subtree-pattern weights are learned for WWL kernels | Strong conceptual precedent for pattern-specific weighting. Exact relation to this candidate cannot be assessed fully from the publisher/author-repository records and accessible abstract; the full text endpoint returned 403 in this review. It is not established here as mathematically equivalent. |
| VeRNAl (Oliver et al., 2021/2022) | Supplement reports trying IDF terms that multiply pairwise comparison scores in an edge-labeled WL graphlet-hashing method for RNA motif matching | Direct precedent for IDF in a WL-based representation. It is not the same documented formula/task: IDF is within graphlet pairwise matching; this candidate weights exact WL histogram coordinates using graph-level document frequency. The supplement does not give enough detail to decide full mathematical equivalence. |
| Proposed candidate | Unsupervised train-fold graph document frequency, then per-coordinate IDF rescaling of exact WL counts | A simple IDF-style transfer to graph feature histograms; its exact prior-art status remains open. |

## What is known

- WL subtree kernels already use explicit feature counts for exact depth-specific labels.
- The generalized WL method changes neighborhood comparison through edit-distance-based similarity and clustering, and reports that most depth-2 EGONETS unfolding trees are unique. That finding supports the sparsity motivation but does not show IDF weighting works.
- Published work has explored learning the importance of subtree patterns in a WL-related Wasserstein kernel. Consequently, the broad claim “weighting WL patterns is novel” is not defensible.
- IDF weighting has been tried in a WL-based edge-graphlet matching method; thus, the broad combination of IDF with WL-derived graph structures has prior art. The reviewed sources do not establish whether it is mathematically equivalent to `df(s) = number of training graphs containing s` followed by this smoothed IDF on exact WL histogram coordinates.
- The literature check was focused rather than systematic. An absence of a hit is not proof of novelty.

## Candidate mathematics and choices

For a WL feature `s`, let `c_G(s)` count its occurrences in graph `G`. In a training fold containing `N` graphs, define `df(s)` as the number of those graphs with `c_G(s) > 0`, and `a(s) = log((N + 1)/(df(s) + 1)) + 1`.

The proposed representation `phi(G)[s] = a(s)c_G(s)` produces

`K(G,H) = sum_s a(s)^2 c_G(s)c_H(s)`.

Thus the stated vector formula applies the square of IDF to each pairwise feature contribution. If the intended contribution weight is one power of IDF, the corresponding feature coordinates would use `sqrt(a(s)) c_G(s)`. Either fixed nonnegative coordinate scaling gives an explicit inner-product PSD kernel. The distinction must be settled before implementation and named unambiguously.

With the proposed smoothing, weights are finite and at least one: a feature present in every training graph has weight 1; a feature absent from all training graphs has maximum weight `log(N + 1) + 1`, though such a feature contributes nothing to any training-training kernel entry. If it appears only in a validation graph, that feature must not change training-derived weights; the unseen feature can be assigned the predeclared zero-training-frequency weight, based on the training fold alone. The formula uses presence/absence across graphs for `df`, while the feature count remains the term-frequency factor. It does not use labels and is unsupervised.

Estimate `df` using training graphs only inside each CV split. Freeze the resulting vocabulary/weights for train and held-out graphs in that split. Computing frequencies using held-out graphs leaks information about the evaluation distribution even without labels.

Raw count inner products can favor larger graphs and graphs with more active WL coordinates. Compare like-for-like kernel normalization, such as cosine normalization `K(G,H)/sqrt(K(G,G)K(H,H))`, and explicitly state how zero diagonals are handled. Normalization changes the effective method; do not conflate its effect with IDF. Per-depth normalization or weights can also change scale between refinement levels and should be separate from per-feature rarity weights.

Recommended ablations: (1) exact WL baseline with raw counts; (2) same baseline with graph-level normalization; (3) IDF weights with raw counts; (4) IDF weights with the same graph normalization; and (5) depth/iteration contributions held constant or separately varied. Fix splits, feature depths, classifier, tuning budget, and random seeds across comparisons. Report fold-level uncertainty and per-dataset results; do not select the IDF exponent or normalizer using test folds.

## What remains unanswered and evidence strength

No reviewed paper tests this exact graph-document-frequency IDF formula on the EGONETS benchmarks. Whether it improves classification, adds value beyond normalization, behaves usefully when many features occur in very few graphs, or is computationally worthwhile remains unmeasured. The integer class-to-network mapping and the dataset-specific license also remain unresolved, as recorded in the dataset documentation.

Evidence strength for the narrow empirical question (“does this specified IDF adaptation help under a leakage-safe protocol on these EGONETS files?”): **moderate**—the method is sufficiently specified to pose a reproducible ablation after the squared-weight choice is settled, and related literature supplies motivation, but the exact prior-art status remains uncertain and Nguyen et al.'s full text was inaccessible. Evidence for a novelty claim: **weak/insufficient**, particularly given the VeRNAl IDF/WL precedent and the learned subtree-pattern weighting work.

## Recommendation

**Refine and retain only as an exploratory baseline comparison, not as a novelty claim.** First obtain the Nguyen et al. full text through an accessible legitimate route and continue exact-formula prior-art checks, particularly around the VeRNAl supplement's IDF weighting. If pursued, frame the question narrowly as an ablation of fold-local unsupervised document-frequency weighting against standard WL, controlling for graph normalization and depth. The generalized WL method is an important comparator in the conceptual discussion, but implementing it is outside this task. Do not proceed to implementation or experiments until the weighting convention, related-work review, label semantics, and data reuse terms are resolved.

## Sources

Bibliographic details and links are maintained in [references](references.md). The detailed source-by-source review is in [literature review](literature_review.md).
