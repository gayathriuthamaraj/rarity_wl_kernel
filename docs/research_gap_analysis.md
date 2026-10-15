# Research-gap analysis: rarity weighting for WL graph features

## 1. Project aim

This project asks whether weighting Weisfeiler–Lehman (WL) graph features by how widely they occur across a training collection changes graph-classification performance on the four EGONETS benchmarks. The standard WL subtree kernel is the baseline. A graph-document-frequency weighting rule is a proposed candidate only: it has not been implemented, validated, or tested here. This document is a focused literature assessment, not a claim of novelty.

## 2. Concepts in plain language

A graph is a set of points (vertices) and links (edges). A social graph can represent people as points and relationships as links. An ego network centers on one person and includes that person's neighbors and links among them. Graph classification assigns one label to a whole graph; here the benchmark labels indicate the source social network, although the integer-to-network mapping is not supplied with the files.

A graph kernel is a rule for measuring similarity between two graphs. It lets a classifier such as an SVM use graph structure without first forcing each graph into a fixed-size hand-designed vector. The WL subtree kernel makes explicit feature counts, then compares two graphs with a dot product. In that sense, each graph has a structural “fingerprint” made of counts of local patterns.

## 3. Standard WL subtree kernel

At depth 0, each node has its initial label (for an unlabeled graph, a common initial label is used). At each next depth, WL combines a node's current label with the sorted multiset of its neighbors' current labels and assigns a new compressed label. Labels at depth 1 therefore describe a node and its immediate neighbors; depth 2 includes information about neighbors' neighborhoods. A graph feature is the count of nodes with each exact label at each included depth. The standard subtree kernel sums dot products of those count vectors over depths. Shervashidze et al. give the relabeling algorithm and feature map in Algorithm 1 and Definition 4, and the depth-summed kernel in Equations (1)–(2), printed pp. 2542–2547 ([paper](https://jmlr.csail.mit.edu/papers/volume12/shervashidze11a/shervashidze11a.pdf)).

Deeper refinement captures larger neighborhoods, but exact patterns can become rare: two similar neighborhoods may receive different labels after one small structural change. The generalized WL paper reports that almost all depth-2 unfolding trees occur only once in each EGONETS variant, so exact matching offers little structural reuse beyond degrees in that setting. This is the authors' reported observation, not evidence that rarity weighting improves results ([Schulz et al.](https://pwelke.de/publications/schulz2022mach.pdf), §6.1.1, printed p. 2616).

## 4. A small example: exact matches and rarity

Imagine two root nodes whose neighborhoods are stars: one root has two leaf neighbors, while the other has three. With a common initial node label, their depth-0 root labels match. At depth 1, their root labels differ because the neighbor multisets have different sizes. Standard WL counts these as two different root features; it does not call them “almost the same.” A generalized method can compare the corresponding neighborhood trees and may treat them as close, depending on its edit cost and clustering. A rarity-weighted method keeps the two exact features separate and only changes their influence based on how many training graphs contain each one. This example describes the root features, not the full kernel value of the two graphs.

## 5. Published generalized WL versus the proposed candidate

Schulz et al.'s generalized WL (including the R-WL* approximation) relaxes exact label equality. It compares unfolding trees using a structure- and depth-preserving edit distance, clusters similar trees using Wasserstein k-means, and counts cluster assignments as features. Their PSD argument follows from an explicit cluster-count feature map. This changes which patterns can match. The standard WL framework's nonnegative coefficients across refinement depths are depth-level weights, not per-pattern document-frequency weights; attribute that general depth weighting to the baseline formulation, not as the candidate method.

The candidate considered here retains standard exact WL labels and counts. For each feature coordinate it would estimate the fraction of *training graphs* containing that feature and apply a smoothed inverse-frequency scale. It changes feature influence, not the definition of a match. It is not generalized WL, R-WL, or an implementation of either paper.

## 6. What prior research says

The [literature review](literature_review.md) records the search limits, publication details, and source inspection extent. Important findings:

| Work | Published method | Relation to this candidate |
|---|---|---|
| Shervashidze et al. (2011) | Exact WL label-count features with a depth-summed inner product | Baseline; no corpus IDF in the standard definition. |
| Schulz et al. (2022) | Unfolding-tree edit distance, clustering, and cluster-count features | Addresses exact-match sparsity by changing similarity, not by weighting exact coordinates. |
| Nguyen et al. (2021) | Learns subtree-pattern importance for a Wasserstein WL kernel | Strong precedent for pattern importance. Full text was not accessible in this review, so exact mathematical overlap is unresolved. |
| Oliver et al. (2021/2022), VeRNAl | Reports IDF-scaled pairwise comparison values in an edge-labeled WL graphlet-hashing method for RNA motif matching | Direct IDF/WL precedent, but its feature unit, similarity task, and documented operation differ from graph-document-frequency weighting of exact WL histogram coordinates. |
| Bulut et al. (2021), SYS2VEC | Applies TF-IDF to node content before WL hashing and Graph2Vec-style system embeddings | Uses TF-IDF and WL in one pipeline, but does not apply IDF to WL subtree histogram coordinates or define a standard WL graph kernel. |
| Kriege et al. (2020) | Survey of graph-kernel methods | Background, not a new weighting rule. |

The exact candidate formula was not found in the sources reviewed. That does not establish novelty: this was a focused search, not a systematic review, and learned pattern weighting and IDF in WL-related methods are already published.

## 7. Research problem and gap assessment

The defensible question is narrow: **does fold-local, unsupervised graph-document-frequency scaling of exact WL subtree counts add predictive value on these EGONETS files, after controlling for graph normalization and refinement depth?** This is an empirical comparison, not a claim that frequency weighting or feature weighting in WL is new. The gap is at most a specific, not-yet-verified formula-and-benchmark comparison. Evidence for a novelty claim is weak; broader prior-art search and the Nguyen full text are still needed.

## 8. Candidate mathematics and design choices

For feature `s`, let `c_G(s)` be its occurrence count in graph `G`. In a training fold with `N_train` graphs, define graph document frequency as

```text
df(s) = number of training graphs G with c_G(s) > 0
a(s)  = log((N_train + 1) / (df(s) + 1)) + 1
```

The candidate vector in the current project brief, `phi(G)[s] = a(s)c_G(s)`, induces

```text
K(G,H) = sum_s a(s)^2 c_G(s)c_H(s)
```

Thus the dot product uses squared IDF as the contribution weight. If the intended kernel contribution is weighted by one power of `a(s)`, use coordinates `sqrt(a(s))c_G(s)`. Both are fixed nonnegative coordinate scalings and therefore define explicit inner-product PSD kernels; choose and document one before implementation. `df` counts graphs containing a feature, not total feature occurrences. The within-graph count remains the term-frequency factor.

Fit vocabulary statistics and weights using only each training fold, then freeze them for held-out graphs. Using held-out graphs to calculate frequencies leaks information about the evaluation distribution even though labels are not used. This rule is unsupervised. A feature absent in training can receive the formula's maximum smoothed weight, but contributes zero against training feature vectors; handling for test-only coordinates should be explicitly fixed.

Raw-count kernels can favor larger graphs or graphs with more feature occurrences. Compare baseline and weighted versions both before and after the same graph-level kernel normalization, with an explicit policy for zero diagonal entries. Keep depth effects separate from feature-frequency effects. A future ablation should use identical folds, classifier, tuning budget, and seeds for raw WL, normalized WL, weighted WL, and weighted-plus-normalized WL. Report fold variation and per-class outcomes. Do not tune the IDF exponent, normalization, or depth on test folds.

## 9. EGONETS fit and unresolved data questions

Schulz et al. describe four sets of 50 egonets sampled from Buzznet, Digg, Flickr, and LiveJournal, labeled by source network. The paper describes sampling and constraints but does not supply a full crawl-to-sample reproduction recipe. The GenWL repository provides processed `EGO-1` through `EGO-4` files, each with 200 graphs and four balanced integer classes. In the checked-in files and inspected loader, the integer-to-network mapping is absent. The code repository is GPL-3.0; no separate dataset license was located. The mapping and reuse terms remain unresolved; see [dataset documentation](dataset_documentation.md). Do not substitute a different dataset or publish/re-distribute these data without resolving terms.

## 10. Recommendation

Retain rarity weighting only as an exploratory baseline comparison, not as a novelty claim or presumed improvement. Before implementation, obtain a legitimate full text of Nguyen et al., extend the targeted prior-art search, settle whether the kernel uses one or two IDF powers, and resolve dataset label semantics and reuse terms. Any later evaluation should be fold-local and separate normalization and depth effects. No kernel, classification experiment, or result is provided by this repository state.

## 11. References and evidence limits

Complete citations and source links are in [references](references.md). Full text was inspected for Shervashidze et al., Schulz et al., VeRNAl's article/supplement, and SYS2VEC. For Nguyen et al., the publisher page, abstract, repository metadata, and indexed excerpts were reviewed, but the linked full-text endpoint returned HTTP 403; do not treat its detailed objective as fully verified. Search coverage was focused, so unlocated work cannot be treated as nonexistent.
