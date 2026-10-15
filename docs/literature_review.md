# Literature review

## Scope and search limits

This focused review checked the two selected WL papers, the Nguyen et al. paper on learned subtree-pattern importance, and a VeRNAl supplementary methods section that explicitly reports trying IDF in a WL graphlet-hashing similarity method. It also uses a graph-kernel survey for context. Full text was inspected for Shervashidze, Schulz, and the VeRNAl supplement. Nguyen et al.'s publisher page, abstract, and metadata were checked, but its linked full-text endpoint returned HTTP 403. Searches used WL/Weisfeiler–Lehman, subtree-pattern importance, feature weighting, frequency, and inverse document frequency. No inspected source exactly specifies training-graph document frequency and smoothed IDF on exact WL subtree histogram coordinates. The VeRNAl result is close enough to rule out a broad claim that IDF has never been used in a WL-based representation. This bounded review cannot prove absence or establish mathematical equivalence.

The full SYS2VEC paper was also inspected after the initial focused review; it describes TF-IDF on node content before WL hashing and graph embedding. This is relevant related work but does not provide the exact graph-document-frequency weighting of WL histogram coordinates considered here. Search coverage remains focused, not systematic.

## Shervashidze et al. (2011): baseline WL subtree kernel

The paper constructs a sequence of vertex relabelings by combining each current label with the sorted multiset of neighbor labels. At each depth the graph is represented by counts of its labels; summing the inner products over depths gives the WL subtree kernel. This explicit count-vector interpretation is the baseline relevant to this project. The standard definition does not introduce corpus document frequency or IDF feature weights. See the [JMLR article](https://www.jmlr.org/papers/v12/shervashidze11a.html) and [PDF](https://jmlr.csail.mit.edu/papers/volume12/shervashidze11a/shervashidze11a.pdf). Full text inspected.

## Schulz et al. (2022): generalized / relaxed WL

The method relaxes exact equality of WL labels by comparing the corresponding unfolding trees with a structure- and depth-preserving edit distance. Similar trees are grouped by Wasserstein k-means; the kernel counts cluster memberships. Its approximate R-WL* version uses cluster representatives to reduce computation. This changes which patterns match; it does not assign rarer exact WL features higher document-frequency weights. The paper also permits nonnegative coefficients for contributions at different refinement depths; this is depth-level weighting, not per-feature IDF. It reports that almost all depth-2 unfolding trees are unique in its EGONETS variants and evaluates R-WL* on them. This motivates studying exact-match sparsity but does not establish that rarity weighting solves it. See the [journal article](https://doi.org/10.1007/s10994-022-06131-w), [author-hosted PDF](https://pwelke.de/publications/schulz2022mach.pdf), [arXiv full text](https://arxiv.org/abs/2101.08104), and [GenWL repository](https://github.com/mlai-bonn/GenWL). Full text inspected.

## Nguyen, Nguyen, and Mamitsuka (2021): learning subtree-pattern importance

The abstract says this work learns subtree-pattern weights in the Wasserstein Weisfeiler–Lehman (WWL) kernel framework, motivated by equal importance given to available substructures by usual kernels. It establishes that feature-importance weighting for WL-related graph kernels is not a new general idea. The accessible abstract does not specify enough detail to determine the exact learning objective or mathematical relationship to the candidate. Its weights are described as learned, rather than the proposed unsupervised training-corpus document-frequency statistic; whether it overlaps with simple exact-label coordinate rescaling was not established. The paper appeared in *Machine Learning* 110(7), 1585–1607 (2021), DOI [10.1007/s10994-021-05991-y](https://doi.org/10.1007/s10994-021-05991-y). The [publisher page](https://link.springer.com/article/10.1007/s10994-021-05991-y), [arXiv abstract](https://arxiv.org/abs/2106.04739), and [Aalto repository record](https://aaltodoc.aalto.fi/items/c0c6fc8f-c123-415e-8cb7-f1e5e32be0e7) were checked. The linked full-text PDF returned HTTP 403; only abstract and metadata assessed.

## Oliver et al. (2021; journal issue 2022): VeRNAl IDF option

VeRNAl is a tool for fuzzy RNA network-motif discovery. Its supplementary methods describe a sparse explicit Weisfeiler–Lehman isomorphism kernel for edge-labeled graphlets and report trying inverse document frequency weighting in matching-based similarity functions: comparison values are scaled by the product of the IDF terms involved, to address frequent noncanonical interactions. This is an important precedent for IDF-like weighting within a WL-based graphlet comparison. It is not shown to be the same operation as multiplying graph-level exact WL subtree count coordinates by graph-corpus IDF: the setting is edge-labeled RNA graphlets, the reported IDF acts within pairwise matching scores, and the supplement gives neither the candidate's graph-document-frequency formula nor the same graph-classification objective. The article appeared in *Bioinformatics* 38(4), 970–976, DOI [10.1093/bioinformatics/btab768](https://doi.org/10.1093/bioinformatics/btab768). The [journal article](https://academic.oup.com/bioinformatics/article/38/4/970/6428528) and [supplementary material](https://academic.oup.com/bioinformatics/article/38/4/970/6428528) were inspected; the IDF note is in supplement §3.3 and WL edge graphlet hashing in §3.4.

## Relationship summary

| Work | What is weighted or changed | Representation | Relation to candidate |
|---|---|---|---|
| Shervashidze et al. (2011) | Counts exact labels at each depth; standard inner product | WL subtree histogram | Baseline; no IDF |
| Schulz et al. (2022) | Tree-edit similarity, clustering, and cluster-match counts; nonnegative depth coefficients also allowed | Generalized WL subtree | Same sparse-match motivation; different matching and weight granularity |
| Nguyen et al. (2021) | Abstract reports learned subtree-pattern importance | Wasserstein WL | Strong precedent for feature-specific weighting; insufficient accessible detail to assess equivalence |
| Oliver et al. (2021/2022) | IDF terms multiply pairwise comparison scores | Edge-labeled WL graphlet hashing / RNA matching | Direct IDF/WL precedent; different feature unit, matching task, and formula as far as documented |
| Kriege, Johansson, and Morris (2020) | Survey, not a new weighting method | Graph-kernel families including WL | Context for explicit feature maps; does not settle IDF prior art |

## Bulut, Tenev, and Becker (2021): SYS2VEC

SYS2VEC describes TF-IDF weighting of node content before producing WL-hashable features and applying a Graph2Vec/Doc2Vec-style embedding to software-system variants. The full conference paper was inspected ([PDF](https://www.thinkmind.org/articles/icsea_2021_2_60_10043.pdf), especially §IV-B, printed pp. 63–65). This is relevant precedent for TF-IDF and WL occurring in one graph representation pipeline, but its TF-IDF acts on node content; it does not document IDF weights on exact WL subtree histogram coordinates in the standard WL graph kernel. Its task is software-system representation, not EGONETS graph classification. No DOI was identified in the inspected proceedings record.

| Work | What is weighted or changed | Representation | Relation to candidate |
|---|---|---|---|
| Bulut, Tenev, and Becker (2021) | TF-IDF weights node content before WL hashing and Graph2Vec-style embedding | Software-system variants | TF-IDF and WL coexist, but IDF is not applied to WL histogram coordinates |

## EGONETS provenance and limits

The Schulz paper attributes Buzznet, Digg, and Flickr to Zafarani and Liu's social-computing repository and LiveJournal to SNAP. It describes sampling 50 egonets per source under size and density constraints, with source network as target class. Its availability statement links to the GenWL repository, which contains processed `EGO-1` through `EGO-4` files. The inspected loader treats graph labels as integer class IDs; neither it nor the data directories provide the integer-to-network mapping. The repository has a GPL-3.0 code license but no separate data license was found. Details and the checked locations are in [dataset documentation](dataset_documentation.md). The original network repositories were not independently used to regenerate these samples.

## Interpretation

The candidate is an incremental IDF-style adaptation with clear precedents for WL pattern weighting and IDF in a WL-based graphlet comparison. The exact formula and graph-level histogram setting differ from the reported VeRNAl setting, but available sources do not allow a conclusive equivalence assessment. If pursued, frame the work as a controlled empirical study of whether fold-local unsupervised graph-document-frequency weighting adds value on these four EGONETS benchmarks after controlling for normalization and depth. The review cannot establish that narrower setting as novel, and no experiment results are claimed.
