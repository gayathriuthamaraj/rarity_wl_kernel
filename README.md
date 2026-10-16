# Rarity-Weighted Weisfeiler–Lehman Graph Kernels for Social Network Graph Classification

Research repository for a controlled comparison of the standard Weisfeiler–Lehman (WL) subtree graph kernel and a proposed feature-frequency weighting variant on the EGONETS-1–4 benchmarks. The title and candidate method are provisional. This repository is in setup and literature review; no method is implemented and no experiments or findings are claimed.

## Research question

Do corpus-level frequency weights on WL subtree features change graph classification performance on EGONETS? The hypothesis is exploratory: reweighting features may affect similarity when exact depth-2-or-higher WL patterns are sparse. It is not an established improvement or a novelty claim. Related work includes generalized WL's tree-edit-distance clustering, learned subtree-pattern weights in WWL, and an IDF option in a WL edge-graphlet hashing method. The candidate is an incremental comparison and needs broader prior-art review and empirical evaluation.

## Data and literature

EGONETS-1, -2, -3, and -4 are the authors' processed benchmark files in GenWL, stored there as `EGO-1` through `EGO-4`. Each has 200 graphs and four source classes. All four variants have been downloaded from the paper-linked repository at a pinned commit and validated locally. Verified statistics and encoding details are in [dataset documentation](docs/dataset_documentation.md); the per-file SHA-256 provenance is in `data/metadata/manifest.json`.

- Base: Shervashidze, N. et al. (2011), [“Weisfeiler-Lehman Graph Kernels,” JMLR 12, 2539–2561](https://www.jmlr.org/papers/v12/shervashidze11a.html); [PDF](https://jmlr.csail.mit.edu/papers/volume12/shervashidze11a/shervashidze11a.pdf).
- Reference: Schulz, T. H. et al. (2022), [“A generalized Weisfeiler-Lehman graph kernel,” Machine Learning 111, 2601–2629](https://doi.org/10.1007/s10994-022-06131-w); [author implementation, GenWL](https://github.com/mlai-bonn/GenWL).

The standard WL subtree kernel iteratively refines each node label using its current label and the multiset of neighboring labels, then compares graphs by matching-label counts across depths. The generalized WL paper describes a different approach: a constrained tree-edit distance over unfolding trees, followed by hard clustering and matching cluster assignments. A focused literature review also found published work on learning subtree-pattern importance inside a Wasserstein WL kernel. The exact corpus-IDF formula remains unverified in prior work, so the candidate is treated as an incremental exploratory comparison rather than a novelty claim. See [research gap analysis](docs/research_gap_analysis.md); the closest paper's full text and broader prior-art search remain outstanding.

## Status

- Dataset selection: EGONETS (EGO-1 through EGO-4).
- Base paper selected: *Weisfeiler-Lehman Graph Kernels* (2011).
- Reference paper selected: *A Generalized Weisfeiler-Lehman Graph Kernel* (2022 journal version).
- Baseline: standard WL subtree kernel.
- Candidate modification: corpus-rarity-weighted WL subtree features (with leakage-safe training-only weights).
- Current phase: Part 4 (experiments, analysis, and report) complete.
- Implementation and experiments: complete.
- Experimental results: available in `results/`, with figures in `results/figures/`.

## Dataset acquisition

The paper-linked [GenWL repository](https://github.com/mlai-bonn/GenWL) contains `Data/EGO-1` through `Data/EGO-4`. The files are separate TU-format text files, not one archive. The recursive tree was verified at commit `044e6f62b3a1ad14888d58907192a360cd4fda52`, and the 12 graph/indicator/label files have been downloaded into `data/raw/`. The source paper does not provide a separate download script or full recipe for regenerating the sampled graphs from original network crawls. A distinct license for the data was not located; keep the data local pending terms review.

When an authoritative direct archive URL is known, run:

```bash
python scripts/download_dataset.py --official-egonets all
python scripts/verify_dataset.py
python scripts/inspect_dataset.py
```

The downloader fetches the four variants from the pinned GenWL source; select one with `--official-egonets EGO-1` (or EGO-2, EGO-3, EGO-4). It records source URL, retrieval date, size, and SHA-256. The verifier checks manifest integrity and the observed TU-format graph and label structure. Raw files remain excluded from version control.

## Setup

Python 3.10–3.12 is intended. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,graphs]"
```

On Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,graphs]"
```

For runtime-only use, `pip install -e .` installs the small core dependency set. The commands above install development and graph-format dependencies needed for setup checks. `pip install -r requirements-dev.txt` is an equivalent development install. No experiment commands are available yet. See `Makefile` for setup-stage helpers.

## Repository guide

- `configs/`: defaults, dataset paths, future experiment settings.
- `data/raw/`: original archives and files (ignored by Git); `data/metadata/`: provenance manifest.
- `docs/`: research plan, literature, protocol, decisions, and report outline.
- `notebooks/`: planned exploration outlines only; no fabricated outputs.
- `scripts/`: acquisition, verification, and inspection entry points.
- `src/egonets_wl/`: package scaffolding. Kernel and evaluation algorithms explicitly raise `NotImplementedError`.
- `tests/`: small infrastructure checks and placeholders for later work.
- `results/`, `reports/`, `logs/`: reserved output locations.

## Planned evaluation (not run)

First establish the exact dataset format and repeatable loader. Implement and validate the standard WL baseline, then specify rarity weighting without consulting held-out fold frequencies. Compare with identical splits, SVM setup, tuning budget, and seeds. Planned metrics are accuracy (for comparability with the reference), macro-F1, per-class performance, and mean ± standard deviation across repeated stratified cross-validation; include paired uncertainty/significance analysis only when justified by the protocol. Depth, normalization, weighting function, and tuning ranges remain pending decisions. Do not interpret this plan as results.

## Reproducibility and citation

Record Python and dependency versions, configuration, split seeds, commands, and data checksums with any future run. Keep raw data and derived results out of source control unless licensing and project policy are reviewed. Cite both papers above and the original data sources identified in `docs/references.md`. The GenWL implementation is GPL-3.0; this project does not copy its source.

## Pipeline

Literature review → dataset acquisition → dataset validation → baseline implementation → candidate modification → controlled evaluation → statistical analysis → final report.

## Limitations and pending decisions

The integer-to-social-network class mapping is not specified in the downloaded files, and a separate data license was not located. Duplicate detection currently covers exact topology after normalization in the stored vertex order, not all isomorphism-equivalent graphs. The candidate IDF formula is documented, but whether the kernel should use IDF or IDF-squared contributions and how to normalize graphs remain open. Any corpus frequencies must be estimated on each training fold only. Resolve these points before implementation or experiments. See [research gap analysis](docs/research_gap_analysis.md), [research plan](docs/research_plan.md), [decisions](docs/decisions.md), and [dataset documentation](docs/dataset_documentation.md).
