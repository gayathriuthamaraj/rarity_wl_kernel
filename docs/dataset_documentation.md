# EGONETS dataset documentation

## Provenance and verified source

Schulz et al. (2022) introduce four ego-network graph classification benchmarks sampled from Buzznet, Digg, Flickr, and LiveJournal. Each variant contains 50 graphs per source network (200 graphs, four classes). They describe the graph as the subgraph induced by a vertex's neighbors, and select graphs under size and density constraints. The paper says Buzznet, Digg, and Flickr are provided by Zafarani and Liu (2009), and LiveJournal by Leskovec and Krevl (2014). The paper does not give a full recipe for recreating the selected samples from those original networks.

The paper's data availability statement points to the authors' [GenWL repository](https://github.com/mlai-bonn/GenWL). Its README documents a MUTAG example and a dataset path argument, but does not document a separate downloader. Inspecting the GitHub API recursive tree at commit `044e6f62b3a1ad14888d58907192a360cd4fda52` verified that it contains `Data/EGO-1` through `Data/EGO-4`. Each directory has the three files `*_A.txt`, `*_graph_indicator.txt`, and `*_graph_labels.txt`. We downloaded all 12 files from those exact paths at that pinned commit into `data/raw/EGO-1/` through `data/raw/EGO-4/`. The local manifest records the direct URL, retrieval timestamp, byte size, and SHA-256 for each file.

The repository names the variants `EGO-1` through `EGO-4`; the paper calls them EGONETS-1 through EGONETS-4. This is a naming difference, not a dataset substitution. No archive is provided for these variants in the inspected GenWL tree, so the individual original text files are retained as downloaded. The repository's code license is GPL-3.0; a distinct data license or redistribution grant for these benchmark files was not found. Keep the raw files local and do not republish them until terms are confirmed.

### Label and terms investigation

The pinned GenWL tree was checked for all files under the four `Data/EGO-n` directories and for repository-level documentation, loader, experiment entry point, and configuration. The EGO directories contain only edge list, graph indicator, and integer graph-label files; no source-network mapping file, per-variant README, or data preparation script is present. `GraphDataToGraphList.py` constructs graphs from the TU-format files and reads graph labels as integers; it does not map class IDs to network names. The paper explains the four source networks and states that graph labels identify source network, but does not state which integer corresponds to which name. Its data availability statement points to the same GenWL repository (journal PDF, pp. 15 and 27). No supplementary mapping table or separately linked data supplement was identified in the paper or repository. Thus the label mapping remains unresolved; do not infer it from numeric order or file layout.

The GenWL root `LICENSE` applies to repository software (GPL-3.0). The checked EGO data directories do not contain an explicit license or usage statement, and the paper does not state dataset-specific reuse terms. The paper's availability statement indicates availability, but is not itself an explicit data license. Do not represent GPL-3.0 as covering the data. The legitimate next route is to request the class-ID mapping and data reuse terms from the GenWL authors; if they cannot provide them, ask the cited original-data maintainers (Zafarani/Liu for Buzznet, Digg, Flickr; SNAP maintainers for LiveJournal) whether these exact processed samples may be used and redistributed. Do not reconstruct or substitute datasets without confirmed provenance and permission.

## File format and graph semantics

The files follow the TU Dortmund graph-kernel text convention used by GenWL's `GraphDataToGraphList.py`:

- `EGO-n_A.txt`: one comma-separated pair of global node IDs per edge row.
- `EGO-n_graph_indicator.txt`: one graph ID per node row, mapping global node IDs to graphs.
- `EGO-n_graph_labels.txt`: one integer graph class per graph row.

The source loader constructs `networkx.Graph` objects, and the paper's formal graph definition is undirected. The edge lists contain no repeated unordered edge rows, self-loops, or cross-graph edges. There are no node-label, edge-label, node-attribute, or edge-attribute files in any variant: these are unlabeled, unattributed graphs in the supplied representation. Graph labels are encoded as integers 1, 2, 3, and 4, with 50 examples of each. The available paper and repository files do not specify which integer maps to each named social network, so this mapping remains unresolved; the source code treats labels as class IDs.

## Verified local statistics

These statistics were computed from the downloaded files, not copied from the paper. `edge count` is the number of rows in `*_A.txt` and matches the simple undirected representation because repeated unordered edges and loops are absent.

| Variant | Graphs | Classes | Total nodes | Mean nodes | Edge rows | Mean edges | Node labels / attributes | Edge labels / attributes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| EGO-1 | 200 | 4 × 50 | 27,793 | 138.965 | 118,706 | 593.530 | none | none |
| EGO-2 | 200 | 4 × 50 | 35,710 | 178.550 | 288,971 | 1,444.855 | none | none |
| EGO-3 | 200 | 4 × 50 | 44,001 | 220.005 | 522,698 | 2,613.490 | none | none |
| EGO-4 | 200 | 4 × 50 | 51,955 | 259.775 | 827,160 | 4,135.800 | none | none |

The validator also checks graph-indicator coverage, class counts, endpoint ranges, cross-graph edges, self-loops, repeated undirected edges, blank/malformed rows, and exact duplicate topology after preserving source node order. It found no failures and no duplicates under that exact normalized representation. In addition, the downloaded bytes were checked against the Git blob IDs returned by the pinned GenWL tree for all 12 files. The validator does not prove absence of graph-isomorphism duplicates under arbitrary vertex renaming; that more expensive check remains outstanding. No missing graph labels or malformed rows were found. The counts, class balance, and steady size/density increase agree with the paper's reported dataset description.

## Reproduce acquisition and checks

From the repository root, with the project installed:

```powershell
python scripts/download_dataset.py --official-egonets all
python scripts/verify_dataset.py
python scripts/inspect_dataset.py
```

Use `--official-egonets EGO-1` (or `EGO-2`, `EGO-3`, `EGO-4`) to fetch one variant. The command fetches the three source files for each selected variant from the pinned GenWL commit. Re-running it replaces those files with the same pinned source content and refreshes their manifest entries. `data/raw/` is gitignored. The manifest is tracked metadata; the dataset files are local-only.

The downloader's separate `--url` option remains available for explicitly verified direct HTTPS files, but must not be used to guess other sources. No files from alternative datasets have been added.
