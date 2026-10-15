"""Dataset loading utilities for TU-format graph benchmarks."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import networkx as nx


def load_tu_dataset(folder_path: str | Path, prefix: str | None = None) -> list[nx.Graph]:
    """Load a graph collection from a directory containing TU-format files.

    Expected files:
        {prefix}_A.txt: Edge list (1-indexed global node IDs)
        {prefix}_graph_indicator.txt: Graph ID mapping for each node
        {prefix}_graph_labels.txt: Integer class label for each graph

    Parameters
    ----------
    folder_path : str or Path
        Directory containing the dataset files.
    prefix : str, optional
        Prefix for dataset files (e.g., 'EGO-1'). If None, inferred from folder name.

    Returns
    -------
    graphs : list of networkx.Graph
        List of undirected simple graphs, each with attributes:
        - graph['id']: 1-indexed graph ID
        - graph['label']: integer class label
        - nodes with 'global_id' and 'local_id'
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    if prefix is None:
        prefix = folder.name

    a_path = folder / f"{prefix}_A.txt"
    ind_path = folder / f"{prefix}_graph_indicator.txt"
    lbl_path = folder / f"{prefix}_graph_labels.txt"

    for path in (a_path, ind_path, lbl_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required TU-format file: {path}")

    # Read labels
    labels = [int(line.strip()) for line in lbl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_graphs = len(labels)

    # Read node-to-graph indicator
    indicator = [int(line.strip()) for line in ind_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Initialize networkx graphs
    graphs: list[nx.Graph] = []
    # Map global node ID (1-indexed) to (graph_index, local_node_id)
    global_to_local: dict[int, tuple[int, int]] = {}

    for g_idx in range(n_graphs):
        g = nx.Graph()
        g.graph["id"] = g_idx + 1
        g.graph["label"] = labels[g_idx]
        graphs.append(g)

    # Add nodes to graphs
    graph_node_counters = [0] * n_graphs
    for global_id, g_id in enumerate(indicator, start=1):
        g_idx = g_id - 1
        local_id = graph_node_counters[g_idx]
        graph_node_counters[g_idx] += 1
        graphs[g_idx].add_node(local_id, global_id=global_id, local_id=local_id)
        global_to_local[global_id] = (g_idx, local_id)

    # Read edges
    edge_lines = a_path.read_text(encoding="utf-8").splitlines()
    for line in edge_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Malformed edge line: {line}")
        u_global, v_global = int(parts[0].strip()), int(parts[1].strip())
        if u_global not in global_to_local or v_global not in global_to_local:
            continue
        g_u, u_local = global_to_local[u_global]
        g_v, v_local = global_to_local[v_global]
        if g_u == g_v and u_local != v_local:
            graphs[g_u].add_edge(u_local, v_local)

    return graphs


def load_dataset(name_or_path: str | Path, raw_dir: str | Path = "data/raw") -> list[nx.Graph]:
    """Load an EGONETS variant by name (e.g. 'EGO-1') or folder path.

    Parameters
    ----------
    name_or_path : str or Path
        Either a variant name ('EGO-1', 'EGO-2', 'EGO-3', 'EGO-4') or a directory path.
    raw_dir : str or Path, default="data/raw"
        Root raw data directory.

    Returns
    -------
    list of networkx.Graph
    """
    target = Path(name_or_path)
    if not target.is_dir():
        # Try relative to raw_dir
        candidate = Path(raw_dir) / name_or_path
        if candidate.is_dir():
            target = candidate
        else:
            raise FileNotFoundError(f"Cannot locate dataset '{name_or_path}'. Searched at {target} and {candidate}.")

    return load_tu_dataset(target)
