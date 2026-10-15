"""Archive and basic local file validation helpers."""
from __future__ import annotations

import gzip
import hashlib
import tarfile
import zipfile
from collections import Counter
from pathlib import Path


def inspect_file(path: str | Path) -> dict[str, object]:
    path = Path(path)
    result: dict[str, object] = {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None}
    if not path.is_file():
        return result
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            result.update(format="zip", members=archive.namelist(), integrity="ok" if bad_member is None else f"bad member: {bad_member}")
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            result.update(format="tar", members=archive.getnames(), integrity="readable")
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            stream.read(1)
        result.update(format="gzip", integrity="readable")
    else:
        result.update(format="unclassified", readable_text=None)
        try:
            path.read_text(encoding="utf-8")
            result["readable_text"] = True
        except UnicodeDecodeError:
            result["readable_text"] = False
    return result


def validate_egonets(root: str | Path = "data/raw") -> dict[str, object]:
    """Validate the verified TU-style files used by the GenWL EGO-1..4 data."""
    root = Path(root)
    variants: dict[str, object] = {}
    errors: list[str] = []
    for index in range(1, 5):
        name = f"EGO-{index}"
        folder = root / name
        expected = {
            f"{name}_A.txt",
            f"{name}_graph_indicator.txt",
            f"{name}_graph_labels.txt",
        }
        present = {p.name for p in folder.iterdir() if p.is_file()} if folder.is_dir() else set()
        missing = sorted(expected - present)
        extra_files = sorted(present - expected)
        if missing:
            errors.append(f"{name}: missing required files: {', '.join(missing)}")
            variants[name] = {"status": "invalid", "missing_files": missing}
            continue
        if extra_files:
            errors.append(f"{name}: unexpected files in variant directory: {', '.join(extra_files)}")
        try:
            indicator = [int(x.strip()) for x in (folder / f"{name}_graph_indicator.txt").read_text(encoding="utf-8").splitlines()]
            labels = [int(x.strip()) for x in (folder / f"{name}_graph_labels.txt").read_text(encoding="utf-8").splitlines()]
            raw_edge_lines = (folder / f"{name}_A.txt").read_text(encoding="utf-8").splitlines()
            if any(not line.strip() for line in raw_edge_lines):
                raise ValueError(f"{name}_A.txt contains blank rows")
            edge_lines = [line.strip() for line in raw_edge_lines]
            edges = []
            for line_no, line in enumerate(edge_lines, 1):
                pair = line.split(",")
                if len(pair) != 2:
                    raise ValueError(f"{name}_A.txt line {line_no}: expected two comma-separated node IDs")
                edges.append((int(pair[0].strip()), int(pair[1].strip())))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{name}: cannot parse expected text format: {exc}")
            variants[name] = {"status": "invalid", "error": str(exc)}
            continue

        n_graphs = len(labels)
        graph_errors = []
        if n_graphs != 200:
            graph_errors.append(f"expected 200 graph labels per variant per paper, found {n_graphs}")
        if not indicator or min(indicator) != 1 or max(indicator) != n_graphs:
            graph_errors.append("graph indicators must cover IDs 1..number of graph labels")
        node_counts = Counter(indicator)
        if len(indicator) != sum(node_counts.values()) or set(node_counts) != set(range(1, n_graphs + 1)):
            graph_errors.append("indicator rows do not map every node to each graph ID")
        edge_counts: Counter[int] = Counter()
        normalized_edges: list[list[tuple[int, int]]] = [[] for _ in range(n_graphs)]
        duplicate_edge_rows = 0
        seen_edges: set[tuple[int, int]] = set()
        self_loops = 0
        cross_graph_edges = 0
        invalid_endpoints = 0
        local_ids: list[int] = []
        local_counts: Counter[int] = Counter()
        for graph_id in indicator:
            local_ids.append(local_counts[graph_id])
            local_counts[graph_id] += 1
        for u, v in edges:
            if not (1 <= u <= len(indicator) and 1 <= v <= len(indicator)):
                invalid_endpoints += 1
                continue
            gu, gv = indicator[u - 1], indicator[v - 1]
            if gu != gv:
                cross_graph_edges += 1
                continue
            if u == v:
                self_loops += 1
            undirected = (min(u, v), max(u, v))
            if undirected in seen_edges:
                duplicate_edge_rows += 1
            seen_edges.add(undirected)
            edge_counts[gu] += 1
            # Graph-local node order signature catches repeated serialized topology.
            local_u = local_ids[u - 1]
            local_v = local_ids[v - 1]
            normalized_edges[gu - 1].append((min(local_u, local_v), max(local_u, local_v)))
        exact_signatures = [(node_counts.get(g, 0), tuple(sorted(es))) for g, es in enumerate(normalized_edges, 1)]
        exact_duplicate_graphs = len(exact_signatures) - len(set(exact_signatures))
        label_counts = Counter(labels)
        if set(label_counts) != {1, 2, 3, 4} or any(count != 50 for count in label_counts.values()):
            graph_errors.append("graph labels must encode four classes with 50 graphs each (paper-reported target)")
        if invalid_endpoints:
            graph_errors.append(f"{invalid_endpoints} edge rows reference node IDs outside indicator range")
        if cross_graph_edges:
            graph_errors.append(f"{cross_graph_edges} edge rows connect nodes assigned to different graphs")
        if duplicate_edge_rows:
            graph_errors.append(f"{duplicate_edge_rows} repeated undirected edge rows")
        if self_loops:
            graph_errors.append(f"{self_loops} self-loop rows; source graphs are expected to be simple")
        if graph_errors:
            errors.extend(f"{name}: {message}" for message in graph_errors)
        variants[name] = {
            "status": "valid" if not graph_errors else "invalid",
            "graph_count": n_graphs,
            "label_counts": dict(sorted(label_counts.items())),
            "node_count_total": len(indicator),
            "node_count_min": min(node_counts.values(), default=0),
            "node_count_max": max(node_counts.values(), default=0),
            "node_count_mean": round(len(indicator) / n_graphs, 3) if n_graphs else None,
            "edge_row_count": len(edges),
            "edge_count_min": min((edge_counts[g] for g in range(1, n_graphs + 1)), default=0),
            "edge_count_max": max((edge_counts[g] for g in range(1, n_graphs + 1)), default=0),
            "edge_count_mean": round(len(edges) / n_graphs, 3) if n_graphs else None,
            "self_loop_rows": self_loops,
            "cross_graph_edge_rows": cross_graph_edges,
            "invalid_endpoint_rows": invalid_endpoints,
            "repeated_undirected_edge_rows": duplicate_edge_rows,
            "exact_local_order_duplicate_graphs": exact_duplicate_graphs,
            "duplicate_check_scope": "identical topology after source node-order normalization; graph-isomorphism duplicates are not tested",
            "extra_files": extra_files,
            "node_labels_present": (folder / f"{name}_node_labels.txt").exists(),
            "edge_labels_present": (folder / f"{name}_edge_labels.txt").exists(),
            "node_attributes_present": (folder / f"{name}_node_attributes.txt").exists(),
            "edge_attributes_present": (folder / f"{name}_edge_attributes.txt").exists(),
        }
    return {"root": str(root), "status": "valid" if not errors and len(variants) == 4 else "invalid", "variants": variants, "errors": errors}


def verify_manifest(manifest_path: str | Path, raw_root: str | Path = "data/raw") -> list[str]:
    """Check that manifest entries exist and their size and SHA-256 still match."""
    import json

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return [f"Manifest not found: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    entries = manifest.get("files", [])
    recorded_paths = {entry.get("relative_path", entry.get("filename")) for entry in entries}
    expected_paths = {
        f"EGO-{index}/EGO-{index}{suffix}"
        for index in range(1, 5)
        for suffix in ("_A.txt", "_graph_indicator.txt", "_graph_labels.txt")
    }
    for missing in sorted(expected_paths - recorded_paths):
        errors.append(f"Manifest entry missing: {missing}")
    for entry in entries:
        path = Path(raw_root) / entry["relative_path"] if entry.get("relative_path") else Path(raw_root) / entry["filename"]
        if not path.is_file():
            errors.append(f"Manifest file missing: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != entry.get("size_bytes") or digest != entry.get("sha256"):
            errors.append(f"Manifest integrity mismatch: {path}")
    return errors
