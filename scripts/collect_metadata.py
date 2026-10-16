#!/usr/bin/env python3
"""Collect the reproducibility record: environment versions + dataset checksums."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import networkx
import scipy
import sklearn
import matplotlib


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "networkx": networkx.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seeds": [42, 43, 44, 45, 46],
        "folds": 10,
        "repetitions": 5,
        "classifier": "SVC(kernel='precomputed', C=1.0)",
    }
    raw = Path("data/raw")
    files = {}
    for p in sorted(raw.rglob("*.txt")):
        rel = p.relative_to(raw).as_posix()
        files[rel] = {"sha256": sha256_of(p), "bytes": p.stat().st_size}
    meta["dataset_files"] = files
    Path("results").mkdir(exist_ok=True)
    Path("results/metadata_part4.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote results/metadata_part4.json with", len(files), "data files")


if __name__ == "__main__":
    sys.exit(main())
