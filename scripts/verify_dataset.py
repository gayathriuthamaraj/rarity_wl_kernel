#!/usr/bin/env python3
"""Check manifest integrity and validate the downloaded EGONETS text files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egonets_wl.data.validation import inspect_file, validate_egonets, verify_manifest


def main() -> None:
    root = Path("data/raw")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep") if root.exists() else []
    report = [inspect_file(path) for path in files]
    graphs = validate_egonets(root)
    manifest_errors = verify_manifest("data/metadata/manifest.json", root)
    print(json.dumps({"raw_directory": str(root), "file_count": len(files), "files": report,
                      "graph_validation": graphs, "manifest_errors": manifest_errors}, indent=2))
    if not files:
        print("No local raw dataset files found; no graph counts or labels were verified.")
    if graphs["status"] != "valid" or manifest_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
