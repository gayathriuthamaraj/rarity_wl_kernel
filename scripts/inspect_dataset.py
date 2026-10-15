#!/usr/bin/env python3
"""Report variant-level statistics for the verified GenWL TU text files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egonets_wl.data.validation import validate_egonets


def main() -> None:
    root = Path("data/raw")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep") if root.exists() else []
    if not files:
        print("No files found under data/raw. No dataset statistics can be reported.")
        return
    print(json.dumps(validate_egonets(root), indent=2))


if __name__ == "__main__":
    main()
