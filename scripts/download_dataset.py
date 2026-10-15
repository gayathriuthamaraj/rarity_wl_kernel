#!/usr/bin/env python3
"""Download verified GenWL EGONETS files or one explicitly supplied HTTPS file."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egonets_wl.data.download import download_file, download_official_egonets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Verified HTTPS direct-file URL; never guessed")
    source.add_argument("--official-egonets", choices=["all", "EGO-1", "EGO-2", "EGO-3", "EGO-4"],
                         help="Download verified graph files from the paper-linked GenWL repository")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/metadata/manifest.json")
    args = parser.parse_args()
    if args.official_egonets:
        paths = download_official_egonets(args.output_dir, args.manifest, args.official_egonets)
    else:
        paths = [download_file(args.url, Path(args.output_dir), Path(args.manifest))]
    for path in paths:
        print(f"Downloaded {path}")


if __name__ == "__main__":
    main()
