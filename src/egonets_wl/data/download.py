"""Explicit-URL download helper with provenance manifest updates."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

GENWL_COMMIT = "044e6f62b3a1ad14888d58907192a360cd4fda52"
EGONETS_FILES = {
    f"EGO-{index}": (
        f"EGO-{index}_A.txt",
        f"EGO-{index}_graph_indicator.txt",
        f"EGO-{index}_graph_labels.txt",
    )
    for index in range(1, 5)
}


def download_file(url: str, output_dir: str | Path, manifest_path: str | Path) -> Path:
    """Download one URL, preserve its basename, and append integrity metadata."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or not Path(parsed.path).name:
        raise ValueError("Provide a verified HTTPS direct-file URL with a filename")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / Path(parsed.path).name
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version": 1, "files": []}
    manifest["status"] = "downloaded_files_present"
    if GENWL_COMMIT in url:
        manifest.update(
            source_repository="https://github.com/mlai-bonn/GenWL",
            source_commit=GENWL_COMMIT,
            source_commit_date="2021-08-30T13:29:21Z",
        )
    try:
        relative_path = target.relative_to(Path("data/raw")).as_posix()
    except ValueError:
        relative_path = target.name
    entry = {
        "filename": target.name,
        "relative_path": relative_path,
        "source_url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }
    entries = manifest.setdefault("files", [])
    entries[:] = [item for item in entries if item.get("relative_path") != relative_path]
    entries.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def download_official_egonets(raw_dir: str | Path, manifest_path: str | Path, variant: str = "all") -> list[Path]:
    """Fetch the four paper-linked EGONETS variants from the pinned GenWL tree."""
    variants = list(EGONETS_FILES) if variant == "all" else [variant]
    if any(name not in EGONETS_FILES for name in variants):
        raise ValueError(f"variant must be one of {', '.join(EGONETS_FILES)} or 'all'")
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    downloaded = []
    for name in variants:
        for filename in EGONETS_FILES[name]:
            relative_path = f"Data/{name}/{filename}"
            url = f"https://raw.githubusercontent.com/mlai-bonn/GenWL/{GENWL_COMMIT}/{relative_path}"
            downloaded.append(download_file(url, raw_dir / name, manifest_path))
    return downloaded
