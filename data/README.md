# Data policy

Raw source files belong in `raw/`; derived or normalized files belong in `interim/` and `processed/`. Large data directories are ignored by Git. Preserve source archives unchanged. Record direct source URL, retrieval date, filename, byte size, and SHA-256 in `metadata/manifest.json`. Do not report dataset statistics until source files have been inspected and parsed. See `../docs/dataset_documentation.md` for current source status.
