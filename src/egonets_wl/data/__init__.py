"""Dataset acquisition, inspection, validation, and loading utilities."""
from .download import download_file, download_official_egonets
from .loader import load_dataset, load_tu_dataset
from .validation import inspect_file, validate_egonets

__all__ = [
    "download_file",
    "download_official_egonets",
    "load_dataset",
    "load_tu_dataset",
    "inspect_file",
    "validate_egonets",
]
