import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Ensure the src directory is on the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from egonets_wl.data.download import download_official_egonets
from egonets_wl.data.loader import load_dataset
from egonets_wl.kernels.wl_subtree import WLSubtreeKernel
from egonets_wl.kernels.rarity_weighted_wl import RarityWeightedWLKernel


def test_egonets_kernels_on_real_data(tmp_path: Path):
    """Integration test that downloads a real EGONETS variant and runs both kernels.

    The test verifies that the download helper works (HTTPS only), that the loader
    produces a non‑empty list of NetworkX graphs, and that the kernel matrices are
    square, symmetric and have the correct shape.
    """
    raw_dir = tmp_path / "raw"
    manifest = tmp_path / "manifest.json"

    # Download a single variant to keep the test lightweight
    downloaded = download_official_egonets(raw_dir, manifest, variant="EGO-1")
    assert downloaded, "No files were downloaded for EGO-1"

    # Load the graphs from the just‑downloaded files
    graphs = load_dataset("EGO-1", raw_dir=raw_dir)
    assert len(graphs) > 0, "Loader returned an empty graph list"

    # Run the standard WL kernel
    wl = WLSubtreeKernel(n_iter=2)
    K_wl = wl.fit_transform(graphs)
    # Run the rarity‑weighted WL kernel
    rw = RarityWeightedWLKernel(n_iter=2)
    K_rw = rw.fit_transform(graphs)

    # Basic sanity checks
    n = len(graphs)
    assert K_wl.shape == (n, n)
    assert K_rw.shape == (n, n)
    # Symmetry
    assert np.allclose(K_wl, K_wl.T)
    assert np.allclose(K_rw, K_rw.T)
    # Off‑diagonal of weighted kernel should be <= standard kernel (same reasoning as unit test)
    off_std = K_wl[np.triu_indices_from(K_wl, k=1)]
    off_rw = K_rw[np.triu_indices_from(K_rw, k=1)]
    assert np.all(off_rw <= off_std + 1e-8)

    # Clean‑up (optional) – the tmp_path fixture will be removed automatically
