import sys
import os
import numpy as np
import networkx as nx

# Ensure the src directory is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from egonets_wl.kernels.wl_subtree import WLSubtreeKernel
from egonets_wl.kernels.rarity_weighted_wl import RarityWeightedWLKernel
from egonets_wl.kernels.wl_edge import WLEdgeKernel, RarityWeightedWLEdgeKernel
from egonets_wl.kernels.wl_shortest_path import WLShortestPathKernel, RarityWeightedWLShortestPathKernel


def _simple_graphs():
    """Create a small set of graphs for testing.
    Returns a list of three NetworkX graphs.
    """
    g1 = nx.path_graph(3)   # 0-1-2
    g2 = nx.cycle_graph(3)  # triangle
    g3 = nx.path_graph(4)   # 0-1-2-3
    return [g1, g2, g3]


def test_fit_transform_shapes():
    graphs = _simple_graphs()
    wl = WLSubtreeKernel(n_iter=2)
    K_wl = wl.fit_transform(graphs)
    rw = RarityWeightedWLKernel(n_iter=2)
    K_rw = rw.fit_transform(graphs)

    # Both kernels should produce a square Gram matrix of shape (n, n)
    assert K_wl.shape == (3, 3)
    assert K_rw.shape == (3, 3)
    # Matrices should be symmetric
    assert np.allclose(K_wl, K_wl.T)
    assert np.allclose(K_rw, K_rw.T)


def test_weighted_vs_standard_off_diagonal():
    graphs = _simple_graphs()
    wl = WLSubtreeKernel(n_iter=2)
    K_wl = wl.fit_transform(graphs)
    rw = RarityWeightedWLKernel(n_iter=2)
    K_rw = rw.fit_transform(graphs)

    # Off‑diagonal entries of the weighted kernel should be less than or equal to the standard kernel
    # (rare features get down‑weighted, common features dominate the standard sum)
    off_diag_std = K_wl[np.triu_indices_from(K_wl, k=1)]
    off_diag_rw = K_rw[np.triu_indices_from(K_rw, k=1)]
    assert np.all(off_diag_rw <= off_diag_std + 1e-8)


def test_normalized_kernel_diagonal_equals_one():
    graphs = _simple_graphs()
    # Use cosine normalization – diagonal should be exactly 1.0 (within tolerance)
    wl_norm = WLSubtreeKernel(n_iter=2, normalize=True)
    K_wl_norm = wl_norm.fit_transform(graphs)
    rw_norm = RarityWeightedWLKernel(n_iter=2, normalize=True)
    K_rw_norm = rw_norm.fit_transform(graphs)

    diag_wl = np.diag(K_wl_norm)
    diag_rw = np.diag(K_rw_norm)
    assert np.allclose(diag_wl, np.ones_like(diag_wl), atol=1e-8)
    assert np.allclose(diag_rw, np.ones_like(diag_rw), atol=1e-8)


def test_feature_weights_computed_correctly():
    graphs = _simple_graphs()
    rw = RarityWeightedWLKernel(n_iter=2)
    rw.fit(graphs)
    # After fitting, feature_weights_ should be a 1‑D array with length equal to number of features
    assert hasattr(rw, "feature_weights_")
    assert isinstance(rw.feature_weights_, np.ndarray)
    assert rw.feature_weights_.ndim == 1
    # No weight should be zero (eps prevents division by zero)
    assert np.all(rw.feature_weights_ > 0)


def test_edge_and_shortest_path_kernels():
    graphs = _simple_graphs()
    
    # Test WL Edge Kernel
    wl_edge = WLEdgeKernel(n_iter=2, normalize=True)
    K_edge = wl_edge.fit_transform(graphs)
    assert K_edge.shape == (3, 3)
    assert np.allclose(K_edge, K_edge.T)
    assert np.allclose(np.diag(K_edge), np.ones(3), atol=1e-8)

    # Test Rarity-Weighted WL Edge Kernel
    rw_edge = RarityWeightedWLEdgeKernel(n_iter=2, weighting="idf", normalize=True)
    K_rw_edge = rw_edge.fit_transform(graphs)
    assert K_rw_edge.shape == (3, 3)
    assert np.allclose(K_rw_edge, K_rw_edge.T)
    assert np.allclose(np.diag(K_rw_edge), np.ones(3), atol=1e-8)

    # Test WL Shortest Path Kernel
    wl_sp = WLShortestPathKernel(n_iter=2, normalize=True)
    K_sp = wl_sp.fit_transform(graphs)
    assert K_sp.shape == (3, 3)
    assert np.allclose(K_sp, K_sp.T)
    assert np.allclose(np.diag(K_sp), np.ones(3), atol=1e-8)

    # Test Rarity-Weighted WL Shortest Path Kernel
    rw_sp = RarityWeightedWLShortestPathKernel(n_iter=2, weighting="idf", normalize=True)
    K_rw_sp = rw_sp.fit_transform(graphs)
    assert K_rw_sp.shape == (3, 3)
    assert np.allclose(K_rw_sp, K_rw_sp.T)
    assert np.allclose(np.diag(K_rw_sp), np.ones(3), atol=1e-8)
