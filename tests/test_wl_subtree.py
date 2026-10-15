"""Unit and validation tests for the standard Weisfeiler-Lehman subtree kernel."""
from __future__ import annotations

import math
import numpy as np
import networkx as nx
import pytest

from egonets_wl.kernels.wl_subtree import WLSubtreeKernel
from egonets_wl.data.loader import load_dataset


def test_hand_verified_star_graphs():
    """Verify exact feature counts and kernel entries against manual derivation.

    G1: 2-star (Path P3): 0-1-2 (node 1 is center with degree 2; 0 and 2 are leaves)
    G2: 3-star (K_1,3): center 0 connected to leaves 1, 2, 3

    Manual derivation with uniform initial label '1':
      Depth 0:
        G1: count('1') = 3
        G2: count('1') = 4
        K0(G1, G2) = 3 * 4 = 12
        K0(G1, G1) = 3^2 = 9
        K0(G2, G2) = 4^2 = 16

      Depth 1:
        G1 leaf signature: ('1', ('1',)) -> count = 2
        G1 center signature: ('1', ('1', '1')) -> count = 1
        G2 leaf signature: ('1', ('1',)) -> count = 3
        G2 center signature: ('1', ('1', '1', '1')) -> count = 1
        Matching feature: leaf signature ('1', ('1',))
        K1(G1, G2) = 2 * 3 = 6
        K1(G1, G1) = 2^2 + 1^2 = 5
        K1(G2, G2) = 3^2 + 1^2 = 10

      Combined (depth 0 + 1):
        K(G1, G2) = 12 + 6 = 18
        K(G1, G1) = 9 + 5 = 14
        K(G2, G2) = 16 + 10 = 26
        Normalized K(G1, G2) = 18 / sqrt(14 * 26) = 18 / sqrt(364) approx 0.9434564
    """
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2)])

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (0, 2), (0, 3)])

    # Test depth 0
    wl0 = WLSubtreeKernel(n_iter=0, normalize=False)
    K0 = wl0.fit_transform([g1, g2])
    assert K0[0, 0] == 9.0
    assert K0[1, 1] == 16.0
    assert K0[0, 1] == 12.0
    assert K0[1, 0] == 12.0

    # Test depth 1
    wl1 = WLSubtreeKernel(n_iter=1, normalize=False)
    K1 = wl1.fit_transform([g1, g2])
    assert K1[0, 0] == 14.0
    assert K1[1, 1] == 26.0
    assert K1[0, 1] == 18.0
    assert K1[1, 0] == 18.0

    # Test normalized kernel at depth 1
    wl1_norm = WLSubtreeKernel(n_iter=1, normalize=True)
    K1_norm = wl1_norm.fit_transform([g1, g2])
    assert math.isclose(K1_norm[0, 0], 1.0, abs_tol=1e-7)
    assert math.isclose(K1_norm[1, 1], 1.0, abs_tol=1e-7)
    expected_norm = 18.0 / math.sqrt(14.0 * 26.0)
    assert math.isclose(K1_norm[0, 1], expected_norm, abs_tol=1e-7)
    assert math.isclose(K1_norm[1, 0], expected_norm, abs_tol=1e-7)


def test_hand_verified_triangle_and_path():
    """Verify against manual calculation for Triangle K3 vs Path P3.

    K3: 3 nodes, all degree 2.
    P3: 3 nodes, two degree 1, one degree 2.

    At depth 1:
      K3 has 3 nodes with ('1', ('1', '1')).
      P3 has 1 node with ('1', ('1', '1')) and 2 nodes with ('1', ('1',)).
      Inner product at depth 1 = 3 * 1 = 3.
      Total K(K3, P3) = 9 (from depth 0) + 3 (from depth 1) = 12.
      K(K3, K3) = 9 + 3^2 = 18.
      K(P3, P3) = 9 + (1^2 + 2^2) = 14.
    """
    k3 = nx.complete_graph(3)
    p3 = nx.path_graph(3)

    wl = WLSubtreeKernel(n_iter=1, normalize=False)
    K = wl.fit_transform([k3, p3])
    assert K[0, 0] == 18.0
    assert K[1, 1] == 14.0
    assert K[0, 1] == 12.0
    assert K[1, 0] == 12.0


def test_isomorphism_invariance():
    """Isomorphic graphs with permuted node IDs must produce identical representations."""
    # Graph A: cycle on 5 nodes with original indices 0..4
    g_a = nx.cycle_graph(5)

    # Graph B: isomorphic permutation of graph A
    perm = [3, 0, 4, 1, 2]
    mapping = {i: perm[i] for i in range(5)}
    g_b = nx.relabel_nodes(g_a, mapping)

    wl = WLSubtreeKernel(n_iter=3, normalize=True)
    K = wl.fit_transform([g_a, g_b])

    # Normalized kernel between isomorphic graphs must be 1.0
    assert math.isclose(K[0, 0], 1.0, abs_tol=1e-7)
    assert math.isclose(K[1, 1], 1.0, abs_tol=1e-7)
    assert math.isclose(K[0, 1], 1.0, abs_tol=1e-7)
    assert math.isclose(K[1, 0], 1.0, abs_tol=1e-7)

    # Feature counts must match exactly
    phi_a = wl.X_train_features_[0]
    phi_b = wl.X_train_features_[1]
    np.testing.assert_array_equal(phi_a, phi_b)


def test_symmetry_and_psd():
    """Kernel matrix must be symmetric and positive semi-definite (all eigenvalues >= 0)."""
    graphs = [
        nx.cycle_graph(3),
        nx.cycle_graph(4),
        nx.cycle_graph(5),
        nx.path_graph(4),
        nx.star_graph(4),
        nx.complete_graph(4),
        nx.wheel_graph(5),
    ]

    for normalize in [False, True]:
        for n_iter in [0, 1, 2, 3]:
            wl = WLSubtreeKernel(n_iter=n_iter, normalize=normalize)
            K = wl.fit_transform(graphs)

            # Check symmetry
            np.testing.assert_allclose(K, K.T, atol=1e-10)

            # Check positive semi-definiteness
            eigenvalues = np.linalg.eigvalsh(K)
            assert np.all(eigenvalues >= -1e-9), f"Negative eigenvalue detected: {eigenvalues.min()}"

            if normalize:
                # Diagonal entries must be 1.0
                np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-7)
                # Off-diagonal in [-1, 1]
                assert np.all(K >= -1.0 - 1e-7) and np.all(K <= 1.0 + 1e-7)


def test_out_of_sample_transform():
    """Test graphs transformed out-of-sample must match the fit_transform entries."""
    train_graphs = [nx.cycle_graph(4), nx.path_graph(4)]
    test_graphs = [nx.cycle_graph(4), nx.star_graph(3)]

    wl = WLSubtreeKernel(n_iter=2, normalize=False)
    K_train = wl.fit_transform(train_graphs)
    K_test = wl.transform(test_graphs)

    assert K_test.shape == (2, 2)
    # The first test graph is identical to the first train graph, so row 0 of K_test must match row 0 of K_train
    np.testing.assert_allclose(K_test[0], K_train[0], atol=1e-10)


def test_initial_labeling_options():
    """Verify 'uniform', 'degree', and 'auto' conventions."""
    p3 = nx.path_graph(3)

    # Degree labeling separates degree-1 from degree-2 nodes already at depth 0
    wl_deg = WLSubtreeKernel(n_iter=0, initial_labeling="degree")
    phi_deg, names_deg = wl_deg.extract_features([p3], fit=True)
    assert phi_deg.shape == (1, 2)  # Two unique degrees: 1 and 2

    # Uniform labeling has only 1 feature at depth 0
    wl_uni = WLSubtreeKernel(n_iter=0, initial_labeling="uniform")
    phi_uni, names_uni = wl_uni.extract_features([p3], fit=True)
    assert phi_uni.shape == (1, 1)
    assert phi_uni[0, 0] == 3.0

    # Auto labeling with custom node labels
    g = nx.Graph()
    g.add_node(0, label="A")
    g.add_node(1, label="B")
    g.add_edge(0, 1)
    wl_auto = WLSubtreeKernel(n_iter=0, initial_labeling="auto")
    phi_auto, _ = wl_auto.extract_features([g], fit=True)
    assert phi_auto.shape == (1, 2)


def test_egonets_e2e_dataset():
    """Validate baseline WL subtree kernel on actual downloaded EGO-1 graphs."""
    graphs = load_dataset("EGO-1")
    assert len(graphs) == 200

    # Test on a subset of 10 graphs
    sample = graphs[:10]
    wl = WLSubtreeKernel(n_iter=2, normalize=True)
    K = wl.fit_transform(sample)

    assert K.shape == (10, 10)
    np.testing.assert_allclose(K, K.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-7)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-9)
