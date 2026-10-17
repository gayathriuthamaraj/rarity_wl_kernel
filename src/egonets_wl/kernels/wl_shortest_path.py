"""Weisfeiler-Lehman (WL) Shortest Path Graph Kernel and Rarity-Weighted Variant.

Reference:
    Shervashidze, N., Schweitzer, P., Leeuwen, E. J. v., Mehlhorn, K., & Borgwardt, K. M. (2011).
    Weisfeiler-Lehman Graph Kernels. Journal of Machine Learning Research, 12, 2539-2561.
    Section 4.3: The Weisfeiler-Lehman Shortest Path Kernel.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence
import numpy as np
import scipy.sparse as sp
import networkx as nx

from .base import GraphKernel, normalize_kernel
from .wl_subtree import WLSubtreeKernel


class WLShortestPathKernel(WLSubtreeKernel):
    """Standard Weisfeiler-Lehman Shortest Path Graph Kernel (Memory-Efficient Sparse).

    At each refinement depth h = 0 ... n_iter, nodes are relabeled using
    the 1-WL coloring. For every connected pair of nodes (u, v) with u < v,
    a shortest-path feature is defined by the triplet:
    ((min(l_h(u), l_h(v)), max(l_h(u), l_h(v))), d(u, v)).
    The kernel counts matching shortest-path triplets across all refinement depths.
    """

    def extract_features(
        self, graphs: Sequence[nx.Graph], fit: bool = False
    ) -> tuple[sp.csr_matrix, list[str]]:
        all_depth_labels = self._refine_labels(graphs, fit=fit)

        graph_sp_counts: list[Counter[tuple[int, tuple[Any, Any], int]]] = []
        for g_idx, g in enumerate(graphs):
            counts: Counter[tuple[int, tuple[Any, Any], int]] = Counter()
            # Cache all-pairs shortest paths on graph to avoid recomputing
            if "_sp_lengths" not in g.graph:
                g.graph["_sp_lengths"] = dict(nx.all_pairs_shortest_path_length(g))
            sp_lengths = g.graph["_sp_lengths"]
            nodes = list(g.nodes())

            for h in range(self.n_iter + 1):
                node_labels_h = all_depth_labels[g_idx][h]
                for i in range(len(nodes)):
                    u = nodes[i]
                    lu = node_labels_h.get(u)
                    if lu is None:
                        continue
                    u_sp = sp_lengths.get(u, {})
                    for j in range(i + 1, len(nodes)):
                        v = nodes[j]
                        if v not in u_sp:
                            continue
                        lv = node_labels_h.get(v)
                        if lv is None:
                            continue
                        d = u_sp[v]
                        # Canonical label pair
                        if type(lu) != type(lv):
                            label_pair = (min(str(lu), str(lv)), max(str(lu), str(lv)))
                        else:
                            label_pair = (min(lu, lv), max(lu, lv))
                        counts[(h, label_pair, d)] += 1
            graph_sp_counts.append(counts)

        if fit:
            unique_features: set[tuple[int, tuple[Any, Any], int]] = set()
            for counts in graph_sp_counts:
                unique_features.update(counts.keys())

            sorted_features = sorted(unique_features, key=lambda x: (x[0], str(x[1]), x[2]))
            self.vocabulary_ = {feat: idx for idx, feat in enumerate(sorted_features)}
            self.feature_names_ = [f"depth_{feat[0]}_sp_{feat[1][0]}_{feat[1][1]}_d{feat[2]}" for feat in sorted_features]
            self.feature_depths_ = np.array([feat[0] for feat in sorted_features], dtype=int)

        n_features = len(self.vocabulary_)
        
        # Build CSR sparse matrix to eliminate memory explosion
        rows, cols, data = [], [], []
        for g_idx, counts in enumerate(graph_sp_counts):
            for feat, cnt in counts.items():
                if feat in self.vocabulary_:
                    rows.append(g_idx)
                    cols.append(self.vocabulary_[feat])
                    data.append(cnt)

        feature_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(graphs), n_features),
            dtype=np.float64,
        )

        return feature_matrix, self.feature_names_

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        if not self.is_fitted_ or self.X_train_features_ is None:
            raise ValueError("Kernel must be fitted before transform")
        X_test, _ = self.extract_features(graphs, fit=False)
        K = (X_test @ self.X_train_features_.T).toarray()
        if self.normalize:
            diag_test = np.asarray(X_test.multiply(X_test).sum(axis=1)).ravel()
            K = normalize_kernel(K, self.train_diag_, diag_test=diag_test)
        return K

    def fit(self, graphs: Sequence[nx.Graph], y: Sequence[Any] | None = None) -> "WLShortestPathKernel":
        self.X_train_features_, _ = self.extract_features(graphs, fit=True)
        self.train_diag_ = np.asarray(self.X_train_features_.multiply(self.X_train_features_).sum(axis=1)).ravel()
        self.is_fitted_ = True
        return self


class RarityWeightedWLShortestPathKernel(WLShortestPathKernel):
    """Rarity-weighted Weisfeiler-Lehman Shortest Path Kernel."""

    WEIGHTING_SCHEMES = ("inverse", "sqrt_inverse", "idf", "binary", "uniform")

    def __init__(
        self,
        n_iter: int = 3,
        initial_labeling: str | Callable[[nx.Graph, Any], str] = "uniform",
        normalize: bool = False,
        node_label_attr: str = "label",
        weighting: str = "inverse",
    ) -> None:
        super().__init__(
            n_iter=n_iter,
            initial_labeling=initial_labeling,
            normalize=normalize,
            node_label_attr=node_label_attr,
        )
        if weighting not in self.WEIGHTING_SCHEMES:
            raise ValueError(f"Unknown weighting scheme '{weighting}'")
        self.weighting = weighting
        self.feature_weights_: np.ndarray | None = None
        self.train_diag_weighted_: np.ndarray | None = None

    def _compute_feature_weights(self) -> None:
        if self.X_train_features_ is None:
            raise ValueError("Training features not available for weight computation")
        eps = 1e-12
        N_train = self.X_train_features_.shape[0]

        total_counts = np.asarray(self.X_train_features_.sum(axis=0)).ravel()
        # Document frequency: number of non-zero rows per column
        doc_freqs = np.diff(self.X_train_features_.tocsc().indptr)

        if self.weighting == "inverse":
            self.feature_weights_ = 1.0 / (total_counts + eps)
        elif self.weighting == "sqrt_inverse":
            self.feature_weights_ = 1.0 / np.sqrt(total_counts + eps)
        elif self.weighting == "idf":
            self.feature_weights_ = np.log((N_train + 1.0) / (doc_freqs + 1.0))
        elif self.weighting == "binary":
            self.feature_weights_ = 1.0 / (doc_freqs + eps)
        elif self.weighting == "uniform":
            self.feature_weights_ = np.ones_like(total_counts, dtype=float)

    def fit(self, graphs: Sequence[nx.Graph], y: Sequence[Any] | None = None) -> "RarityWeightedWLShortestPathKernel":
        super().fit(graphs, y=y)
        self._compute_feature_weights()
        sqrt_w = np.sqrt(self.feature_weights_)
        X_train_weighted = self.X_train_features_.multiply(sqrt_w)
        self.train_diag_weighted_ = np.asarray(X_train_weighted.multiply(X_train_weighted).sum(axis=1)).ravel()
        return self

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        if not self.is_fitted_ or self.feature_weights_ is None:
            raise ValueError("Kernel must be fitted before transform")
        X_test, _ = self.extract_features(graphs, fit=False)
        sqrt_w = np.sqrt(self.feature_weights_)
        X_test_weighted = X_test.multiply(sqrt_w)
        X_train_weighted = self.X_train_features_.multiply(sqrt_w)

        K = (X_test_weighted @ X_train_weighted.T).toarray()
        if self.normalize:
            diag_test_weighted = np.asarray(X_test_weighted.multiply(X_test_weighted).sum(axis=1)).ravel()
            K = normalize_kernel(K, self.train_diag_weighted_, diag_test=diag_test_weighted)
        return K
