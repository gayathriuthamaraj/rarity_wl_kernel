"""Weisfeiler-Lehman (WL) Edge Graph Kernel and Rarity-Weighted Variant.

Reference:
    Shervashidze, N., Schweitzer, P., Leeuwen, E. J. v., Mehlhorn, K., & Borgwardt, K. M. (2011).
    Weisfeiler-Lehman Graph Kernels. Journal of Machine Learning Research, 12, 2539-2561.
    Section 4.2: The Weisfeiler-Lehman Edge Kernel.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence
import numpy as np
import networkx as nx

from .base import GraphKernel, normalize_kernel
from .wl_subtree import WLSubtreeKernel


class WLEdgeKernel(WLSubtreeKernel):
    """Standard Weisfeiler-Lehman Edge Graph Kernel.

    At each refinement depth h = 0 ... n_iter, nodes are relabeled using
    the 1-WL coloring. For each edge (u, v) in the graph, an edge feature
    is defined by the canonical pair of endpoint labels (min(l_h(u), l_h(v)), max(l_h(u), l_h(v))).
    The kernel counts matching edges across all refinement depths.

    Parameters
    ----------
    n_iter : int, default=3
        Number of WL refinement iterations (refinement depth H).
    initial_labeling : {"uniform", "degree", "auto"} or callable, default="uniform"
        Convention for initial node labeling.
    normalize : bool, default=False
        Whether to apply cosine normalization to the kernel matrix.
    node_label_attr : str, default="label"
        Node attribute key for initial labels.
    """

    def extract_features(
        self, graphs: Sequence[nx.Graph], fit: bool = False
    ) -> tuple[np.ndarray, list[str]]:
        """Extract WL edge count feature matrix for the given graphs.

        Parameters
        ----------
        graphs : Sequence of networkx.Graph
            Input graphs.
        fit : bool, default=False
            Whether to discover new vocabulary features (used during fit).

        Returns
        -------
        feature_matrix : np.ndarray of shape (len(graphs), n_features)
            Integer occurrence counts of edge features.
        feature_names : list of str
            Feature descriptions.
        """
        all_depth_labels = self._refine_labels(graphs, fit=fit)

        # Count occurrences of edge label pairs at each depth for each graph
        graph_edge_counts: list[Counter[tuple[int, tuple[Any, Any]]]] = []
        for g_idx, g in enumerate(graphs):
            counts: Counter[tuple[int, tuple[Any, Any]]] = Counter()
            for h in range(self.n_iter + 1):
                node_labels_h = all_depth_labels[g_idx][h]
                for u, v in g.edges():
                    lu, lv = node_labels_h.get(u), node_labels_h.get(v)
                    if lu is None or lv is None:
                        continue
                    # Canonical edge representation (sorted endpoint labels)
                    edge_key = (min(str(lu), str(lv)), max(str(lu), str(lv)))
                    counts[(h, edge_key)] += 1
            graph_edge_counts.append(counts)

        if fit:
            # Build vocabulary from training graphs
            unique_features: set[tuple[int, tuple[Any, Any]]] = set()
            for counts in graph_edge_counts:
                unique_features.update(counts.keys())

            sorted_features = sorted(unique_features, key=lambda x: (x[0], x[1]))
            self.vocabulary_ = {feat: idx for idx, feat in enumerate(sorted_features)}
            self.feature_names_ = [f"depth_{feat[0]}_edge_{feat[1][0]}_{feat[1][1]}" for feat in sorted_features]
            self.feature_depths_ = np.array([feat[0] for feat in sorted_features], dtype=int)

        n_features = len(self.vocabulary_)
        feature_matrix = np.zeros((len(graphs), n_features), dtype=np.float64)

        for g_idx, counts in enumerate(graph_edge_counts):
            for feat, cnt in counts.items():
                if feat in self.vocabulary_:
                    feature_matrix[g_idx, self.vocabulary_[feat]] = cnt

        return feature_matrix, self.feature_names_


class RarityWeightedWLEdgeKernel(WLEdgeKernel):
    """Rarity-weighted Weisfeiler-Lehman Edge Kernel.

    Weights each WL edge feature by its rarity across the training split.
    """

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
            raise ValueError(
                f"Unknown weighting scheme '{weighting}'. "
                f"Must be one of {', '.join(self.WEIGHTING_SCHEMES)}."
            )
        self.weighting = weighting
        self.feature_weights_: np.ndarray | None = None
        self.train_diag_weighted_: np.ndarray | None = None

    def _compute_feature_weights(self) -> None:
        """Derive rarity weights strictly from training fold edge counts."""
        if self.X_train_features_ is None:
            raise ValueError("Training features not available for weight computation")
        eps = 1e-12
        N_train = self.X_train_features_.shape[0]

        total_counts = np.sum(self.X_train_features_, axis=0)
        doc_freqs = np.sum(self.X_train_features_ > 0, axis=0)

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

    def fit(self, graphs: Sequence[nx.Graph], y: Sequence[Any] | None = None) -> "RarityWeightedWLEdgeKernel":
        super().fit(graphs, y=y)
        self._compute_feature_weights()
        sqrt_w = np.sqrt(self.feature_weights_)
        X_train_weighted = self.X_train_features_ * sqrt_w[None, :]
        self.train_diag_weighted_ = np.sum(X_train_weighted ** 2, axis=1)
        return self

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        if not self.is_fitted_ or self.feature_weights_ is None:
            raise ValueError("Kernel must be fitted before transform")
        X_test, _ = self.extract_features(graphs, fit=False)
        sqrt_w = np.sqrt(self.feature_weights_)
        X_test_weighted = X_test * sqrt_w[None, :]
        X_train_weighted = self.X_train_features_ * sqrt_w[None, :]

        K = np.dot(X_test_weighted, X_train_weighted.T)
        if self.normalize:
            diag_test_weighted = np.sum(X_test_weighted ** 2, axis=1)
            K = normalize_kernel(K, self.train_diag_weighted_, diag_test=diag_test_weighted)
        return K
