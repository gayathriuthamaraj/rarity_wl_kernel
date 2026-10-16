"""Standard Weisfeiler-Lehman (WL) subtree graph kernel.

Reference:
    Shervashidze, N., Schweitzer, P., Leeuwen, E. J. v., Mehlhorn, K., & Borgwardt, K. M. (2011).
    Weisfeiler-Lehman Graph Kernels. Journal of Machine Learning Research, 12, 2539-2561.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence
import numpy as np
import networkx as nx

from .base import GraphKernel, normalize_kernel


class WLSubtreeKernel(GraphKernel):
    """Standard Weisfeiler-Lehman Subtree Graph Kernel.

    Computes subtree pattern counts across refinement depths h = 0 ... n_iter
    and evaluates the graph similarity by the inner product of their subtree
    feature vectors.

    Parameters
    ----------
    n_iter : int, default=3
        Number of WL refinement iterations (refinement depth H).
    initial_labeling : {"uniform", "degree", "auto"} or callable, default="uniform"
        Convention for initial node labeling on graphs:
        - "uniform": Assign the constant label "1" to all nodes (standard for unlabeled graphs).
        - "degree": Assign the node degree as string (e.g. "2").
        - "auto": Use node attribute `node_label_attr` if present on all nodes;
          otherwise fall back to "uniform".
        - callable: Function f(graph, node) -> str returning a string label.
    normalize : bool, default=False
        Whether to apply cosine normalization to the kernel matrix:
        K_norm(G, H) = K(G, H) / sqrt(K(G, G) * K(H, H)).
    node_label_attr : str, default="label"
        Node attribute key to read initial labels from when initial_labeling="auto".

    Attributes
    ----------
    vocabulary_ : dict[tuple[int, Any], int]
        Mapping from (depth, compressed_label) to column index in feature matrix.
    feature_names_ : list[str]
        List of human-readable feature names corresponding to columns of feature matrix.
    feature_depths_ : np.ndarray of shape (n_features,)
        Refinement depth (0 ... n_iter) associated with each feature column.
    X_train_features_ : np.ndarray of shape (n_train, n_features)
        Subtree feature count matrix of the fitted training graphs.
    train_diag_ : np.ndarray of shape (n_train,)
        Self-kernel values K(G_i, G_i) of the training graphs.
    """

    def __init__(
        self,
        n_iter: int = 3,
        initial_labeling: str | Callable[[nx.Graph, Any], str] = "uniform",
        normalize: bool = False,
        node_label_attr: str = "label",
    ) -> None:
        if n_iter < 0:
            raise ValueError(f"n_iter must be non-negative, got {n_iter}")
        self.n_iter = n_iter
        self.initial_labeling = initial_labeling
        self.normalize = normalize
        self.node_label_attr = node_label_attr

        # State learned during fit
        self.label_compression_dicts_: list[dict[tuple[Any, tuple[Any, ...]], int]] = []
        self.vocabulary_: dict[tuple[int, Any], int] = {}
        self.feature_names_: list[str] = []
        self.feature_depths_: np.ndarray = np.array([], dtype=int)
        self.X_train_features_: np.ndarray | None = None
        self.train_diag_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def _get_initial_label(self, g: nx.Graph, node: Any) -> str:
        """Assign initial label for a node based on the configured convention."""
        if callable(self.initial_labeling):
            return str(self.initial_labeling(g, node))
        if self.initial_labeling == "degree":
            return str(g.degree(node))
        if self.initial_labeling == "auto":
            if self.node_label_attr in g.nodes[node]:
                return str(g.nodes[node][self.node_label_attr])
            return "1"
        if self.initial_labeling == "uniform":
            return "1"
        raise ValueError(
            f"Unsupported initial_labeling: '{self.initial_labeling}'. "
            "Must be 'uniform', 'degree', 'auto', or a callable."
        )

    def _refine_labels(
        self, graphs: Sequence[nx.Graph], fit: bool
    ) -> list[list[dict[Any, Any]]]:
        """Run WL label refinement across all graphs up to depth n_iter.

        Returns
        -------
        all_graph_labels : list of list of dict
            all_graph_labels[graph_idx][depth][node] = label_at_depth
        """
        if fit:
            self.label_compression_dicts_ = [{} for _ in range(self.n_iter + 1)]

        # Depth 0: Initial labeling
        depth_0_labels: list[dict[Any, Any]] = []
        for g in graphs:
            node_labels = {v: self._get_initial_label(g, v) for v in g.nodes()}
            depth_0_labels.append(node_labels)

        # Structure to hold labels for each graph at each depth: [n_graphs][depth]
        all_depth_labels: list[list[dict[Any, Any]]] = [
            [depth_0_labels[i]] for i in range(len(graphs))
        ]

        # Iterative WL refinement for depth h = 1 ... n_iter
        for h in range(1, self.n_iter + 1):
            comp_dict = self.label_compression_dicts_[h] if not fit else {}
            for g_idx, g in enumerate(graphs):
                prev_labels = all_depth_labels[g_idx][h - 1]
                current_h_labels: dict[Any, Any] = {}
                for v in g.nodes():
                    # Multiset of neighbor labels
                    prev_lbl = prev_labels[v]
                    if prev_lbl is None:
                        # Node's own (h-1)-subtree was unseen in training; it
                        # cannot produce a h-subtree matching any train feature.
                        current_h_labels[v] = None
                        continue
                    neigh_labels = [prev_labels[u] for u in g.neighbors(v)]
                    if None in neigh_labels:
                        # A neighbor's (h-1)-subtree is unseen in training, so the
                        # node's h-subtree (which contains it) cannot match either.
                        current_h_labels[v] = None
                        continue
                    neighbor_labels = sorted(neigh_labels)
                    signature = (prev_lbl, tuple(neighbor_labels))

                    if fit:
                        if signature not in comp_dict:
                            comp_dict[signature] = len(comp_dict) + 1
                        current_h_labels[v] = comp_dict[signature]
                    else:
                        # For test graphs, map to existing compressed label or None if unseen
                        current_h_labels[v] = self.label_compression_dicts_[h].get(signature, None)

                all_depth_labels[g_idx].append(current_h_labels)

            if fit:
                self.label_compression_dicts_[h] = comp_dict

        return all_depth_labels

    def extract_features(
        self, graphs: Sequence[nx.Graph], fit: bool = False
    ) -> tuple[np.ndarray, list[str]]:
        """Extract WL subtree count feature matrix for the given graphs.

        Parameters
        ----------
        graphs : Sequence of networkx.Graph
            Input graphs.
        fit : bool, default=False
            Whether to discover new vocabulary features (used during fit).

        Returns
        -------
        feature_matrix : np.ndarray of shape (len(graphs), n_features)
            Integer occurrence counts of subtree features.
        feature_names : list of str
            Feature descriptions.
        """
        all_depth_labels = self._refine_labels(graphs, fit=fit)

        # Count occurrences of labels at each depth for each graph
        graph_counts: list[Counter[tuple[int, Any]]] = []
        for g_idx in range(len(graphs)):
            counts: Counter[tuple[int, Any]] = Counter()
            for h in range(self.n_iter + 1):
                node_labels = all_depth_labels[g_idx][h]
                for lbl in node_labels.values():
                    if lbl is not None:
                        counts[(h, lbl)] += 1
            graph_counts.append(counts)

        if fit:
            # Build global vocabulary sorted deterministically
            unique_features: set[tuple[int, Any]] = set()
            for counts in graph_counts:
                unique_features.update(counts.keys())

            # Sort deterministically by (depth, label_id)
            sorted_features = sorted(unique_features, key=lambda x: (x[0], str(x[1])))
            self.vocabulary_ = {feat: idx for idx, feat in enumerate(sorted_features)}
            self.feature_names_ = [f"depth_{h}_label_{lbl}" for h, lbl in sorted_features]
            self.feature_depths_ = np.array([h for h, _ in sorted_features], dtype=int)

        n_features = len(self.vocabulary_)
        X = np.zeros((len(graphs), n_features), dtype=float)

        for g_idx, counts in enumerate(graph_counts):
            for feat, cnt in counts.items():
                if feat in self.vocabulary_:
                    col = self.vocabulary_[feat]
                    X[g_idx, col] = cnt

        return X, self.feature_names_

    def fit(self, graphs: Sequence[nx.Graph], y: Sequence[Any] | None = None) -> "WLSubtreeKernel":
        """Fit the WL subtree kernel on training graphs.

        Learns label compressions, builds the feature vocabulary, and
        stores training feature counts and diagonal entries.
        """
        graphs = list(graphs)
        self.X_train_features_, _ = self.extract_features(graphs, fit=True)
        # Compute self-kernels: K(G_i, G_i) = sum_j (X_ij)^2
        self.train_diag_ = np.sum(self.X_train_features_ ** 2, axis=1)
        self.is_fitted_ = True
        return self

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        """Compute the kernel matrix between input graphs and the training graphs."""
        if not self.is_fitted_:
            raise ValueError("WLSubtreeKernel must be fitted before calling transform().")

        graphs = list(graphs)
        X_test_features, _ = self.extract_features(graphs, fit=False)

        # Raw linear kernel: K = X_test @ X_train.T
        K = np.dot(X_test_features, self.X_train_features_.T)

        if self.normalize:
            test_diag = np.sum(X_test_features ** 2, axis=1)
            K = normalize_kernel(K, diag_train=self.train_diag_, diag_test=test_diag)

        return K

    def fit_transform(
        self, graphs: Sequence[nx.Graph], y: Sequence[Any] | None = None
    ) -> np.ndarray:
        """Fit and return the training Gram matrix."""
        self.fit(graphs, y)
        K = np.dot(self.X_train_features_, self.X_train_features_.T)
        if self.normalize:
            K = normalize_kernel(K, diag_train=self.train_diag_)
        return K

    def get_feature_names_out(self) -> list[str]:
        """Return feature names for extracted subtree features."""
        if not self.is_fitted_:
            raise ValueError("WLSubtreeKernel is not fitted yet.")
        return list(self.feature_names_)
