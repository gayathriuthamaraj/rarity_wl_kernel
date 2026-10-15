"""
Rarity-weighted Weisfeiler-Lehman (WL) subtree graph kernel.

This kernel builds upon the standard WLSubtreeKernel by weighting each
subtree feature according to its rarity (inverse frequency) across the
training dataset. Rare patterns receive higher weight, while common
patterns are down‑weighted. The weighting is computed after fitting on
the training graphs and is applied during kernel computation.
"""

from __future__ import annotations
import numpy as np
from .wl_subtree import WLSubtreeKernel

class RarityWeightedWLKernel(WLSubtreeKernel):
    """Rarity‑weighted WL subtree kernel.

    Parameters
    ----------
    n_iter : int, default=3
        Number of WL refinement iterations.
    initial_labeling : {"uniform", "degree", "auto"} or callable, default="uniform"
        Initial node labeling strategy.
    normalize : bool, default=False
        Whether to apply cosine normalization to the kernel matrix.
    node_label_attr : str, default="label"
        Node attribute key for ``auto`` labeling.
    weighting : str, default="inverse"
        Weighting scheme. Currently only ``"inverse"`` is supported, which
        uses ``1 / (frequency + eps)`` where frequency is the total count of
        a feature across the training graphs.
    """

    def __init__(self, n_iter: int = 3, initial_labeling: str | callable = "uniform", normalize: bool = False, node_label_attr: str = "label", weighting: str = "inverse") -> None:
        super().__init__(n_iter=n_iter, initial_labeling=initial_labeling, normalize=normalize, node_label_attr=node_label_attr)
        if weighting != "inverse":
            raise ValueError("Only 'inverse' weighting is currently implemented")
        self.weighting = weighting
        self.feature_weights_: np.ndarray | None = None

    def _compute_feature_weights(self) -> None:
        """Compute rarity weights from the training feature matrix.

        The weight for a feature ``j`` is defined as ``1 / (freq_j + eps)``
        where ``freq_j`` is the sum of counts of that feature over all
        training graphs. ``eps`` avoids division by zero.
        """
        if self.X_train_features_ is None:
            raise ValueError("Training features not available for weight computation")
        freq = np.sum(self.X_train_features_, axis=0)
        eps = 1e-12
        self.feature_weights_ = 1.0 / (freq + eps)

    def fit(self, graphs, y=None):
        """Fit the kernel on training graphs and compute rarity weights."""
        super().fit(graphs, y)
        self._compute_feature_weights()
        return self

    def transform(self, graphs):
        """Compute weighted kernel matrix between ``graphs`` and training graphs.

        Weighted inner product: K(i, j) = sum_k w_k * x_i_k * x_j_k
        """
        if not self.is_fitted_:
            raise ValueError("RarityWeightedWLKernel must be fitted before calling transform().")
        if self.feature_weights_ is None:
            raise ValueError("Feature weights have not been computed.")
        X_test, _ = self.extract_features(list(graphs), fit=False)
        X_test_weighted = X_test * self.feature_weights_
        K = np.dot(X_test_weighted, self.X_train_features_.T)
        if self.normalize:
            test_diag = np.sum(X_test_weighted ** 2, axis=1)
            K = normalize_kernel(K, diag_train=self.train_diag_, diag_test=test_diag)
        return K

    def fit_transform(self, graphs, y=None):
        """Fit on ``graphs`` and return weighted training Gram matrix."""
        self.fit(graphs, y)
        X_train_weighted = self.X_train_features_ * self.feature_weights_
        K = np.dot(X_train_weighted, X_train_weighted.T)
        if self.normalize:
            K = normalize_kernel(K, diag_train=self.train_diag_)
        return K
