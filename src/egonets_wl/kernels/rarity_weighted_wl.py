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
from .base import normalize_kernel

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
        Rarity weighting scheme computed from the TRAINING graphs only:
        - "inverse":     w_j = 1 / (total_count_j + eps)
        - "sqrt_inverse": w_j = 1 / sqrt(total_count_j + eps)
        - "idf":          w_j = log((N_train + 1) / (df_j + 1))   (document-frequency rarity)
        - "binary":       w_j = 1 / (df_j + eps)                  (df_j = # train graphs containing feature)
        - "uniform":      w_j = 1 (reproduces the standard WL kernel; sanity check)
    """

    WEIGHTING_SCHEMES = ("inverse", "sqrt_inverse", "idf", "binary", "uniform")

    def __init__(self, n_iter: int = 3, initial_labeling: str | callable = "uniform", normalize: bool = False, node_label_attr: str = "label", weighting: str = "inverse") -> None:
        super().__init__(n_iter=n_iter, initial_labeling=initial_labeling, normalize=normalize, node_label_attr=node_label_attr)
        if weighting not in self.WEIGHTING_SCHEMES:
            raise ValueError(
                f"Unknown weighting scheme '{weighting}'. "
                f"Must be one of {', '.join(self.WEIGHTING_SCHEMES)}."
            )
        self.weighting = weighting
        self.feature_weights_: np.ndarray | None = None
        self.train_diag_weighted_: np.ndarray | None = None

    def _compute_feature_weights(self) -> None:
        """Compute rarity weights from the training feature matrix.

        All statistics are derived exclusively from the fitted training
        feature matrix X_train_features_, so no test-graph information can
        leak into the weights.
        """
        if self.X_train_features_ is None:
            raise ValueError("Training features not available for weight computation")
        eps = 1e-12
        freq = np.sum(self.X_train_features_, axis=0)
        if self.weighting == "inverse":
            self.feature_weights_ = 1.0 / (freq + eps)
        elif self.weighting == "sqrt_inverse":
            self.feature_weights_ = 1.0 / np.sqrt(freq + eps)
        elif self.weighting == "idf":
            df = np.sum(self.X_train_features_ > 0, axis=0)
            n_train = self.X_train_features_.shape[0]
            self.feature_weights_ = np.log((n_train + 1.0) / (df + 1.0))
        elif self.weighting == "binary":
            df = np.sum(self.X_train_features_ > 0, axis=0)
            self.feature_weights_ = 1.0 / (df + eps)
        elif self.weighting == "uniform":
            self.feature_weights_ = np.ones_like(freq, dtype=float)
        else:  # pragma: no cover - constructor validates the scheme
            raise ValueError(f"Unknown weighting scheme '{self.weighting}'")

    def fit(self, graphs, y=None):
        """Fit the kernel on training graphs and compute rarity weights."""
        super().fit(graphs, y)
        self._compute_feature_weights()
        # Train-side self-kernels must use the WEIGHTED feature vectors so that
        # cosine normalization of the weighted Gram matrix has unit diagonal.
        X_w = self.X_train_features_ * self.feature_weights_
        self.train_diag_weighted_ = np.sum(X_w ** 2, axis=1)
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
            K = normalize_kernel(K, diag_train=self.train_diag_weighted_, diag_test=test_diag)
        return K

    def fit_transform(self, graphs, y=None):
        """Fit on ``graphs`` and return weighted training Gram matrix."""
        self.fit(graphs, y)
        X_train_weighted = self.X_train_features_ * self.feature_weights_
        K = np.dot(X_train_weighted, X_train_weighted.T)
        if self.normalize:
            K = normalize_kernel(K, diag_train=self.train_diag_weighted_)
        return K
