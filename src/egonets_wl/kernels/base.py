"""Base classes and utilities for graph kernels."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np


def normalize_kernel(
    K: np.ndarray,
    diag_train: np.ndarray,
    diag_test: np.ndarray | None = None,
) -> np.ndarray:
    """Apply cosine normalization to a kernel matrix K.

    K_norm(i, j) = K(i, j) / sqrt(diag_test[i] * diag_train[j])

    Parameters
    ----------
    K : np.ndarray of shape (n_test, n_train)
        Raw kernel matrix.
    diag_train : np.ndarray of shape (n_train,)
        Self-kernel values of training graphs: K(G_j, G_j).
    diag_test : np.ndarray of shape (n_test,), optional
        Self-kernel values of test graphs: K(G_i, G_i).
        If None, K is assumed square and diag_train is used for both.

    Returns
    -------
    np.ndarray of shape (n_test, n_train)
        Normalized kernel matrix with diagonal entries equal to 1.0
        (for non-zero self-kernels) and values in [-1, 1].
    """
    K = np.asarray(K, dtype=float)
    diag_tr = np.asarray(diag_train, dtype=float)
    diag_te = diag_tr if diag_test is None else np.asarray(diag_test, dtype=float)

    # Protect against negative values from numerical round-off
    diag_tr = np.maximum(diag_tr, 0.0)
    diag_te = np.maximum(diag_te, 0.0)

    # Compute outer product of norms: sqrt(diag_te[:, None] * diag_tr[None, :])
    norm_te = np.sqrt(diag_te)
    norm_tr = np.sqrt(diag_tr)
    denom = np.outer(norm_te, norm_tr)

    K_norm = np.zeros_like(K, dtype=float)
    non_zero = denom > 0.0
    K_norm[non_zero] = K[non_zero] / denom[non_zero]

    # Clip to [-1.0, 1.0] for exact numerical stability
    np.clip(K_norm, -1.0, 1.0, out=K_norm)

    # If square and diag_test is None, ensure exact 1.0 on diagonal for non-zero graphs
    if diag_test is None and K.shape[0] == K.shape[1]:
        for i in range(K.shape[0]):
            if diag_tr[i] > 0.0:
                K_norm[i, i] = 1.0

    return K_norm


class GraphKernel(ABC):
    """Abstract base class for graph kernels."""

    @abstractmethod
    def fit(self, graphs: Sequence[Any], y: Sequence[Any] | None = None) -> "GraphKernel":
        """Fit the kernel to the training graphs.

        Parameters
        ----------
        graphs : Sequence of graphs (e.g. networkx.Graph)
            Training graphs.
        y : Sequence of labels, optional
            Ignored; present for scikit-learn pipeline compatibility.

        Returns
        -------
        self : GraphKernel
            Fitted kernel estimator.
        """
        pass

    @abstractmethod
    def transform(self, graphs: Sequence[Any]) -> np.ndarray:
        """Compute the kernel matrix between input graphs and fitted training graphs.

        Parameters
        ----------
        graphs : Sequence of graphs
            Test/evaluation graphs.

        Returns
        -------
        K : np.ndarray of shape (len(graphs), len(self.X_train_))
            Kernel Gram matrix.
        """
        pass

    def fit_transform(
        self, graphs: Sequence[Any], y: Sequence[Any] | None = None
    ) -> np.ndarray:
        """Fit the kernel to graphs and return the training Gram matrix.

        Parameters
        ----------
        graphs : Sequence of graphs
            Training graphs.
        y : Sequence of labels, optional
            Ignored.

        Returns
        -------
        K : np.ndarray of shape (len(graphs), len(graphs))
            Symmetric positive semi-definite Gram matrix.
        """
        return self.fit(graphs, y).transform(graphs)

    @abstractmethod
    def extract_features(
        self, graphs: Sequence[Any], fit: bool = False
    ) -> tuple[np.ndarray, list[str]]:
        """Extract explicit subtree feature count representations.

        Parameters
        ----------
        graphs : Sequence of graphs
            Input graphs.
        fit : bool, default=False
            Whether to update the feature vocabulary during extraction.

        Returns
        -------
        features : np.ndarray of shape (len(graphs), n_features)
            Count matrix of subtree features.
        feature_names : list of str
            Identities of the extracted feature columns.
        """
        pass
