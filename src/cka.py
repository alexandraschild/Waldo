"""Linear CKA (Centered Kernel Alignment) between two sets of layer activations.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited" (2019). CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L)),
computed here with a linear kernel, which reduces to the closed form below
and avoids ever materializing the n x n Gram matrices.
"""

import numpy as np


def _center_columns(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between activation matrices x (n, d1) and y (n, d2).

    n is the number of samples (must match between x and y); d1/d2 are the
    (possibly different) feature dimensionalities for each side.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: {x.shape[0]} vs {y.shape[0]}")

    x = _center_columns(x.astype(np.float64))
    y = _center_columns(y.astype(np.float64))

    # ||X^T Y||_F^2 == HSIC(K, L) up to the constant shared by all three terms below,
    # which cancels in the ratio - see Kornblith et al. eq. 3-4.
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    norm_x = np.linalg.norm(x.T @ x, ord="fro")
    norm_y = np.linalg.norm(y.T @ y, ord="fro")

    denom = norm_x * norm_y
    if denom == 0:
        return float("nan")
    return float(cross / denom)
