"""Sample-level filters applicable to the evaluation pool.

Kept at top-level so that ``save.loader`` can depend on it without creating
a reverse dependency into ``save.paper_experiment`` (imports only NumPy).
"""
from __future__ import annotations

import numpy as np


def build_ce_nll_mask(losses: np.ndarray, threshold: float) -> np.ndarray:
    """Return boolean mask of shape (N,), True where losses <= threshold.

    Parameters
    ----------
    losses : np.ndarray or array-like
        Per-sample cross-entropy loss values (any float dtype).
    threshold : float
        Inclusive upper bound. Samples with ``losses > threshold`` are masked
        out (mask[i] == False).

    Returns
    -------
    np.ndarray
        Boolean ndarray with ``dtype == np.bool_`` and ``shape == losses.shape``.
    """
    return (np.asarray(losses) <= threshold).astype(bool)
