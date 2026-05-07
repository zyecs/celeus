"""
SAVE surrogate pipeline utilities.

Currently provides:
  - degrade_surrogate(): surrogate quality degradation for ablation A4

Source: Spec section 5 A4; Blueprint section 3.2.
"""

from __future__ import annotations

import numpy as np


def degrade_surrogate(
    scores: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Degrade surrogate scores by mixing with uniform noise.

    s_hat^(gamma) = gamma * s_hat + (1 - gamma) * U(0, 1)

    Parameters
    ----------
    scores : np.ndarray
        Shape (N,) original surrogate scores in [0, 1].
    gamma : float
        Degradation parameter in [0, 1].
        gamma=1.0: original scores unchanged.
        gamma=0.0: pure random noise.
    rng : np.random.Generator
        Seeded random generator. [CLAUDE.md Rule 5]

    Returns
    -------
    np.ndarray
        Shape (N,) degraded scores, clipped to [0, 1].

    Raises
    ------
    ValueError
        If gamma is outside [0, 1] or if any score is outside [0, 1].

    Source: Spec section 5 A4; Blueprint section 3.2.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError(
            f"scores must be in [0, 1], got range "
            f"[{float(scores.min()):.6f}, {float(scores.max()):.6f}]"
        )

    if gamma == 1.0:
        return scores.copy()

    noise = rng.uniform(0.0, 1.0, size=scores.shape)
    degraded = gamma * scores + (1.0 - gamma) * noise
    return np.clip(degraded, 0.0, 1.0)
