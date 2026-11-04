"""
Residual-variance importance-sampling acquisition policy.

Surrogate-guided within-stratum acquisition that oversamples items
with large surrogate residuals |s_i - s_bar_k|, floored at beta_min/N_k.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from celeus.core.state import StratumState
from celeus.acquisition.base import AcquisitionPolicy


class ResidualVarianceAcquisition(AcquisitionPolicy):
    """
    Residual-variance IS acquisition policy.

    Oversamples items with large surrogate residuals |s_i - s_bar_k| as a
    label-free approximation of the variance-minimizing proposal of Theorem 4.

    q_k(i) = alpha + (1 - N_rem * alpha) * score_i / sum(scores)
    where alpha = beta_min / N_k, score_i = |s_i - s_bar_k| + 1e-8.

    Floor guarantee: q_k(i) >= beta_min / N_k for all i, which is the
    minimum-probability condition of Assumption 2.
    """

    def __init__(self, beta_min: float = 0.05):
        self.beta_min = beta_min

    def _compute_initial_probs(
        self,
        stratum: StratumState,
    ) -> np.ndarray:
        """
        Compute the step-0 proposal distribution over remaining items.

        Returns a shape (|remaining|,) array of probabilities summing to 1,
        or np.empty(0, dtype=np.float64) when nothing remains.
        """
        remaining = stratum.remaining_indices()
        N_rem = len(remaining)

        if N_rem == 0:
            return np.empty(0, dtype=np.float64)

        alpha = self.beta_min / stratum.N_k

        # Mixture of a uniform floor and the score-proportional part.
        if N_rem * alpha >= 1.0:
            # Defensive guard: the floor alone would exceed total mass. This is
            # unreachable with beta_min=0.05, but guards pathological configs.
            pool_probs = np.full(N_rem, 1.0 / N_rem)
        else:
            # Raw scores: |s_i - s_bar_k| + epsilon.
            scores = np.abs(stratum.surrogate_scores[remaining] - stratum.surrogate_mean)
            scores = scores + 1e-8  # keep the score sum away from zero
            score_sum = scores.sum()
            # q_i = alpha + (1 - N_rem * alpha) * score_i / score_sum, which
            # gives q_i >= alpha and sum q_i = 1 algebraically.
            pool_probs = alpha + (1.0 - N_rem * alpha) * (scores / score_sum)

        return pool_probs

    def select(
        self,
        stratum: StratumState,
        n_k: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select n_k unlabeled items via residual-variance acquisition.

        Parameters
        ----------
        stratum : StratumState
            Current stratum state.
        n_k : int
            Requested number of items to select.
        rng : np.random.Generator
            Seeded random generator.

        Returns
        -------
        local_indices : np.ndarray
            Shape (n_select,) 0-indexed local indices selected.
        q_values : np.ndarray
            Shape (n_select,) sequential without-replacement sampling probabilities.
        """
        # Remaining unlabeled items.
        remaining = stratum.remaining_indices()  # 0-indexed local indices
        n_select = min(n_k, len(remaining))

        if n_select == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        pool_probs = self._compute_initial_probs(stratum)

        # Sequential without-replacement loop. After each removal the
        # probabilities are renormalized, not recomputed, so the floor still
        # holds: q_i/(1-q_j) > q_i >= alpha.
        pool = remaining.copy()
        chosen_local = np.empty(n_select, dtype=int)
        q_values = np.empty(n_select, dtype=np.float64)

        for j in range(n_select):
            idx_in_pool = rng.choice(len(pool), p=pool_probs)
            chosen_local[j] = pool[idx_in_pool]
            q_values[j] = pool_probs[idx_in_pool]
            pool = np.delete(pool, idx_in_pool)
            pool_probs = np.delete(pool_probs, idx_in_pool)
            if len(pool) > 0:
                pool_probs = pool_probs / pool_probs.sum()

        return chosen_local, q_values

    def get_proposal(
        self,
        stratum: StratumState,
        n_k: int,  # noqa: ARG002 — honoured for interface parity with select
    ) -> np.ndarray:
        """
        Return the current proposal distribution over unlabeled items.

        Unlike select(), this does not draw. It exposes the probability mass
        function that select() would use on its first draw. The replay
        diagnostics need it to compute conditional variances over the pool at
        step 0, before the within-batch sequential renormalization.

        Replay runs with n_select == 1, so the step-0 distribution is all that
        is required.

        Parameters
        ----------
        stratum : StratumState
            Current stratum state.
        n_k : int
            Requested number of draws; unused (kept for signature parity).

        Returns
        -------
        pool_probs : np.ndarray
            Shape (|remaining|,) float64 probability mass function.
            All entries ≥ 0; entries sum to 1. Returns np.empty(0) when
            remaining is empty.
        """
        return self._compute_initial_probs(stratum)
