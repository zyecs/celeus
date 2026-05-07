"""
Residual-variance IS acquisition policy (Eq. 12).

Surrogate-guided within-stratum acquisition that oversamples items
with large surrogate residuals |s_i - s_bar_k|, floored at beta_min/N_k.

Source: Spec §2.4 Eq. (12); Blueprint lines 615-663.
CLAUDE.md G1: every formula cites source.
CLAUDE.md Rule 5: rng passed explicitly, never global state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from save.core.state import StratumState
from save.acquisition.base import AcquisitionPolicy


class ResidualVarianceAcquisition(AcquisitionPolicy):
    """
    Residual-variance IS acquisition policy.

    Oversamples items with large surrogate residuals |s_i - s_bar_k|
    to approximate the oracle-optimal acquisition (Paper Theorem 5).

    q_k(i) = alpha + (1 - N_rem * alpha) * score_i / sum(scores)
    where alpha = beta_min / N_k, score_i = |s_i - s_bar_k| + 1e-8.

    Floor guarantee: q_k(i) >= beta_min / N_k for all i.
    [Spec §4.5; Paper Theorem 4: minimum-probability condition]

    Source: Spec §2.4 Eq. (12); Blueprint lines 615-663.
    """

    def __init__(self, beta_min: float = 0.05):
        self.beta_min = beta_min

    def _compute_initial_probs(
        self,
        stratum: StratumState,
    ) -> np.ndarray:
        """
        Compute initial (step-0) proposal distribution over remaining items.

        Helper extracted from select() to expose the first-draw probabilities.
        Returns shape (|remaining|,) array of probabilities summing to 1.

        When remaining is empty, returns np.empty(0, dtype=np.float64).

        Source: Spec §2.4 Eq. (12); Blueprint lines 615-663.
        """
        remaining = stratum.remaining_indices()
        N_rem = len(remaining)

        if N_rem == 0:
            return np.empty(0, dtype=np.float64)

        alpha = self.beta_min / stratum.N_k

        # Compute initial probabilities via mixture distribution
        # [Spec §2.4 Eq. (12)]
        if N_rem * alpha >= 1.0:
            # Defensive guard: uniform fallback
            # [Unreachable with beta_min=0.05, but guards pathological configs]
            pool_probs = np.full(N_rem, 1.0 / N_rem)
        else:
            # Compute raw scores: |s_i - s_bar_k| + epsilon
            scores = np.abs(stratum.surrogate_scores[remaining] - stratum.surrogate_mean)
            scores = scores + 1e-8  # [Blueprint line 639: prevent all-zero]
            score_sum = scores.sum()
            # Mixture distribution: q_i = alpha + (1 - N_rem * alpha) * score_i / score_sum
            # Guarantees: q_i >= alpha (algebraically), sum q_i = 1 (algebraically)
            # [Spec §2.4 Eq. (12); Paper Theorem 4: minimum-probability condition]
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
            Seeded random generator. [CLAUDE.md Rule 5]

        Returns
        -------
        local_indices : np.ndarray
            Shape (n_select,) 0-indexed local indices selected.
        q_values : np.ndarray
            Shape (n_select,) sequential without-replacement sampling probabilities.

        Source: Spec §2.4 Eq. (12); Blueprint lines 615-663.
        """
        # Step 1: Get remaining unlabeled items
        remaining = stratum.remaining_indices()  # 0-indexed local indices
        n_select = min(n_k, len(remaining))

        # Step 2: Handle empty case
        if n_select == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        # Step 3: Compute initial probabilities
        pool_probs = self._compute_initial_probs(stratum)

        # Step 4: Sequential WOR loop
        # [Blueprint lines 647-663; Spec §2.4]
        # After removal: renormalize only, do NOT recompute scores.
        # Floor preserved: q_i/(1-q_j) > q_i >= alpha
        pool = remaining.copy()
        chosen_local = np.empty(n_select, dtype=int)
        q_values = np.empty(n_select, dtype=np.float64)

        for j in range(n_select):
            idx_in_pool = rng.choice(len(pool), p=pool_probs)
            chosen_local[j] = pool[idx_in_pool]
            q_values[j] = pool_probs[idx_in_pool]
            # Remove selected item from pool and probability arrays
            pool = np.delete(pool, idx_in_pool)
            pool_probs = np.delete(pool_probs, idx_in_pool)
            if len(pool) > 0:
                pool_probs = pool_probs / pool_probs.sum()  # Floor preserved: q_i/(1-q_j) > q_i >= alpha

        return chosen_local, q_values

    def get_proposal(
        self,
        stratum: StratumState,
        n_k: int,  # noqa: ARG002 — honoured for interface parity with select
    ) -> np.ndarray:
        """
        Return the current proposal distribution over unlabeled items.

        Unlike select(), this does NOT draw. It exposes the probability mass
        function q_t(j | F_{t-1}, D_N) that select() would use on its first draw.
        Needed by rq4-rq6 replay analysis to compute conditional variances over
        the pool at step-0 (before within-batch sequential renormalization).

        The G4 coupling (n_select == 1) ensures this step-0 distribution is
        all that's needed — no information about subsequent draws within a batch
        is required. [rq4-rq6 spec §4.3; design §5.6]

        Parameters
        ----------
        stratum : StratumState
            Current stratum state.
        n_k : int
            Requested number of draws; unused (honoured for signature parity).

        Returns
        -------
        pool_probs : np.ndarray
            Shape (|remaining|,) float64 probability mass function.
            All entries ≥ 0; entries sum to 1. Returns np.empty(0) when remaining is empty.

        Source: rq4-rq6 design §5.6 + §4.3.
        """
        return self._compute_initial_probs(stratum)
