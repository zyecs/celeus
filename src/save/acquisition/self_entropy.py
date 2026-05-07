"""
Self-entropy IS acquisition policy (Eq. 14).

Surrogate-guided within-stratum acquisition that oversamples items
with high binary entropy H_bin(s_hat_i), floored at beta_min/N_k.

Source: Spec section 2.4 Eq. (14); Berrada et al. SelfEntropyAcquisition.
CLAUDE.md G1: every formula cites source.
CLAUDE.md Rule 5: rng passed explicitly, never global state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from save.acquisition.base import AcquisitionPolicy
from save.core.state import StratumState


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Binary entropy H_bin(p) = -p*ln(p) - (1-p)*ln(1-p).

    Boundary convention: 0*ln(0) = 0. Uses clipping for numerically
    stable vectorized computation.

    Parameters
    ----------
    p : np.ndarray
        Values in [0, 1].

    Returns
    -------
    np.ndarray
        H_bin(p) in [0, ln(2)].

    Source: Spec section 2.4 Eq. (14).
    [DERIVED -- verify]: standard binary entropy formula.
    """
    safe_p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -safe_p * np.log(safe_p) - (1.0 - safe_p) * np.log(1.0 - safe_p)


class SelfEntropyAcquisition(AcquisitionPolicy):
    """
    Self-entropy IS acquisition policy.

    Oversamples items with large H_bin(s_hat_i) to focus on the most
    uncertain surrogate predictions.

    q_k(i) = alpha + (1 - N_rem * alpha) * score_i / sum(scores)
    where alpha = beta_min / N_k, score_i = H_bin(s_hat_i) + 1e-8.

    Floor guarantee: q_k(i) >= beta_min / N_k for all i.
    [Spec section 4.5; Paper Theorem 4: minimum-probability condition]

    Source: Spec section 2.4 Eq. (14); Berrada et al. SelfEntropyAcquisition.
    [Berrada et al. SelfEntropyAcquisition; adapted for scalar surrogates]
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

        Source: Spec section 2.4 Eq. (14); Blueprint lines 647-663.
        """
        remaining = stratum.remaining_indices()
        N_rem = len(remaining)

        if N_rem == 0:
            return np.empty(0, dtype=np.float64)

        alpha = self.beta_min / stratum.N_k

        # Compute initial probabilities via mixture distribution
        # [Spec section 2.4 Eq. (14)]
        if N_rem * alpha >= 1.0:
            # Defensive guard: uniform fallback
            pool_probs = np.full(N_rem, 1.0 / N_rem)
        else:
            # Compute raw scores: H_bin(s_hat_i) + epsilon
            # [Spec section 2.4 Eq. (14)]
            scores = _binary_entropy(stratum.surrogate_scores[remaining])
            scores = scores + 1e-8
            score_sum = scores.sum()
            pool_probs = alpha + (1.0 - N_rem * alpha) * (scores / score_sum)

        return pool_probs

    def select(
        self,
        stratum: StratumState,
        n_k: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select n_k unlabeled items via self-entropy acquisition.

        Source: Spec section 2.4 Eq. (14); Blueprint lines 647-663 pattern.
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
        # [Same procedure as ResidualVarianceAcquisition; Blueprint lines 647-663]
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
