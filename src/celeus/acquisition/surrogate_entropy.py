"""
Surrogate-entropy importance-sampling acquisition policy.

Surrogate-guided within-stratum acquisition that oversamples items
where target and surrogate models disagree (high cross-entropy),
floored at beta_min/N_k.

Falls back to SelfEntropyAcquisition when only scalar surrogates
are available (no distribution arrays on the stratum).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from celeus.acquisition.base import AcquisitionPolicy
from celeus.acquisition.self_entropy import SelfEntropyAcquisition
from celeus.core.state import StratumState


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Cross-entropy H_cross(p, q) = -sum_c p(c) * ln(q(c)).

    Parameters
    ----------
    p : np.ndarray
        Shape (N, C) target distributions.
    q : np.ndarray
        Shape (N, C) surrogate distributions.

    Returns
    -------
    np.ndarray
        Shape (N,) cross-entropy values.
    """
    q_safe = np.clip(q, 1e-12, None)
    return -np.sum(p * np.log(q_safe), axis=1)


class SurrogateEntropyAcquisition(AcquisitionPolicy):
    """
    Surrogate-entropy IS acquisition policy.

    Oversamples items with large H_cross(p_surr, p_target) to focus
    on examples where surrogate is confident but target assigns low
    probability — indicating likely high-loss items for the target model.

    Falls back to SelfEntropyAcquisition when distribution arrays are absent.

    Source: Berrada et al. SurrogateEntropyAcquisition (scaling-up-active-testing).
    Superseded by ResidualMagnitudeAcquisition, which follows Theorem 4. This
    class is kept so that earlier runs stay reproducible; benchmark.py selects
    it whenever surrogate_type is not a `remark1_*` value.
    """

    def __init__(self, beta_min: float = 0.05):
        self.beta_min = beta_min
        self._fallback = SelfEntropyAcquisition(beta_min=beta_min)

    def _compute_initial_probs(
        self,
        stratum: StratumState,
    ) -> np.ndarray:
        """
        Compute the step-0 proposal distribution over remaining items.

        Delegates to the self-entropy fallback when distribution arrays are
        absent. Returns a shape (|remaining|,) array of probabilities summing
        to 1, or np.empty(0, dtype=np.float64) when nothing remains.
        """
        # Per-class distributions are optional on the stratum.
        if (
            stratum.target_distributions is None
            or stratum.surrogate_distributions is None
        ):
            return self._fallback._compute_initial_probs(stratum)

        remaining = stratum.remaining_indices()
        N_rem = len(remaining)

        if N_rem == 0:
            return np.empty(0, dtype=np.float64)

        alpha = self.beta_min / stratum.N_k

        # Mixture of a uniform floor and the score-proportional part.
        if N_rem * alpha >= 1.0:
            pool_probs = np.full(N_rem, 1.0 / N_rem)
        else:
            # Raw scores: H_cross(p_surr, p_target) = -Σ p_surr(c) log p_target(c),
            # plus an epsilon that keeps the sum positive.
            p_target = stratum.target_distributions[remaining]
            p_surr = stratum.surrogate_distributions[remaining]
            scores = _cross_entropy(p_surr, p_target)
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
        Select n_k unlabeled items via surrogate-entropy acquisition.
        """
        # Per-class distributions are optional on the stratum.
        if (
            stratum.target_distributions is None
            or stratum.surrogate_distributions is None
        ):
            return self._fallback.select(stratum, n_k, rng)

        # Remaining unlabeled items.
        remaining = stratum.remaining_indices()
        n_select = min(n_k, len(remaining))

        if n_select == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        pool_probs = self._compute_initial_probs(stratum)

        # Sequential without-replacement loop; renormalize after each removal.
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
