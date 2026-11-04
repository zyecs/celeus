"""
Residual-magnitude acquisition policy with label-free loss proxies.

The proposal mixes a probability floor with a term proportional to the
absolute residual between the label-free loss proxy and the surrogate score:

    q_t(j) = alpha + (1 - N_rem*alpha) * |ell_proxy(j) - hat_ell(j)|
                                         / sum_s |ell_proxy(s) - hat_ell(s)|

with alpha = beta_min / N_k. The implemented floor is therefore beta_min/N_k
per remaining item, while the paper's Eq. (8) uses beta/m_t + (1 - beta) p_t;
the two coincide only when the remaining count equals N_k. A uniform fallback
applies when the residual sum is below 1e-12 or when N_rem*alpha >= 1.

Theorem 4 motivates sampling proportionally to the residual magnitude, and
Assumption 2 requires a probability floor. The label-free proxies are the
0-1 loss strategies of Remark 1 (config prefix `remark2_`) and the
cross-entropy strategies of Remark 2 (config prefix `remark1_`).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from celeus.acquisition.base import AcquisitionPolicy
from celeus.core.state import StratumState


class ResidualMagnitudeAcquisition(AcquisitionPolicy):
    """
    Sample proportionally to |ell_proxy - hat_ell| with a beta_min/N_k floor.

    Reads `stratum.ell_proxy` and `stratum.surrogate_scores`. Both must be
    populated; the policy raises RuntimeError otherwise. The loader sets
    `ell_proxy` for every `remark1_*` and `remark2_*` surrogate_type.
    """

    def __init__(self, beta_min: float = 0.05) -> None:
        self.beta_min = beta_min

    def _compute_initial_probs(
        self,
        stratum: StratumState,
    ) -> np.ndarray:
        """
        Compute the step-0 proposal distribution over remaining items.

        Returns a shape (|remaining|,) array of probabilities summing to 1,
        or np.empty(0, dtype=np.float64) when nothing remains.

        Requires stratum.ell_proxy; raises RuntimeError otherwise.
        """
        # ell_proxy is an optional StratumState field, so read it with getattr.
        ell_proxy = getattr(stratum, "ell_proxy", None)
        if ell_proxy is None:
            raise RuntimeError(
                "ResidualMagnitudeAcquisition requires stratum.ell_proxy to be "
                "populated by the loader. Set surrogate_type to one of "
                "remark1_strategy{1,2,3} (loss_type='cross_entropy') or "
                "remark2_strategy{1,2,3,4,5} (loss_type='accuracy')."
            )

        remaining = stratum.remaining_indices()
        N_rem = len(remaining)

        if N_rem == 0:
            return np.empty(0, dtype=np.float64)

        alpha = self.beta_min / stratum.N_k

        # Probability floor plus residual-proportional mass.
        if N_rem * alpha >= 1.0:
            # Degenerate: the floor consumes the entire mass. Fall back to
            # uniform, matching the guard in surrogate_entropy.py.
            pool_probs = np.full(N_rem, 1.0 / N_rem)
        else:
            ell_proxy_rem = ell_proxy[remaining]
            hat_ell_rem = stratum.surrogate_scores[remaining]
            residuals = np.abs(ell_proxy_rem - hat_ell_rem)
            score_sum = residuals.sum()
            if score_sum < 1e-12:
                # All residuals are effectively zero, so every item carries the
                # same correction and the proposal does not matter. Use uniform.
                pool_probs = np.full(N_rem, 1.0 / N_rem)
            else:
                pool_probs = alpha + (1.0 - N_rem * alpha) * (residuals / score_sum)

        return pool_probs

    def select(
        self,
        stratum: StratumState,
        n_k: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        remaining = stratum.remaining_indices()
        n_select = min(n_k, len(remaining))
        if n_select == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        pool_probs = self._compute_initial_probs(stratum)

        # Sequential without-replacement draws; renormalize after each removal.
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
            Current stratum state, with ell_proxy populated by the loader.
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
