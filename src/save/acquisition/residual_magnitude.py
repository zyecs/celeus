"""
Residual-magnitude acquisition policy — Theorem 5 form with Remark 1
label-free proxies.

Implements the practical oracle proposal:

    q_t(j) = alpha + (1 - N_rem*alpha) * |ell_proxy(j) - hat_ell(j)|
                                         / sum_s |ell_proxy(s) - hat_ell(s)|

with alpha = beta_min / N_k, and a uniform fallback when the residuals
sum is below 1e-12 or when N_rem*alpha >= 1.

Source: tmp/draft_save_v0410.tex Theorem 5 (thm:optimal, lines 411-420),
Theorem 4 (thm:consistency, lines 404-406), Remark 1 (lines 433-453),
all-zero remark (line 422).
CLAUDE.md G1: every formula cites source; Rule 5: rng passed explicitly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from save.acquisition.base import AcquisitionPolicy
from save.core.state import StratumState


class ResidualMagnitudeAcquisition(AcquisitionPolicy):
    """
    Sample proportionally to |ell_proxy - hat_ell| with a beta_min/N_k floor.

    Reads `stratum.ell_proxy` and `stratum.surrogate_scores`. Requires both
    to be populated; raises RuntimeError otherwise — the caller (loader)
    must set `ell_proxy` for any `remark1_*` surrogate_type.

    Source: draft Theorem 5 lines 411-420, Remark 1 lines 433-453.
    """

    def __init__(self, beta_min: float = 0.05) -> None:
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

        Requires stratum.ell_proxy to be populated; raises RuntimeError otherwise.

        Source: draft Theorem 5 lines 411-420, Remark 1 lines 433-453.
        """
        # Stage 13: ell_proxy is added to StratumState in Task 4. For now,
        # we read via getattr so the policy works both before and after the
        # field exists.
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

        # Compute initial probabilities.
        # [Draft Theorem 5 + Theorem 4 floor; spec §3.4]
        if N_rem * alpha >= 1.0:
            # Degenerate: floor consumes entire mass. Fall back to uniform
            # to match legacy guard at surrogate_entropy.py:99.
            pool_probs = np.full(N_rem, 1.0 / N_rem)
        else:
            ell_proxy_rem = ell_proxy[remaining]
            hat_ell_rem = stratum.surrogate_scores[remaining]
            residuals = np.abs(ell_proxy_rem - hat_ell_rem)
            score_sum = residuals.sum()
            if score_sum < 1e-12:
                # All residuals effectively zero — draft line 422 says the
                # choice of proposal distribution is immaterial; pick uniform.
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

        # Compute initial probabilities
        pool_probs = self._compute_initial_probs(stratum)

        # Sequential WOR loop (identical in structure to SurrogateEntropyAcquisition).
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
            Current stratum state, with ell_proxy populated by the loader.
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
