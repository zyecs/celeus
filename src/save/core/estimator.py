"""
SAVE AIPW estimator.

Implements Draft Eq. (1):
  R_hat_k = surrogate_mean + sum_past_residuals/N_k + (L_t - s_t)/(N_k * q_t)

No clipping — the rescale-then-map-back mechanism in EValueCS handles
boundedness without sacrificing conditional unbiasedness. [Draft §3.2]

Source: Draft Eq. (1); [Draft Theorem 1].
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

from typing import List

import numpy as np

from save.core.state import StratumState


class AIPWEstimator:
    """
    SAVE AIPW risk estimator.

    Implements Draft Eq. (1): a mixture of true losses for past
    labeled items and surrogate scores for unlabeled items, plus a
    single-item correction for the most recently drawn item.

    No clipping is applied — the EValueCS rescaling mechanism handles
    boundedness without breaking conditional unbiasedness. [Draft §3.2]

    Parameters
    ----------
    u_max : float
        Kept for API compatibility. Not used.
    """

    def __init__(self, u_max: float = 20.0) -> None:
        self.u_max = u_max  # retained for API compat; unused
        self.total_output_count: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_stratum(
        self,
        stratum: StratumState,
        new_local_indices: np.ndarray,
        new_losses: np.ndarray,
        new_q_values: np.ndarray,
    ) -> float:
        """
        Update stratum state with newly acquired labels.

        Implements Paper v0320 Eq. (1):
          R_hat_k = surrogate_mean + sum_past_residuals/N_k + (L_t - s_t)/(N_k * q_t)

        For batches of n_new > 1 items, processes sequentially: each item's
        residual is added to sum_past_residuals after its R_hat_k is computed,
        and only the LAST item's R_hat_k is retained. This is consistent with
        the proof that conditional unbiasedness holds only at m = M_k^t.

        Parameters
        ----------
        stratum : StratumState
            Stratum to update (mutated in place).
        new_local_indices : np.ndarray
            Local 0-indexed positions within stratum.pool_indices.
        new_losses : np.ndarray
            Ground-truth loss values.
        new_q_values : np.ndarray
            Sampling probabilities q_k(m) for each item.

        Returns
        -------
        float
            Updated R_hat_k. [Paper v0320 Eq. (1)]

        Source: Paper v0320 Eq. (1); spec/equations.md Eq. (1).
        """
        n_new = len(new_local_indices)
        if n_new == 0:
            return stratum.R_hat_k

        N_k = stratum.N_k
        R_hat_k = stratum.R_hat_k

        for j in range(n_new):
            local_idx = int(new_local_indices[j])
            loss = float(new_losses[j])
            q = float(new_q_values[j])
            s = float(stratum.surrogate_scores[local_idx])

            residual = loss - s

            # v0320 Eq. (1): mixture + single-item correction
            # [Paper v0320 Eq. (1); spec/equations.md Eq. (1')]
            R_hat_k = (
                stratum.surrogate_mean
                + stratum.sum_past_residuals / N_k
                + residual / (N_k * q)
            )

            # Update running sum for NEXT round [Eq. (1') incremental]
            stratum.sum_past_residuals += residual

            # Record history
            stratum.label_order.append(local_idx)
            stratum.losses.append(loss)
            stratum.q_values.append(q)
            stratum.labeled_mask[local_idx] = True
            stratum.M_k += 1

        stratum.R_hat_k = R_hat_k
        return stratum.R_hat_k

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self, strata: List[StratumState]) -> float:
        """
        Aggregate per-stratum estimates into a global R_hat.

        R_hat = Σ_k w_k * R_hat_k   [Draft Eq. (1), aggregation]

        No clipping — the EValueCS rescaling handles boundedness.
        R_hat may exceed [0, 1] due to AIPW correction; this is expected.
        [Draft §3.2: rescale-then-map-back preserves unbiasedness]

        Source: Draft Eq. (1); Draft Theorem 1.
        """
        r_hat = sum(s.w_k * s.R_hat_k for s in strata)
        self.total_output_count += 1
        return float(r_hat)

    def get_plugin_state(self, strata: List[StratumState]) -> dict:
        """
        Return quantities needed for adaptive bounds (a_t, b_t) computation.

        Returns the sum of all past observed losses and the surrogate scores
        of all remaining (unlabeled) items across all strata.

        Source: Paper v0414 §3.2, Eqs. (4a)-(4b).
        """
        sum_past_losses = 0.0
        remaining_surrogates = []
        for s in strata:
            sum_past_losses += sum(s.losses)
            remaining = s.remaining_indices()
            if len(remaining) > 0:
                remaining_surrogates.append(s.surrogate_scores[remaining])
        remaining_surrogate_scores = (
            np.concatenate(remaining_surrogates)
            if remaining_surrogates
            else np.array([], dtype=np.float64)
        )
        return {
            "sum_past_losses": sum_past_losses,
            "remaining_surrogate_scores": remaining_surrogate_scores,
        }

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def weight_clip_rate(self) -> float:
        """Always 0.0 — AIPW estimator has no IS weights."""
        return 0.0

    @property
    def output_clip_rate(self) -> float:
        """Always 0.0 — no clipping applied (rescaling handles boundedness)."""
        return 0.0
