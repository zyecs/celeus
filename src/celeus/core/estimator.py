"""
CELEUS AIPW estimator.

Computes the per-stratum inferential signal of Eq. (2):
  R_hat_k = surrogate_mean + sum_past_residuals/N_k + (L_t - s_t)/(N_k * q_t)

No clipping is applied. The rescale-then-map-back step in EValueCS keeps the
signal inside the betting range without breaking the conditional unbiasedness
of Thm. 3.
"""

from __future__ import annotations

from typing import List

import numpy as np

from celeus.core.state import StratumState


class AIPWEstimator:
    """
    CELEUS AIPW risk estimator.

    Computes the per-stratum signal of Eq. (2): a mixture of true losses for
    past labeled items and surrogate scores for unlabeled items, plus a
    single-item correction for the most recently drawn item.

    No clipping is applied; the EValueCS rescaling keeps the signal bounded
    without breaking conditional unbiasedness.

    Parameters
    ----------
    u_max : float
        Kept for API compatibility. Not used.
    """

    def __init__(self, u_max: float = 20.0) -> None:
        self.u_max = u_max  # retained for API compat; unused
        self.total_output_count: int = 0

    def update_stratum(
        self,
        stratum: StratumState,
        new_local_indices: np.ndarray,
        new_losses: np.ndarray,
        new_q_values: np.ndarray,
    ) -> float:
        """
        Update stratum state with newly acquired labels.

        Implements Eq. (2) restricted to one stratum:
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
            Updated R_hat_k.
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

            # Mixture plus single-item correction.
            R_hat_k = (
                stratum.surrogate_mean
                + stratum.sum_past_residuals / N_k
                + residual / (N_k * q)
            )

            # Update the running residual sum for the next round.
            stratum.sum_past_residuals += residual

            # Record history
            stratum.label_order.append(local_idx)
            stratum.losses.append(loss)
            stratum.q_values.append(q)
            stratum.labeled_mask[local_idx] = True
            stratum.M_k += 1

        stratum.R_hat_k = R_hat_k
        return stratum.R_hat_k

    def aggregate(self, strata: List[StratumState]) -> float:
        """
        Aggregate per-stratum estimates into a global R_hat.

        R_hat = Σ_k w_k * R_hat_k

        No clipping — the EValueCS rescaling handles boundedness.
        R_hat may fall outside [0, 1] because of the correction term; that is
        expected. Rescale-then-map-back in EValueCS preserves unbiasedness.
        """
        r_hat = sum(s.w_k * s.R_hat_k for s in strata)
        self.total_output_count += 1
        return float(r_hat)

    def get_plugin_state(self, strata: List[StratumState]) -> dict:
        """
        Return quantities needed for adaptive bounds (a_t, b_t) computation.

        Returns the sum of all past observed losses and the surrogate scores
        of all remaining (unlabeled) items across all strata.
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

    @property
    def weight_clip_rate(self) -> float:
        """Always 0.0 — AIPW estimator has no IS weights."""
        return 0.0

    @property
    def output_clip_rate(self) -> float:
        """Always 0.0 — no clipping applied (rescaling handles boundedness)."""
        return 0.0
