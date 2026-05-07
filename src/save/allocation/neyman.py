"""
NeymanAllocation: allocates labels via Neyman allocation.

n_k proportional to N_k * sigma_hat_k   [Spec §2.4 Eq. (13)]

Fallback: proportional when min(M_k) < 2 or sum(N_k * sigma_hat_k) == 0.
Uses _proportional_round for integer rounding with capacity clamping.
Calls _check_invariants for AL1-AL3. [Spec §6.3]
CLAUDE.md G1: every formula cites source.

v0320: uses raw (unweighted) residuals (L_m - s_m) — no IS weights.
"""

from __future__ import annotations

from typing import List

import numpy as np

from save.allocation.base import AllocationStrategy, _proportional_round
from save.core.state import StratumState


def _compute_sigma_hat_k(stratum: StratumState) -> float:
    """Compute sigma_hat_k from raw residuals. [Spec §2.4 Eq. (13)]

    Under v0320, the estimator uses raw (unweighted) residuals (L_m - s_m).
    Neyman allocation variance estimate follows suit.

    Index mapping (StratumState docstring, state.py):
      losses[j] is in query order.
      surrogate_scores is indexed by local position.
      label_order[j] maps query index j -> local position.

    Returns 0.0 if M_k < 2 (insufficient data for ddof=1).
    """
    M_k = stratum.M_k
    if M_k < 2:
        return 0.0  # insufficient data for std with ddof=1

    # Raw residuals: (L_m - s_m) [Paper v0320 Eq. (1)]
    raw_residuals = np.array(
        [
            stratum.losses[j]
            - stratum.surrogate_scores[stratum.label_order[j]]
            for j in range(M_k)
        ]
    )
    return float(np.std(raw_residuals, ddof=1))


class NeymanAllocation(AllocationStrategy):
    """
    Neyman allocation: n_k proportional to N_k * sigma_hat_k.

    Source: Spec §2.4 Eq. (13).

    Fallback to proportional when:
      - min(M_k) < 2: insufficient data for variance estimation
      - sum(N_k * sigma_hat_k) == 0: degenerate case (all zero variance)

    [POTENTIAL ISSUE]: Spec §2.4 says "Falls back to proportional when M_k < 2"
    which could be per-stratum. This uses GLOBAL fallback (all proportional when
    ANY M_k < 2) to avoid mixing incomparable weight scales.
    """

    def allocate(
        self,
        strata: List[StratumState],
        B_round: int,
    ) -> np.ndarray:
        """
        Allocate B_round labels via Neyman allocation.

        n_k proportional to N_k * sigma_hat_k   [Spec §2.4 Eq. (13)]
        """
        capacities = np.array([s.N_k - s.M_k for s in strata], dtype=int)

        # Check warm-up: all strata need M_k >= 2 for variance estimation
        # [Spec §2.4 Eq. (13): fallback when M_k < 2]
        min_M_k = min(s.M_k for s in strata)
        if min_M_k < 2:
            # Proportional fallback during warm-up
            raw_weights = np.array([float(s.N_k) for s in strata])  # [Spec §2.4 fallback]
            alloc = _proportional_round(raw_weights, B_round, capacities)
            self._check_invariants(alloc, strata, B_round)
            return alloc

        # Compute sigma_hat_k for each stratum [Spec §2.4 Eq. (13)]
        sigma_hats = np.array([_compute_sigma_hat_k(s) for s in strata])
        N_ks = np.array([float(s.N_k) for s in strata])

        # Neyman weights: N_k * sigma_hat_k  [Spec §2.4 Eq. (13)]
        neyman_weights = N_ks * sigma_hats

        # Guard: if all Neyman weights are zero, fall back to proportional
        if np.sum(neyman_weights) == 0.0:
            raw_weights = N_ks  # [Spec §2.4 fallback]
            alloc = _proportional_round(raw_weights, B_round, capacities)
            self._check_invariants(alloc, strata, B_round)
            return alloc

        alloc = _proportional_round(neyman_weights, B_round, capacities)
        self._check_invariants(alloc, strata, B_round)
        return alloc
