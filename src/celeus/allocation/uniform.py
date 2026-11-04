"""
UniformAllocation: allocates labels equally across strata.

For K=1, all budget B_round goes to the single stratum.
For K>1, uses floor division with remainder distributed to highest-capacity strata.
"""

from __future__ import annotations

from typing import List

import numpy as np

from celeus.core.state import StratumState
from celeus.allocation.base import AllocationStrategy


class UniformAllocation(AllocationStrategy):
    """
    Trivial allocation: distribute B_round labels uniformly across K strata.

    For K=1: n_0 = min(B_round, N_0 - M_0).
    For K>1: base = B_round // K per stratum; remainder allocated one-by-one
    to strata with the most remaining capacity (ties broken by stratum index).

    When the total remaining capacity is below B_round, the allocator does not
    over-allocate: min(base, capacity) caps each stratum.
    """

    def allocate(
        self,
        strata: List[StratumState],
        B_round: int,
    ) -> np.ndarray:
        """
        Allocate B_round labels uniformly across strata.

        Parameters
        ----------
        strata : list[StratumState]
            Current stratum states.
        B_round : int
            Total label budget for this round.

        Returns
        -------
        np.ndarray
            Shape (K,) int array of per-stratum budgets n_k.
            Invariants: sum(n_k) <= B_round, n_k <= N_k - M_k.
        """
        K = len(strata)
        # Per-stratum remaining capacity: N_k - M_k.
        capacities = np.array([s.N_k - s.M_k for s in strata], dtype=int)

        # Base allocation: floor(B_round / K) per stratum.
        base = B_round // K
        remainder = B_round % K

        # Cap the base allocation so exhausted strata are not over-allocated.
        alloc = np.minimum(base, capacities)

        # Distribute the remainder one-by-one to the strata with the most
        # remaining capacity. Ties break by stratum index, so this is deterministic.
        if remainder > 0:
            # Residual capacity after base allocation
            residual_cap = capacities - alloc  # how much more each stratum can take
            # Sort by descending residual capacity. A stable sort keeps the
            # index order for ties.
            order = np.argsort(-residual_cap, kind="stable")
            for i in range(min(remainder, int(np.sum(residual_cap > 0)))):
                k = order[i]
                if residual_cap[k] > 0:
                    alloc[k] += 1
                    residual_cap[k] -= 1

        # Validate the AL1-AL3 invariants.
        self._check_invariants(alloc, strata, B_round)

        return alloc
