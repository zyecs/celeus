"""
UniformAllocation: allocates labels equally across strata.

For K=1, all budget B_round goes to the single stratum.
For K>1, uses floor division with remainder distributed to highest-capacity strata.

Source: plan-final.md §3.3; Blueprint §1.3 (UniformAllocation pseudo-code).
Spec §6.3 AL1-AL2 (invariants).
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

from typing import List

import numpy as np

from save.core.state import StratumState
from save.allocation.base import AllocationStrategy


class UniformAllocation(AllocationStrategy):
    """
    Trivial allocation: distribute B_round labels uniformly across K strata.

    For K=1: n_0 = min(B_round, N_0 - M_0).
    For K>1: base = B_round // K per stratum; remainder allocated one-by-one
    to strata with the most remaining capacity (ties broken by stratum index).

    Design decision [NON-BLOCKING]: when total remaining capacity < B_round
    (pool nearly exhausted), does not over-allocate. The inner min(base, capacity)
    handles this correctly. [plan-final.md §3.3]

    Source: plan-final.md §3.3; Blueprint §1.3 (UniformAllocation pseudo-code).
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
            Total label budget for this round. [Spec §4.7 Algorithm 1]

        Returns
        -------
        np.ndarray
            Shape (K,) int array of per-stratum budgets n_k.
            Invariants: sum(n_k) <= B_round, n_k <= N_k - M_k.
            [Spec §6.3 AL1-AL2; plan-final.md §3.3]

        Source: plan-final.md §3.3; Blueprint §1.3.
        """
        K = len(strata)
        # Per-stratum remaining capacity: N_k - M_k [Spec §2.1 Eq. (1)]
        capacities = np.array([s.N_k - s.M_k for s in strata], dtype=int)

        # Base allocation: floor(B_round / K) per stratum [Blueprint §1.3]
        base = B_round // K
        remainder = B_round % K

        # Allocate base, capped by capacity [plan-final.md §3.3 design decision]
        # [NON-BLOCKING]: min(base, capacity) prevents over-allocation for exhausted strata
        alloc = np.minimum(base, capacities)

        # Distribute remainder one-by-one to strata with most remaining capacity
        # (after base allocation). Ties broken by stratum index (deterministic).
        # [Blueprint §1.3; plan-final.md §3.3]
        if remainder > 0:
            # Residual capacity after base allocation
            residual_cap = capacities - alloc  # how much more each stratum can take
            # Sort strata by descending residual capacity, then by index for tie-breaking
            # [DERIVED — verify]: argsort with stable sort preserves index ordering for ties
            order = np.argsort(-residual_cap, kind="stable")
            for i in range(min(remainder, int(np.sum(residual_cap > 0)))):
                k = order[i]
                if residual_cap[k] > 0:
                    alloc[k] += 1
                    residual_cap[k] -= 1

        # Validate invariants [Spec §6.3 AL1-AL2; plan-final.md §3.2]
        self._check_invariants(alloc, strata, B_round)

        return alloc
