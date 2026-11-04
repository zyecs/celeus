"""
AllocationStrategy abstract base class for CELEUS.

Defines the interface for per-stratum budget allocation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from celeus.core.state import StratumState


def _proportional_round(
    raw_weights: np.ndarray,
    B_round: int,
    capacities: np.ndarray,
) -> np.ndarray:
    """Floor + largest-remainder rounding with capacity clamping.

    Parameters: raw_weights (K,) float, B_round int, capacities (K,) int.
    Returns: (K,) int array with sum <= B_round, each <= capacity.
    Deterministic tie-breaking via stable argsort.
    """
    K = len(raw_weights)

    # Guard 1: trivial cases — return zeros immediately
    if B_round == 0 or np.sum(capacities) == 0 or np.sum(raw_weights) == 0:
        return np.zeros(K, dtype=int)

    # Guard 2: effective budget cannot exceed total remaining capacity
    B_eff = min(B_round, int(np.sum(capacities)))

    total_w = np.sum(raw_weights)
    proportions = raw_weights / total_w * B_eff

    alloc = np.minimum(np.floor(proportions).astype(int), capacities)

    # Distribute remainder by largest fractional part
    remainder = B_eff - np.sum(alloc)
    fractional = proportions - alloc

    # Zero out fractional for capacity-full strata
    fractional[alloc >= capacities] = -1.0
    order = np.argsort(-fractional, kind="stable")

    for i in range(int(remainder)):
        k = order[i]
        if alloc[k] < capacities[k]:
            alloc[k] += 1

    return alloc


class AllocationStrategy(ABC):
    """
    Abstract base class for allocation strategies.

    Subclasses implement allocate() to distribute B_round labels across K strata.

    Invariants enforced:
      AL1: sum(n_k) <= B_round  (total budget consumed — may be less if pool exhausted)
      AL2: n_k <= N_k - M_k for all k  (cannot over-allocate beyond capacity)
      AL3: n_k >= 0 for all k
    """

    @abstractmethod
    def allocate(
        self,
        strata: List[StratumState],
        B_round: int,
    ) -> np.ndarray:
        """
        Allocate B_round labels across strata.

        Parameters
        ----------
        strata : list[StratumState]
            Current stratum states. len(strata) == K.
        B_round : int
            Total label budget for this round.

        Returns
        -------
        np.ndarray
            Shape (K,) int array of per-stratum budgets n_k.
            Invariants: n_k >= 0, n_k <= N_k - M_k, sum(n_k) <= B_round.
        """
        ...

    def _check_invariants(
        self,
        alloc: np.ndarray,
        strata: List[StratumState],
        B_round: int,
    ) -> None:
        """
        Verify the AL1-AL3 invariants; raise ValueError on violation.
        """
        capacities = np.array([s.N_k - s.M_k for s in strata], dtype=int)

        # AL3: no negative allocations
        if np.any(alloc < 0):
            raise ValueError(
                f"AllocationStrategy invariant AL3 violated: negative n_k in {alloc}"
            )

        # AL2: cannot exceed capacity
        if np.any(alloc > capacities):
            raise ValueError(
                f"AllocationStrategy invariant AL2 violated: "
                f"alloc={alloc} exceeds capacities={capacities}"
            )

        # AL1: total does not exceed B_round (may be less if pool is exhausted)
        if int(np.sum(alloc)) > B_round:
            raise ValueError(
                f"AllocationStrategy invariant AL1 violated: "
                f"sum(alloc)={int(np.sum(alloc))} > B_round={B_round}"
            )
