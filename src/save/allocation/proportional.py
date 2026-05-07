"""
ProportionalAllocation: allocates labels proportional to stratum size.

n_k proportional to N_k.  [Spec §2.4 fallback; Spec §5 A1]

Uses _proportional_round for integer rounding with capacity clamping.
Calls _check_invariants for AL1-AL3. [Spec §6.3]
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

from typing import List

import numpy as np

from save.allocation.base import AllocationStrategy, _proportional_round
from save.core.state import StratumState


class ProportionalAllocation(AllocationStrategy):
    """
    Proportional allocation: n_k proportional to N_k.

    Source: Spec §2.4 (fallback); Spec §5 A1.
    """

    def allocate(
        self,
        strata: List[StratumState],
        B_round: int,
    ) -> np.ndarray:
        """
        Allocate B_round labels proportional to stratum sizes N_k.

        [Spec §2.4 fallback; Spec §5 A1]
        """
        capacities = np.array([s.N_k - s.M_k for s in strata], dtype=int)
        raw_weights = np.array([float(s.N_k) for s in strata])  # [Spec §5 A1]
        alloc = _proportional_round(raw_weights, B_round, capacities)
        self._check_invariants(alloc, strata, B_round)
        return alloc
