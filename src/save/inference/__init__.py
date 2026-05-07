"""
SAVE baseline inference method interfaces — Stage 6.

BaseCS: abstract base class for confidence sequence / interval methods.
population_correction: finite-population correction (Spec §2.3 Eqs. 10-11).

Source: Blueprint §3.3; Spec §A5; Spec §2.3 Eqs. (10)-(11).
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from save.core.confidence import pool_feasibility_intersection


class BaseCS(ABC):
    """
    Abstract base class for confidence sequence / interval methods.

    All inference methods implement update() which processes a new risk estimate
    and returns (lower, upper, width).

    Source: Blueprint §3.3 interface specification.
    """

    @abstractmethod
    def update(self, R_hat_t: float) -> tuple[float, float, float]:
        """
        Process new risk estimate, return (lower, upper, width).

        Parameters
        ----------
        R_hat_t : float
            Current aggregate risk estimate, clipped to [0, 1].

        Returns
        -------
        tuple[float, float, float]
            (lower, upper, width) of the CI at the current round.

        Source: Blueprint §3.3 BaseCS interface.
        """
        ...


def population_correction(
    lower: float,
    upper: float,
    N: int,
    alpha_2: float,
    loss_bound: float = 1.0,
    loss_lower: float = 0.0,
) -> tuple[float, float, float]:
    """
    Finite-population correction via Hoeffding ball.

    delta_N = (U-L) * sqrt(ln(2 / alpha_2) / (2 * N))   [Draft page 7]
    pop_lower = max(L, lower - delta_N)                 [Draft page 7]
    pop_upper = min(U, upper + delta_N)                 [Draft page 7]

    This is the same formula as EValueCS.population_correction(),
    extracted here as a standalone function for reuse by all baseline methods.

    Source: Draft page 7; Hoeffding 1963.
    """
    loss_range = loss_bound - loss_lower
    delta_N = loss_range * math.sqrt(math.log(2.0 / alpha_2) / (2.0 * N))
    pop_lower = max(loss_lower, lower - delta_N)
    pop_upper = min(loss_bound, upper + delta_N)
    return (pop_lower, pop_upper, pop_upper - pop_lower)
