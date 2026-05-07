"""
Hoeffding confidence interval — Stage 6 baseline.

Wraps the existing hoeffding_ci_width() function from save.baselines into the
BaseCS interface. This is a fixed-sample CI applied at each round.

For bounded [0,1] observations:
  half_width = sqrt(ln(2/alpha) / (2*t))
  width = min(1.0, 2 * half_width)

NOT anytime-valid. Any observed over-coverage under data-dependent stopping is
due to the bound's conservatism (excessive width), not a theoretical property.

Source: Hoeffding (1963); Spec §4.3 B0; existing baselines.py.
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

import math

from save.inference import BaseCS


class HoeffdingCS(BaseCS):
    """
    Hoeffding CI for bounded [0, 1] observations.

    At round t:
      half_width = sqrt(ln(2/alpha) / (2*t))   [Hoeffding 1963; Eq. H-2]
      lower = max(0, X_bar_t - half_width)      [Eq. H-3]
      upper = min(1, X_bar_t + half_width)      [Eq. H-3]

    Source: Hoeffding (1963); Spec §4.3 B0.

    Parameters
    ----------
    alpha : float
        Error probability for the CI.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.t = 0
        self._sum = 0.0

    def update(self, R_hat_t: float) -> tuple[float, float, float]:
        """
        Update with new observation and return (lower, upper, width).

        Source: Hoeffding (1963); Eq. H-2 to H-4 in plan.
        """
        self.t += 1
        t = self.t
        self._sum += R_hat_t

        X_bar = self._sum / t

        # Half-width from Hoeffding bound [Hoeffding 1963; Eq. H-2]
        half_width = math.sqrt(math.log(2.0 / self.alpha) / (2.0 * t))

        lower = max(0.0, X_bar - half_width)
        upper = min(1.0, X_bar + half_width)
        width = upper - lower

        return (lower, upper, width)
