"""
Empirical Bernstein confidence interval — Stage 6 baseline.

Implements the two-sided empirical Bernstein bound derived from Maurer & Pontil
(2009), Theorem 3 via union bound for [0,1]-bounded observations. This is a
fixed-sample CI applied at each round — NOT anytime-valid under optional stopping.

Two-sided derivation: Theorem 3 gives one-sided error delta with ln(3/delta).
For two-sided error alpha, set delta = alpha/2, giving ln(3/(alpha/2)) = ln(6/alpha).

Source: Maurer & Pontil (2009), Theorem 3; union bound for two-sided [DERIVED -- verify].
Secondary ref: Audibert, Munos, & Szepesvári (2009) use similar Bernstein constants.
Spec §4.3 B2; [CerEval25].
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

import math

from save.inference import BaseCS


class EmpiricalBernsteinCS(BaseCS):
    """
    Two-sided Empirical Bernstein CI for bounded [0, 1] observations.

    At round t >= 2, with sample mean X_bar_t and sample variance V_t:

      log_term = ln(6 / alpha)
      half_width = sqrt(2 * V_t * log_term / (t - 1)) + 3 * log_term / (t - 1)
      lower = max(0, X_bar_t - half_width)
      upper = min(1, X_bar_t + half_width)

    At t = 1: returns trivial interval [0, 1].

    Source: Maurer & Pontil (2009), Theorem 3 (one-sided);
            two-sided via union bound with delta = alpha/2 [DERIVED -- verify].

    Parameters
    ----------
    alpha : float
        Two-sided error probability for the CI.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.t = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, R_hat_t: float) -> tuple[float, float, float]:
        """
        Update with new observation and return (lower, upper, width).

        Source: Maurer & Pontil (2009), Theorem 3 + union bound
                (Eqs. EB-1 to EB-3 in plan).
        """
        self.t += 1
        t = self.t
        self._sum += R_hat_t
        self._sum_sq += R_hat_t ** 2

        if t == 1:
            return (0.0, 1.0, 1.0)

        X_bar = self._sum / t

        # V_t = (Q_t - S_t^2/t) / (t - 1)
        # [DERIVED -- verify]: standard one-pass sample variance
        V_t = max(0.0, (self._sum_sq - self._sum ** 2 / t) / (t - 1))

        # [Maurer & Pontil 2009, Theorem 3] + union bound [DERIVED -- verify]
        log_term = math.log(6.0 / self.alpha)

        # [Maurer & Pontil 2009, Theorem 3]: denominator is (t - 1)
        half_width = (
            math.sqrt(2.0 * V_t * log_term / (t - 1))
            + 3.0 * log_term / (t - 1)
        )

        lower = max(0.0, X_bar - half_width)
        upper = min(1.0, X_bar + half_width)
        width = upper - lower

        return (lower, upper, width)
