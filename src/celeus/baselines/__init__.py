"""
Baseline confidence interval utilities for CELEUS experiments.

Implements the fixed-sample Hoeffding CI width baseline:

  width_H(t, alpha) = min(1.0, 2 * sqrt(log(2 / alpha) / (2 * t)))

Source: Hoeffding (1963), "Probability inequalities for sums of bounded
random variables".
"""

from __future__ import annotations

import math


def hoeffding_ci_width(t: int, alpha: float) -> float:
    """
    Return the width of the fixed-sample Hoeffding CI at label count ``t``.

    For [0, 1]-valued observations, the two-sided Hoeffding width is

      min(1.0, 2 * sqrt(log(2 / alpha) / (2 * t)))

    Source: Hoeffding (1963).
    """
    if t < 1:
        return float("inf")

    width = 2.0 * math.sqrt(math.log(2.0 / alpha) / (2.0 * t))
    return min(1.0, width)


from celeus.baselines.evalue import NaiveEValueBaseline  # noqa: E402, F401
from celeus.baselines.cereval import CerEvalBaseline  # noqa: E402, F401
