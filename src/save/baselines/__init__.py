"""
Baseline confidence interval utilities for SAVE experiments.

Implements the fixed-sample Hoeffding CI width baseline:

  width_H(t, alpha) = min(1.0, 2 * sqrt(log(2 / alpha) / (2 * t)))

Source: Hoeffding (1963), "Probability inequalities for sums of bounded
random variables"; Spec §4.3 baseline B0.
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

import math


def hoeffding_ci_width(t: int, alpha: float) -> float:
    """
    Return the width of the fixed-sample Hoeffding CI at label count ``t``.

    For [0, 1]-valued observations, the two-sided Hoeffding width is

      min(1.0, 2 * sqrt(log(2 / alpha) / (2 * t)))

    Source: Hoeffding (1963); Spec §4.3 baseline B0; Stage 3 review items R2/R4.
    """
    if t < 1:
        return float("inf")

    width = 2.0 * math.sqrt(math.log(2.0 / alpha) / (2.0 * t))
    return min(1.0, width)


from save.baselines.evalue import NaiveEValueBaseline  # noqa: E402, F401
from save.baselines.cereval import CerEvalBaseline  # noqa: E402, F401
