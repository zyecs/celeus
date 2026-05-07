"""
Bootstrap confidence interval — Stage 6 baseline.

Implements nonparametric percentile bootstrap CI for the mean of [0,1]-bounded
observations. NOT anytime-valid — expected to under-cover under data-dependent
stopping.

Source: Efron (1979); [F21] Active Testing; Spec §4.3 B1.
CLAUDE.md G1: every formula cites source.
CLAUDE.md Rule 4: all randomness via np.random.Generator.
"""

from __future__ import annotations

import numpy as np

from save.inference import BaseCS


class BootstrapCI(BaseCS):
    """
    Nonparametric percentile bootstrap CI.

    At round t, resamples X_1, ..., X_t with replacement n_bootstrap times,
    computes bootstrap means, and returns percentile CI.

      lower = quantile(alpha/2, bootstrap_means)   [Eq. Boot-1]
      upper = quantile(1-alpha/2, bootstrap_means) [Eq. Boot-2]

    Source: Efron (1979); [F21] Active Testing; Spec §4.3 B1.

    Parameters
    ----------
    alpha : float
        Error probability for the CI.
    n_bootstrap : int
        Number of bootstrap resamples. Default 1000.
    seed : int
        Seed for bootstrap resampling RNG. Separate from the main sampling RNG
        to avoid correlation. [CLAUDE.md Rule 4]
    """

    def __init__(self, alpha: float, n_bootstrap: int = 1000, seed: int = 0) -> None:
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(seed)
        self._observations: list[float] = []

    def update(self, R_hat_t: float) -> tuple[float, float, float]:
        """
        Update with new observation and return (lower, upper, width).

        Source: Efron (1979); Eqs. Boot-1, Boot-2 in plan.
        CLAUDE.md Rule 4: bootstrap RNG is self._rng (seeded Generator).
        """
        self._observations.append(R_hat_t)
        t = len(self._observations)

        if t == 1:
            return (max(0.0, R_hat_t), min(1.0, R_hat_t), 0.0)

        X = np.array(self._observations, dtype=np.float64)

        # [DERIVED -- verify]: standard vectorized bootstrap
        indices = self._rng.choice(t, size=(self.n_bootstrap, t), replace=True)
        boot_means = X[indices].mean(axis=1)

        lower = float(np.percentile(boot_means, 100.0 * self.alpha / 2.0))
        upper = float(np.percentile(boot_means, 100.0 * (1.0 - self.alpha / 2.0)))

        lower = max(0.0, lower)
        upper = min(1.0, upper)
        width = upper - lower

        return (lower, upper, width)
