"""
CELEUS e-value confidence sequence with adaptive local scaling.

Supports two modes:
  - adaptive_bounds=True (the mode used for the paper results):
    per-round scaling bounds (a_t, b_t), grid on the original scale [L, U],
    candidate-specific variance σ̂²(υ), time-varying caps on the betting
    fractions.

  - adaptive_bounds=False:
    fixed bounds (a, b) derived from q_min, grid on [0, 1], one shared
    variance estimate with running-mean centering, affine map-back.

Both modes build the hedged capital process of Eq. (3), invert it as in
Eq. (4), and use the betting fractions of Remark 3. Remark 4 states the
predictability and boundedness conditions the scaling bounds must satisfy.
The construction follows Waudby-Smith and Ramdas (2024).
"""

from __future__ import annotations

import math

import numpy as np


class EValueCS:
    """
    E-value confidence sequence for CELEUS.

    Supports two modes:
    - adaptive_bounds=True: per-round (a_t, b_t), grid on the original scale,
      per-grid-point variance.
    - adaptive_bounds=False: fixed (a, b), grid on [0,1], one shared variance
      with running-mean centering.
    """

    def __init__(
        self,
        alpha_1: float,
        grid_size: int,
        c: float,
        theta: float,
        a: float = 0.0,
        b: float = 1.0,
        fixed_horizon: bool = False,
        T_max: int = 5000,
        c_fixed: float = 0.5,
        adaptive_bounds: bool = False,
    ) -> None:
        self.alpha_1 = alpha_1
        self.grid_size = grid_size
        self.c = c
        self.theta = theta
        self.fixed_horizon = fixed_horizon
        self.T_max = T_max
        self.c_fixed = c_fixed
        self.adaptive_bounds = adaptive_bounds

        if b <= a:
            raise ValueError(f"EValueCS: b={b} must be > a={a}")
        self.a = a
        self.b = b

        if adaptive_bounds:
            # Grid on the original scale [a, b].
            self.grid = np.linspace(
                a + (b - a) / (grid_size + 1),
                b - (b - a) / (grid_size + 1),
                grid_size,
            )
            # Per-grid-point variance accumulator (Remark 3).
            self.sum_sq_centered = np.zeros(grid_size)
        else:
            # Grid on [0, 1]; results are mapped back at the end of update.
            self.grid = np.linspace(
                1 / (grid_size + 1), grid_size / (grid_size + 1), grid_size
            )
            self.scale = b - a
            # One shared scalar variance for all grid points (Remark 3).
            self.sum_R = 0.0
            self.sum_sq_dev = 0.0

        # Log-wealth accumulators, shape (G,)
        self.log_wealth_upper = np.zeros(grid_size)
        self.log_wealth_lower = np.zeros(grid_size)
        self.ever_rejected = np.zeros(grid_size, dtype=bool)

        self.t = 0
        self.log_threshold = np.log(1.0 / alpha_1)
        self.log_theta = np.log(theta)
        self.log_one_minus_theta = np.log(1.0 - theta)

    def update(
        self, R_hat_t: float, a_t: float | None = None, b_t: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Update CS with new estimate. Returns (lower, upper, width) on original scale.

        When adaptive_bounds=True, a_t and b_t are required (per-round bounds).
        When adaptive_bounds=False, a_t/b_t are ignored (fixed bounds from __init__).
        """
        if self.adaptive_bounds:
            return self._update_adaptive(R_hat_t, a_t, b_t)
        else:
            return self._update_fixed(R_hat_t)

    def _update_adaptive(
        self, R_hat_t: float, a_t: float, b_t: float,
    ) -> tuple[float, float, float]:
        """Update with per-round adaptive scaling bounds (a_t, b_t)."""
        if a_t is None or b_t is None:
            raise ValueError("adaptive_bounds=True requires a_t and b_t")

        self.t += 1
        t = self.t
        scale_t = b_t - a_t
        if scale_t <= 0:
            raise ValueError(f"EValueCS: b_t={b_t} must be > a_t={a_t}")

        # Step 1: Scale the signal to [0, 1] with the round-t bounds.
        R_bar_t = (R_hat_t - a_t) / scale_t

        # Step 2: Scale every candidate with the same round-t bounds (Remark 3).
        v_bar_t = (self.grid - a_t) / scale_t

        # Step 3: Candidates outside (a_t, b_t) are rejected immediately.
        self.ever_rejected |= (v_bar_t <= 0.0) | (v_bar_t >= 1.0)

        # Step 4: Candidate-specific variance estimate σ̂²_{t-1}(υ) (Remark 3).
        sigma_sq = (0.25 + self.sum_sq_centered) / t
        sigma_sq = np.maximum(sigma_sq, 1e-6)

        # Step 5: Betting fraction before capping (Remark 3).
        if self.fixed_horizon:
            ons_bet = np.sqrt(
                2.0 * np.log(2.0 / self.alpha_1) / (sigma_sq * self.T_max)
            )
        else:
            ons_bet = np.sqrt(
                2.0 * np.log(2.0 / self.alpha_1) / (sigma_sq * t * np.log(t + 1))
            )

        # Step 6: Candidate-specific caps on the betting fraction (Remark 3).
        cap_upper = 1.0 / (np.maximum(v_bar_t, 1e-10) + self.c)
        cap_lower = 1.0 / (np.maximum(1.0 - v_bar_t, 1e-10) + self.c)

        lambda_upper = np.minimum(ons_bet, cap_upper)
        lambda_lower = np.minimum(ons_bet, cap_lower)

        # Step 7: Per-round capital factors of Eq. (3).
        diff = R_bar_t - v_bar_t

        factor_upper = 1.0 + lambda_upper * diff
        factor_lower = 1.0 - lambda_lower * diff

        log_factor_upper = np.where(factor_upper > 0, np.log(factor_upper), -np.inf)
        log_factor_lower = np.where(factor_lower > 0, np.log(factor_lower), -np.inf)

        # Step 8: Accumulate log-wealth
        self.log_wealth_upper += log_factor_upper
        self.log_wealth_lower += log_factor_lower

        # Step 9: Hedged log-wealth of Eq. (3).
        log_hedged = np.logaddexp(
            self.log_theta + self.log_wealth_upper,
            self.log_one_minus_theta + self.log_wealth_lower,
        )

        # Step 10: Running intersection of Eq. (4).
        self.ever_rejected |= (log_hedged >= self.log_threshold)

        # Step 11: Extract CI bounds on ORIGINAL scale (no map-back)
        active = ~self.ever_rejected
        if active.any():
            active_values = self.grid[active]
            lower = float(active_values[0])
            upper = float(active_values[-1])
        else:
            import warnings

            warnings.warn(
                f"EValueCS: all {self.grid_size} hypotheses rejected at t={t}. "
                "Returning full-range fallback.",
                stacklevel=2,
            )
            lower = float(self.a)
            upper = float(self.b)

        width = upper - lower

        # Step 12: Update the variance accumulator for the next round (Remark 3).
        self.sum_sq_centered += (R_bar_t - v_bar_t) ** 2

        return (lower, upper, width)

    def _update_fixed(self, R_hat_t: float) -> tuple[float, float, float]:
        """Update with fixed scaling bounds (a, b)."""
        # Rescale the signal to [0, 1].
        R_bar_t = (R_hat_t - self.a) / self.scale

        self.t += 1
        t = self.t

        # One shared variance estimate for all grid points (Remark 3).
        sigma_sq = (0.25 + self.sum_sq_dev) / t
        sigma_sq = max(sigma_sq, 1e-6)

        # ONS betting
        if self.fixed_horizon:
            ons_bet = np.sqrt(
                2.0 * np.log(2.0 / self.alpha_1) / (sigma_sq * self.T_max)
            )
        else:
            ons_bet = np.sqrt(
                2.0 * np.log(2.0 / self.alpha_1) / (sigma_sq * t * np.log(t + 1))
            )

        # Caps on the betting fraction (Remark 3).
        if self.fixed_horizon:
            MAX_CAP = 1e4
            cap_upper = np.minimum(self.c_fixed / self.grid, MAX_CAP)
            cap_lower = np.minimum(self.c_fixed / (1.0 - self.grid), MAX_CAP)
        else:
            cap_upper = 1.0 / (self.grid + self.c)
            cap_lower = 1.0 / (1.0 - self.grid + self.c)

        lambda_upper = np.minimum(ons_bet, cap_upper)
        lambda_lower = np.minimum(ons_bet, cap_lower)

        diff = R_bar_t - self.grid

        factor_upper = 1.0 + lambda_upper * diff
        factor_lower = 1.0 - lambda_lower * diff

        log_factor_upper = np.where(factor_upper > 0, np.log(factor_upper), -np.inf)
        log_factor_lower = np.where(factor_lower > 0, np.log(factor_lower), -np.inf)

        self.log_wealth_upper += log_factor_upper
        self.log_wealth_lower += log_factor_lower

        log_hedged = np.logaddexp(
            self.log_theta + self.log_wealth_upper,
            self.log_one_minus_theta + self.log_wealth_lower,
        )

        self.ever_rejected |= (log_hedged >= self.log_threshold)

        active = ~self.ever_rejected
        if active.any():
            active_values = self.grid[active]
            lower_bar = float(active_values[0])
            upper_bar = float(active_values[-1])
        else:
            import warnings

            warnings.warn(
                f"EValueCS: all {self.grid_size} hypotheses rejected at t={t}. "
                "Returning full-range fallback.",
                stacklevel=2,
            )
            lower_bar = 0.0
            upper_bar = 1.0

        # Map the grid bounds back to the original scale.
        lower = self.scale * lower_bar + self.a
        upper = self.scale * upper_bar + self.a
        width = upper - lower

        # Update the variance estimate for the next round.
        self.sum_R += R_bar_t
        R_mean_t = (0.5 + self.sum_R) / (t + 1)
        self.sum_sq_dev += (R_bar_t - R_mean_t) ** 2

        return (lower, upper, width)

    def population_correction(
        self,
        lower: float,
        upper: float,
        N: int,
        alpha_2: float,
        loss_bound: float = 1.0,
        loss_lower: float = 0.0,
    ) -> tuple[float, float, float]:
        """Widen the interval by the Hoeffding finite-pool-to-population term."""
        loss_range = loss_bound - loss_lower
        delta_N = loss_range * math.sqrt(math.log(2.0 / alpha_2) / (2.0 * N))
        pop_lower = max(loss_lower, lower - delta_N)
        pop_upper = min(loss_bound, upper + delta_N)
        return (pop_lower, pop_upper, pop_upper - pop_lower)


def pool_feasibility_intersection(
    lower: float,
    upper: float,
    sum_queried_losses: float,
    t: int,
    N: int,
    loss_lower: float = 0.0,
    loss_upper: float = 1.0,
) -> tuple[float, float, float]:
    """
    Intersect CS bounds with deterministic pool feasibility interval.

    I_t^pool = [(Σ ℓ_m + (N−t)·L) / N,  (Σ ℓ_m + (N−t)·U) / N]

    Coverage-preserving: R_N ∈ I_t^pool deterministically.
    """
    feasible_lower = (sum_queried_losses + (N - t) * loss_lower) / N
    feasible_upper = (sum_queried_losses + (N - t) * loss_upper) / N

    lower_out = max(lower, feasible_lower)
    upper_out = min(upper, feasible_upper)

    if lower_out > upper_out:  # numerical guard
        mid = (lower_out + upper_out) / 2.0
        lower_out = mid
        upper_out = mid

    return (lower_out, upper_out, upper_out - lower_out)
