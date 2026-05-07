"""
SAVE e-value confidence sequence with adaptive local scaling.

Supports two modes:
  - SAVE-Ada (v0414, adaptive_bounds=True):
    Per-round bounds (a_t, b_t), grid on original scale [L, U],
    candidate-specific variance σ̂²(υ), time-varying caps v̄_t.
    [Paper v0414 §3.2, Remark 6]

  - Legacy (v0413, adaptive_bounds=False):
    Fixed bounds (a, b) from q_min, grid on [0, 1],
    shared variance with running mean centering, affine map-back.
    [Paper v0413 §3.2, Remark 6]

Source: Paper v0414 §3.2; [WSR24] Theorems 1-2; spec/equations.md §2.2.
CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

import math

import numpy as np


class EValueCS:
    """
    E-value confidence sequence for SAVE.

    Supports two modes:
    - adaptive_bounds=True (SAVE-Ada, v0414): per-round (a_t, b_t),
      grid on original scale, per-grid-point variance.
    - adaptive_bounds=False (v0413): fixed (a, b), grid on [0,1],
      shared variance with running mean centering.

    Source: Paper v0414 §3.2, Remark 6; [WSR24] Theorems 1-2.
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
            # Grid on ORIGINAL scale [a, b] [v0414 §3.2]
            self.grid = np.linspace(
                a + (b - a) / (grid_size + 1),
                b - (b - a) / (grid_size + 1),
                grid_size,
            )
            # Per-grid-point variance accumulator [v0414 Remark 6, Eq. (10c)]
            self.sum_sq_centered = np.zeros(grid_size)
        else:
            # Legacy grid on [0, 1] [v0413]
            self.grid = np.linspace(
                1 / (grid_size + 1), grid_size / (grid_size + 1), grid_size
            )
            self.scale = b - a
            # Shared scalar variance [v0413 Remark 6]
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

        Source: Paper v0414 §3.2, Remark 6.
        """
        if self.adaptive_bounds:
            return self._update_adaptive(R_hat_t, a_t, b_t)
        else:
            return self._update_fixed(R_hat_t)

    def _update_adaptive(
        self, R_hat_t: float, a_t: float, b_t: float,
    ) -> tuple[float, float, float]:
        """Per-round adaptive bounds (v0414). [Paper v0414 §3.2, Remark 6]"""
        if a_t is None or b_t is None:
            raise ValueError("adaptive_bounds=True requires a_t and b_t")

        self.t += 1
        t = self.t
        scale_t = b_t - a_t
        if scale_t <= 0:
            raise ValueError(f"EValueCS: b_t={b_t} must be > a_t={a_t}")

        # Step 1: Local scaling [v0414 §3.2 Eq. (4c)]
        R_bar_t = (R_hat_t - a_t) / scale_t

        # Step 2: Time-varying candidate image [v0414 Eq. (4d)]
        v_bar_t = (self.grid - a_t) / scale_t

        # Step 3: Immediate rejection for points outside (a_t, b_t) [v0414 L-322]
        self.ever_rejected |= (v_bar_t <= 0.0) | (v_bar_t >= 1.0)

        # Step 4: Candidate-specific variance σ̂²_{t-1}(υ) [v0414 Remark 6 Eq. (10c)]
        sigma_sq = (0.25 + self.sum_sq_centered) / t
        sigma_sq = np.maximum(sigma_sq, 1e-6)

        # Step 5: ONS betting term [v0414 Remark 6 Eqs. (10a)-(10b)]
        if self.fixed_horizon:
            ons_bet = np.sqrt(
                2.0 * np.log(2.0 / self.alpha_1) / (sigma_sq * self.T_max)
            )
        else:
            ons_bet = np.sqrt(
                2.0 * np.log(2.0 / self.alpha_1) / (sigma_sq * t * np.log(t + 1))
            )

        # Step 6: Adaptive caps using v̄_t [v0414 Remark 6]
        cap_upper = 1.0 / (np.maximum(v_bar_t, 1e-10) + self.c)
        cap_lower = 1.0 / (np.maximum(1.0 - v_bar_t, 1e-10) + self.c)

        lambda_upper = np.minimum(ons_bet, cap_upper)
        lambda_lower = np.minimum(ons_bet, cap_lower)

        # Step 7: Capital factors [v0414 Eq. (5)-(6)]
        diff = R_bar_t - v_bar_t

        factor_upper = 1.0 + lambda_upper * diff
        factor_lower = 1.0 - lambda_lower * diff

        log_factor_upper = np.where(factor_upper > 0, np.log(factor_upper), -np.inf)
        log_factor_lower = np.where(factor_lower > 0, np.log(factor_lower), -np.inf)

        # Step 8: Accumulate log-wealth
        self.log_wealth_upper += log_factor_upper
        self.log_wealth_lower += log_factor_lower

        # Step 9: Hedged log-wealth [v0414 Eq. (7)]
        log_hedged = np.logaddexp(
            self.log_theta + self.log_wealth_upper,
            self.log_one_minus_theta + self.log_wealth_lower,
        )

        # Step 10: Running intersection [v0414 Eq. (8)]
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

        # Step 12: Update variance accumulator for NEXT round [v0414 Remark 6 Eq. (10c)]
        self.sum_sq_centered += (R_bar_t - v_bar_t) ** 2

        return (lower, upper, width)

    def _update_fixed(self, R_hat_t: float) -> tuple[float, float, float]:
        """Fixed bounds (v0413 backward compat). [Paper v0413 §3.2, Remark 6]"""
        # Rescale [Draft v0413 §3.2]
        R_bar_t = (R_hat_t - self.a) / self.scale

        self.t += 1
        t = self.t

        # Shared variance [v0413 Remark 6]
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

        # Fixed caps [v0413 Remark 6]
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

        # Map back [v0413 page 7]
        lower = self.scale * lower_bar + self.a
        upper = self.scale * upper_bar + self.a
        width = upper - lower

        # Update variance [v0413 Remark 6]
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
        """Apply Hoeffding population correction. [Draft page 7]"""
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
    [Draft v0403 §3.2 page 7]
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
