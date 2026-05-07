"""
Naive e-value baseline — online mean estimation via e-value CS.

No AIPW, no surrogate, no strata. Uses α₁/α₂ split matching SAVE:
  α₁ for the e-value CS targeting R_N (pool risk)
  α₂ for Hoeffding population correction R_N → R

Source: Draft §3.2; [WSR24] — WOR preserves supermartingale property.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from save.core.confidence import EValueCS, pool_feasibility_intersection
from save.core.state import SAVEConfig
from save.partition import EvaluationPool


class NaiveEValueBaseline:
    """
    Online mean estimation baseline using e-value confidence sequence.

    Each round: draw one item uniformly WOR, observe L_t, feed directly
    into EValueCS.update(L_t). Uses α₁/α₂ split matching SAVE, with
    population correction via Hoeffding Δ_N.

    For accuracy loss (L=1): a=0, b=1 (raw losses already in [0,1]).
    For CE loss (L>1): a=0, b=L (rescales losses to [0,1]).

    Parameters
    ----------
    config : SAVEConfig
        SAVE configuration for CS parameters.
    pool : EvaluationPool
        Evaluation pool (only N used for budget).
    seed : int | None
        Random seed. CLAUDE.md Rule 4.
    rng : np.random.Generator | None
        Injected random generator. Mutually exclusive with seed.

    Source: Draft §3.2; [WSR24] — WOR supermartingale property.
    """

    def __init__(
        self,
        config: SAVEConfig,
        pool: EvaluationPool,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config
        self.pool = pool
        if rng is not None and seed is not None:
            raise ValueError("pass exactly one of seed= or rng=")
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            raise ValueError("must pass seed= or rng=")

        # α₁/α₂ split matching SAVE [Draft §3.2]
        L = config.loss_bound

        # For uniform WOR sampling, raw losses L_t ∈ [0, L].
        # Rescale to [0, 1] via a=0, b=L.
        # Scale grid to match SAVE resolution (~0.0005 per unit range)
        cs_range = L - config.loss_lower
        grid_size = min(max(config.cs_grid_size, int(np.ceil(cs_range * config.cs_grid_size))), 200_000)
        if grid_size > config.cs_grid_size:
            warnings.warn(
                f"Baseline grid scaled to {grid_size} for [a,b] range={cs_range:.1f}.",
                stacklevel=2,
            )
        self.cs = EValueCS(
            alpha_1=config.alpha_1,
            grid_size=grid_size,
            c=config.c_betting,
            theta=config.theta,
            a=config.loss_lower,
            b=L,
            adaptive_bounds=False,
        )

    def run(self, oracle_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
        """
        Run the naive e-value baseline.

        Parameters
        ----------
        oracle_fn : callable
            Takes array of global indices, returns array of losses.

        Returns
        -------
        dict
            7-key trajectory dict: t, R_hat, lower, upper,
            pop_lower, pop_upper, total_labels.
        """
        N = self.pool.N
        max_steps = min(self.config.T_max, N)
        perm = self.rng.permutation(N)
        L = self.config.loss_bound

        t_list = []
        r_hat_list = []
        lower_list = []
        upper_list = []
        pop_lower_list = []
        pop_upper_list = []
        total_labels_list = []
        running_sum = 0.0

        for t in range(1, max_steps + 1):
            idx = int(perm[t - 1])
            loss = float(oracle_fn(np.array([idx]))[0])

            # Feed raw loss into CS — rescaling handled inside EValueCS
            # [WSR24]: WOR supermartingale → CS valid (conservative)
            lower, upper, _ = self.cs.update(loss)

            # Pool feasibility intersection [Draft v0403 §3.2 page 7]
            lower, upper, _ = pool_feasibility_intersection(
                lower=lower,
                upper=upper,
                sum_queried_losses=running_sum + loss,
                t=t,
                N=N,
                loss_lower=self.config.loss_lower,
                loss_upper=self.config.loss_bound,
            )

            # Population correction [Draft page 7: Δ_N = L·√(log(2/α₂)/(2N))]
            pop_lower, pop_upper, pop_w = self.cs.population_correction(
                lower=lower,
                upper=upper,
                N=N,
                alpha_2=self.config.alpha_2,
                loss_bound=L,
                loss_lower=self.config.loss_lower,
            )

            running_sum += loss
            t_list.append(t)
            r_hat_list.append(running_sum / t)  # running mean, not raw loss
            lower_list.append(lower)
            upper_list.append(upper)
            pop_lower_list.append(pop_lower)
            pop_upper_list.append(pop_upper)
            total_labels_list.append(t)

            # paper_experiment spec §4: monitor_to_T_max suppresses the break.
            # Also mirror SAVERunner's fixed_horizon semantics — when True, the
            # baseline MUST run to T_max because its CI is valid only at t=T.
            # (Current legacy behaviour was to break unconditionally, which is
            # a pre-existing divergence from SAVERunner; this patch fixes both.)
            if (
                pop_w <= self.config.epsilon
                and not self.config.fixed_horizon
                and not getattr(self.config, "monitor_to_T_max", False)
            ):
                break

        return {
            "t": np.array(t_list),
            "R_hat": np.array(r_hat_list),
            "lower": np.array(lower_list),
            "upper": np.array(upper_list),
            "pop_lower": np.array(pop_lower_list),
            "pop_upper": np.array(pop_upper_list),
            "total_labels": np.array(total_labels_list),
        }
