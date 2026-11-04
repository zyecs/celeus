"""
CELEUS runner.

Implements the Algorithm 1 control flow:
  - Pluggable AllocationStrategy; None falls back to a trivial K=1 allocation
  - Pluggable AcquisitionPolicy; None falls back to an inline uniform draw
    without replacement
  - Early stopping once the population-corrected width is at most epsilon
  - All randomness goes through the injected np.random.Generator
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from celeus.acquisition.surrogate_entropy import SurrogateEntropyAcquisition
from celeus.core.confidence import EValueCS, pool_feasibility_intersection
from celeus.core.estimator import AIPWEstimator
from celeus.core.state import CeleusConfig, StratumState
from celeus.diagnostics import TrajectoryRecorder
from celeus.partition import EvaluationPool


def compute_adaptive_bounds(
    sum_past_losses: float,
    remaining_surrogates: np.ndarray,
    remaining_q_values: np.ndarray,
    N: int,
    L: float,
    U: float,
) -> tuple[float, float]:
    """
    Compute tightest predictable bounds (a_t, b_t) for R̂_t.

    The bounds use pre-round information only, so they are predictable as
    Remark 4 and Assumption 3 require.
    """
    plug_in = (sum_past_losses + remaining_surrogates.sum()) / N
    correction_lo = (L - remaining_surrogates) / (N * remaining_q_values)
    correction_hi = (U - remaining_surrogates) / (N * remaining_q_values)
    a_t = plug_in + float(correction_lo.min())
    b_t = plug_in + float(correction_hi.max())
    return a_t, b_t


class CeleusRunner:
    """
    Orchestrates the CELEUS evaluation loop (Algorithm 1).

    Parameters
    ----------
    config : CeleusConfig
        All hyperparameters.
    pool : EvaluationPool
        Full evaluation pool.
    strata : list[StratumState]
        Pre-computed stratum states (from make_strata / make_trivial_partition).
    estimator : AIPWEstimator
        AIPW estimator.
    cs : EValueCS
        E-value confidence sequence.
    recorder : TrajectoryRecorder
        Trajectory logger.
    rng : np.random.Generator
        Seeded random generator.
    allocator : AllocationStrategy or None
        Budget allocation strategy. If None, falls back to a trivial K=1
        allocation that gives all of B_round to stratum 0.
    acquirer : AcquisitionPolicy or None
        Acquisition policy. Defaults to SurrogateEntropyAcquisition(beta_min=0.05),
        which falls back to SelfEntropy when distribution arrays are absent.
        If None, falls back to an inline uniform draw without replacement.
    """

    def __init__(
        self,
        config: CeleusConfig,
        pool: EvaluationPool,
        strata: List[StratumState],
        estimator: AIPWEstimator,
        cs: EValueCS,
        recorder: TrajectoryRecorder,
        rng: np.random.Generator,
        allocator=None,  # Optional[AllocationStrategy] — avoids circular import
        acquirer="default",  # Optional[AcquisitionPolicy]; "default" → SurrogateEntropyAcquisition
    ) -> None:
        self.config = config
        self.pool = pool
        self.strata = strata
        self.estimator = estimator
        self.cs = cs
        self.recorder = recorder
        self.rng = rng
        self.allocator = allocator
        # Default acquisition: SurrogateEntropyAcquisition, which falls back
        # to SelfEntropy when distribution arrays are absent.
        if acquirer == "default":
            self.acquirer = SurrogateEntropyAcquisition(beta_min=config.beta_min)
        else:
            self.acquirer = acquirer

        # The algorithm queries one unlabeled sample from each stratum.
        # Enforce B_round = K (one item per stratum per round).
        K = len(strata)
        if config.B_round != K and config.B_round != 1:
            import warnings
            warnings.warn(
                f"CELEUS requires B_round = K = {K} (one item per stratum per round). "
                f"Overriding B_round={config.B_round} → {K}.",
                stacklevel=2,
            )
        config.B_round = K

    def run(self, oracle_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Run the CELEUS evaluation loop until `pop_w <= epsilon` or `total_labels >= T_max`.

        Algorithm 1 control flow:
          For each round t:
            1. ALLOCATE: assign B_round labels to strata via allocator or trivial fallback
            2. ACQUIRE: call acquirer.select() or inline uniform WOR
            3. LABEL: call oracle_fn on global pool indices of selected items
            4. UPDATE: update estimator, aggregate, update CS, apply population correction
            5. RECORD: log trajectory
            6. STOP: break if the population-corrected width `pop_w` is at most `epsilon`

        Parameters
        ----------
        oracle_fn : Callable[[np.ndarray], np.ndarray]
            Takes global pool indices (int array, shape (m,)), returns loss values
            (float array, shape (m,)).
            In tests: closure over pool.ground_truth_losses.
        """
        total_labels = 0
        t = 0  # round counter

        while total_labels < self.config.T_max:
            # Step 1: ALLOCATE labels to strata.
            K = len(self.strata)
            budgets = np.zeros(K, dtype=int)

            # Determine how many labels remain allocatable
            remaining_total = self.config.T_max - total_labels
            b_this_round = min(self.config.B_round, remaining_total)
            if b_this_round <= 0:
                break

            if self.allocator is not None:
                # Use the pluggable allocation strategy.
                budgets = self.allocator.allocate(self.strata, b_this_round)
            else:
                # Trivial K=1 fallback: all budget to stratum 0.
                budgets[0] = b_this_round

            # Step 2 and 3: ACQUIRE and LABEL per stratum.
            # Compute the adaptive bounds before acquisition.
            # Must be F_{t-1}-measurable: uses pre-round state only.
            if self.config.adaptive_bounds:
                plugin = self.estimator.get_plugin_state(self.strata)
                rem_surr = plugin["remaining_surrogate_scores"]
                if len(rem_surr) > 0:
                    # Build per-item q-floor: β_min / n_remaining_in_stratum
                    # Uses len(remaining) not N_k — tighter, still F_{t-1}-measurable.
                    remaining_q = []
                    for stratum in self.strata:
                        rem = stratum.remaining_indices()
                        n_rem = len(rem)
                        if n_rem > 0:
                            q_floor = self.config.beta_min / n_rem
                            remaining_q.append(np.full(n_rem, q_floor))
                    all_q = np.concatenate(remaining_q)
                    a_t, b_t = compute_adaptive_bounds(
                        plugin["sum_past_losses"], rem_surr, all_q,
                        self.pool.N, self.config.loss_lower, self.config.loss_bound,
                    )
                else:
                    a_t = b_t = None  # All items labeled

            labels_this_round = 0  # track actual labels acquired (not requested)
            for stratum_idx, stratum in enumerate(self.strata):
                b_k = int(budgets[stratum_idx])
                if b_k == 0:
                    continue

                # Remaining (unlabeled) local indices within this stratum
                remaining_local = stratum.remaining_indices()
                if len(remaining_local) == 0:
                    continue

                # Clip budget to available unlabeled items
                b_k = min(b_k, len(remaining_local))

                if self.acquirer is not None:
                    # Pluggable acquisition policy; it returns both the chosen
                    # local indices and their sampling probabilities.
                    selected_local, q_values = self.acquirer.select(
                        stratum,
                        b_k,
                        self.rng,
                    )
                    selected_local = np.asarray(selected_local, dtype=int)
                    q_values = np.asarray(q_values, dtype=np.float64)
                else:
                    # Inline uniform without-replacement fallback with
                    # q_k(m) = 1/|remaining|.
                    selected_local = self.rng.choice(
                        remaining_local,
                        size=b_k,
                        replace=False,
                    )
                    n_remaining = len(remaining_local)
                    q_values = np.full(
                        b_k,
                        fill_value=1.0 / n_remaining,
                        dtype=np.float64,
                    )

                # Convert local indices to global pool indices for oracle
                global_indices = stratum.pool_indices[selected_local]

                # LABEL: call oracle function on global indices
                new_losses = oracle_fn(global_indices)
                new_losses = np.asarray(new_losses, dtype=np.float64)

                # UPDATE: update stratum state via the estimator.
                self.estimator.update_stratum(
                    stratum=stratum,
                    new_local_indices=selected_local,
                    new_losses=new_losses,
                    new_q_values=q_values,
                )
                labels_this_round += len(selected_local)

            # Step 4: AGGREGATE, UPDATE CS, POPULATION CORRECTION.
            # Aggregate the per-stratum estimates.
            R_hat_t = self.estimator.aggregate(self.strata)

            # Update the confidence sequence.
            if self.config.adaptive_bounds:
                if a_t is not None and b_t is not None:
                    lower, upper, _ = self.cs.update(R_hat_t, a_t=a_t, b_t=b_t)
                else:
                    lower = upper = R_hat_t
            else:
                lower, upper, _ = self.cs.update(R_hat_t)

            # Pool feasibility intersection.
            sum_queried_losses = sum(sum(s.losses) for s in self.strata)
            lower, upper, _ = pool_feasibility_intersection(
                lower=lower,
                upper=upper,
                sum_queried_losses=sum_queried_losses,
                t=total_labels + labels_this_round,
                N=self.pool.N,
                loss_lower=self.config.loss_lower,
                loss_upper=self.config.loss_bound,
            )

            # Population correction: Δ_N = L·√(log(2/α₂)/(2N)).
            pop_lower, pop_upper, pop_w = self.cs.population_correction(
                lower=lower,
                upper=upper,
                N=self.pool.N,
                alpha_2=self.config.alpha_2,
                loss_bound=self.config.loss_bound,
                loss_lower=self.config.loss_lower,
            )

            # Update total label count (actual labels acquired, not requested budget)
            total_labels += labels_this_round
            t += 1

            # Guard: if no labels were acquired this round, all strata are exhausted
            if labels_this_round == 0:
                break

            # Step 5: RECORD.
            ess_per_stratum = [TrajectoryRecorder.compute_ess(s) for s in self.strata]

            self.recorder.record(
                t=t,
                R_hat=R_hat_t,
                lower=lower,
                upper=upper,
                pop_lower=pop_lower,
                pop_upper=pop_upper,
                total_labels=total_labels,
                weight_clip_rate=self.estimator.weight_clip_rate,
                output_clip_rate=self.estimator.output_clip_rate,
                ess_per_stratum=ess_per_stratum,
            )

            # Early stopping, applied after RECORD.
            # Fixed-horizon mode must run to T_max: the CI is valid only at t=T.
            # monitor_to_T_max also suppresses the break, so the full
            # trajectory stays available for coverage-over-time analysis. The
            # crossing time is recovered from the recorded pop_w series by the
            # cell-schema writer.
            if (
                not self.config.fixed_horizon
                and not self.config.monitor_to_T_max
                and pop_w <= self.config.epsilon
            ):
                break
