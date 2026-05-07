"""Per-seed method runner wrappers for M1-M5 (paper_experiment spec section 2).

Both entry points (``run_save_method_for_seed`` and ``run_cereval_for_seed``)
call ``spawn_role_rngs(seed)`` internally and pass the spawned Generators
to the downstream SAVE + baseline code. This satisfies stream separation.
"""
from __future__ import annotations

import socket
import time

import numpy as np

from save.baselines.cereval import CerEvalBaseline
from save.diagnostics import _get_git_hash
from save.partition import EvaluationPool

from .cell_schema import PerSeedResult, compute_labels_to_stop
from .rng_streams import spawn_role_rngs
from .traj_utils import pad_trajectory


def _pad_traj(traj: dict, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, r_hat, lo, hi), each length T."""
    labels = pad_trajectory(np.asarray(traj["total_labels"], dtype=np.int64), T, fill="last")
    r_hat = pad_trajectory(np.asarray(traj["R_hat"], dtype=np.float64), T, fill="last")
    lo = pad_trajectory(np.asarray(traj["pop_lower"], dtype=np.float64), T, fill="last")
    hi = pad_trajectory(np.asarray(traj["pop_upper"], dtype=np.float64), T, fill="last")
    return labels, r_hat, lo, hi


def _count_pop_inversions(lo: np.ndarray, hi: np.ndarray) -> int:
    """Count pop_lower > pop_upper rows (diagnostic only - do not clamp).

    SAVE's maintained EValueCS is monotone-valid by construction; inversions
    only arise from pool_feasibility_intersection conflicts (an early-
    trajectory pathology where the CS disagrees with labeled observations).
    Spec section 9's original "clamp to [0, 1]" rule is unsafe for CE loss
    (where R_N can exceed 1, turning a numerical pathology into a false
    miscoverage). Instead we record the inversion count so downstream QC can
    flag it, but leave the trajectory arrays untouched. Analysis code skips
    inverted rows via the ``hi >= lo`` mask. This is a deliberate deviation
    from spec section 9; see v2 changelog item I3.
    """
    return int(np.sum(lo > hi))


def run_save_method_for_seed(
    *,
    method_id: str,
    pool: EvaluationPool,
    T_max: int,
    loss_type: str,
    beta_min: float,
    surrogate_type: str,
    adaptive_bounds: bool,
    uniform_acquisition: bool,
    epsilon: float,
    alpha_1: float,
    alpha_2: float,
    theta: float,
    c_betting: float,
    c_fixed: float,
    cs_grid_size: int,
    monitor_to_T_max: bool,
    seed: int,
) -> PerSeedResult:
    """Run one SAVE-family method (M1/M2/M3/M4) end-to-end for one seed."""
    from save.benchmark import run_benchmark

    config = {
        "dataset": "__inline__",
        "data_root": "__unused__",
        "target_model": "__inline__",
        "surrogate_model": "__inline__",
        "surrogate_type": surrogate_type,
        "loss_type": loss_type,
        "uniform_acquisition": uniform_acquisition,
        "skip_cereval": True,
        "save_config": {
            "K": 1,
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "epsilon": epsilon,
            "T_max": T_max,
            "cs_grid_size": cs_grid_size,
            "c_betting": c_betting,
            "c_fixed": c_fixed,
            "theta": theta,
            "seed": seed,
            "fixed_horizon": False,
            "beta_min": beta_min,
            "adaptive_bounds": adaptive_bounds,
            "monitor_to_T_max": monitor_to_T_max,
        },
    }

    # Spawn independent role streams from seed and hand them to run_benchmark,
    # which threads them to SAVE (save_acq) and baseline (baseline_order).
    rngs = spawn_role_rngs(seed)
    t0 = time.time()
    result = run_benchmark(config, pool=pool, rngs=rngs)
    elapsed = time.time() - t0

    save_labels, save_rhat, save_lo, save_hi = _pad_traj(result.save_trajectory, T_max)
    if method_id == "M4":
        base_labels, base_rhat, base_lo, base_hi = _pad_traj(
            result.baseline_trajectory, T_max
        )
        # For M4 the save_* slot reports the naive e-value curve. Downstream
        # analysis keys on method_id, not save_* versus base_* naming.
        save_labels, save_rhat, save_lo, save_hi = base_labels, base_rhat, base_lo, base_hi
    else:
        base_labels, base_rhat, base_lo, base_hi = _pad_traj(
            result.baseline_trajectory, T_max
        )

    save_inverted = _count_pop_inversions(save_lo, save_hi)
    base_inverted = _count_pop_inversions(base_lo, base_hi)

    true_R = float(result.true_risk)
    rho = float(result.surrogate_correlation)
    labels_to_stop = compute_labels_to_stop(
        save_lo, save_hi, save_labels, epsilon=epsilon,
    )
    did_stop = labels_to_stop > 0

    if save_hi[-1] >= save_lo[-1]:
        final_width = float(save_hi[-1] - save_lo[-1])
    else:
        final_width = float("nan")

    if did_stop:
        cross_idx = int(np.searchsorted(save_labels, labels_to_stop))
        cross_idx = min(cross_idx, save_lo.shape[0] - 1)
        width_at_stop = float(save_hi[cross_idx] - save_lo[cross_idx])
        coverage_at_stop = bool(save_lo[cross_idx] <= true_R <= save_hi[cross_idx])
    else:
        width_at_stop = float("nan")
        coverage_at_stop = False

    valid = save_lo <= save_hi
    ever_miss = bool(np.any(valid & ((save_lo > true_R) | (save_hi < true_R))))

    return PerSeedResult(
        save_labels=save_labels, save_rhat=save_rhat, save_lo=save_lo, save_hi=save_hi,
        base_labels=base_labels, base_rhat=base_rhat, base_lo=base_lo, base_hi=base_hi,
        true_R=true_R, rho=rho,
        did_stop=did_stop, labels_to_stop=labels_to_stop,
        width_at_stop=width_at_stop, final_width=final_width,
        coverage_at_stop=coverage_at_stop, ever_miss=ever_miss,
        pop_inverted_count=save_inverted + base_inverted,
        elapsed_seconds=elapsed,
        git_commit=_get_git_hash(), hostname=socket.gethostname(),
    )


def run_cereval_for_seed(
    *,
    pool: EvaluationPool,
    T_max: int,
    epsilon: float,
    alpha_1: float,
    alpha_2: float,
    m_init: int,
    seed: int,
    monitor_to_T_max: bool,
) -> PerSeedResult:
    """Run M5 (Cer-Eval) end-to-end for one seed."""
    rngs = spawn_role_rngs(seed)
    bl = CerEvalBaseline(
        pool=pool, C_full=epsilon, delta=alpha_1 + alpha_2,
        m_init=m_init,
        rng=rngs["baseline_order"],
        T_max=T_max, monitor_to_T_max=monitor_to_T_max,
    )
    t0 = time.time()
    traj = bl.run()
    elapsed = time.time() - t0

    # Paper §6.5 item #5 (Task 9): preserve raw per-iteration wall-clock.
    # ``traj.get`` keeps backwards compatibility with older trajectory dicts
    # that lacked the key (pre-Task-9 fixture data).
    round_times_raw = traj.get("round_times")
    if round_times_raw is not None:
        round_times_arr = np.asarray(round_times_raw, dtype=np.float64)
    else:
        round_times_arr = None
    labels, r_hat, lo, hi = _pad_traj(traj, T_max)
    inverted = _count_pop_inversions(lo, hi)
    true_R = float(np.mean(pool.ground_truth_losses))
    surr = pool.surrogate_scores
    rho_mat = np.corrcoef(surr, pool.ground_truth_losses)
    rho = float(rho_mat[0, 1]) if np.isfinite(rho_mat[0, 1]) else 0.0
    labels_to_stop = compute_labels_to_stop(lo, hi, labels, epsilon=epsilon)
    did_stop = labels_to_stop > 0

    if hi[-1] >= lo[-1]:
        final_width = float(hi[-1] - lo[-1])
    else:
        final_width = float("nan")

    if did_stop:
        cross_idx = int(np.searchsorted(labels, labels_to_stop))
        cross_idx = min(cross_idx, lo.shape[0] - 1)
        width_at_stop = float(hi[cross_idx] - lo[cross_idx])
        coverage_at_stop = bool(lo[cross_idx] <= true_R <= hi[cross_idx])
    else:
        width_at_stop = float("nan")
        coverage_at_stop = False

    valid = lo <= hi
    ever_miss = bool(np.any(valid & ((lo > true_R) | (hi < true_R))))

    return PerSeedResult(
        save_labels=labels, save_rhat=r_hat, save_lo=lo, save_hi=hi,
        base_labels=labels, base_rhat=r_hat, base_lo=lo, base_hi=hi,
        true_R=true_R, rho=rho,
        did_stop=did_stop, labels_to_stop=labels_to_stop,
        width_at_stop=width_at_stop, final_width=final_width,
        coverage_at_stop=coverage_at_stop, ever_miss=ever_miss,
        pop_inverted_count=inverted,
        elapsed_seconds=elapsed,
        git_commit=_get_git_hash(), hostname=socket.gethostname(),
        round_times=round_times_arr,
    )
