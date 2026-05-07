"""Pure metric functions for rq4-rq6 estimator diagnostics.

All functions are side-effect-free and operate on ReplayedTrajectory
objects produced by save.analysis.replay.

"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from save.analysis.replay import ReplayedTrajectory


def estimator_bias_terms(
    traj: ReplayedTrajectory,
    t_grid: np.ndarray,
) -> dict:
    """LURE-weighted and unweighted R̂_t at every t in t_grid.

    Formula (rq4-rq6 design §4.1 / draft_save_v0422-a.tex:665-683):

        u_{m,t} = 1 + (N-t)/(N-m) * (1/((N-m+1)*q_m) - 1)
        R_hat_lure_t        = ell_hat_bar + (1/t) * Σ_m u_{m,t} * (ell_m - ell_hat_m)
        R_hat_unweighted_t  = ell_hat_bar + (1/t) * Σ_m         (ell_m - ell_hat_m)

    Returns a dict with keys 'R_hat_lure' and 'R_hat_unweighted', each a
    np.ndarray of shape t_grid.shape.

    For t values exceeding len(traj.steps), NaN is returned for that entry.

    Source: rq4-rq6 design §4.1; draft_save_v0422-a.tex:665-683.
    """
    t_grid = np.asarray(t_grid, dtype=np.int64)
    N = traj.N
    ell_hat_bar = float(traj.ell_hat_full.mean())
    T = len(traj.steps)

    # Precompute per-step residuals and the (N-m+1)*q_m denominator.
    residuals = np.array(
        [s.loss_chosen - s.hat_chosen for s in traj.steps],
        dtype=np.float64,
    )  # shape (T,)
    q_at = np.array([s.q_m_at_chosen for s in traj.steps], dtype=np.float64)  # (T,)
    m_arr = np.arange(1, T + 1, dtype=np.float64)                             # (T,)

    # Denominator in u_{m,t}'s inner factor: (N - m + 1) * q_m
    denom_core = (N - m_arr + 1.0) * q_at  # (T,)

    R_hat_lure = np.full(t_grid.shape, np.nan, dtype=np.float64)
    R_hat_unweighted = np.full(t_grid.shape, np.nan, dtype=np.float64)

    for k, t in enumerate(t_grid):
        if t <= 0 or t > T:
            continue
        m_up_to_t = m_arr[:t]
        r_up_to_t = residuals[:t]
        q_up_to_t = denom_core[:t]
        # u_{m,t} = 1 + (N - t)/(N - m) * (1/((N-m+1)*q_m) - 1)
        u = 1.0 + (N - t) / (N - m_up_to_t) * (1.0 / q_up_to_t - 1.0)
        R_hat_lure[k] = ell_hat_bar + float((u * r_up_to_t).sum()) / float(t)
        R_hat_unweighted[k] = ell_hat_bar + float(r_up_to_t.sum()) / float(t)

    return {"R_hat_lure": R_hat_lure, "R_hat_unweighted": R_hat_unweighted}


def signal_sequence(
    traj: ReplayedTrajectory,
    t_grid: np.ndarray,
) -> dict:
    """IS-corrected signal S_hat_t and naive signal ell_t at every t in t_grid.

    Formula (rq4-rq6 design §4.2 / draft_save_v0422-a.tex:709-722):

        S_hat_t = (Σ_{m=1..t-1} ell_m + Σ_{j ∈ J_{t-1}} ell_hat[j]) / N
                  + (ell_t - ell_hat_t) / (N * q_t(i_t))

        S_naive_t = ell_t

    where J_{t-1} = [N] \ {i_1, ..., i_{t-1}}.

    Returns dict with 'S_hat' and 'naive', shape t_grid.shape.

    Source: rq4-rq6 design §4.2 (typo fix: both targeted at R_N, not R).
    """
    t_grid = np.asarray(t_grid, dtype=np.int64)
    N = traj.N
    T = len(traj.steps)
    ell_hat_full_sum = float(traj.ell_hat_full.sum())

    losses = np.array([s.loss_chosen for s in traj.steps], dtype=np.float64)
    hats = np.array([s.hat_chosen for s in traj.steps], dtype=np.float64)
    q_at = np.array([s.q_m_at_chosen for s in traj.steps], dtype=np.float64)
    sampled_i = np.array([s.i_m for s in traj.steps], dtype=np.int64)

    S_hat = np.full(t_grid.shape, np.nan, dtype=np.float64)
    naive = np.full(t_grid.shape, np.nan, dtype=np.float64)

    for k, t in enumerate(t_grid):
        if t < 1 or t > T:
            continue
        # Σ_{m=1..t-1} ell_m
        sum_past_losses = float(losses[:t - 1].sum()) if t >= 2 else 0.0
        # Σ_{j ∈ J_{t-1}} ell_hat[j] = total ell_hat minus those at labeled i_{1..t-1}
        sum_hat_unlabeled = ell_hat_full_sum - float(
            traj.ell_hat_full[sampled_i[:t - 1]].sum()
        ) if t >= 2 else ell_hat_full_sum
        # Correction term at t
        correction = (losses[t - 1] - hats[t - 1]) / (N * q_at[t - 1])
        S_hat[k] = (sum_past_losses + sum_hat_unlabeled) / N + correction
        naive[k] = losses[t - 1]

    return {"S_hat": S_hat, "naive": naive}


def _compute_proposal_over_J(
    ell_proxy_full: "Optional[np.ndarray]",
    ell_hat_full: np.ndarray,
    J_remaining: np.ndarray,
    N: int,
    policy: Literal["ada", "uniform", "oracle_accuracy"],
    beta_min: float,
) -> np.ndarray:
    """Return q_t(j) over j ∈ J_{t-1} for the given policy.

    For 'ada' and 'oracle_accuracy' the closed form is identical; what differs
    is the PROXY array ell_proxy_full that drives the |ell_proxy - ell_hat|
    score:
      - 'ada':             ell_proxy_full = campaign's surrogate-derived proxy
                           (e.g. remark2_strategy4). MUST NOT be ground truth.
      - 'oracle_accuracy': ell_proxy_full = ground_truth_losses (baked into
                           the loader via surrogate_type="remark2_oracle_strategy4").
      - 'uniform':         ell_proxy_full is ignored (may be None).

    F7 fix: previously this function took ``ell_full`` (ground truth) and used
    it as the proxy for ada cells — mathematically wrong. ada replay must
    reconstruct q_t(j) from the SURROGATE-derived proxy the campaign actually
    consumed. Callers now pass ``traj.ell_proxy_full`` which is populated from
    ``pool.ell_proxy`` (see replay.replay_trajectory + rq4-rq6 design §4.3).

    Returns a shape (|J_remaining|,) array of floats that sum to 1.
    """
    N_rem = len(J_remaining)
    if policy == "uniform":
        return np.full(N_rem, 1.0 / N_rem, dtype=np.float64)

    if ell_proxy_full is None:
        raise ValueError(
            f"_compute_proposal_over_J requires ell_proxy_full for policy={policy!r}; "
            f"got None. Ensure traj.ell_proxy_full is populated by replay_trajectory."
        )

    alpha = beta_min / N
    residuals = np.abs(ell_proxy_full[J_remaining] - ell_hat_full[J_remaining])
    s = residuals.sum()
    if s < 1e-12 or N_rem * alpha >= 1.0:
        return np.full(N_rem, 1.0 / N_rem, dtype=np.float64)
    return alpha + (1.0 - N_rem * alpha) * (residuals / s)


def conditional_variance_signal(
    ell_full: np.ndarray,
    ell_hat_full: np.ndarray,
    ell_proxy_full: "Optional[np.ndarray]",
    J_remaining: np.ndarray,
    N: int,
    policy: Literal["ada", "uniform", "oracle_accuracy"],
    beta_min: float,
) -> float:
    """Conditional variance of S_hat_t given F_{t-1}.

    Formula (rq4-rq6 design §4.3 / draft_save_v0422-a.tex:743-755):

        Var(S_hat_t | F_{t-1}) = Σ_{j ∈ J_{t-1}} r_j^2 / (N^2 * q_t(j))
                                 - (Σ_{j ∈ J_{t-1}} r_j / N)^2

    where r_j = ell(f(x_j), y_j) - ell_hat(f(x_j), x_j), USING GROUND-TRUTH
    losses in the residual numerator. The proposal q_t(j), however, is
    reconstructed from the campaign's actual acquisition proxy — for ada
    that is the surrogate-derived proxy, not ground truth (F7 fix).

    Source: rq4-rq6 design §4.3.
    """
    q_t = _compute_proposal_over_J(
        ell_proxy_full=ell_proxy_full, ell_hat_full=ell_hat_full,
        J_remaining=J_remaining, N=N, policy=policy, beta_min=beta_min,
    )
    residuals = ell_full[J_remaining] - ell_hat_full[J_remaining]
    term1 = float(((residuals ** 2) / (N * N * q_t)).sum())
    term2 = (float(residuals.sum()) / N) ** 2
    return term1 - term2


def empirical_variance_rhat(rhats_by_seed: np.ndarray) -> np.ndarray:
    """Sample variance across seeds (ddof=1).

    Input shape (M_seeds, G); output shape (G,).

    Source: rq4-rq6 design §4.3.
    """
    return rhats_by_seed.var(axis=0, ddof=1)


def estimator_aipw_running(traj: ReplayedTrajectory) -> np.ndarray:
    """Replay the campaign's AIPW running estimate step by step.

    Matches the single-stratum (K=1, N_k=N) formula from
    `src/save/core/estimator.py:5` and `:103-108`:

        R_hat_t = surrogate_mean + sum_past_residuals/N + (ell_t - hat_t) / (N * q_t)
        sum_past_residuals_{t+1} = sum_past_residuals_t + (ell_t - hat_t)

    With `sum_past_residuals_1 = 0`, so at t=1 it reduces to
        R_hat_1 = surrogate_mean + (ell_1 - hat_1) / (N * q_1).

    Parameters
    ----------
    traj : ReplayedTrajectory
        From replay_trajectory(). Uses traj.N, traj.ell_hat_full (for
        surrogate_mean), and traj.steps.

    Returns
    -------
    r_hat : np.ndarray, shape (len(traj.steps),)
        The AIPW running estimate after each of the T steps. Intended for
        trajectory-equivalence comparison against the campaign's stored
        save_rhat array (rq4-rq6 design §8 gate).

    NOTE: This differs from `estimator_bias_terms["R_hat_lure"]` (which
    implements the LURE re-weighting for §6.2.3 bias analysis). The two
    are mathematically distinct; this function exists solely for replay
    bit-exactness validation, not for any statistical claim in the paper.

    Source: src/save/core/estimator.py:5 + :103-108 single-stratum form.
    """
    N = traj.N
    T = len(traj.steps)
    ell_hat_bar = float(traj.ell_hat_full.mean())
    out = np.full(T, np.nan, dtype=np.float64)
    sum_past_residuals = 0.0
    for t_idx, step in enumerate(traj.steps):
        residual = step.loss_chosen - step.hat_chosen
        out[t_idx] = ell_hat_bar + sum_past_residuals / N + residual / (N * step.q_m_at_chosen)
        sum_past_residuals += residual
    return out
