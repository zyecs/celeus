"""Fixed-seed cases for the trajectory regression test.

Each case runs one method end to end on a small synthetic pool and returns a
flat dict of arrays. The keys are independent of the package's internal field
names, so the stored fixture stays valid when identifiers are renamed.
"""
from __future__ import annotations

import numpy as np

from save.paper_experiment.runners import run_cereval_for_seed, run_save_method_for_seed
from save.partition import EvaluationPool

N_POOL = 1500
T_MAX = 1500
SEEDS = (0, 1)

# Method settings copied from configs/paper_experiment_v0504.yaml (M1-M4).
# epsilon=0.15 (not the paper's 0.05) so that the small pools reach the
# stopping rule and the stop-time fields are exercised.
METHODS = {
    "M1": dict(uniform_acquisition=False, adaptive_bounds=True),
    "M2": dict(uniform_acquisition=False, adaptive_bounds=False),
    "M3": dict(uniform_acquisition=True, adaptive_bounds=True),
    "M4": dict(uniform_acquisition=True, adaptive_bounds=False),
}

PROTOCOL = dict(
    epsilon=0.15, alpha_1=0.025, alpha_2=0.025, theta=0.5,
    c_betting=0.5, c_fixed=0.5, cs_grid_size=2000, monitor_to_T_max=True,
)


def _accuracy_pool() -> EvaluationPool:
    rng = np.random.default_rng(12345)
    pool = EvaluationPool.from_synthetic(N=N_POOL, R_true=0.3, rho_surr=0.8, rng=rng)
    # Label-free loss proxy for the residual-magnitude acquisition rule.
    pool.ell_proxy = np.clip(pool.surrogate_scores + 0.1 * rng.uniform(size=N_POOL), 0.0, 1.0)
    pool.embeddings = rng.normal(size=(N_POOL, 8))
    pool.embeddings /= np.linalg.norm(pool.embeddings, axis=1, keepdims=True)
    return pool


def _ce_pool() -> EvaluationPool:
    rng = np.random.default_rng(54321)
    pool = EvaluationPool.from_synthetic_ce(N=N_POOL, R_true=1.2, rho_surr=0.7, rng=rng)
    pool.ell_proxy = pool.surrogate_scores + 0.2 * rng.uniform(size=N_POOL)
    return pool


POOLS = {"accuracy": _accuracy_pool, "cross_entropy": _ce_pool}


def _flatten(prefix: str, res) -> dict[str, np.ndarray]:
    out = {
        f"{prefix}/method_labels": res.save_labels,
        f"{prefix}/method_rhat": res.save_rhat,
        f"{prefix}/method_lo": res.save_lo,
        f"{prefix}/method_hi": res.save_hi,
        f"{prefix}/baseline_labels": res.base_labels,
        f"{prefix}/baseline_rhat": res.base_rhat,
        f"{prefix}/baseline_lo": res.base_lo,
        f"{prefix}/baseline_hi": res.base_hi,
    }
    scalars = [res.true_R, res.rho, float(res.did_stop), float(res.labels_to_stop),
               res.width_at_stop, res.final_width, float(res.coverage_at_stop),
               float(res.ever_miss), float(res.pop_inverted_count)]
    out[f"{prefix}/scalars"] = np.asarray(scalars, dtype=np.float64)
    return out


def run_all_cases() -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for loss, make_pool in POOLS.items():
        for method_id, flags in METHODS.items():
            surrogate_type = "none" if flags["uniform_acquisition"] else "remark1_strategy1"
            for seed in SEEDS:
                res = run_save_method_for_seed(
                    method_id=method_id, pool=make_pool(), T_max=T_MAX, loss_type=loss,
                    beta_min=0.4, surrogate_type=surrogate_type, seed=seed,
                    **flags, **PROTOCOL,
                )
                results.update(_flatten(f"{loss}/{method_id}/seed{seed}", res))
    for seed in SEEDS:
        res = run_cereval_for_seed(
            pool=_accuracy_pool(), T_max=T_MAX, epsilon=0.15, alpha_1=0.025,
            alpha_2=0.025, m_init=50, seed=seed, monitor_to_T_max=True,
        )
        results.update(_flatten(f"accuracy/M5/seed{seed}", res))
    return results
