"""
CELEUS benchmark runner — runs CELEUS + baselines for comparison.

"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml

from celeus.allocation.uniform import UniformAllocation
from celeus.acquisition.residual_magnitude import ResidualMagnitudeAcquisition
from celeus.acquisition.surrogate_entropy import SurrogateEntropyAcquisition
from celeus.baselines.evalue import NaiveEValueBaseline
from celeus.config_keys import method_config
from celeus.core.confidence import EValueCS
from celeus.core.estimator import AIPWEstimator
from celeus.core.state import CeleusConfig
from celeus.diagnostics import TrajectoryRecorder, _get_git_hash
from celeus.partition import make_quantile_partition, make_trivial_partition
from celeus.runner import CeleusRunner

if TYPE_CHECKING:
    from celeus.partition import EvaluationPool


@dataclass
class BenchmarkResult:
    """Results from a single (target, surrogate, dataset) benchmark run."""
    config: dict
    true_risk: float
    surrogate_correlation: float
    celeus_estimate: float
    celeus_ci: tuple
    celeus_width: float
    celeus_labels: int
    celeus_trajectory: dict
    baseline_estimate: float
    baseline_ci: tuple
    baseline_width: float
    baseline_labels: int
    baseline_trajectory: dict
    celeus_ci_mid: float = 0.0
    baseline_ci_mid: float = 0.0
    # New baseline fields — Source: spec §Benchmark Integration
    evalue_baseline_estimate: Optional[float] = None
    evalue_baseline_ci: Optional[tuple] = None
    evalue_baseline_width: Optional[float] = None
    evalue_baseline_labels: Optional[int] = None
    evalue_baseline_trajectory: Optional[dict] = None
    evalue_baseline_ci_mid: Optional[float] = None
    cereval_baseline_estimate: Optional[float] = None
    cereval_baseline_ci: Optional[tuple] = None
    cereval_baseline_width: Optional[float] = None
    cereval_baseline_labels: Optional[int] = None
    cereval_baseline_trajectory: Optional[dict] = None
    cereval_baseline_ci_mid: Optional[float] = None
    # Runtime-computed values for reporting [Stage 11 Fix 2]
    runtime_grid_size: Optional[int] = None
    runtime_cs_range: Optional[float] = None


def _extract_trajectory(recorder: TrajectoryRecorder) -> dict:
    """Extract trajectory arrays from a TrajectoryRecorder."""
    return {
        "t": np.array(recorder._t),
        "R_hat": np.array(recorder._R_hat),
        "lower": np.array(recorder._lower),
        "upper": np.array(recorder._upper),
        "pop_lower": np.array(recorder._pop_lower),
        "pop_upper": np.array(recorder._pop_upper),
        "total_labels": np.array(recorder._total_labels),
    }


def _extract_from_traj(traj: dict) -> tuple:
    """Extract final values from a 7-key trajectory dict."""
    if len(traj["t"]) == 0:
        return 0.0, (0.0, 1.0), 1.0, 0, 0.5
    estimate = float(traj["R_hat"][-1])
    ci = (float(traj["pop_lower"][-1]), float(traj["pop_upper"][-1]))
    width = ci[1] - ci[0]
    labels = int(traj["total_labels"][-1])
    ci_mid = (ci[0] + ci[1]) / 2.0
    return estimate, ci, width, labels, ci_mid


def _build_celeus_config(raw_config: dict) -> CeleusConfig:
    """Build CeleusConfig from the celeus_config subtree, merging with defaults."""
    celeus_dict = dict(method_config(raw_config))
    if raw_config.get("surrogate_type", "auto") == "none":
        celeus_dict["K"] = 1
    K = celeus_dict.get("K", 1)
    celeus_dict["B_round"] = K
    known = {f.name for f in dataclasses.fields(CeleusConfig)}
    filtered = {k: v for k, v in celeus_dict.items() if k in known}
    return CeleusConfig(**filtered)


def _run_single(
    pool: EvaluationPool,
    config: CeleusConfig,
    allocator,
    acquirer,
    label: str,
    surrogate_type: str = "auto",
    rng: np.random.Generator | None = None,
) -> tuple:
    """Run one CELEUS evaluation (either full CELEUS or baseline)."""
    # Always copy config to avoid mutation by CeleusRunner.__init__
    config = dataclasses.replace(config)

    if label == "celeus":
        strata = make_quantile_partition(pool, K=config.K)
    else:
        strata = make_trivial_partition(pool)
        config = dataclasses.replace(config, K=1, B_round=1)

    if rng is None:
        rng = np.random.default_rng(config.seed)
    estimator = AIPWEstimator(u_max=config.u_max)

    # Compute rescaling bounds [a, b] for the AIPW estimator
    L_lo = config.loss_lower
    L_hi = config.loss_bound
    loss_range = L_hi - L_lo
    beta = config.beta_min
    adaptive = config.adaptive_bounds
    if adaptive:
        cs_a = L_lo
        cs_b = L_hi
    elif surrogate_type == "none":
        # With ℓ̂ ≡ 0 and uniform WOR, R̂_t is naturally bounded in
        # [L_lo, L_hi] — the AIPW correction term cannot push it outside
        # the raw loss range. Use tight bounds matching the e-value baseline.
        cs_a = L_lo
        cs_b = L_hi
    else:
        # Conservative bound, exact in large-N limit. [Draft v0403 §3.2 page 6]
        cs_a = L_lo - loss_range / beta
        cs_b = L_hi + loss_range / beta

    # Scale grid_size to maintain ~0.0005 original-scale resolution
    # regardless of [a,b] range. [Remark 3; Stage 11 plan §1.2]
    cs_range = cs_b - cs_a
    grid_size = max(config.cs_grid_size, int(np.ceil(cs_range * config.cs_grid_size)))
    grid_size = min(grid_size, 200_000)  # Safety cap for CE loss
    if grid_size > config.cs_grid_size:
        warnings.warn(
            f"Grid size scaled to {grid_size} (from {config.cs_grid_size}) "
            f"to match [a,b] range={cs_range:.1f}.",
            stacklevel=2,
        )

    cs = EValueCS(
        alpha_1=config.alpha_1,
        grid_size=grid_size,
        c=config.c_betting,
        theta=config.theta,
        a=cs_a,
        b=cs_b,
        fixed_horizon=config.fixed_horizon,
        T_max=config.T_max,
        c_fixed=config.c_fixed,
        adaptive_bounds=adaptive,
    )
    recorder = TrajectoryRecorder(
        K=len(strata),
        config=config,
        git_hash=_get_git_hash(),
    )
    runner = CeleusRunner(
        config=config,
        pool=pool,
        strata=strata,
        estimator=estimator,
        cs=cs,
        recorder=recorder,
        rng=rng,
        allocator=allocator,
        acquirer=acquirer,
    )

    ground_truth = pool.ground_truth_losses
    runner.run(lambda indices: ground_truth[indices])

    traj = _extract_trajectory(recorder)

    if len(traj["t"]) == 0:
        return 0.5, (0.0, 1.0), 1.0, 0, traj, cs_range, grid_size

    final_lower = float(traj["pop_lower"][-1])
    final_upper = float(traj["pop_upper"][-1])
    # Use CI midpoint as point estimate — stable, unlike volatile last-round AIPW
    estimate = (final_lower + final_upper) / 2.0
    width = final_upper - final_lower
    labels = int(traj["total_labels"][-1])

    return estimate, (final_lower, final_upper), width, labels, traj, cs_range, grid_size


def run_benchmark(
    config: dict,
    pool: Optional[EvaluationPool] = None,
    rngs: Optional[dict[str, "np.random.Generator"]] = None,
) -> BenchmarkResult:
    """
    Run CELEUS + all baselines for one experiment config.

    Parameters
    ----------
    config : dict
        Full experiment config (from YAML or constructed directly).
    pool : EvaluationPool or None
        If provided, use this pool directly (for testing).
        If None, load from .pt files via loader.load_experiment.
    """
    if pool is None:
        from celeus.loader import load_experiment
        pool = load_experiment(config)

    if rngs is None:
        # Back-compat: legacy single-seed path uses independent
        # default_rng(seed) per site (existing behaviour).
        rngs = {"celeus_acq": None, "baseline_order": None}

    celeus_config = _build_celeus_config(config)

    # Set loss_bound from data for CE loss [Draft page 5: 0 ≤ ℓ ≤ L]
    # For accuracy loss, L=1.0 (default). For CE loss, L=max(losses).
    loss_type = config.get("loss_type", "accuracy")
    if loss_type == "cross_entropy":
        celeus_config.loss_bound = float(np.max(pool.ground_truth_losses))

    true_risk = float(np.mean(pool.ground_truth_losses))
    corr_matrix = np.corrcoef(pool.surrogate_scores, pool.ground_truth_losses)
    corr = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else 0.0

    # CELEUS run: Uniform allocation + acquisition policy
    # uniform_acquisition=True -> CELEUS-uniform ablation (acquirer=None, uniform WOR)
    # [Stage 11 plan §4.4: isolate acquisition vs estimator benefit]
    surrogate_type = config.get("surrogate_type", "auto")
    uniform_acq = config.get("uniform_acquisition", False)
    if surrogate_type == "none" or uniform_acq:
        acquirer = None
    else:
        beta_min_val = celeus_config.beta_min
        if surrogate_type.startswith(("remark1_", "remark2_")):
            acquirer = ResidualMagnitudeAcquisition(beta_min=beta_min_val)
        else:
            acquirer = SurrogateEntropyAcquisition(beta_min=beta_min_val)
    run_single_kwargs = dict(
        pool=pool,
        config=celeus_config,
        allocator=UniformAllocation(),
        acquirer=acquirer,
        label="celeus",
        surrogate_type=surrogate_type,
    )
    if rngs["celeus_acq"] is not None:
        run_single_kwargs["rng"] = rngs["celeus_acq"]
    celeus_est, celeus_ci, celeus_w, celeus_n, celeus_traj, celeus_cs_range, celeus_grid_size = (
        _run_single(**run_single_kwargs)
    )

    # E-value baseline — anytime-valid, uniform WOR + e-value CS
    # Uses α₁/α₂ split + population correction matching CELEUS. [Draft §3.2]
    # Replaces CLT baseline: we only care about anytime-valid guarantees.
    baseline_seed = method_config(config).get("seed", celeus_config.seed)
    if rngs["baseline_order"] is not None:
        evalue_bl = NaiveEValueBaseline(celeus_config, pool, rng=rngs["baseline_order"])
    else:
        evalue_bl = NaiveEValueBaseline(celeus_config, pool, seed=baseline_seed)
    base_traj = evalue_bl.run(lambda idx: pool.ground_truth_losses[idx])
    base_est, base_ci, base_w, base_n, baseline_ci_mid = _extract_from_traj(base_traj)

    celeus_ci_mid = (celeus_ci[0] + celeus_ci[1]) / 2.0

    # evalue_baseline_* fields alias baseline_* (both are the e-value baseline now)
    ev_est, ev_ci, ev_w, ev_n, ev_mid = base_est, base_ci, base_w, base_n, baseline_ci_mid
    evalue_traj = base_traj

    # Cer-Eval baseline — only if embeddings available and not skipped
    # Source: spec §3
    ce_est = ce_ci = ce_w = ce_n = ce_traj = ce_mid = None
    skip_cereval = config.get("skip_cereval", False)
    if not skip_cereval and pool.embeddings is not None:
        from celeus.baselines.cereval import CerEvalBaseline
        ce_kwargs = dict(
            pool=pool,
            C_full=celeus_config.epsilon,
            delta=celeus_config.alpha_1 + celeus_config.alpha_2,
            m_init=config.get("cereval_m_init", 100),
        )
        if rngs["baseline_order"] is not None:
            ce_kwargs["rng"] = rngs["baseline_order"]
        else:
            ce_kwargs["seed"] = baseline_seed
        cereval_bl = CerEvalBaseline(**ce_kwargs)
        ce_traj = cereval_bl.run()
        ce_est, ce_ci, ce_w, ce_n, ce_mid = _extract_from_traj(ce_traj)
    else:
        warnings.warn(
            "Cer-Eval baseline skipped: pool.embeddings is None. "
            "Set embedding_model in config or provide embeddings.",
            stacklevel=2,
        )

    return BenchmarkResult(
        config=config,
        true_risk=true_risk,
        surrogate_correlation=corr,
        celeus_estimate=celeus_est,
        celeus_ci=celeus_ci,
        celeus_width=celeus_w,
        celeus_labels=celeus_n,
        celeus_trajectory=celeus_traj,
        baseline_estimate=base_est,
        baseline_ci=base_ci,
        baseline_width=base_w,
        baseline_labels=base_n,
        baseline_trajectory=base_traj,
        celeus_ci_mid=celeus_ci_mid,
        baseline_ci_mid=baseline_ci_mid,
        evalue_baseline_estimate=ev_est,
        evalue_baseline_ci=ev_ci,
        evalue_baseline_width=ev_w,
        evalue_baseline_labels=ev_n,
        evalue_baseline_trajectory=evalue_traj,
        evalue_baseline_ci_mid=ev_mid,
        cereval_baseline_estimate=ce_est,
        cereval_baseline_ci=ce_ci,
        cereval_baseline_width=ce_w,
        cereval_baseline_labels=ce_n,
        cereval_baseline_trajectory=ce_traj,
        cereval_baseline_ci_mid=ce_mid,
        runtime_grid_size=celeus_grid_size,
        runtime_cs_range=celeus_cs_range,
    )


def run_batch(config_dir: str) -> list[BenchmarkResult]:
    """Run all YAML experiment configs in a directory."""
    config_path = Path(config_dir)
    results = []
    for yaml_file in sorted(config_path.glob("*.yaml")):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        if "data_root" in config:
            data_root = Path(config["data_root"])
            if not data_root.is_absolute():
                config["data_root"] = str(yaml_file.parent / data_root)
        print(f"Running: {yaml_file.name} ...")
        result = run_benchmark(config)
        results.append(result)
        print(
            f"  CELEUS: {result.celeus_labels} labels, width={result.celeus_width:.4f} | "
            f"Baseline: {result.baseline_labels} labels, width={result.baseline_width:.4f}"
        )
    return results
