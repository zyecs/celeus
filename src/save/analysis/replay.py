"""Deterministic trajectory replay for rq4-rq6 estimator diagnostics.


Key guarantees (guards G1-G6):
  - G1: q_m reconstructed by ORIGINAL item identity
  - G2: u_{m,t} computed fresh per horizon t in metrics.py, not here
  - G3: proportional sampling (not greedy)
  - G4: single-draw semantics verified (hard for future cells, inferred for legacy)
  - G5: RNG is spawn_role_rngs(seed)["save_acq"], never default_rng
  - G6: NumPy version hard-checked against config_json when present
"""
from __future__ import annotations

import os

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from save.acquisition.residual_magnitude import ResidualMagnitudeAcquisition
from save.acquisition.uniform import UniformAcquisition
from save.core.state import StratumState
from save.paper_experiment.pool_loader import load_pool_for_cell, pool_sha256
from save.paper_experiment.rng_streams import spawn_role_rngs


logger = logging.getLogger(__name__)


# Campaign invariants; see rq4-rq6 design §4.4 G4.
_CAMPAIGN_K = 1
_CAMPAIGN_B_ROUND = 1


# Default data_root from CLAUDE.md memory (user env).
DEFAULT_DATA_ROOT = Path(os.environ.get("SAVE_DATA_ROOT", "./data"))


def _parse_config_and_check_gates(
    config_json: str,
    runtime_numpy: str,
) -> dict:
    """Parse config_json and enforce the G4/G6 two-tier gates.

    For future cells (key present): hard-abort on mismatch.
    For legacy cells (key absent): log WARNING and record the inferred value.

    Returns a dict of flags:
      {"K_inferred": bool, "B_round_inferred": bool, "numpy_inferred": bool}

    Source: rq4-rq6 design §4.4 (G4, G6), §8 failure matrix.
    """
    cfg = json.loads(config_json)
    flags = {"K_inferred": False, "B_round_inferred": False, "numpy_inferred": False}

    if "K" in cfg:
        if cfg["K"] != 1:
            raise ValueError(
                f"G4: config_json['K'] must be 1 for LURE single-draw semantics; "
                f"got {cfg['K']}. See rq4-rq6 design §4.4 G4."
            )
    else:
        logger.warning(
            "G4: legacy cell — config_json lacks 'K'; inferring campaign default (K=1)"
        )
        flags["K_inferred"] = True

    if "B_round" in cfg:
        if cfg["B_round"] != 1:
            raise ValueError(
                f"G4: config_json['B_round'] must be 1 for LURE single-draw semantics; "
                f"got {cfg['B_round']}. See rq4-rq6 design §4.4 G4."
            )
    else:
        logger.warning(
            "G4: legacy cell — config_json lacks 'B_round'; inferring campaign default (B_round=1)"
        )
        flags["B_round_inferred"] = True

    if "numpy_version" in cfg:
        if cfg["numpy_version"] != runtime_numpy:
            raise ValueError(
                f"G6: config_json['numpy_version']={cfg['numpy_version']!r} != "
                f"runtime numpy.__version__={runtime_numpy!r}. RNG streams may diverge; "
                f"replay aborted. See rq4-rq6 design §4.4 G6."
            )
    else:
        logger.warning(
            "G6: legacy cell — config_json lacks 'numpy_version'; recording runtime %s",
            runtime_numpy,
        )
        flags["numpy_inferred"] = True

    return flags


def _load_pool_and_check_sha(
    meta,
    data_root: Path,
    pool_override=None,
):
    """Reload the pool for this cell and verify pool_sha256.

    Honours CE NLL filter metadata (rq4-rq6 design §4.2): when ``ce_nll_filter_kept``
    is set, passes the corresponding filter dict to load_pool_for_cell so that the
    returned pool is already filtered. The replayed trajectory then indexes into
    the filtered arrays directly.

    Raises ValueError on SHA mismatch.

    Parameters
    ----------
    meta : CellMetadata
        Loaded via cell_schema.load_cell().
    data_root : Path
        Pool data root (defaults to DEFAULT_DATA_ROOT).
    pool_override : EvaluationPool or None
        For testing: skip the real loader and use this pool directly.

    Source: rq4-rq6 design §4.2 CE NLL-filter handling + §8 hash gate.
    """
    if pool_override is not None:
        pool = pool_override
    else:
        ce_filter = None
        if meta.ce_nll_filter_kept is not None:
            ce_filter = {
                "enabled": True,
                "threshold": float(meta.ce_nll_filter_threshold),
            }
        pool = load_pool_for_cell(
            data_root=data_root,
            dataset=meta.dataset,
            surrogate=meta.surrogate,
            target=meta.target,
            loss=meta.loss,
            surrogate_type=meta.surrogate_type,
            load_embeddings=False,
            ce_nll_filter=ce_filter,
        )

    sha = pool_sha256(pool)
    if sha != meta.pool_sha256:
        raise ValueError(
            f"pool_sha256 mismatch for {meta.dataset}/{meta.surrogate}/{meta.target}/"
            f"{meta.loss}: stored={meta.pool_sha256!r}, recomputed={sha!r}. "
            f"Pool data has drifted since the cell was written. Cannot replay."
        )

    # CE filter round-trip check (rq4-rq6 design §4.2): recompute the kept
    # mask from threshold + raw NLL values and assert kept_mask.sum() equals
    # the stored kept-count. This catches silent drift in either the filter
    # threshold or the build_ce_nll_mask implementation — stronger than a
    # bare pool.N == ce_nll_filter_kept check.
    if meta.ce_nll_filter_kept is not None:
        if pool.N != int(meta.ce_nll_filter_kept):
            raise ValueError(
                f"CE NLL filter kept-count mismatch: stored={meta.ce_nll_filter_kept}, "
                f"loaded N={pool.N}. Filter threshold or loader behaviour drifted."
            )
        if pool_override is None:
            # Load the UNFILTERED pool to access raw CE NLL values, then
            # recompute the mask independently and verify consistency.
            from save.filters import build_ce_nll_mask

            raw_pool = load_pool_for_cell(
                data_root=data_root,
                dataset=meta.dataset,
                surrogate=meta.surrogate,
                target=meta.target,
                loss=meta.loss,
                surrogate_type=meta.surrogate_type,
                load_embeddings=False,
                ce_nll_filter=None,  # raw, unfiltered
            )
            raw_nll = np.asarray(raw_pool.ground_truth_losses, dtype=np.float64)
            recomputed_mask = build_ce_nll_mask(
                raw_nll, float(meta.ce_nll_filter_threshold)
            )
            recomputed_kept = int(recomputed_mask.sum())
            if recomputed_kept != int(meta.ce_nll_filter_kept):
                raise ValueError(
                    f"CE NLL filter recompute mismatch: stored kept="
                    f"{meta.ce_nll_filter_kept}, recomputed kept={recomputed_kept} "
                    f"at threshold={meta.ce_nll_filter_threshold}. "
                    f"The filter semantics have drifted since the cell was written."
                )
            if meta.ce_nll_filter_original_n is not None:
                if raw_pool.N != int(meta.ce_nll_filter_original_n):
                    raise ValueError(
                        f"CE NLL filter original_n mismatch: stored="
                        f"{meta.ce_nll_filter_original_n}, raw pool N={raw_pool.N}."
                    )

    return pool


def _make_save_acq_rng(seed: int) -> np.random.Generator:
    """Return the same RNG the campaign's SAVE acquisition stream consumed.

    Campaign threading: runners.run_save_method_for_seed calls
    spawn_role_rngs(seed) then hands rngs['save_acq'] to run_benchmark.
    Replay must use the same child to be bit-exact (G5).

    Source: rq4-rq6 design §4.4 G5; src/save/paper_experiment/rng_streams.py.
    """
    return spawn_role_rngs(int(seed))["save_acq"]


def _make_acquisition_policy(
    acquisition: str,
    beta_min: float,
):
    """Map the 'ada' | 'uniform' | 'oracle_accuracy' label to an AcquisitionPolicy.

    For 'ada' and 'oracle_accuracy', both use ResidualMagnitudeAcquisition.
    The difference is what fills ``stratum.ell_proxy``:
      - 'ada':             surrogate-derived (e.g. remark2 strategy4) proxy
      - 'oracle_accuracy': ell_proxy = ground_truth_losses

    Which proxy is correct is determined by the original run's surrogate_type
    (already baked into the reloaded pool — see load_pool_for_cell). So the
    same policy object works for both and the distinction is handled by the
    pool, not by switching policies here.

    Source: rq4-rq6 design §4.1 / §6; spec
    2026-04-22-accuracy-oracle-acquisition-design.md.
    """
    if acquisition == "uniform":
        return UniformAcquisition()
    return ResidualMagnitudeAcquisition(beta_min=beta_min)


def _build_stratum_from_pool(pool) -> StratumState:
    """Construct a single K=1 stratum covering the whole pool (campaign invariant).

    Mirrors benchmark.make_trivial_partition but doesn't rely on it to keep
    this module self-contained. Attaches ``ell_proxy`` when the pool carries
    one — for oracle_accuracy cells this is ``ground_truth_losses`` (set by
    ``load_experiment`` when ``surrogate_type == "remark2_oracle_strategy4"``);
    for ada cells it is the surrogate-derived proxy; for M3 uniform cells
    the loader leaves it as None and the uniform policy does not read it.
    """
    N = pool.N
    ell_hat = np.asarray(pool.surrogate_scores, dtype=np.float64)
    stratum = StratumState(
        k=0,
        pool_indices=np.arange(N, dtype=np.int64),
        N_k=N,
        w_k=1.0,
        surrogate_scores=ell_hat,
        surrogate_mean=float(ell_hat.mean()),
        labeled_mask=np.zeros(N, dtype=bool),
    )
    ell_proxy = getattr(pool, "ell_proxy", None)
    if ell_proxy is not None:
        stratum.ell_proxy = np.asarray(ell_proxy, dtype=np.float64)
    return stratum


@dataclass(frozen=True)
class ReplayedStep:
    """One step of a replayed trajectory.

    Attributes
    ----------
    m : int
        1-indexed step number.
    i_m : int
        Chosen item identity in the ORIGINAL pool [0, N).
    q_m_at_chosen : float
        q_m(i_m | F_{m-1}, D_N), the proposal probability at the chosen
        item evaluated AT step m (not step t). Always strictly positive.
    loss_chosen : float
        ell(f(x_{i_m}), y_{i_m}) from the labeled pool.
    hat_chosen : float
        ell_hat(f(x_{i_m}), x_{i_m}) from the surrogate scores.
    """
    m: int
    i_m: int
    q_m_at_chosen: float
    loss_chosen: float
    hat_chosen: float


@dataclass(frozen=True)
class ReplayedTrajectory:
    """Full replayed trajectory for one (cell, seed, policy) triple.

    Three pool arrays are carried, not two. This is critical for §6.4's
    conditional-variance formula:

      - ``ell_full``: GROUND-TRUTH losses for the full pool.
        Used to form residuals r_j = ell_full[j] - ell_hat_full[j] in the
        numerator of Var(S_hat_t | F_{t-1}).
      - ``ell_hat_full``: surrogate scores hat_ell(f(x), x) for the full pool.
        Used both in residuals and in R_hat = (1/N) sum hat_ell + correction.
      - ``ell_proxy_full``: the PROXY array the campaign's acquisition policy
        actually consumed. For ada cells this is a surrogate-derived array
        (e.g. remark2_strategy4); for oracle_accuracy cells it equals
        ell_full; for uniform cells it is None (policy ignores it).
        Used in _compute_proposal_over_J to reconstruct q_t(j) over J_{t-1}.

    Conflating ell_full and ell_proxy_full produces mathematically wrong
    conditional variances for ada cells — see rq4-rq6 design §4.3 and the
    round-4 plan review (F7 fix).

    Attributes
    ----------
    cell_path : Optional[Path]
        Original .npz path. ``None`` when the trajectory is constructed
        synthetically (used in tests).
    method_id : str
        From CellMetadata, e.g. 'M1' / 'M3' / 'oracle_accuracy'.
    seed : int
        Seed used for spawn_role_rngs().
    acquisition : str
        One of 'ada', 'uniform', 'oracle_accuracy' — the logical policy label
        used by downstream metrics.
    N : int
        Pool size (CE-filtered when applicable).
    T : int
        Number of replayed steps (== len(steps)).
    ell_full : np.ndarray, shape (N,) float64
        Ground-truth loss for the FULL pool (filtered if applicable).
    ell_hat_full : np.ndarray, shape (N,) float64
        Surrogate scores for the FULL pool (filtered if applicable).
    ell_proxy_full : np.ndarray or None, shape (N,) float64
        Acquisition proxy array used by the campaign. None when
        ``acquisition == "uniform"`` (policy ignores the proxy).
    steps : list[ReplayedStep]
        One entry per step m=1..T.
    R_N : float
        Finite-pool risk for this (cell, seed): (1/N) sum_j ell_full[j].
        Matches CellMetadata.true_R for the seed.
    legacy_gate_flags : dict
        {"K_inferred": bool, "B_round_inferred": bool, "numpy_inferred": bool}
        True when the stored config_json did not carry the field and we fell
        back to campaign invariants or runtime value.
    """
    cell_path: "Optional[Path]"
    method_id: str
    seed: int
    acquisition: str
    N: int
    T: int
    ell_full: np.ndarray
    ell_hat_full: np.ndarray
    ell_proxy_full: "Optional[np.ndarray]"
    steps: list
    R_N: float
    legacy_gate_flags: dict = field(default_factory=dict)


def replay_trajectory(
    cell_path: Path,
    seed: int,
    acquisition: Literal["ada", "uniform", "oracle_accuracy"],
    T_horizon: int,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> ReplayedTrajectory:
    """Deterministic replay of a stored trajectory. See module docstring."""
    from save.paper_experiment.cell_schema import load_cell

    meta, _results = load_cell(cell_path)

    # Gate: batch size (G4) + NumPy version (G6).
    flags = _parse_config_and_check_gates(meta.config_json, runtime_numpy=np.__version__)

    # Gate: pool SHA (plus CE NLL-filter recomputation; see _load_pool_and_check_sha).
    pool = _load_pool_and_check_sha(meta=meta, data_root=Path(data_root))

    # Build single stratum + RNG + acquisition policy.
    stratum = _build_stratum_from_pool(pool)
    rng = _make_save_acq_rng(seed)
    policy = _make_acquisition_policy(acquisition, beta_min=meta.beta_min)

    # Pool arrays. Note three arrays — see ReplayedTrajectory docstring (F7 fix).
    N = pool.N
    T = min(T_horizon, N)
    ell_full = np.asarray(pool.ground_truth_losses, dtype=np.float64)
    ell_hat_full = stratum.surrogate_scores.copy()
    ell_proxy_full = None
    if acquisition != "uniform":
        pool_proxy = getattr(pool, "ell_proxy", None)
        if pool_proxy is None:
            raise ValueError(
                f"acquisition={acquisition!r} requires pool.ell_proxy to be populated "
                f"by the loader (via surrogate_type). Got None for cell {cell_path}."
            )
        ell_proxy_full = np.asarray(pool_proxy, dtype=np.float64)

    steps = []
    for m_idx in range(T):
        chosen_local, q_vals = policy.select(stratum, n_k=1, rng=rng)
        if len(chosen_local) == 0:
            break
        local = int(chosen_local[0])
        i_m = int(stratum.pool_indices[local])
        q = float(q_vals[0])
        # Guards: floor respect, NaN, non-positive.
        floor = meta.beta_min / stratum.N_k
        if q < floor - 1e-12:
            raise ValueError(
                f"G1/G3 floor violation at m={m_idx+1}: q={q} < floor={floor}. "
                f"Acquisition policy mutation suspected."
            )
        if not np.isfinite(q) or q <= 0.0:
            raise ValueError(f"Invalid q_m={q} at m={m_idx+1} (NaN or non-positive)")

        steps.append(ReplayedStep(
            m=m_idx + 1,
            i_m=i_m,
            q_m_at_chosen=q,
            loss_chosen=float(ell_full[i_m]),
            hat_chosen=float(ell_hat_full[i_m]),
        ))
        # Advance stratum: only ``labeled_mask`` is consumed by the next
        # select() call. The other mutable fields (M_k, label_order, losses,
        # q_values) are only used by the estimator pipeline, which replay
        # deliberately bypasses — leave them at their initial default values.
        stratum.labeled_mask[local] = True

    return ReplayedTrajectory(
        cell_path=cell_path,
        method_id=meta.method_id,
        seed=seed,
        acquisition=acquisition,
        N=N,
        T=len(steps),
        ell_full=ell_full,
        ell_hat_full=ell_hat_full,
        ell_proxy_full=ell_proxy_full,
        steps=steps,
        R_N=float(ell_full.mean()),
        legacy_gate_flags=flags,
    )
