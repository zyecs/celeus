"""
Canonical trajectory .npz filename and path helpers.

All trajectory-writing and trajectory-reading code in the repo should go
through this module so that filenames stay consistent across SAVE, the
e-value baseline, and Cer-Eval, and across loss types and hyperparameter
sweeps.

Canonical schemes:

    save_trajectory_{dataset}_{target}_{surrogate}_{loss}_beta{beta}_{mode}_{acq}_seed{seed}.npz
    cereval_trajectory_{dataset}_{target}_{surrogate}_{loss}_seed{seed}.npz

Cer-Eval filenames intentionally omit beta/mode/acq because Cer-Eval's
output does not depend on them — it depends only on
(dataset, target, surrogate, loss_type, seed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Defaults mirrored from the SAVE spec experiment matrix.
DEFAULT_DATASETS: tuple[str, ...] = ("sst2", "mmlu", "agnews")
DEFAULT_SURROGATES: tuple[str, ...] = ("llama2_7b", "llama3_8b")
DEFAULT_TARGETS: tuple[str, ...] = (
    "Mixtral_8x7b",
    "deepseek_67b",
    "llama3_70b",
    "qwen25_72b",
)

_VALID_LOSS_TYPES = frozenset({"accuracy", "cross_entropy"})
_VALID_MODES = frozenset({"anytime", "fixed"})
_VALID_ACQUISITIONS = frozenset({"active", "uniform"})
_TAGGED_SURROGATE_PREFIXES = ("remark1_", "remark2_")


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

# src/save/trajectory_paths.py → repo root is two parents up.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical output directory for all trajectory .npz files. Override at import
# time via the SAVE_TRAJECTORY_DIR environment variable.
TRAJECTORY_DIR: Path = Path(
    os.environ.get(
        "SAVE_TRAJECTORY_DIR",
        _DEFAULT_REPO_ROOT / "results" / "llm_experiments" / "trajectories",
    )
)

# Legacy directory where old-scheme files still live until the migration
# script runs. Only used by the read-side enumerators (*_read_paths).
_LEGACY_ANALYSIS_DIR: Path = (
    _DEFAULT_REPO_ROOT / "results" / "llm_experiments" / "analysis"
)


# ---------------------------------------------------------------------------
# Normalization and validation
# ---------------------------------------------------------------------------

def _normalize_loss_type(loss_type: str) -> str:
    if loss_type not in _VALID_LOSS_TYPES:
        raise ValueError(
            f"Unknown loss_type {loss_type!r}; expected one of {sorted(_VALID_LOSS_TYPES)}."
        )
    return loss_type


def _normalize_mode(mode: Union[str, bool]) -> str:
    """
    Normalize any of ``"anytime"``, ``"fixed"``, ``"fixed_horizon"``, or a
    bool (``fixed_horizon`` flag) to the canonical ``"anytime"`` / ``"fixed"``.
    """
    if isinstance(mode, bool):
        return "fixed" if mode else "anytime"
    if mode == "fixed_horizon":
        return "fixed"
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of {sorted(_VALID_MODES)} "
            f"(or the bool fixed_horizon flag, or the legacy string 'fixed_horizon')."
        )
    return mode


def _normalize_acquisition(acquisition: Union[str, bool]) -> str:
    """
    Normalize to ``"active"`` / ``"uniform"``. Accepts a bool where
    ``True`` means ``"uniform"`` (matching the ``uniform_acquisition`` flag
    used in the benchmark config).
    """
    if isinstance(acquisition, bool):
        return "uniform" if acquisition else "active"
    if acquisition not in _VALID_ACQUISITIONS:
        raise ValueError(
            f"Unknown acquisition {acquisition!r}; expected one of "
            f"{sorted(_VALID_ACQUISITIONS)} (or the bool uniform_acquisition flag)."
        )
    return acquisition


def _format_beta(beta_min: float) -> str:
    """
    Format beta_min for inclusion in a filename. Uses ``str(float(x))`` so
    that 0.2 → "0.2" and 0.05 → "0.05", matching the legacy convention.
    """
    return str(float(beta_min))


def _format_seed(seed: int) -> str:
    return str(int(seed))


def _has_surrogate_type_tag(surrogate_type: Optional[str]) -> bool:
    return surrogate_type == "none" or (
        surrogate_type is not None
        and surrogate_type.startswith(_TAGGED_SURROGATE_PREFIXES)
    )


# ---------------------------------------------------------------------------
# Canonical filename builders
# ---------------------------------------------------------------------------

def save_trajectory_filename(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    beta_min: float,
    mode: Union[str, bool],
    acquisition: Union[str, bool],
    seed: int,
    surrogate_type: Optional[str] = None,
    adaptive_bounds: bool = False,
) -> str:
    """Build the canonical SAVE+baseline trajectory filename (no directory)."""
    loss = _normalize_loss_type(loss_type)
    mode_s = _normalize_mode(mode)
    acq_s = _normalize_acquisition(acquisition)
    surrogate_tag = (
        f"_{surrogate_type}"
        if _has_surrogate_type_tag(surrogate_type)
        else ""
    )
    ada_tag = "_ada" if adaptive_bounds else ""
    return (
        f"save_trajectory_{dataset}_{target}_{surrogate}_{loss}"
        f"_beta{_format_beta(beta_min)}_{mode_s}_{acq_s}"
        f"{surrogate_tag}{ada_tag}_seed{_format_seed(seed)}.npz"
    )


def cereval_trajectory_filename(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    seed: int,
) -> str:
    """Build the canonical Cer-Eval trajectory filename (no directory)."""
    loss = _normalize_loss_type(loss_type)
    return (
        f"cereval_trajectory_{dataset}_{target}_{surrogate}_{loss}"
        f"_seed{_format_seed(seed)}.npz"
    )


# ---------------------------------------------------------------------------
# Write-side paths (always the new canonical location)
# ---------------------------------------------------------------------------

def save_trajectory_path(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    beta_min: float,
    mode: Union[str, bool],
    acquisition: Union[str, bool],
    seed: int,
    base_dir: Union[Path, None] = None,
    surrogate_type: Optional[str] = None,
    adaptive_bounds: bool = False,
) -> Path:
    """
    Return the canonical write path for a SAVE+baseline trajectory.

    ``base_dir`` overrides :data:`TRAJECTORY_DIR` (useful in tests).
    """
    root = Path(base_dir) if base_dir is not None else TRAJECTORY_DIR
    return root / save_trajectory_filename(
        dataset,
        target,
        surrogate,
        loss_type,
        beta_min,
        mode,
        acquisition,
        seed,
        surrogate_type=surrogate_type,
        adaptive_bounds=adaptive_bounds,
    )


def cereval_trajectory_path(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    seed: int,
    base_dir: Union[Path, None] = None,
) -> Path:
    """
    Return the canonical write path for a Cer-Eval trajectory.

    ``base_dir`` overrides :data:`TRAJECTORY_DIR` (useful in tests).
    """
    root = Path(base_dir) if base_dir is not None else TRAJECTORY_DIR
    return root / cereval_trajectory_filename(
        dataset, target, surrogate, loss_type, seed
    )


# ---------------------------------------------------------------------------
# Read-side path enumerators — new path first, then legacy fallbacks
# ---------------------------------------------------------------------------
#
# These exist so that, during the cutover window between the code deploy
# and the bulk filename migration, readers (and the merge-with-existing
# step in run_all_experiments.py) can transparently find files under
# their old names. Once the migration script has finished moving every
# legacy file, the fallback paths return nothing and can be dropped in a
# follow-up cleanup commit.
#
# Callers iterate the returned list and pick the first path that
# ``.is_file()``.


def _save_legacy_candidates(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    beta_min: float,
    mode: str,
    acquisition: str,
    seed: int,
) -> Iterable[Path]:
    """
    Legacy SAVE+baseline paths that could hold data for this config.

    Legacy sources (see plan §Migration inventory):
      * ``trajectories/{ce_?}{d}_{t}_{s}_beta{B}_{mode}_{acq}_s{S}.npz``
        from ``examples/run_all_experiments.py``.
      * ``analysis/{ce_?}trajectory_{d}_{t}_{s}_beta{B}_s{S}.npz``
        from ``examples/plot_all_trajectories.py`` — only meaningful when
        ``mode == "anytime"`` and ``acquisition == "active"``.
      * ``analysis/trajectory_{d}_{t}_{s}_beta{B}_s{S}.npz``
        from ``examples/plot_trajectory.py`` — accuracy only, anytime/active.
    """
    beta_str = _format_beta(beta_min)
    seed_str = _format_seed(seed)
    ce_prefix = "ce_" if loss_type == "cross_entropy" else ""

    # run_all_experiments.py legacy path (still under trajectories/)
    yield TRAJECTORY_DIR / (
        f"{ce_prefix}{dataset}_{target}_{surrogate}"
        f"_beta{beta_str}_{mode}_{acquisition}_s{seed_str}.npz"
    )

    # plot_all_trajectories.py and plot_trajectory.py only ever produced
    # anytime+active data. For accuracy the two scripts emit *identical*
    # filenames (bare ``trajectory_`` prefix), so one yield covers both;
    # for cross_entropy only plot_all_trajectories.py emitted files (with
    # the ``ce_trajectory_`` prefix).
    if mode == "anytime" and acquisition == "active":
        yield _LEGACY_ANALYSIS_DIR / (
            f"{ce_prefix}trajectory_{dataset}_{target}_{surrogate}"
            f"_beta{beta_str}_s{seed_str}.npz"
        )


def save_trajectory_read_paths(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    beta_min: float,
    mode: Union[str, bool],
    acquisition: Union[str, bool],
    seed: int,
    base_dir: Union[Path, None] = None,
    surrogate_type: Optional[str] = None,
    adaptive_bounds: bool = False,
) -> list[Path]:
    """
    Enumerate ``[canonical_new_path, *legacy_candidates]`` for a SAVE
    trajectory. The caller is responsible for picking the first entry that
    exists on disk.
    """
    loss = _normalize_loss_type(loss_type)
    mode_s = _normalize_mode(mode)
    acq_s = _normalize_acquisition(acquisition)
    paths: list[Path] = [
        save_trajectory_path(
            dataset,
            target,
            surrogate,
            loss,
            beta_min,
            mode_s,
            acq_s,
            seed,
            base_dir=base_dir,
            surrogate_type=surrogate_type,
            adaptive_bounds=adaptive_bounds,
        )
    ]
    if _has_surrogate_type_tag(surrogate_type):
        paths.append(
            save_trajectory_path(
                dataset,
                target,
                surrogate,
                loss,
                beta_min,
                mode_s,
                acq_s,
                seed,
                base_dir=base_dir,
                surrogate_type=None,
            )
        )
    paths.extend(
        _save_legacy_candidates(
            dataset, target, surrogate, loss, beta_min, mode_s, acq_s, seed
        )
    )
    return paths


def _cereval_legacy_candidates(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    seed: int,
) -> Iterable[Path]:
    """
    Legacy Cer-Eval path. Only ``analysis/cereval_trajectory_{d}_{t}_{s}.npz``
    was ever written, and only for accuracy at seed 42 (both scripts that
    wrote it hard-coded those values).
    """
    if loss_type == "accuracy" and int(seed) == 42:
        yield _LEGACY_ANALYSIS_DIR / (
            f"cereval_trajectory_{dataset}_{target}_{surrogate}.npz"
        )


def cereval_trajectory_read_paths(
    dataset: str,
    target: str,
    surrogate: str,
    loss_type: str,
    seed: int,
    base_dir: Union[Path, None] = None,
) -> list[Path]:
    """
    Enumerate ``[canonical_new_path, *legacy_candidates]`` for a Cer-Eval
    trajectory.
    """
    loss = _normalize_loss_type(loss_type)
    paths: list[Path] = [
        cereval_trajectory_path(
            dataset, target, surrogate, loss, seed, base_dir=base_dir
        )
    ]
    paths.extend(
        _cereval_legacy_candidates(dataset, target, surrogate, loss, seed)
    )
    return paths


__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_SURROGATES",
    "DEFAULT_TARGETS",
    "TRAJECTORY_DIR",
    "save_trajectory_filename",
    "cereval_trajectory_filename",
    "save_trajectory_path",
    "cereval_trajectory_path",
    "save_trajectory_read_paths",
    "cereval_trajectory_read_paths",
]
