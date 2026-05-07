"""Stage index -> cell-tuple mappings (deterministic iteration order).

v2: adds ``seed_chunks_of`` helper so SLURM arrays can assign one
(cell, seed-chunk) to each task. Each chunk is a contiguous slice of
the seed schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import PaperExperimentConfig


DEFAULT_SEED_CHUNK_SIZE = 5  # v2: keeps CE tasks <= 25 h walltime.


def seed_chunks(
    seeds: Sequence[int], chunk_size: int = DEFAULT_SEED_CHUNK_SIZE
) -> list[list[int]]:
    seeds = [int(s) for s in seeds]
    return [seeds[i:i + chunk_size] for i in range(0, len(seeds), chunk_size)]


@dataclass(frozen=True)
class MainCell:
    method_id: str
    dataset: str
    surrogate: str
    target: str
    loss: str


@dataclass(frozen=True)
class CeSweepCell:
    dataset: str
    surrogate: str
    target: str
    surrogate_type: str
    beta_min: float


@dataclass(frozen=True)
class BetaSweepCell:
    dataset: str
    surrogate: str
    target: str
    loss: str
    beta_min: float


@dataclass(frozen=True)
class OracleAccuracyCell:
    dataset: str
    surrogate: str
    target: str


@dataclass(frozen=True)
class AcquisitionSweepCell:
    """One acquisition-sweep trajectory key (§6.5 item #3, Task 7).

    The Cartesian product of YAML ``acquisition_sweep.cells`` x
    ``variants_{accuracy,cross_entropy}`` yields all sweep trajectories.
    """
    dataset: str
    surrogate: str
    target: str
    loss: str
    surrogate_type: str


@dataclass(frozen=True)
class WallclockCell:
    """One wallclock-stage trajectory key (§6.5 item #5, Task 9).

    Cer-Eval-only: the wallclock stage runs M5 with per-round timing
    instrumentation so we can plot Cer-Eval's compute cost alongside
    SAVE/IID. ``loss`` is always ``"accuracy"`` (Cer-Eval scope per the
    paper campaign) but is kept explicit for filename round-trip.
    """
    dataset: str
    surrogate: str
    target: str
    loss: str


@dataclass(frozen=True)
class HparamSweepCell:
    """One hparam-sweep trajectory key (§6.5 item #4, Task 8).

    The Cartesian product of YAML ``hparam_sweep.cells`` x ``configs`` yields
    all hparam-sweep trajectories. Accuracy-only; ``surrogate_type`` is fixed
    at the deployed accuracy strategy in the runner (``remark2_strategy4``).
    """
    dataset: str
    surrogate: str
    target: str
    loss: str
    config_name: str
    alpha_1: float
    theta: float
    c_betting: float
    c_fixed: float


def main_cells_for_loss(
    cfg: PaperExperimentConfig,
    *,
    loss: str,
    include_cereval: bool,
    only_cereval: bool = False,
) -> list[MainCell]:
    """Return cells for one loss.

    M5 is included only if ``include_cereval``. Cer-Eval scope
    (``cfg.cereval_scope``) further restricts M5:
    - ``losses`` filters which losses M5 runs on (paper campaign: accuracy only).
    - ``collapse_by_target`` runs M5 with one canonical surrogate per target
      (K=1 makes the surrogate choice irrelevant for the estimator).
    """
    methods = ["M5"] if only_cereval else ["M1", "M2", "M3", "M4"]
    if include_cereval and not only_cereval:
        methods = methods + ["M5"]
    if "M5" in methods and loss not in cfg.cereval_losses:
        methods = [m for m in methods if m != "M5"]

    cells: list[MainCell] = []
    for method in methods:
        pairs = cfg.cereval_pairs if method == "M5" else list(cfg.all_pairs)
        for dataset in cfg.datasets:
            for pair in pairs:
                cells.append(
                    MainCell(
                        method_id=method,
                        dataset=dataset,
                        surrogate=pair["surrogate"],
                        target=pair["target"],
                        loss=loss,
                    )
                )
    return cells


def ce_sweep_cells(cfg: PaperExperimentConfig) -> list[CeSweepCell]:
    pairs = list(cfg.ce_sweep["cells_weak"]) + list(cfg.ce_sweep["cells_strong"])
    out: list[CeSweepCell] = []
    for strategy in cfg.ce_sweep["strategies"]:
        for beta in cfg.ce_sweep["beta_mins"]:
            for cell in pairs:
                out.append(
                    CeSweepCell(
                        dataset=cell["dataset"],
                        surrogate=cell["surrogate"],
                        target=cell["target"],
                        surrogate_type=strategy,
                        beta_min=float(beta),
                    )
                )
    return out


def beta_sweep_cells(cfg: PaperExperimentConfig) -> list[BetaSweepCell]:
    out: list[BetaSweepCell] = []
    for beta in cfg.beta_sweep["beta_mins"]:
        for cell in cfg.beta_sweep["pairs_representative"]:
            for loss in cfg.losses:
                out.append(
                    BetaSweepCell(
                        dataset=cell["dataset"],
                        surrogate=cell["surrogate"],
                        target=cell["target"],
                        loss=loss,
                        beta_min=float(beta),
                    )
                )
    return out


def acquisition_sweep_cells(
    cfg: PaperExperimentConfig, *, loss: str, scope: str = "legacy"
) -> list[AcquisitionSweepCell]:
    """Enumerate (cell × variant) trajectories for one loss × scope.

    Outer loop is YAML cells, inner is variants — preserves config order so
    SLURM array indices are stable. The `scope` argument selects between the
    legacy 6-cell sweep (`scope="legacy"`) and the v0502 12-cell sweep
    (`scope="v0502"`); see spec §7.3.
    """
    block = cfg.acquisition_sweep
    if scope == "legacy":
        cells = block["cells_legacy"]
        if loss == "accuracy":
            variants = block["variants_legacy_accuracy"]
        elif loss == "cross_entropy":
            variants = block["variants_legacy_cross_entropy"]
        else:
            raise ValueError(f"unknown loss {loss!r}")
    elif scope == "v0502":
        cells = block["cells_v0502"]
        if loss not in block["strategies_v0502"]:
            raise ValueError(f"unknown loss {loss!r}")
        variants = block["strategies_v0502"][loss]
    else:
        raise ValueError(f"unknown scope {scope!r}")
    out: list[AcquisitionSweepCell] = []
    for cell in cells:
        for v in variants:
            out.append(
                AcquisitionSweepCell(
                    dataset=cell["dataset"],
                    surrogate=cell["surrogate"],
                    target=cell["target"],
                    loss=loss,
                    surrogate_type=v,
                )
            )
    return out


def hparam_sweep_cells(cfg: PaperExperimentConfig) -> list[HparamSweepCell]:
    """Enumerate (cell x config) trajectories for the hparam OAT sweep.

    Outer loop is YAML cells, inner is configs — preserves config order so
    SLURM array indices are stable.
    """
    block = cfg.hparam_sweep
    out: list[HparamSweepCell] = []
    for cell in block["cells"]:
        for hcfg in block["configs"]:
            out.append(
                HparamSweepCell(
                    dataset=cell["dataset"],
                    surrogate=cell["surrogate"],
                    target=cell["target"],
                    loss=block["loss"],
                    config_name=hcfg["name"],
                    alpha_1=float(hcfg["alpha_1"]),
                    theta=float(hcfg["theta"]),
                    c_betting=float(hcfg["c_betting"]),
                    c_fixed=float(hcfg["c_fixed"]),
                )
            )
    return out


def wallclock_cells(cfg: PaperExperimentConfig) -> list[WallclockCell]:
    """Enumerate wallclock-stage trajectories (§6.5 item #5, Task 9).

    Outer loop is YAML cells; ``loss`` comes from the block ('accuracy' in
    the paper campaign). Each cell maps to one Cer-Eval (M5) trajectory.
    """
    block = cfg.wallclock
    loss = str(block.get("loss", "accuracy"))
    out: list[WallclockCell] = []
    for cell in block["cells"]:
        out.append(
            WallclockCell(
                dataset=cell["dataset"],
                surrogate=cell["surrogate"],
                target=cell["target"],
                loss=loss,
            )
        )
    return out


def oracle_accuracy_cells(cfg: PaperExperimentConfig) -> list[OracleAccuracyCell]:
    out: list[OracleAccuracyCell] = []
    for dataset in cfg.datasets:
        for pair in cfg.all_pairs:
            out.append(
                OracleAccuracyCell(
                    dataset=dataset,
                    surrogate=pair["surrogate"],
                    target=pair["target"],
                )
            )
    return out


def smoke_cells(cfg: PaperExperimentConfig) -> list[MainCell]:
    s = cfg.smoke
    return [
        MainCell(
            method_id=m,
            dataset=s["dataset"],
            surrogate=s["surrogate"],
            target=s["target"],
            loss=s["loss"],
        )
        for m in s["methods"]
    ]


def pick_cell(cells: list, index: int):
    if not (0 <= index < len(cells)):
        raise IndexError(f"--index {index} out of range for {len(cells)} cells")
    return cells[index]


@dataclass(frozen=True)
class ChunkedAssignment:
    """One SLURM array task unit: a cell + a contiguous seed chunk."""

    cell: object
    chunk_index: int
    seeds: tuple


def chunked_assignments(
    cells: list,
    seeds: Sequence[int],
    chunk_size: int = DEFAULT_SEED_CHUNK_SIZE,
) -> list[ChunkedAssignment]:
    """Cartesian product: ``len(cells) * ceil(len(seeds)/chunk_size)`` tasks."""
    out: list[ChunkedAssignment] = []
    chunks = seed_chunks(seeds, chunk_size=chunk_size)
    for cell in cells:
        for i, ch in enumerate(chunks):
            out.append(ChunkedAssignment(cell=cell, chunk_index=i, seeds=tuple(ch)))
    return out


def pick_assignment(
    cells: list,
    seeds: Sequence[int],
    index: int,
    chunk_size: int = DEFAULT_SEED_CHUNK_SIZE,
) -> ChunkedAssignment:
    assignments = chunked_assignments(cells, seeds, chunk_size=chunk_size)
    if not (0 <= index < len(assignments)):
        raise IndexError(
            f"--index {index} out of range for {len(assignments)} "
            "(cell x chunk) assignments"
        )
    return assignments[index]
