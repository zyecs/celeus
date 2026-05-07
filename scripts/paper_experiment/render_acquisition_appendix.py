"""Render acquisition-strategy appendix for §6.5 item #3 (label app:acquisition-strategy).

Produces two artifacts under ``<out_root>/acquisition_appendix/``:

* ``tab_strategies.tex`` — per-strategy median labels-to-epsilon and
  miscoverage-at-tau_eps, split into Cross-entropy (Remark 1 variants)
  and Accuracy (Remark 2 variants) blocks. The deployed strategies pull
  rows from ``trajectories/main`` (existing campaign), the new variants
  pull from ``trajectories/acquisition_sweep``, and the accuracy oracle
  pulls from ``trajectories/oracle_accuracy``.
* ``fig_strategies.pdf`` — two-panel scatter (median labels-to-eps vs
  miscoverage rate); one panel per loss; one point per strategy.

All filtering goes through ``cfg.acquisition_sweep`` so the cell list
and seed window stay configurable. Rows that didn't materialise on disk
fall through with em-dash placeholders rather than crashing.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Make src/ and the repo root importable when running as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config
from scripts.paper_experiment.plot_style import apply_rc_helvetica


# Display labels for each strategy (table column ordering).
LABELS = {
    "remark2_strategy1": "R2-S1 (unc gap)",
    "remark2_strategy2": "R2-S2 (unc + soft)",
    "remark2_strategy3": "R2-S3 (soft + unc)",
    "remark2_strategy4": "R2-S4 (deployed)",
    "remark2_strategy5": "R2-S5 (hard + soft)",
    "remark2_oracle_strategy4": "R2-Oracle",
    "remark1_strategy1": "R1-S1 (entropy gap)",
    "remark1_strategy2": "R1-S2 (deployed)",
    "remark1_strategy3": "R1-S3 (entropy + mode)",
    "remark1_oracle": "R1-Oracle",
}

# Accuracy strategies in display order. Includes the deployed variant
# (S4) and the oracle baseline; the remaining four come from
# trajectories/acquisition_sweep.
ACC_STRATS = [
    "remark2_strategy1",
    "remark2_strategy2",
    "remark2_strategy3",
    "remark2_strategy4",
    "remark2_strategy5",
    "remark2_oracle_strategy4",
]

# CE strategies in display order. S2 is deployed; S1, S3, oracle are new.
CE_STRATS = [
    "remark1_strategy1",
    "remark1_strategy2",
    "remark1_strategy3",
    "remark1_oracle",
]

# v0502 top-3 strategies per loss (deployed + 2 next-best + Oracle).
ACC_STRATS_TOP3 = [
    "remark2_strategy2",
    "remark2_strategy3",
    "remark2_strategy4",   # deployed
    "remark2_oracle_strategy4",
]
CE_STRATS_TOP3 = [
    "remark1_strategy1",
    "remark1_strategy2",   # deployed
    "remark1_strategy3",
    "remark1_oracle",
]

# Source-folder dispatch per strategy: deployed strategies come from the
# §6.1 main campaign, the oracle for accuracy comes from the
# oracle_accuracy stage, everything else comes from acquisition_sweep.
_SOURCE_BY_STRATEGY: dict[str, list[str]] = {
    "remark2_strategy4":         ["trajectories/main"],
    "remark2_oracle_strategy4":  ["trajectories/oracle_accuracy"],
    "remark1_strategy2":         ["trajectories/main"],
}
_DEFAULT_SOURCE = ["trajectories/acquisition_sweep"]


def _sources_for(strategy: str) -> list[str]:
    return _SOURCE_BY_STRATEGY.get(strategy, _DEFAULT_SOURCE)


def _filter_to_acquisition_cells(
    summary: pd.DataFrame,
    cells_or_cfg,
    *,
    loss: str,
    surrogate_type: str,
    sources: list[str],
    seeds_acq: set[int] | None = None,
) -> pd.DataFrame:
    """Filter summary.csv rows to a specific cell list x seed window x strategy.

    Accepts two call styles for backwards compatibility:

    * New style (explicit): pass ``cells`` as a list of dicts and
      ``seeds_acq`` as a set of ints.
    * Legacy style (cfg object): pass a cfg namespace whose
      ``acquisition_sweep["cells_legacy"]`` and
      ``acquisition_sweep["seeds"]`` attributes supply the cell list and
      seed window.  ``seeds_acq`` must be omitted (or None).
    """
    # Resolve cells and seeds from either call style.
    if isinstance(cells_or_cfg, list):
        cells = cells_or_cfg
        if seeds_acq is None:
            raise TypeError(
                "_filter_to_acquisition_cells: seeds_acq is required when "
                "cells_or_cfg is a list"
            )
    else:
        # Legacy cfg-object path.
        cfg = cells_or_cfg
        cells = cfg.acquisition_sweep["cells_legacy"]
        seeds_acq = set(cfg.acquisition_sweep["seeds"])

    pair_keys = {(c["dataset"], c["surrogate"], c["target"]) for c in cells}

    df = summary[
        summary.source.isin(sources)
        & (summary.loss == loss)
        & (summary.surrogate_type == surrogate_type)
        & summary.seed.isin(seeds_acq)
    ]
    if df.empty:
        return df
    keys = list(zip(df.dataset, df.surrogate, df.target))
    mask = [k in pair_keys for k in keys]
    return df[mask]


def _strategy_stats(filtered: pd.DataFrame) -> tuple[str, str]:
    """Format ``(median labels-to-stop, miscoverage rate)`` for one strategy slice.

    Returns em-dash placeholders when the slice is empty (cells haven't
    materialised on disk yet). Median is taken over rows with
    ``did_stop == 1``; miscoverage is the mean of ``ever_miss`` over all
    rows in the slice (whether or not they stopped) so unstopped seeds
    still count toward the miscoverage rate denominator.
    """
    if filtered.empty:
        return "--", "--"
    stopped = filtered[filtered.did_stop == 1]
    if stopped.empty:
        cells_str = "--"
    else:
        med_labels = int(stopped.labels_to_stop.median())
        cells_str = f"{med_labels:,}"
    miscov_rate = float(filtered.ever_miss.mean())
    miscov_str = f"{miscov_rate:.3f}"
    return cells_str, miscov_str


def build_strategies_table(
    summary: pd.DataFrame,
    cells_or_cfg,
    seeds_acq: set[int] | None = None,
    acc_strats: list[str] | None = None,
    ce_strats: list[str] | None = None,
) -> str:
    """Two-block LaTeX table: CE strategies, then accuracy strategies.

    Columns: Strategy | Median labels-to-eps | Miscoverage at tau_eps.

    Accepts two call styles:

    * New style: ``cells_or_cfg`` is a list of cell dicts; ``seeds_acq``,
      ``acc_strats``, and ``ce_strats`` are explicit.
    * Legacy style: ``cells_or_cfg`` is a cfg namespace; remaining args
      default to None and are resolved internally from the cfg.
    """
    if acc_strats is None:
        acc_strats = ACC_STRATS
    if ce_strats is None:
        ce_strats = CE_STRATS

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Strategy & Median labels-to-$\epsilon$ & Miscoverage at $\tau_\epsilon$ \\",
        r"\midrule",
    ]
    # CE block.
    lines.append(r"\multicolumn{3}{l}{\textbf{Cross-entropy}} \\")
    lines.append(r"\midrule")
    for st in ce_strats:
        sub = _filter_to_acquisition_cells(
            summary, cells_or_cfg, loss="cross_entropy",
            surrogate_type=st, sources=_sources_for(st),
            seeds_acq=seeds_acq,
        )
        cells_str, miscov_str = _strategy_stats(sub)
        lines.append(f"{LABELS[st]} & {cells_str} & {miscov_str} \\\\")

    # Accuracy block.
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\textbf{Accuracy}} \\")
    lines.append(r"\midrule")
    for st in acc_strats:
        sub = _filter_to_acquisition_cells(
            summary, cells_or_cfg, loss="accuracy",
            surrogate_type=st, sources=_sources_for(st),
            seeds_acq=seeds_acq,
        )
        cells_str, miscov_str = _strategy_stats(sub)
        lines.append(f"{LABELS[st]} & {cells_str} & {miscov_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render_strategies_figure(
    summary: pd.DataFrame,
    cells_or_cfg,
    out_path_or_seeds: "Path | set[int] | None",
    acc_strats: list[str] | None = None,
    ce_strats: list[str] | None = None,
    out_path: Path | None = None,
    seeds_acq: set[int] | None = None,
) -> None:
    """Two-panel scatter: x = median labels-to-eps, y = miscoverage rate.

    Strategies that never stopped on the acquisition cells are dropped
    from the figure (no plottable x-coord). Each panel is an isolated
    legend so the eight (acc) / four (CE) markers can each be labeled.

    Accepts two call styles:

    * New style: ``cells_or_cfg`` is a list of cell dicts; positional
      arg 3 is ``seeds_acq`` (set[int]); positional args 4-5 are
      ``acc_strats`` and ``ce_strats``; ``out_path`` is the final kwarg.
    * Legacy style: ``cells_or_cfg`` is a cfg namespace; positional arg 3
      is ``out_path`` (Path); remaining strategy args are None.
    """
    # Detect call style from the third positional argument.
    if isinstance(out_path_or_seeds, Path) or out_path_or_seeds is None:
        # Legacy call: render_strategies_figure(summary, cfg, out_path)
        resolved_out_path = out_path_or_seeds  # type: ignore[assignment]
        resolved_seeds_acq = None
        resolved_acc = acc_strats if acc_strats is not None else ACC_STRATS
        resolved_ce = ce_strats if ce_strats is not None else CE_STRATS
    else:
        # New call: render_strategies_figure(summary, cells, seeds_acq, acc, ce, out_path)
        resolved_seeds_acq = out_path_or_seeds
        resolved_acc = acc_strats if acc_strats is not None else ACC_STRATS
        resolved_ce = ce_strats if ce_strats is not None else CE_STRATS
        resolved_out_path = out_path

    apply_rc_helvetica()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=False)
    panels = (
        (axes[0], "accuracy", resolved_acc),
        (axes[1], "cross_entropy", resolved_ce),
    )
    for ax, loss, strats in panels:
        for st in strats:
            sub = _filter_to_acquisition_cells(
                summary, cells_or_cfg, loss=loss,
                surrogate_type=st, sources=_sources_for(st),
                seeds_acq=resolved_seeds_acq,
            )
            if sub.empty:
                continue
            stopped = sub[sub.did_stop == 1]
            if stopped.empty:
                continue
            x = float(stopped.labels_to_stop.median())
            y = float(sub.ever_miss.mean())
            ax.scatter([x], [y], s=80, label=LABELS[st], alpha=0.85)
        ax.set_xlabel(r"Median labels-to-$\epsilon$")
        ax.set_ylabel(r"Miscoverage at $\tau_\epsilon$")
        ax.set_title({"accuracy": "Accuracy", "cross_entropy": "Cross-entropy"}[loss])
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    if resolved_out_path is not None:
        fig.savefig(resolved_out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root.")
    ap.add_argument("--cells", choices=("legacy", "v0502"), default="legacy",
                    help="Which cell list (cfg.acquisition_sweep.cells_legacy / cells_v0502).")
    ap.add_argument("--strategies", choices=("all", "top3"), default="all",
                    help="all = full strategy universe; top3 = curated top-3 + Oracle per loss.")
    args = ap.parse_args(argv)

    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    summary = pd.read_csv(out_root / "summary.csv")
    out_dir = out_root / "acquisition_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cells == "legacy":
        cells = cfg.acquisition_sweep["cells_legacy"]
        seeds_acq = set(cfg.acquisition_sweep["seeds"])
    else:
        cells = cfg.acquisition_sweep["cells_v0502"]
        # v0502 uses the full 50-seed window from main, not the legacy 15-seed window.
        seeds_acq = set(cfg.seeds_main)

    if args.strategies == "all":
        acc_strats, ce_strats = ACC_STRATS, CE_STRATS
    else:
        acc_strats, ce_strats = ACC_STRATS_TOP3, CE_STRATS_TOP3

    # Suffix rules:
    # - Default --out-root + legacy + all → no suffix (preserves v0426 filenames)
    # - Non-default --out-root + legacy + all → "_legacy"
    # - Anything else → f"_{cells}_{strategies}"
    is_default_root = (args.out_root is None
                       or Path(args.out_root) == Path(cfg.paths["out_root"]))
    if args.cells == "legacy" and args.strategies == "all":
        suffix = "" if is_default_root else "_legacy"
    else:
        suffix = f"_{args.cells}_{args.strategies}"
    tab_path = out_dir / f"tab_strategies{suffix}.tex"
    fig_path = out_dir / f"fig_strategies{suffix}.pdf"
    tab_path.write_text(build_strategies_table(summary, cells, seeds_acq, acc_strats, ce_strats))
    render_strategies_figure(summary, cells, seeds_acq, acc_strats, ce_strats, out_path=fig_path)
    print(f"wrote acquisition_appendix artifacts -> {out_dir} (cells={args.cells}, strategies={args.strategies}, suffix={suffix!r})")


if __name__ == "__main__":
    main()
