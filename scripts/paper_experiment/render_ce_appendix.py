"""Render CE appendix tables + figure for §6.5 item #6 (label app:ce-results).

Produces three artifacts under ``<out_root>/ce_appendix/``:

* ``tab_ce_coverage.tex`` — empirical miscoverage rate at the stopping
  epsilon for {M1, M4} x {SST-2, MMLU, AG News}.
* ``tab_ce_labels_to_eps.tex`` — median labels-to-epsilon for {Celeus,
  IID+EValue} per dataset (median taken over seeds and pairs that
  actually stopped before the budget).
* ``fig_ce_width.pdf`` — three-panel CE width-vs-labels figure (M1 in
  blue, M4 in orange, one panel per dataset).

Design mirrors the §6.2 / §6.3 main-paper figures so the appendix slot
referenced by ``\\label{app:ce-results}`` lines up cleanly.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make src/ and the repo root importable when running as a script
# (parents[2] is the repo root containing both ``src/`` and ``scripts/``).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config
from save.paper_experiment.cell_paths import main_cell_path
from save.paper_experiment.cell_schema import load_cell
from scripts.paper_experiment.plot_style import apply_rc_helvetica


DATASET_LABEL = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
METHOD_LABEL = {
    "M1": "Celeus",
    "M2": "Celeus-fixed",
    "M3": "Celeus-no-surr",
    "M4": "IID+EValue",
}


def _load_method_curves(
    out_root: Path,
    method: str,
    loss: str,
    dataset: str,
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate width-vs-labels across pairs and seeds for one (method, loss, dataset).

    Loads ``trajectories/main/cell__{method}__{dataset}__{surrogate}__{target}__{loss}.npz``
    for each pair, extracts ``(save_labels, save_lo, save_hi)`` per seed,
    computes ``width = hi - lo`` and returns
    ``(label_grid, mean_width, lo_width, hi_width)`` where the bands are
    the 5th / 95th percentiles across the (pair, seed) population.

    Returns ``None`` if no cells were found on disk for the request.
    """
    pairs = list(cfg.pairs_weak) + list(cfg.pairs_strong)
    all_labels: list[np.ndarray] = []
    all_widths: list[np.ndarray] = []
    for pair in pairs:
        path = main_cell_path(
            out_root,
            method=method,
            dataset=dataset,
            surrogate=pair["surrogate"],
            target=pair["target"],
            loss=loss,
        )
        if not path.is_file():
            continue
        _meta, results = load_cell(path)
        for _seed, r in results.items():
            labels = np.asarray(r.save_labels, dtype=np.int64)
            widths = np.asarray(r.save_hi - r.save_lo, dtype=np.float64)
            valid = (widths >= 0) & np.isfinite(widths) & (labels >= 0)
            all_labels.append(labels[valid])
            all_widths.append(widths[valid])
    if not all_labels:
        return None
    grid = np.array(
        sorted({int(x) for arr in all_labels for x in arr}),
        dtype=np.int64,
    )
    matrix = np.full((len(all_labels), len(grid)), np.nan, dtype=np.float64)
    for i, (labs, ws) in enumerate(zip(all_labels, all_widths)):
        if labs.size == 0:
            continue
        # For each grid point, take the most recent observation at-or-before
        # that point (step interpolation; trajectories record only updates).
        idx = np.searchsorted(labs, grid, side="right") - 1
        valid_grid = idx >= 0
        matrix[i, valid_grid] = ws[idx[valid_grid]]
    mean_w = np.nanmean(matrix, axis=0)
    lo_w = np.nanpercentile(matrix, 5, axis=0)
    hi_w = np.nanpercentile(matrix, 95, axis=0)
    return grid, mean_w, lo_w, hi_w


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% interval for proportion k/n.

    Returns (lo, hi) clipped to [0, 1]. n=0 -> (0, 1).
    """
    if n <= 0:
        return (0.0, 1.0)
    from math import sqrt
    # 1.96 ≈ z_{1-α/2} for α=0.05.
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# CE coverage table mirrors the accuracy tab_rq1_coverage.tex aesthetic:
# point estimate + Wilson 95% interval in grey, paper-pair-filtered counts.
# Method roster: M1 / M3 / M4 from trajectories/main, plus an Oracle-CE row
# pulled from trajectories/acquisition_sweep (surrogate_type=remark1_oracle).
# CE has no dedicated trajectories/oracle_cross_entropy stage, so the Oracle
# CE statistics live in the sweep cells instead.
_CE_COVERAGE_METHODS = (
    ("M1",         "main",  None,             r"\ourmethod"),
    ("M3",         "main",  None,             r"\nosurr"),
    ("ORACLE_CE",  "sweep", "remark1_oracle", r"\oracleacq"),
    ("M4",         "main",  None,             r"\evbaseline"),
)


def build_coverage_table(summary: pd.DataFrame) -> str:
    """Empirical miscoverage rate at τ_ε for CE methods × 3 datasets, with
    Wilson 95% intervals; pooled across all paper_pairs the snapshot
    contains. M1/M3/M4 read from ``trajectories/main`` (M1 implicitly
    restricted to surrogate_type=remark1_strategy2 via the main path).
    Oracle CE pulls from ``trajectories/acquisition_sweep`` rows with
    surrogate_type=remark1_oracle, since no trajectories/oracle_cross_entropy
    stage exists. Format matches the accuracy tab_rq1_coverage.tex."""
    ce = summary[summary.loss == "cross_entropy"]
    df_main = ce[ce.source == "trajectories/main"]
    df_sweep = ce[ce.source == "trajectories/acquisition_sweep"]
    datasets = ("sst2", "mmlu", "agnews")
    rows: list[tuple[str, list[str]]] = []
    for m, kind, surrogate_type, label in _CE_COVERAGE_METHODS:
        cells: list[str] = []
        for ds in datasets:
            if kind == "main":
                slc = df_main[(df_main.method == m) & (df_main.dataset == ds)]
            else:  # kind == "sweep" — Oracle CE
                slc = df_sweep[
                    (df_sweep.dataset == ds)
                    & (df_sweep.surrogate_type == surrogate_type)
                ]
            n = len(slc)
            k = int(slc.ever_miss.sum()) if n else 0
            if n == 0:
                cells.append("--")
                continue
            p = k / n
            lo, hi = _wilson_ci(k, n)
            cells.append(
                f"${p:.3f}$~"
                f"{{\\color{{gray!70}}\\tiny$[{lo:.3f},\\,{hi:.3f}]$}}"
            )
        rows.append((label, cells))

    header = (
        r"\textbf{Method} & "
        + " & ".join(rf"\textsc{{{DATASET_LABEL[d]}}}" for d in datasets)
        + r" \\"
    )
    lines = [
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for label, cells in rows:
        lines.append(f"{label}")
        for cell in cells:
            lines[-1] += "\n  & " + cell
        lines[-1] += r" \\"
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_labels_to_eps_table(summary: pd.DataFrame) -> str:
    """Median labels-to-eps for Celeus (M1) and IID+EValue (M4) per dataset."""
    df = summary[(summary.loss == "cross_entropy") & (summary.did_stop == 1)]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Dataset & Celeus & IID+EValue \\",
        r"\midrule",
    ]
    for ds in ("sst2", "mmlu", "agnews"):
        cells: list[str] = []
        for m in ("M1", "M4"):
            slc = df[(df.method == m) & (df.dataset == ds)]
            med = int(slc.labels_to_stop.median()) if len(slc) else None
            cells.append(f"{med:,}" if med is not None else r"$t_{\max}$")
        lines.append(f"{DATASET_LABEL[ds]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render_width_figure(out_root: Path, cfg, out_path: Path) -> None:
    """Three-panel CE width-vs-labels figure (one panel per dataset)."""
    apply_rc_helvetica()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, ds in zip(axes, ("sst2", "mmlu", "agnews")):
        for m, color in [("M1", "C0"), ("M4", "C1")]:
            curve = _load_method_curves(out_root, m, "cross_entropy", ds, cfg)
            if curve is None:
                continue
            labels, mean_w, lo_w, hi_w = curve
            ax.plot(labels, mean_w, color=color, label=METHOD_LABEL[m])
            ax.fill_between(labels, lo_w, hi_w, color=color, alpha=0.2)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlabel("labels acquired")
    axes[0].set_ylabel(r"$\mathrm{width}(C_t)$")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root. Default: cfg.paths['out_root'].")
    ap.add_argument("--paper-pairs", action="store_true",
                    help="Filter to cfg.paper_pair_keys (v0502 scope).")
    args = ap.parse_args(argv)
    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    pair_filter = cfg.paper_pair_keys if args.paper_pairs else None
    summary = pd.read_csv(out_root / "summary.csv")
    if pair_filter:
        summary = summary[
            summary.apply(lambda r: (r.surrogate, r.target) in pair_filter, axis=1)
        ]
    out_dir = out_root / "ce_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_ce_coverage.tex").write_text(build_coverage_table(summary))
    (out_dir / "tab_ce_labels_to_eps.tex").write_text(
        build_labels_to_eps_table(summary)
    )
    render_width_figure(out_root, cfg, out_dir / "fig_ce_width.pdf")
    print(f"wrote ce_appendix artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
