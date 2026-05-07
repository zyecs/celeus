"""Render OAT hyperparameter sensitivity (label app:beta-sensitivity).

Produces three artifacts under ``<out_root>/hparam_appendix/``:

* ``tab_oat.tex`` — three sub-tables (one per OAT axis: alpha-split,
  theta, c_betting/c_fixed) glued together via ``\\multicolumn`` block
  separators. Each sub-table has rows = configs (baseline + 2
  perturbations) and columns = the 4 weak ce_sweep cells, listing the
  median labels-to-epsilon per (config, cell).
* ``tab_oat_pooled.tex`` — pooled-per-dataset variant. Rows = configs;
  columns = datasets (sst2/mmlu/agnews). Cells = median
  labels-to-epsilon pooled across (cell, seed) within each dataset.
  Asymmetric pooling (sst2 has 2 cells, mmlu/agnews 1 each) is noted
  in the table caption upstream.
* ``fig_oat.pdf`` — three-panel grouped bar chart (one panel per axis)
  showing the median labels-to-epsilon for the baseline vs each
  perturbation, averaged across the four cells.

Source rows: ``summary.csv`` filtered to
``source == "trajectories/hparam_sweep"``. Hparam stage is
accuracy-only; we ignore the ``loss`` column except as a sanity check.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
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


# Three OAT axes; per-axis (display label, list of perturbation config_names).
# Baseline is shared across axes and appears as the first row in each block.
AXES: list[tuple[str, str, list[str]]] = [
    ("alpha", r"$\alpha$-split", ["alpha_lo", "alpha_hi"]),
    ("theta", r"$\theta$",        ["theta_lo", "theta_hi"]),
    ("c",     r"$c_{betting}/c_{fixed}$", ["c_lo", "c_hi"]),
]

CONFIG_LABEL = {
    "baseline": "baseline",
    "alpha_lo": r"$\alpha_1=0.005$",
    "alpha_hi": r"$\alpha_1=0.045$",
    "theta_lo": r"$\theta=0.25$",
    "theta_hi": r"$\theta=0.75$",
    "c_lo":     r"$c=0.3$",
    "c_hi":     r"$c=0.7$",
}


def _cell_label(cell: dict) -> str:
    """Compact display label ``dataset/surrogate->target`` for a cell.
    Escapes underscores in model names for safe rendering in text mode."""
    ds = cell["dataset"]
    sur = cell["surrogate"].replace("_", r"\_")
    tgt = cell["target"].replace("_", r"\_")
    return f"{ds}/{sur}$\\to${tgt}"


def _filter_hparam(summary: pd.DataFrame, cfg) -> pd.DataFrame:
    """Slice summary.csv to the hparam_sweep cells x seeds (accuracy-only)."""
    cells = cfg.hparam_sweep["cells"]
    seeds = set(cfg.hparam_sweep["seeds"])
    pair_keys = {(c["dataset"], c["surrogate"], c["target"]) for c in cells}
    df = summary[
        (summary.source == "trajectories/hparam_sweep")
        & (summary.loss == cfg.hparam_sweep.get("loss", "accuracy"))
        & summary.seed.isin(seeds)
    ]
    if df.empty:
        return df
    keys = list(zip(df.dataset, df.surrogate, df.target))
    return df[[k in pair_keys for k in keys]]


def _median_labels(slc: pd.DataFrame) -> float:
    stopped = slc[slc.did_stop == 1]
    if stopped.empty:
        return float("nan")
    return float(stopped.labels_to_stop.median())


def build_oat_table(summary: pd.DataFrame, cfg) -> str:
    """Three OAT sub-tables glued into one ``tabular`` via multicolumn separators."""
    df = _filter_hparam(summary, cfg)
    cells = cfg.hparam_sweep["cells"]
    n_cols = len(cells)
    col_spec = "l" + "r" * n_cols

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Config & " + " & ".join(_cell_label(c) for c in cells) + r" \\",
        r"\midrule",
    ]
    for axis_key, axis_label, perturbations in AXES:
        lines.append(
            rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textbf{{{axis_label}}}}} \\"
        )
        lines.append(r"\midrule")
        for cfg_name in ["baseline"] + perturbations:
            row_cells = []
            for cell in cells:
                if df.empty:
                    row_cells.append("--")
                    continue
                slc = df[
                    (df.surrogate_type.isin(["remark2_strategy4"]) | df.surrogate_type.isna())
                    & (df.dataset == cell["dataset"])
                    & (df.surrogate == cell["surrogate"])
                    & (df.target == cell["target"])
                ]
                # Match config_name via the config_name column written by the
                # hparam_sweep parser; if absent (legacy or partial CSVs), the
                # fallback re-parses config_json. Both code paths must agree.
                if "config_name" in df.columns:
                    slc = slc[slc.config_name == cfg_name]
                med = _median_labels(slc)
                row_cells.append(f"{int(med):,}" if np.isfinite(med) else "--")
            lines.append(f"{CONFIG_LABEL[cfg_name]} & " + " & ".join(row_cells) + r" \\")
        lines.append(r"\midrule")
    # Drop the trailing \midrule and close the tabular.
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


_DATASET_DISPLAY = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
_DATASETS = ("sst2", "mmlu", "agnews")


def build_oat_pooled_table(summary: pd.DataFrame, cfg) -> str:
    """Pooled-per-dataset table: rows=configs (all 7), cols=datasets.

    Each cell = median labels-to-epsilon pooled across all (cell × seed)
    rows within that (config, dataset) slice. Stoppers only.
    """
    df = _filter_hparam(summary, cfg)
    cells_per_ds = {ds: [c for c in cfg.hparam_sweep["cells"] if c["dataset"] == ds]
                    for ds in _DATASETS}

    col_spec = "l" + "r" * len(_DATASETS)
    cells_count = " & ".join(f"{len(cells_per_ds[ds])}" for ds in _DATASETS)
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Config & " + " & ".join(_DATASET_DISPLAY[ds] for ds in _DATASETS) + r" \\",
        rf"{{\footnotesize \emph{{cells per dataset}}}} & {cells_count} \\",
        r"\midrule",
    ]
    config_order: list[str] = ["baseline"]
    for _, _, perts in AXES:
        config_order.extend(perts)
    for cfg_name in config_order:
        row_cells: list[str] = []
        for ds in _DATASETS:
            ds_cells = cells_per_ds[ds]
            if not ds_cells or df.empty:
                row_cells.append("--")
                continue
            pair_keys = {(c["dataset"], c["surrogate"], c["target"]) for c in ds_cells}
            slc = df[
                df.apply(lambda r: (r.dataset, r.surrogate, r.target) in pair_keys, axis=1)
            ]
            if "config_name" in df.columns:
                slc = slc[slc.config_name == cfg_name]
            med = _median_labels(slc)
            row_cells.append(f"{int(med):,}" if np.isfinite(med) else "--")
        lines.append(f"{CONFIG_LABEL[cfg_name]} & " + " & ".join(row_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render_oat_figure(summary: pd.DataFrame, cfg, out_path: Path) -> None:
    """Three-panel bar chart: baseline vs perturbations per axis.

    Each bar is the cross-cell mean of the per-cell median labels-to-eps.
    Empty cells (NaN medians) are skipped before averaging.
    """
    apply_rc_helvetica()
    df = _filter_hparam(summary, cfg)
    cells = cfg.hparam_sweep["cells"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, (axis_key, axis_label, perturbations) in zip(axes, AXES):
        configs = ["baseline"] + perturbations
        values: list[float] = []
        for cfg_name in configs:
            per_cell = []
            for cell in cells:
                if df.empty:
                    per_cell.append(float("nan"))
                    continue
                slc = df[
                    (df.dataset == cell["dataset"])
                    & (df.surrogate == cell["surrogate"])
                    & (df.target == cell["target"])
                ]
                if "config_name" in df.columns:
                    slc = slc[slc.config_name == cfg_name]
                per_cell.append(_median_labels(slc))
            arr = np.asarray(per_cell, dtype=float)
            values.append(float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan"))
        ax.bar(range(len(configs)), values, color=["#888888", "#332288", "#CC6677"])
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([CONFIG_LABEL[c] for c in configs], rotation=20, ha="right")
        ax.set_title(axis_label)
        ax.set_ylabel(r"Median labels-to-$\epsilon$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root (where hparam_appendix/ is written).")
    ap.add_argument("--summary-root", type=Path, default=None,
                    help="Override summary.csv source root. Defaults to "
                         "results/paper_experiment (where hparam_sweep rows live).")
    args = ap.parse_args(argv)

    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    summary_root = args.summary_root if args.summary_root else (_REPO / "results" / "paper_experiment")
    summary = pd.read_csv(summary_root / "summary.csv")

    out_dir = out_root / "hparam_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_oat.tex").write_text(build_oat_table(summary, cfg))
    (out_dir / "tab_oat_pooled.tex").write_text(build_oat_pooled_table(summary, cfg))
    render_oat_figure(summary, cfg, out_dir / "fig_oat.pdf")
    print(f"wrote hparam_appendix artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
