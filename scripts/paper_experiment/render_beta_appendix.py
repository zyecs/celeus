"""Render β-sensitivity appendix.

Mirrors the structure of render_hparam_appendix.py: produces a per-cell
table, a pooled-per-(dataset×surr-quality) table, and a 2×3 line-plot
figure (loss rows × dataset columns) with weak vs strong surrogate
overlays. Reads ``results/paper_experiment/summary.csv`` cross-snapshot
because the β-sweep cell set is independent of v0504's paper_pairs
(scope-agnostic; see beta_sweep["pairs_representative"]).

Outputs under ``<out_root>/beta_appendix/``:

* ``tab_beta.tex`` — rows = β values, cols = cells (6 per loss × 2 loss
  blocks); cells = median labels-to-ε.
* ``tab_beta_pooled.tex`` — same rows/cols but datasets × {weak, strong}
  (6 cols per loss block; pooled over the single cell each combo has).
* ``fig_beta.pdf`` — 2×3 grid (loss × dataset); per panel shows median
  labels-to-ε vs β with weak vs strong as separate lines.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_TMP_CACHE = Path("/tmp/save-rq1-mpl")
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config
from scripts.paper_experiment.plot_style import apply_rc_helvetica


_DATASET_DISPLAY = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
_DATASETS = ("sst2", "mmlu", "agnews")
_LOSS_DISPLAY = {"accuracy": "Accuracy", "cross_entropy": "Cross-entropy"}


def _quality_for_pair(cfg, surrogate: str, target: str) -> str | None:
    """Return 'weak' or 'strong' based on cfg.beta_sweep.pairs_representative
    annotation (the YAML keeps a comment but no field; we infer by re-reading
    the original cfg.pairs_weak / pairs_strong sets)."""
    weak_set = {(p["surrogate"], p["target"]) for p in cfg.pairs_weak}
    strong_set = {(p["surrogate"], p["target"]) for p in cfg.pairs_strong}
    if (surrogate, target) in weak_set:
        return "weak"
    if (surrogate, target) in strong_set:
        return "strong"
    return None


def _filter_beta(summary: pd.DataFrame, cfg) -> pd.DataFrame:
    pairs = cfg.beta_sweep["pairs_representative"]
    seeds = set(cfg.seeds_beta_sweep)
    pair_keys = {(c["dataset"], c["surrogate"], c["target"]) for c in pairs}
    df = summary[
        (summary.source == "trajectories/beta_sweep")
        & summary.seed.isin(seeds)
    ]
    if df.empty:
        return df
    keys = list(zip(df.dataset, df.surrogate, df.target))
    df = df[[k in pair_keys for k in keys]]
    return df


def _median_labels(slc: pd.DataFrame) -> float:
    stopped = slc[slc.did_stop == 1]
    if stopped.empty:
        return float("nan")
    return float(stopped.labels_to_stop.median())


def _cell_label(cell: dict) -> str:
    sur = cell["surrogate"].replace("_", r"\_")
    tgt = cell["target"].replace("_", r"\_")
    return f"{cell['dataset']}/{sur}$\\to${tgt}"


def build_beta_table(summary: pd.DataFrame, cfg) -> str:
    """Per-cell median labels-to-ε. Two blocks (acc / CE), 6 cell columns each."""
    df = _filter_beta(summary, cfg)
    cells = cfg.beta_sweep["pairs_representative"]
    betas = list(cfg.beta_sweep["beta_mins"])
    n_cols = len(cells)
    col_spec = "l" + "r" * n_cols

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"$\beta_{\min}$ & " + " & ".join(_cell_label(c) for c in cells) + r" \\",
        r"\midrule",
    ]
    for loss_key, loss_label in _LOSS_DISPLAY.items():
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textbf{{{loss_label}}}}} \\")
        lines.append(r"\midrule")
        for beta in betas:
            row_cells = []
            for cell in cells:
                slc = df[
                    (df.loss == loss_key)
                    & (df.dataset == cell["dataset"])
                    & (df.surrogate == cell["surrogate"])
                    & (df.target == cell["target"])
                    & (df.beta_min == beta)
                ]
                med = _median_labels(slc)
                row_cells.append(f"{int(med):,}" if np.isfinite(med) else "--")
            lines.append(f"{beta:.1f} & " + " & ".join(row_cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_beta_pooled_table(summary: pd.DataFrame, cfg) -> str:
    """Cols = dataset × {weak, strong}; cells = median labels-to-ε.

    Each (dataset × surr-quality) combo has exactly one cell in
    cfg.beta_sweep.pairs_representative, so 'pooled' here is just the
    seed-level median for that single cell. Layout matches the request:
    accuracy block + cross-entropy block.
    """
    df = _filter_beta(summary, cfg)
    cells = cfg.beta_sweep["pairs_representative"]
    betas = list(cfg.beta_sweep["beta_mins"])

    # Build (dataset, quality) -> cell mapping.
    by_quality: dict[tuple[str, str], dict] = {}
    for c in cells:
        q = _quality_for_pair(cfg, c["surrogate"], c["target"])
        if q is not None:
            by_quality[(c["dataset"], q)] = c
    cols = [(ds, q) for ds in _DATASETS for q in ("weak", "strong")
            if (ds, q) in by_quality]
    n_cols = len(cols)
    col_spec = "l" + "r" * n_cols

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"$\beta_{\min}$ & "
        + " & ".join(rf"{_DATASET_DISPLAY[ds]} ({q})" for ds, q in cols) + r" \\",
        r"\midrule",
    ]
    for loss_key, loss_label in _LOSS_DISPLAY.items():
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textbf{{{loss_label}}}}} \\")
        lines.append(r"\midrule")
        for beta in betas:
            row_cells = []
            for ds, q in cols:
                cell = by_quality[(ds, q)]
                slc = df[
                    (df.loss == loss_key)
                    & (df.dataset == cell["dataset"])
                    & (df.surrogate == cell["surrogate"])
                    & (df.target == cell["target"])
                    & (df.beta_min == beta)
                ]
                med = _median_labels(slc)
                row_cells.append(f"{int(med):,}" if np.isfinite(med) else "--")
            lines.append(f"{beta:.1f} & " + " & ".join(row_cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _pair_label(surrogate: str, target: str) -> str:
    """Pretty surrogate→target string for legends."""
    pretty = {
        "llama2_7b":    "Llama2-7B",
        "llama3_8b":    "Llama3-8B",
        "llama3_70b":   "Llama3-70B",
        "qwen25_72b":   "Qwen2.5-72B",
        "deepseek_67b": "DeepSeek-67B",
        "Mixtral_8x7b": "Mixtral-8×7B",
    }
    s = pretty.get(surrogate, surrogate.replace("_", "-"))
    t = pretty.get(target, target.replace("_", "-"))
    return rf"{s} $\to$ {t}"


def render_beta_figure(summary: pd.DataFrame, cfg, out_path: Path) -> None:
    """1 × 3 grid (datasets as columns), 0-1 loss only. Each panel overlays
    the two representative surrogate-target pairs for that dataset; legend
    labels carry the surrogate→target identity (no weak/strong text).
    Vertical hairline at deployed β=0.4."""
    apply_rc_helvetica()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.titleweight": "semibold",
        "axes.labelsize": 17,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })
    df = _filter_beta(summary, cfg)
    cells = cfg.beta_sweep["pairs_representative"]
    betas = list(cfg.beta_sweep["beta_mins"])

    # Order cells per dataset: lower-quality surrogate first (red, circle),
    # higher-quality second (teal, triangle). Drives only color/marker;
    # the legend always reads as the actual surrogate→target identity.
    by_dataset: dict[str, list[dict]] = {ds: [] for ds in _DATASETS}
    for c in cells:
        q = _quality_for_pair(cfg, c["surrogate"], c["target"])
        if q is None:
            continue
        if q == "weak":
            by_dataset[c["dataset"]].insert(0, c)
        else:
            by_dataset[c["dataset"]].append(c)

    slot_colors = ["#B64A3B", "#2B7A78"]
    slot_markers = ["o", "^"]

    fig, axes = plt.subplots(
        1, len(_DATASETS),
        figsize=(12.6, 3.4),
        sharex=True, sharey=False,
    )
    deployed_beta = 0.4

    for c_idx, ds in enumerate(_DATASETS):
        ax = axes[c_idx]
        for slot, cell in enumerate(by_dataset[ds]):
            ys = []
            for beta in betas:
                slc = df[
                    (df.loss == "accuracy")
                    & (df.dataset == cell["dataset"])
                    & (df.surrogate == cell["surrogate"])
                    & (df.target == cell["target"])
                    & (df.beta_min == beta)
                ]
                ys.append(_median_labels(slc))
            ax.plot(
                betas, ys,
                color=slot_colors[slot],
                marker=slot_markers[slot],
                linewidth=2.0,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=_pair_label(cell["surrogate"], cell["target"]),
                zorder=3,
            )
        ax.axvline(deployed_beta, color="#999999",
                   linewidth=0.9, linestyle="--", zorder=1)
        ax.set_title(_DATASET_DISPLAY[ds], pad=6)
        ax.set_xlabel(r"$\beta$")
        if c_idx == 0:
            ax.set_ylabel("0-1 loss\n" + r"median labels to $\epsilon$")
        ax.set_xticks(betas)
        ax.tick_params(axis="x", which="both", length=2.5)
        ax.tick_params(axis="y", which="both", length=2.5)
        ax.legend(loc="best", borderaxespad=0.4, handlelength=1.6)

    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.085, right=0.995,
                        wspace=0.20)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root (where beta_appendix/ is written).")
    ap.add_argument("--summary-root", type=Path, default=None,
                    help="Override summary.csv source root. Defaults to "
                         "results/paper_experiment (where beta_sweep rows live).")
    args = ap.parse_args(argv)

    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    summary_root = args.summary_root if args.summary_root else (_REPO / "results" / "paper_experiment")
    summary = pd.read_csv(summary_root / "summary.csv")

    out_dir = out_root / "beta_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_beta.tex").write_text(build_beta_table(summary, cfg))
    (out_dir / "tab_beta_pooled.tex").write_text(build_beta_pooled_table(summary, cfg))
    render_beta_figure(summary, cfg, out_dir / "fig_beta.pdf")
    print(f"wrote beta_appendix artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
