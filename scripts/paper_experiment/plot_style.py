"""Shared plot styles for paper-experiment figures."""
from __future__ import annotations

METHOD_STYLE = {
    "M1": {"label": "SAVE-ADA",          "color": "#d62728", "linestyle": "-",  "marker": "o"},
    "M2": {"label": "SAVE-fixed",        "color": "#d62728", "linestyle": "--", "marker": "s"},
    "M3": {"label": "save-no-surrogate", "color": "#7f7f7f", "linestyle": ":",  "marker": "^"},
    "M4": {"label": "naive e-value",     "color": "#000000", "linestyle": "--", "marker": "x"},
    "M5": {"label": "Cer-Eval",          "color": "#1f77b4", "linestyle": "-",  "marker": "D"},
}
ALPHA = 0.05
LOSS_ORDER = ("accuracy", "cross_entropy")
DATASET_ORDER = ("sst2", "mmlu", "agnews")


def apply_rc():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# rq4-rq6 plotting enhancement (spec 2026-04-25, commit 4c5596c).
# Tol-muted palette, semantic acquisition / kind / pair mappings,
# Helvetica-family typography via TeX-Gyre-Heros fallback chain.
# ---------------------------------------------------------------------------

import logging

_log = logging.getLogger(__name__)

TOL_MUTED = {
    "indigo": "#332288",   "green":  "#117733",
    "teal":   "#44AA99",   "sky":    "#88CCEE",
    "sand":   "#DDCC77",   "wine":   "#CC6677",
    "purple": "#AA4499",   "grey":   "#888888",
}

# Semantic mapping for rq4/rq5/rq6 acquisition policies.
# Color stays constant across sections; LURE-vs-unweighted is encoded
# by linestyle (KIND_LINESTYLE), not by hue.
ACQUISITION_STYLE = {
    "ada":             {"color": TOL_MUTED["indigo"], "label": "ADA (ours)"},
    "oracle_accuracy": {"color": TOL_MUTED["sand"],   "label": "Oracle"},
    "uniform":         {"color": "#000000",            "label": "Uniform"},
}

KIND_LINESTYLE = {
    "lure":         {"linestyle": "-",  "linewidth": 1.4, "label": "LURE-weighted"},
    "unweighted":   {"linestyle": "--", "linewidth": 1.2, "label": "Unweighted"},
    "is_corrected": {"linestyle": "-",  "linewidth": 1.4, "label": r"$\hat{S}_t$ IS-corrected"},
    "naive":        {"linestyle": "--", "linewidth": 1.2, "label": r"naive $\ell_{i_t}$"},
}

# Hand-picked demo pairs used in showcase.pdf views — same 3 pairs across all
# 3 datasets and all 4 sections. CIE Lab L* gaps: indigo↔sand=60, sand↔wine=26,
# indigo↔wine=33 — all >15, satisfying the §4.3 greyscale luminance test.
#
# Sand is reused from ACQUISITION_STYLE["oracle_accuracy"] but never collides
# inside showcase: showcase legends label sand as the model-pair; pooled
# legends label sand as the oracle policy. The two figure types are mutually
# exclusive on the page; readers see only one legend at a time.
PAIR_DEFS = [
    {"slot": "cross_arch", "surrogate": "llama2_7b",  "target": "Mixtral_8x7b",
     "color": TOL_MUTED["indigo"], "label": "llama2-7b → Mixtral-8×7b"},
    {"slot": "weak",       "surrogate": "llama3_8b",  "target": "qwen25_72b",
     "color": TOL_MUTED["sand"],   "label": "llama3-8b → qwen2.5-72b"},
    {"slot": "strong",     "surrogate": "qwen25_72b", "target": "llama3_70b",
     "color": TOL_MUTED["wine"],   "label": "qwen2.5-72b → llama3-70b"},
]

BAND_ALPHA = 0.18


_HELVETICA_FALLBACK = [
    "Helvetica",
    "TeX Gyre Heros",
    "Liberation Sans",
    "Nimbus Sans",
    "DejaVu Sans",
]


def apply_rc_helvetica():
    """Apply Nature-style rcparams using Helvetica-family font.

    Resolves the first available font from `[$SAVE_PLOT_FONT?] + _HELVETICA_FALLBACK`
    and applies 7pt body / 6pt ticks / 0.6pt axes / spines top+right off / no grid /
    pdf.fonttype=42 for editor-editable PDFs / 600dpi save.

    `regen_plots_only.py --font "Liberation Sans"` propagates the choice via
    the `SAVE_PLOT_FONT` environment variable, which we prepend to the chain.

    Logs the resolved font name at INFO level so authors notice if matplotlib's
    font cache resolves differently than expected.

    Returns the resolved font family name.
    """
    import os
    import matplotlib as mpl
    from matplotlib import font_manager

    user_font = os.environ.get("SAVE_PLOT_FONT")
    chain = ([user_font] if user_font else []) + _HELVETICA_FALLBACK
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((c for c in chain if c in available), "DejaVu Sans")
    chain_pos = chain.index(family) + 1 if family in chain else len(chain)
    _log.info(
        "[plot_style] font resolved to: %s (chain position %d of %d, SAVE_PLOT_FONT=%r)",
        family, chain_pos, len(chain), user_font,
    )

    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    [family] + chain,
        "font.size":          7.0,
        "axes.titlesize":     8.0,
        "axes.labelsize":     7.0,
        "legend.fontsize":    6.0,
        "xtick.labelsize":    6.0,
        "ytick.labelsize":    6.0,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.frameon":     False,
        "legend.handlelength": 1.6,
        "lines.linewidth":    1.2,
        "lines.markersize":   2.5,
        "figure.dpi":         150,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })
    return family


_DATASET_POS = {
    "topleft":  (0.02, 0.95, "top",    "left"),
    "topright": (0.98, 0.95, "top",    "right"),
}


def finalize_panel(ax, *, dataset: str = "", ylabel=None,
                   show_zero_line: bool = False,
                   xlabel=None,
                   dataset_pos: str = "topleft") -> None:
    """Apply Nature-style polish to every panel.

    - Dataset annotation (8pt bold) placed per `dataset_pos` ('topleft' or
      'topright'). Use 'topright' when curves are downward-sloping so the
      label doesn't overlap them.
    - Optional zero line at y=0 (thin grey).
    - Minor ticks on; no grid (Nature convention).
    - Tick params match apply_rc_helvetica.
    """
    if show_zero_line:
        ax.axhline(0.0, color=TOL_MUTED["grey"], lw=0.4, zorder=0)
    if dataset:
        x_anchor, y_anchor, va, ha = _DATASET_POS[dataset_pos]
        ax.text(x_anchor, y_anchor, dataset, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va=va, ha=ha)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.tick_params(direction="out", length=2.5, width=0.6)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=1.4, width=0.4)
    ax.grid(False)


# --- runtime overrides -----------------------------------------------------
# regen_plots_only.py propagates --font / --no-bands / --pairs through these
# environment variables; plot scripts and apply_rc_helvetica honour them.

def bands_enabled() -> bool:
    """Return False iff `SAVE_PLOT_NO_BANDS` is truthy ('1', 'true', etc.)."""
    import os
    return os.environ.get("SAVE_PLOT_NO_BANDS", "").lower() not in {"1", "true", "yes"}


def get_runtime_pairs(default_pairs: list[dict] = PAIR_DEFS) -> list[dict]:
    """If `SAVE_PLOT_PAIRS` env var is set (comma-separated slot names),
    return the subset of `default_pairs` whose `slot` is in that set;
    otherwise return all `default_pairs`.

    Slot names match `PAIR_DEFS[*]['slot']`: cross_arch / weak / strong.
    """
    import os
    spec = os.environ.get("SAVE_PLOT_PAIRS", "").strip()
    if not spec:
        return list(default_pairs)
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    return [p for p in default_pairs if p["slot"] in wanted]


# Color cycle for v0502 paper_pair_defs() — 4 muted-Tol colors that work
# alongside ACQUISITION_STYLE (which uses indigo/sand for ada/oracle).
_PAPER_PAIR_COLORS = [
    TOL_MUTED["indigo"],
    TOL_MUTED["wine"],
    TOL_MUTED["green"],
    TOL_MUTED["purple"],
]


def paper_pair_defs(paper_pairs: list[dict]) -> list[dict]:
    """Build PAIR_DEFS-shaped showcase entries from cfg.paper_pairs.

    Each output entry has the keys consumed by _render_showcase / per_pair_filter:
    slot, surrogate, target, color, label. Slot names use the index (pair0, …).
    Colors cycle through _PAPER_PAIR_COLORS; if more than 4 paper_pairs are
    supplied, colors repeat (the showcase can comfortably fit ~3–4 lines per
    panel, so this is rarely an issue in practice).
    """
    out: list[dict] = []
    for i, p in enumerate(paper_pairs):
        surr = p["surrogate"]
        tgt = p["target"]
        out.append({
            "slot": f"pair{i}",
            "surrogate": surr,
            "target": tgt,
            "color": _PAPER_PAIR_COLORS[i % len(_PAPER_PAIR_COLORS)],
            "label": f"{surr} → {tgt}",
        })
    return out
