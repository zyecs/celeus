"""Compute pool-level MAE axis for §6.5 item #1 and emit the S3 decision gate."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.config import default_config_path, load_config  # noqa: E402
from save.paper_experiment.pool_loader import load_pool_for_cell  # noqa: E402

LOG = logging.getLogger(__name__)

# Module-level constants for gate thresholds.
LOG10_SEP_MIN = 0.2  # Pass A threshold (log10 MAE units)
TAU_MIN = 0.4  # Pass B Kendall tau threshold (signed)
TERTILE_PCTS = (33.33, 66.67)

DEPLOYED = {
    "accuracy": "remark2_strategy4",
    "cross_entropy": "remark1_strategy2",
}

# Trajectory file path scheme used by RQ6 per_cell_curves.csv (cell column).
# {out_root} is parameterized so --out-root overrides work correctly (Opus MAJOR #4).
_V_T_CELL_PATH_FMT = (
    "{out_root}/trajectories/main/"
    "cell__M1__{dataset}__{surrogate}__{target}__accuracy.npz"
)


@dataclass(frozen=True)
class GateResult:
    passes: bool
    label: str
    diagnostics: dict


def _compute_pool_mae(cfg, dataset, surrogate, target, loss, *, axis: str = "mae_lure"):
    """Compute MAE under the deployed strategy.

    axis="mae_lure": E|ground_truth - surrogate_score|  (LURE-aligned, target-driven for S4)
    axis="mae_acq":  E|ell_proxy - surrogate_score|     (acquisition-aligned, surrogate-dependent for S4)
    """
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=dataset, surrogate=surrogate, target=target,
        loss=loss, surrogate_type=DEPLOYED[loss],
        ce_nll_filter=cfg.ce_nll_filter if loss == "cross_entropy" else None,
    )
    if axis == "mae_lure":
        return float(np.mean(np.abs(pool.ground_truth_losses - pool.surrogate_scores)))
    elif axis == "mae_acq":
        if pool.ell_proxy is None:
            return float("nan")
        return float(np.mean(np.abs(pool.ell_proxy - pool.surrogate_scores)))
    else:
        raise ValueError(f"unknown axis {axis!r}")


def load_cell_arrays(cfg, dataset, surrogate, target, *, loss: str = "accuracy") -> dict[str, np.ndarray]:
    """Load the RQ7 per-item arrays for one deployed-strategy cell.

    Returns 1D arrays named:
      - ell_proxy: acquisition proxy used by the deployed strategy.
      - hat_ell: target-only baseline score used in the LURE residual.
      - ell: finite-pool ground-truth loss.

    This intentionally reuses the same pool-loader path as ``_compute_pool_mae``
    so RQ7 sees exactly the same deployed-strategy arrays as the MAE audit.
    """
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=dataset,
        surrogate=surrogate,
        target=target,
        loss=loss,
        surrogate_type=DEPLOYED[loss],
        ce_nll_filter=cfg.ce_nll_filter if loss == "cross_entropy" else None,
    )
    ell_proxy = getattr(pool, "ell_proxy", None)
    if ell_proxy is None:
        raise ValueError(
            f"pool.ell_proxy is missing for {dataset}/{surrogate}/{target}/{loss} "
            f"under deployed strategy {DEPLOYED[loss]}"
        )
    return {
        "ell_proxy": np.asarray(ell_proxy, dtype=np.float64).reshape(-1),
        "hat_ell": np.asarray(pool.surrogate_scores, dtype=np.float64).reshape(-1),
        "ell": np.asarray(pool.ground_truth_losses, dtype=np.float64).reshape(-1),
    }


def _per_cell_width_at_stop_from(summary_df: pd.DataFrame,
                                 dataset, surrogate, target, loss) -> float:
    sub = summary_df[(summary_df.method == "M1")
                     & (summary_df.dataset == dataset)
                     & (summary_df.surrogate == surrogate)
                     & (summary_df.target == target)
                     & (summary_df.loss == loss)]
    return float(sub.width_at_stop.median())


def _per_cell_v_t(rq6_csv: Path, out_root: Path, dataset, surrogate, target) -> float:
    """Median V_t over t in [500, t_max] from rq6-variance accuracy cells.
    Returns NaN if not present (e.g., CE cells)."""
    if not rq6_csv.is_file():
        return float("nan")
    df = pd.read_csv(rq6_csv)
    df = df[df.acquisition == "ada"]
    # Normalize out_root: drop trailing slashes via Path(...), then make relative
    # to _REPO if possible so the path matches how cell paths are stored in the CSV
    # (which uses repo-relative paths when the RQ6 script is run from the repo root).
    norm_root = Path(out_root)
    try:
        norm_root = norm_root.resolve().relative_to(_REPO)
    except ValueError:
        pass  # out_root is outside repo; use as-is (absolute)
    expected_cell = _V_T_CELL_PATH_FMT.format(
        out_root=str(norm_root),
        dataset=dataset, surrogate=surrogate, target=target,
    )
    sub = df[(df.cell == expected_cell) & (df.t >= 500)]
    if sub.empty:
        return float("nan")
    return float(sub.cond_var_S_mean.median())


def _tertile_bin_means(log_mae: np.ndarray):
    """Bin log10(MAE) into tertiles via np.searchsorted with explicit ties policy.

    Cuts are the (33.33, 66.67) percentiles of the sorted values. Ties at a cut
    go to the lower bin (side='right') so the partition is unambiguous and the
    tertile counts are as balanced as possible. Returns (means, sizes) where
    means[i] = mean of bin i, sizes[i] = count in bin i; means[i] is NaN if the
    bin is empty.
    """
    cuts = np.percentile(log_mae, list(TERTILE_PCTS))
    # side='right': values equal to a cut go to the LOWER bin.
    idx = np.searchsorted(cuts, log_mae, side="right")
    means = np.full(3, np.nan, dtype=float)
    sizes = np.zeros(3, dtype=int)
    for b in range(3):
        bucket = log_mae[idx == b]
        sizes[b] = bucket.size
        if bucket.size:
            means[b] = float(bucket.mean())
    return means, sizes


def decide_gate(df: pd.DataFrame) -> GateResult:
    """Pass A (log10 separation > LOG10_SEP_MIN between adjacent tertile bins) AND
    Pass B (signed Kendall tau >= TAU_MIN for width; CE skips V_t arm of B).

    Slices with degenerate inputs (empty/under-populated tertile bins, or NaN
    Kendall tau from a constant input) are reported as INSUFFICIENT_VARIATION
    rather than treated as a true Pass A/B failure. If any slice is degenerate,
    the overall label is INSUFFICIENT_VARIATION_<slices>.
    """
    diagnostics = {}
    fails = []
    insufficient = []
    for (ds, loss), slc in df.groupby(["dataset", "loss"]):
        log_mae = np.log10(slc.MAE.values)
        means, sizes = _tertile_bin_means(log_mae)
        bin_lo, bin_mid, bin_hi = means
        sep_lo_mid = bin_mid - bin_lo
        sep_mid_hi = bin_hi - bin_mid

        # Degenerate tertile binning: any empty bin or fewer than 2 elements.
        bins_degenerate = bool(np.any(sizes < 2))

        tau_w, _ = kendalltau(slc.MAE, slc.width_at_stop_med)
        if loss == "accuracy":
            tau_v, _ = kendalltau(slc.MAE, slc.V_t_med)
            tau_v_required = True
        else:
            tau_v = float("nan")
            tau_v_required = False

        # NaN tau (constant input) is degenerate, not a Pass B failure.
        tau_w_degenerate = bool(np.isnan(tau_w))
        tau_v_degenerate = tau_v_required and bool(np.isnan(tau_v))

        diagnostics[f"{ds}_{loss}"] = {
            "log_mae_sep_lo_mid": float(sep_lo_mid),
            "log_mae_sep_mid_hi": float(sep_mid_hi),
            "tertile_sizes": [int(s) for s in sizes],
            "tau_width": float(tau_w),
            "tau_V_t": float(tau_v),
        }

        if bins_degenerate or tau_w_degenerate or tau_v_degenerate:
            insufficient.append(f"{ds}_{loss}")
            continue

        pass_a = (sep_lo_mid > LOG10_SEP_MIN) and (sep_mid_hi > LOG10_SEP_MIN)
        # Signed Kendall tau: negative correlation (lower MAE -> wider widths)
        # is the wrong direction and is a bug signal, not a pass.
        if tau_v_required:
            pass_b = (tau_w >= TAU_MIN) and (tau_v >= TAU_MIN)
        else:
            pass_b = tau_w >= TAU_MIN

        if not (pass_a and pass_b):
            fails.append(f"{ds}_{loss}")

    if insufficient:
        return GateResult(False,
                          f"INSUFFICIENT_VARIATION_{'+'.join(insufficient)}",
                          diagnostics)
    if not fails:
        return GateResult(True, "NATURAL_GRADIENT_OK", diagnostics)
    return GateResult(False, f"ADD_EXTREMES_{'+'.join(fails)}", diagnostics)


def emit_extreme_pairs_yaml(out_path: Path) -> None:
    """Write 6 extreme-MAE cells (3 datasets x 2 regimes)."""
    cells = [
        {"dataset": ds, "kind": kind, "surrogate": tgt, "target": tgt}
        for ds in ("sst2", "mmlu", "agnews")
        for kind, tgt in (("self_as_surrogate", "llama3_70b"),
                          ("shuffled_target",   "llama3_70b"))
    ]
    out_path.write_text(yaml.safe_dump({"extreme_pairs": cells}, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(default_config_path()))
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root.")
    ap.add_argument("--paper-pairs", action="store_true",
                    help="Filter to cfg.paper_pair_keys (v0502 scope). Default: all_pairs.")
    ap.add_argument("--axis", choices=("mae_lure", "mae_acq"), default="mae_lure",
                    help="MAE flavor: mae_lure = E|gt - hat_ell|; mae_acq = E|ell_proxy - hat_ell|.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    rq6_csv = out_root / "rq6-variance/accuracy/per_cell_curves.csv"
    summary_df = pd.read_csv(out_root / "summary.csv")

    pairs = cfg.paper_pairs if args.paper_pairs else cfg.all_pairs

    rows = []
    for ds in cfg.datasets:
        for pair in pairs:
            for loss in cfg.losses:
                rows.append({
                    "dataset": ds, "surrogate": pair["surrogate"],
                    "target": pair["target"], "loss": loss,
                    "MAE": _compute_pool_mae(
                        cfg, ds, pair["surrogate"], pair["target"], loss,
                        axis=args.axis,
                    ),
                    "deployed_strategy": DEPLOYED[loss],
                    "width_at_stop_med": _per_cell_width_at_stop_from(
                        summary_df, ds, pair["surrogate"], pair["target"], loss),
                    "V_t_med": _per_cell_v_t(
                        rq6_csv, out_root, ds, pair["surrogate"], pair["target"]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "mae_axis.csv", index=False)

    # gate decision (only meaningful for the legacy axis)
    if args.axis == "mae_lure":
        res = decide_gate(df)
        payload = {
            "label": res.label,
            "passes": res.passes,
            "diagnostics": res.diagnostics,
        }
        # First line is the bare label so the existing read-and-grep idiom works;
        # blank line then JSON blob with full structured details.
        (out_root / "mae_audit_decision.txt").write_text(
            res.label + "\n\n" + json.dumps(payload, indent=2, default=float) + "\n"
        )
        if not res.passes:
            emit_extreme_pairs_yaml(out_root / "extreme_pairs.yaml")
        LOG.info("MAE audit: %s", res.label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
