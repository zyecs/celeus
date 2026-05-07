#!/usr/bin/env python3
# scripts/paper_experiment/verify_smoke.py
"""Stage 1 smoke-test pass checklist (spec §9 Stage 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.cell_paths import main_cell_path  # noqa: E402
from save.paper_experiment.cell_schema import load_cell  # noqa: E402
from save.paper_experiment.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root", default=None,
        help="Override cfg.paths['out_root'] (for testing).",
    )
    args = p.parse_args(argv)

    cfg = load_config(_REPO / "configs" / "paper_experiment.yaml")
    base = Path(args.out_root) if args.out_root else Path(cfg.paths["out_root"])
    failed = []
    s = cfg.smoke

    # (1) expected cell files exist
    for method in s["methods"]:
        path = main_cell_path(base, method, s["dataset"], s["surrogate"], s["target"], s["loss"])
        if not path.exists():
            failed.append(f"missing cell: {path}")

    if failed:
        for f in failed:
            print(f)
        return 1

    # Load everything.
    all_ok = True
    r_n_path = base / "R_N.csv"
    if not r_n_path.exists():
        print("missing R_N.csv — run --stage compute-rn first")
        return 1
    import csv
    r_n: dict[tuple, dict] = {}
    with open(r_n_path) as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["target"], row["loss"])
            kept_raw = row.get("ce_nll_filter_kept")
            if kept_raw in (None, "", "-1"):
                kept = None
            else:
                kept = int(kept_raw)
            r_n[key] = {
                "R_N": float(row["R_N"]),
                "enabled": (
                    row.get("ce_nll_filter_enabled", "False").lower() == "true"
                ),
                "kept": kept,
            }

    elapsed_samples = []
    pool_shas = {}
    for method in s["methods"]:
        path = main_cell_path(base, method, s["dataset"], s["surrogate"], s["target"], s["loss"])
        meta, results = load_cell(path)
        rn_row = r_n[(meta.dataset, meta.target, meta.loss)]
        rn = rn_row["R_N"]
        # Conditional filter-kept audit — only when filter active AND cell recorded a count.
        if rn_row["enabled"] and meta.ce_nll_filter_kept is not None:
            if meta.ce_nll_filter_kept != rn_row["kept"]:
                print(
                    f"{method}: ce_nll_filter_kept mismatch — cell="
                    f"{meta.ce_nll_filter_kept}, R_N.csv={rn_row['kept']}"
                )
                all_ok = False
        # (4) pool identity: every seed in a cell must share the same pool_sha256.
        pool_shas[method] = meta.pool_sha256
        if not meta.pool_sha256:
            print(f"{method}: meta.pool_sha256 is empty (expected SHA-256 hex)")
            all_ok = False
        for seed, r in results.items():
            # (2) coverage at every step
            lo, hi = r.save_lo, r.save_hi
            bad = np.sum((lo > rn) | (hi < rn))
            if bad > 0:
                print(f"{method} seed={seed}: miscovers R_N={rn:.4f} at {bad} steps")
                all_ok = False
            # (3) schema shape
            assert r.save_labels.shape == (meta.T_max,)
            # (5) git_commit + hostname populated
            if r.git_commit in ("", "unknown"):
                print(f"{method} seed={seed}: git_commit missing")
                all_ok = False
            if r.hostname == "":
                print(f"{method} seed={seed}: hostname missing")
                all_ok = False
            # (7) timing collected
            if r.elapsed_seconds <= 0:
                print(f"{method} seed={seed}: elapsed_seconds not recorded")
                all_ok = False
            elapsed_samples.append(r.elapsed_seconds)

    # Item 4 (spec §9 Stage 1): pool identity must be consistent within each
    # surrogate-type group. pool_sha256 incorporates surrogate_scores, so
    # methods with different surrogate_type_* legitimately hash differently
    # (v2.1 plan deferred item c). Group by surrogate_type_<loss> and check
    # within-group consistency; divergence within a group signals a
    # non-deterministic pool load.
    loss_key = f"surrogate_type_{s['loss']}"
    groups: dict[str, dict[str, str]] = {}
    for method in s["methods"]:
        st = cfg.methods[method].get(loss_key) or "none"
        groups.setdefault(st, {})[method] = pool_shas[method]
    for st, method_shas in groups.items():
        if len(set(method_shas.values())) > 1:
            print(f"Pool SHA inconsistency within surrogate_type={st}: {method_shas}")
            all_ok = False

    # Item 7 (spec §9 Stage 1): extrapolate whole-campaign wall-clock.
    if elapsed_samples:
        import statistics
        avg = statistics.mean(elapsed_samples)
        campaign_trajectories = 15000 + 450 + 1800  # spec §3 grid totals
        total_h = avg * campaign_trajectories / 3600.0
        print(f"Per-trajectory avg elapsed: {avg:.1f} s; campaign estimate ~{total_h:,.0f} CPU-h")

    if not all_ok:
        return 2
    print("Stage 1 smoke checklist PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
