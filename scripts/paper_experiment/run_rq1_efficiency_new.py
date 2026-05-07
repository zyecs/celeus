#!/usr/bin/env python3
"""Plan + render the RQ1 label-efficiency artifacts for the CELEUS-anchored
new variant.

Sibling of ``run_rq1_efficiency.py``. Differences:

- Methods include ``M3`` (CELEUS w/o surrogate).
- No Cer-Eval scope.
- No oracle planner gating: failures surface as ``FileNotFoundError`` /
  ``RuntimeError`` from ``common.load_selected_seed_trajectories`` when the
  rendering call runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from save.paper_experiment.config import default_config_path, load_config  # noqa: E402

import plot_rq1_efficiency_new as plot_mod  # noqa: E402
import rq1_efficiency_common as common  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=default_config_path(),
        help="Main paper_experiment config (must contain paper_pairs).",
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502",
        help="Root holding merged main paper_experiment trajectories.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502" / "rq1-efficiency-new",
        help="Output directory for the new RQ1 artifacts.",
    )
    parser.add_argument(
        "--loss", choices=("accuracy", "cross_entropy"), default="accuracy",
        help="Which loss to render (default: accuracy). CE drops the Oracle "
             "method since no oracle_cross_entropy stage exists.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan-only", action="store_true",
                       help="Only verify input cells and write the readiness "
                            "manifest; do not render the figure.")
    group.add_argument("--plot-only", action="store_true",
                       help="Skip planning messages and render immediately.")
    return parser.parse_args(argv)


def _readiness_manifest(
    *, cfg_main, input_root: Path, output_root: Path,
) -> dict[str, object]:
    """Walk ``METHOD_ORDER`` × ``cfg.paper_pairs`` × ``selected datasets``
    and record cell status. ``ready=True`` only when every selected cell
    exists with the full expected seed set."""
    pairs = tuple(
        common.SelectedPair(p["surrogate"], p["target"])
        for p in cfg_main.paper_pairs
    )
    datasets = common.selected_dataset_names(cfg_main)

    statuses: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for method in plot_mod.METHOD_ORDER:
        for status in common.collect_selected_cell_statuses(
            cfg_main, input_root, method, datasets=datasets, pairs=pairs,
        ):
            entry = common.status_to_dict(status)
            entry["method"] = method
            statuses.append(entry)
            if not status.complete:
                missing.append(entry)

    ready = len(missing) == 0
    manifest = {
        "variant": "rq1-efficiency-new",
        "selected_datasets": list(datasets),
        "selected_pairs": [
            {"surrogate": p.surrogate, "target": p.target} for p in pairs
        ],
        "methods": list(plot_mod.METHOD_ORDER),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "selected_cell_statuses": statuses,
        "missing_or_incomplete": missing,
        "ready": ready,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rq1_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_main = load_config(args.config)

    if args.plan_only:
        manifest = _readiness_manifest(
            cfg_main=cfg_main,
            input_root=args.input_root,
            output_root=args.output_root,
        )
        if not manifest["ready"]:
            print(
                f"RQ1-new inputs not ready ({len(manifest['missing_or_incomplete'])} "
                f"missing/incomplete cells). See "
                f"{args.output_root / 'rq1_manifest.json'}"
            )
            return 1
        print(f"wrote RQ1-new readiness manifest to {args.output_root / 'rq1_manifest.json'}")
        return 0

    plot_mod.build_rq1_artifacts(
        cfg_main=cfg_main,
        input_root=args.input_root,
        output_root=args.output_root,
        validate_only=False,
        loss=args.loss,
    )
    print(f"wrote RQ1 (new variant, {args.loss}) artifacts to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
