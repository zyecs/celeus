#!/usr/bin/env python3
"""Plan and render the standalone RQ1 efficiency artifact set."""
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

import plot_rq1_efficiency as plot_mod  # noqa: E402
import rq1_efficiency_common as common  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=default_config_path(),
        help="Main paper_experiment config.",
    )
    parser.add_argument(
        "--cereval-config", type=Path,
        default=_REPO / "configs" / "paper_experiment_cereval_3seed_cpu.yaml",
        help="Cer-Eval config.",
    )
    parser.add_argument(
        "--paper-root", type=Path, default=None,
        help="Override the canonical results/paper_experiment root.",
    )
    parser.add_argument(
        "--cereval-input-root", type=Path,
        default=_REPO / "results" / "paper_experiment_cereval_3seed_cpu",
        help="Cer-Eval input root.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_REPO / "results" / "paper_experiment" / "rq1-efficiency",
        help="Output directory for RQ1 artifacts (singular paper_experiment).",
    )
    parser.add_argument(
        "--skip-cereval", action="store_true",
        help="Skip the Cer-Eval scope entirely (planner + figure).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan-only", action="store_true",
                       help="Only write the RQ1 manifest; do not render.")
    group.add_argument("--plot-only", action="store_true",
                       help="Skip planning messages and render immediately.")
    parser.add_argument(
        "--paper-pairs", action="store_true",
        help="Filter cells to cfg.paper_pair_keys (v0502 scope).",
    )
    return parser.parse_args(argv)


def _oracle_requirements(cfg, paper_root: Path):
    statuses = common.collect_selected_cell_statuses(cfg, paper_root, "ORACLE_ACC")
    manifest = common.oracle_stage_manifest(cfg)
    manifest["selected_cell_statuses"] = [
        common.status_to_dict(status) for status in statuses
    ]
    manifest["selected_cells_ready"] = all(status.complete for status in statuses)
    manifest["selected_missing_or_incomplete"] = [
        common.status_to_dict(status) for status in statuses if not status.complete
    ]
    manifest["execution_rule"] = "full_stage_rerun_if_any_selected_oracle_cell_is_missing_or_incomplete"
    return statuses, manifest


def _cereval_repair_target_to_dict(target, cereval_config_path: Path) -> dict:
    """Like ``common.repair_target_to_dict`` but the merge command embeds
    ``--config <cereval cfg>`` so the index is interpreted against the cereval
    cfg's iteration order. The default ``run_paper_experiment.py --config`` is
    the main paper_experiment cfg, which would resolve a different cell at the
    same index."""
    from dataclasses import asdict
    data = asdict(target)
    data["run_commands"] = [
        f"python scripts/run_paper_experiment.py --config {cereval_config_path} "
        f"--stage cereval-fresh --index {idx}"
        for idx in target.run_indices
    ]
    data["merge_command"] = (
        f"python scripts/run_paper_experiment.py --config {cereval_config_path} "
        f"--stage merge-cells --index {target.merge_index}"
    )
    return data


def build_rq1_plan_manifest(
    *,
    cfg_main,
    cfg_cereval,
    cereval_config_path: Path,
    paper_root: Path,
    cereval_input_root: Path,
    output_root: Path,
    skip_cereval: bool = False,
) -> dict[str, object]:
    main_repair_targets = common.collect_selected_main_repair_targets(cfg_main, paper_root)
    _, oracle_manifest = _oracle_requirements(cfg_main, paper_root)

    next_steps: list[str] = []
    for target in main_repair_targets:
        next_steps.extend(
            [
                f"rerun full main-accuracy cell: {target.method_id} "
                f"{target.dataset} {target.surrogate}->{target.target}",
                *[
                    f"  python scripts/run_paper_experiment.py --stage main-accuracy --index {idx}"
                    for idx in target.run_indices
                ],
                f"  python scripts/run_paper_experiment.py --stage merge-cells --index {target.merge_index}",
            ]
        )
    if not oracle_manifest["selected_cells_ready"]:
        next_steps.extend(
            [
                "run the full oracle stage because at least one selected oracle cell is missing or incomplete",
                "  python scripts/run_paper_experiment.py --stage oracle-accuracy",
                "  python scripts/run_paper_experiment.py --stage merge-oracle-accuracy",
            ]
        )

    cereval_block: dict[str, object] | None = None
    if not skip_cereval:
        cereval_targets = common.collect_selected_cereval_repair_targets(
            cfg_cereval, cereval_input_root,
        )
        cereval_block = {
            "config_path": str(cereval_config_path),
            "input_root": str(cereval_input_root),
            "seeds_expected": [int(s) for s in cfg_cereval.seeds_main],
            "repairs": [
                _cereval_repair_target_to_dict(t, cereval_config_path)
                for t in cereval_targets
            ],
            "ready": len(cereval_targets) == 0,
        }
        for target in cereval_targets:
            next_steps.extend(
                [
                    f"merge cereval cell: {target.method_id} "
                    f"{target.dataset} {target.surrogate}->{target.target}",
                    f"  python scripts/run_paper_experiment.py "
                    f"--config {cereval_config_path} "
                    f"--stage merge-cells --index {target.merge_index}",
                ]
            )

    if not next_steps:
        next_steps.append("inputs are complete; run plot-only or the default command to render the figures")

    plot_ready = (
        not main_repair_targets
        and bool(oracle_manifest["selected_cells_ready"])
        and (skip_cereval or (cereval_block is not None and cereval_block["ready"]))
    )
    manifest = {
        "selected_datasets": list(common.selected_dataset_names(cfg_main)),
        "selected_pairs": [
            {"surrogate": pair.surrogate, "target": pair.target}
            for pair in common.selected_pairs()
        ],
        "required_methods": list(common.RQ1_METHODS),
        "paper_root": str(paper_root),
        "output_root": str(output_root),
        "main_repairs": [
            common.repair_target_to_dict(target) for target in main_repair_targets
        ],
        "oracle": oracle_manifest,
        "cereval": cereval_block,
        "plot_ready": plot_ready,
        "next_steps": next_steps,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rq1_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_main = load_config(args.config)
    cfg_cereval = load_config(args.cereval_config)
    paper_root = args.paper_root or Path(cfg_main.paths["out_root"])

    manifest = build_rq1_plan_manifest(
        cfg_main=cfg_main,
        cfg_cereval=cfg_cereval,
        cereval_config_path=args.cereval_config,
        paper_root=paper_root,
        cereval_input_root=args.cereval_input_root,
        output_root=args.output_root,
        skip_cereval=args.skip_cereval,
    )

    if args.plan_only:
        print(f"wrote RQ1 plan manifest to {args.output_root / 'rq1_manifest.json'}")
        return 0

    if not manifest["plot_ready"]:
        print(f"RQ1 inputs are not ready. See {args.output_root / 'rq1_manifest.json'}")
        return 1

    # --paper-pairs builds pairs_override directly from cfg.paper_pairs.
    # Do NOT intersect with selected_pairs() (legacy 4 pairs); the overlap
    # may be smaller than 4, collapsing per-dataset std (B6 bug fix).
    pairs_override = None
    if args.paper_pairs:
        pairs_override = tuple(
            common.SelectedPair(p["surrogate"], p["target"])
            for p in cfg_main.paper_pairs
        )

    plot_mod.build_rq1_artifacts(
        cfg_main=cfg_main,
        cfg_cereval=cfg_cereval,
        input_root=paper_root,
        cereval_input_root=args.cereval_input_root,
        cereval_config_path=args.cereval_config,
        output_root=args.output_root,
        validate_only=False,
        skip_cereval=args.skip_cereval,
        pairs=pairs_override,
    )
    print(f"wrote RQ1 artifacts to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
