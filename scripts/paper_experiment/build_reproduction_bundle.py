#!/usr/bin/env python3
# scripts/paper_experiment/build_reproduction_bundle.py
"""Build results/paper_experiment/reproduction_bundle.tar.gz (spec §9)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_checksums(data_root: Path) -> dict:
    out: dict[str, str] = {}
    for dataset in ("sst2", "mmlu", "agnews"):
        d = data_root / dataset
        if not d.exists():
            continue
        for p in sorted(d.rglob("all_set_*.pt")):
            rel = p.relative_to(data_root)
            out[str(rel)] = _sha256(p)
    return out


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build reproduction bundle.")
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help="Override results root for input/output paths. "
             "Default: results/paper_experiment.",
    )
    args = parser.parse_args(argv)
    out_root = args.out_root if args.out_root else _REPO / "results" / "paper_experiment"
    bundle_path = out_root / "reproduction_bundle.tar.gz"
    manifest_path = out_root / "reproduction_manifest.json"

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True,
    ).strip()

    # v2: read data_root from YAML rather than hardcoded relative path.
    import sys as _sys
    _sys.path.insert(0, str(_REPO / "src"))
    from save.paper_experiment.config import load_config
    cfg = load_config(_REPO / "configs" / "paper_experiment.yaml")
    data_root = Path(cfg.paths["data_root"])

    manifest = {
        "git_commit": git_hash,
        "config": str(_REPO / "configs" / "paper_experiment.yaml"),
        "data_root": str(data_root),
        "dataset_checksums": _dataset_checksums(data_root),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="reproduction_manifest.json")
        tar.add(_REPO / "configs" / "paper_experiment.yaml",
                arcname="paper_experiment.yaml")
        for sub in ("R_N.csv", "ce_sweep_winner.json", "summary.csv",
                    "beta_sensitivity.csv", "trajectories", "figures"):
            src = out_root / sub
            if src.exists():
                tar.add(src, arcname=sub)
    print(f"wrote {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
