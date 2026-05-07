"""Load configs/paper_experiment.yaml into a typed object."""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    """Resolve the default paper_experiment config path.

    If ``SAVE_PE_CONFIG`` is set, return that; otherwise return the canonical
    ``configs/paper_experiment.yaml`` under the repo root. Lets v0504 (and any
    future snapshot) swap the active config without touching every call site
    that previously hardcoded the default path.
    """
    env = os.environ.get("SAVE_PE_CONFIG")
    if env:
        return Path(env)
    return _REPO_ROOT / "configs" / "paper_experiment.yaml"


_REQUIRED_TOP_LEVEL_KEYS = frozenset({
    "paths", "datasets", "pairs_weak", "pairs_strong", "losses",
    "seeds_main", "seeds_beta_sweep", "seeds_ce_sweep", "seeds_smoke",
    "protocol", "methods", "ce_sweep", "beta_sweep", "smoke", "disk_audit",
    "cereval_scope", "paper_pairs",
})
_OPTIONAL_KEYS_WITH_DEFAULTS: dict[str, Any] = {
    "ce_nll_filter": {"enabled": False, "threshold": 3.0},
    # Acquisition sweep (§6.5 item #3) — populated by Task 1 (cells) and Task 7
    # (variants/seeds/beta_min). Optional because the block is added
    # incrementally; harness defaults keep load_config() backwards-compatible.
    "acquisition_sweep": {
        "cells_legacy": [],
        "variants_legacy_accuracy": [],
        "variants_legacy_cross_entropy": [],
        "cells_v0502": [],
        "strategies_v0502": {"accuracy": [], "cross_entropy": []},
        "seeds": [],
        "beta_min": 0.4,
    },
    # Hyperparameter sweep (§6.5 item #4, Task 8). OAT recipe knobs over
    # (α₁, θ, c_betting/c_fixed) on the 4 weak ce_sweep cells. Accuracy-only.
    # Optional default keeps load_config() backwards-compatible.
    "hparam_sweep": {
        "cells": [],
        "configs": [],
        "seeds": [],
        "beta_min": 0.4,
        "loss": "accuracy",
    },
    # Wallclock stage (§6.5 item #5, Task 9). Cer-Eval-only with per-round
    # wall-clock instrumentation, so we can plot M5's compute cost alongside
    # SAVE/IID. Accuracy-only by paper-campaign convention.
    "wallclock": {
        "cells": [],
        "seeds": [],
        "loss": "accuracy",
    },
}
_TOP_LEVEL_KEYS = _REQUIRED_TOP_LEVEL_KEYS | frozenset(_OPTIONAL_KEYS_WITH_DEFAULTS)


@dataclass
class PaperExperimentConfig:
    paths: dict
    datasets: dict
    pairs_weak: list
    pairs_strong: list
    losses: list
    seeds_main: list
    seeds_beta_sweep: list
    seeds_ce_sweep: list
    seeds_smoke: list
    protocol: dict
    methods: dict
    ce_sweep: dict
    beta_sweep: dict
    smoke: dict
    disk_audit: dict
    cereval_scope: dict
    paper_pairs: list
    ce_nll_filter: dict = field(
        default_factory=lambda: {"enabled": False, "threshold": 3.0}
    )
    acquisition_sweep: dict = field(
        default_factory=lambda: {
            "cells_legacy": [],
            "variants_legacy_accuracy": [],
            "variants_legacy_cross_entropy": [],
            "cells_v0502": [],
            "strategies_v0502": {"accuracy": [], "cross_entropy": []},
            "seeds": [],
            "beta_min": 0.4,
        }
    )
    hparam_sweep: dict = field(
        default_factory=lambda: {
            "cells": [],
            "configs": [],
            "seeds": [],
            "beta_min": 0.4,
            "loss": "accuracy",
        }
    )
    wallclock: dict = field(
        default_factory=lambda: {
            "cells": [],
            "seeds": [],
            "loss": "accuracy",
        }
    )

    @property
    def all_pairs(self) -> list:
        return list(self.pairs_weak) + list(self.pairs_strong)

    @property
    def paper_pair_keys(self) -> set[tuple[str, str]]:
        """Set of (surrogate, target) tuples drawn from `paper_pairs`."""
        return {(p["surrogate"], p["target"]) for p in self.paper_pairs}

    @property
    def cereval_losses(self) -> list:
        return list(self.cereval_scope.get("losses") or self.losses)

    @property
    def cereval_pairs(self) -> list:
        if not self.cereval_scope.get("collapse_by_target", False):
            return list(self.all_pairs)
        seen: set = set()
        out: list = []
        for pair in self.all_pairs:
            if pair["target"] not in seen:
                out.append(pair)
                seen.add(pair["target"])
        return out


def _validate_ce_nll_filter(cfg: object) -> None:
    """Strict validation of the ce_nll_filter config block.

    Raises ValueError (not bare assert — strippable under python -O) for every
    pathological YAML pattern: non-dict, unknown keys, non-bool enabled, bool
    threshold (Python bool is int subclass — reject first), non-finite
    threshold, negative threshold, string threshold.
    """
    if not isinstance(cfg, dict):
        raise ValueError("ce_nll_filter must be a dict")
    unknown = set(cfg.keys()) - {"enabled", "threshold"}
    if unknown:
        raise ValueError(
            f"ce_nll_filter has unknown keys: {sorted(unknown)}"
        )
    if not isinstance(cfg.get("enabled"), bool):
        raise ValueError("ce_nll_filter.enabled must be bool")
    thr = cfg.get("threshold")
    # bool is a subclass of int in Python — reject BEFORE (int, float) check.
    if isinstance(thr, bool):
        raise ValueError(
            "ce_nll_filter.threshold must be a number, not bool"
        )
    if (
        not isinstance(thr, (int, float))
        or not math.isfinite(thr)
        or thr < 0
    ):
        raise ValueError(
            "ce_nll_filter.threshold must be finite non-negative number"
        )


def _validate_paper_pairs(cfg: object) -> None:
    """Strict validation of the paper_pairs config block.

    Raises ValueError for pathological patterns: not a list, entry not a dict,
    missing 'surrogate' or 'target' keys, or non-string values.
    """
    if not isinstance(cfg, list):
        raise ValueError("paper_pairs must be a list")
    for i, entry in enumerate(cfg):
        if not isinstance(entry, dict):
            raise ValueError(
                f"paper_pairs[{i}] must be a dict, got {type(entry).__name__}"
            )
        if "surrogate" not in entry:
            raise ValueError(
                f"paper_pairs[{i}] missing required key 'surrogate'"
            )
        if "target" not in entry:
            raise ValueError(
                f"paper_pairs[{i}] missing required key 'target'"
            )
        if not isinstance(entry.get("surrogate"), str):
            raise ValueError(
                f"paper_pairs[{i}]['surrogate'] must be a string"
            )
        if not isinstance(entry.get("target"), str):
            raise ValueError(
                f"paper_pairs[{i}]['target'] must be a string"
            )


def load_config(path: str | Path) -> PaperExperimentConfig:
    with open(path, "r") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    unknown = set(data.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level keys in {path}: {sorted(unknown)}")
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"missing top-level keys in {path}: {sorted(missing)}")
    for k, default in _OPTIONAL_KEYS_WITH_DEFAULTS.items():
        if data.get(k) is None:
            data[k] = copy.deepcopy(default)
        if k == "ce_nll_filter":
            _validate_ce_nll_filter(data[k])
    # Validate paper_pairs
    _validate_paper_pairs(data.get("paper_pairs"))
    return PaperExperimentConfig(
        **{k: data[k] for k in _TOP_LEVEL_KEYS}
    )
