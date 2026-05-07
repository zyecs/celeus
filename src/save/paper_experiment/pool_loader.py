"""Cell-level pool loader — wraps save.loader.load_experiment for reuse across seeds.

Also computes a SHA-256 of the loaded pool's ``(ground_truth_losses,
surrogate_scores)`` arrays as a fingerprint; spec §9 mandates a pool-identity
hash for reproducibility.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from save.partition import EvaluationPool


from save.loader import load_experiment


def load_pool_for_cell(
    data_root: Path,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    surrogate_type: str,
    load_embeddings: bool = False,
    ce_nll_filter: dict | None = None,
) -> EvaluationPool:
    """Return an EvaluationPool for one (dataset, surrogate, target, loss, surrogate_type).

    ``load_embeddings=False`` (default) skips the optional embedding tensor
    (large; ~560 MB for sst2), suitable for M1-M4 paths. ``True`` is required
    for M5 / Cer-Eval, which uses embeddings for stratification.

    ``ce_nll_filter`` (default None) gates the per-target NLL filter. Forwarded
    verbatim to ``load_experiment``; accuracy loss ignores it.
    """
    cfg = {
        "data_root": str(data_root),
        "dataset": dataset,
        "target_model": target,
        "surrogate_model": surrogate,
        "surrogate_type": surrogate_type,
        "loss_type": loss,
    }
    if not load_embeddings:
        # Point embedding lookup at a non-existent subdir so _load_embeddings
        # returns None gracefully (see save/loader.py:219-236).
        cfg["embedding_model"] = "__pool_loader_no_embeddings__"
    return load_experiment(cfg, ce_nll_filter=ce_nll_filter)


def pool_sha256(pool: EvaluationPool) -> str:
    """Deterministic fingerprint of the loaded pool (spec §9 Stage 0)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pool.ground_truth_losses).tobytes())
    h.update(b"|")
    h.update(np.ascontiguousarray(pool.surrogate_scores).tobytes())
    return h.hexdigest()
