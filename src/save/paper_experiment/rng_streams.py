# src/save/paper_experiment/rng_streams.py
"""RNG stream spawning per spec §4 (four-role separation)."""
from __future__ import annotations

import numpy as np


_ROLE_NAMES = ("save_acq", "baseline_order", "tiebreak", "reserve")


def spawn_role_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Return four independent Generators spawned from seed (spec §4)."""
    root = np.random.default_rng(int(seed))
    children = root.spawn(len(_ROLE_NAMES))
    return dict(zip(_ROLE_NAMES, children))
