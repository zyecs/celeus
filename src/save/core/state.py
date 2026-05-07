"""
SAVE core state dataclasses: SAVEConfig and StratumState.

Blueprint §1.1: field layout for StratumState.
Spec §4.5: hyperparameter defaults for SAVEConfig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# SAVEConfig
# ---------------------------------------------------------------------------

@dataclass
class SAVEConfig:
    """
    All SAVE hyperparameters loaded from YAML config.
    No magic numbers: every field corresponds 1-to-1 with default.yaml.

    Source: Spec §4.5, configs/paper_experiment.yaml.
    """
    # Spec §2.1: number of strata
    K: int = 1
    # Spec §2.1 Eq. (2): IS weight clipping threshold
    u_max: float = 20.0
    # Loss lower bound L (v0403 §3.1): min possible per-item loss. Default 0.
    loss_lower: float = 0.0
    # Loss bound L: max possible per-item loss. Accuracy=1.0, CE=max(losses).
    # Used for rescaling [a,b] and Hoeffding Δ_N. [Draft §3.2, page 6]
    loss_bound: float = 1.0
    # Paper v0320: labels per round (runner enforces B_round = K; one per stratum)
    B_round: int = 1
    # Spec §4.5: minimum allocation fraction per stratum
    beta_min: float = 0.05
    # Spec §2.2: CS coverage parameter (sample-based)
    alpha_1: float = 0.025
    # Spec §2.3 Eq. (11): population correction parameter
    alpha_2: float = 0.025
    # Spec §4.5: grid density for e-value CS (points per unit original scale)
    cs_grid_size: int = 2000
    # Spec §4.5: betting fraction for hedged capital (anytime mode)
    c_betting: float = 0.5
    # Remark 5: cap multiplier for fixed-horizon mode; 0 < c_fixed < 1
    c_fixed: float = 0.5
    # Spec §4.5: mixture weight for e-value CS
    theta: float = 0.5
    # Spec §4.5: width threshold for early stopping
    epsilon: float = 0.02
    # Spec §4.7 Algorithm 1: maximum total labels
    T_max: int = 5000
    # Remark 5: fixed-horizon mode (CI valid only at t=T_max)
    fixed_horizon: bool = False
    # Spec §4.5: surrogate model update cadence (0 = never update)
    surrogate_update_interval: int = 0
    # CLAUDE.md Rule 4: reproducibility seed
    seed: int = 42
    # v0414 §3.2: use per-round adaptive local scaling bounds (a_t, b_t)
    # When False, falls back to fixed bounds (v0413 behavior).
    adaptive_bounds: bool = True
    # Paper-experiment (spec §4): when True, SAVERunner / baselines continue
    # past the epsilon-width stopping rule up to T_max so the full trajectory
    # is observable. The crossing time is still recorded in the trajectory.
    monitor_to_T_max: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "SAVEConfig":
        """
        Load SAVEConfig from a YAML file.

        Raises ValueError on unknown keys (strict mapping).
        Source: CLAUDE.md Rule 3 (no magic numbers), Rule 4 (seed in config).
        """
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"SAVEConfig.from_yaml: unknown keys in {path}: {sorted(unknown)}"
            )
        missing = known_fields - set(data.keys())
        if missing:
            raise ValueError(
                f"SAVEConfig.from_yaml: missing keys in {path}: {sorted(missing)}"
            )
        return cls(**data)


# ---------------------------------------------------------------------------
# StratumState
# ---------------------------------------------------------------------------

@dataclass
class StratumState:
    """
    Mutable per-stratum state for the AIPW estimator.

    Identity fields (set at construction, immutable by convention):
      - k, pool_indices, N_k, w_k, surrogate_scores, surrogate_mean
      - target_distributions, surrogate_distributions (optional, see note below)

    Mutable labeling state (updated by AIPWEstimator.update_stratum):
      - labeled_mask, M_k, label_order, losses, q_values,
        sum_past_residuals, R_hat_k

    Note on target_distributions / surrogate_distributions:
      These are logically immutable identity fields but are placed after
      labeled_mask (in the defaulted section) because Python dataclass
      ordering requires fields with defaults to follow fields without.

    Index convention (Blueprint §1.1, Spec §7.2):
      label_order stores LOCAL 0-indexed positions within pool_indices.
      The IS-weight formula Eq. (2) uses 1-indexed m (cumulative within-stratum
      label count); callers must convert: m = position_in_label_order + 1.

    Source: Blueprint §1.1 (field layout), Spec §2.1 (w_k, N_k, Eq. (1)-(2)).
    """

    # --- Identity (immutable after construction) ---

    # Stratum index, 0-based [DERIVED from Blueprint §1.1]
    k: int
    # Global pool indices for items in this stratum [Blueprint §1.1]
    pool_indices: np.ndarray          # shape (N_k,) int
    # Stratum size [Spec §2.1]
    N_k: int
    # Stratum weight: N_k / N [Spec §2.1 setup]
    w_k: float
    # Surrogate scores for items in this stratum, in [0,1] [Spec §1.3]
    surrogate_scores: np.ndarray      # shape (N_k,) float
    # Cached surrogate mean: (1/N_k) Σ s_n [Spec §2.1 Eq. (1) note]
    surrogate_mean: float

    # --- Mutable labeling state ---

    # Boolean mask; True where item has been labeled [Blueprint §1.1]
    # NOT a dataclass default — must be passed explicitly (depends on N_k).
    labeled_mask: np.ndarray          # shape (N_k,) bool

    # Number of labels acquired so far [Spec §2.1 Eq. (1)]
    M_k: int = 0

    # Local indices in query order, 0-indexed within pool_indices [DERIVED — verify]
    # IS-weight formula uses 1-indexed m; caller converts: m = idx_in_list + 1
    label_order: list = field(default_factory=list)

    # L_m values (ground-truth losses) in query order [Spec §2.1 Eq. (1)]
    losses: list = field(default_factory=list)

    # q_k(m) sampling probabilities [Spec §2.1 Eq. (2)]
    q_values: list = field(default_factory=list)

    # --- Running AIPW state ---

    # Σ_{m=1}^{t-1} (L_m - s_m) — running sum of raw residuals [Paper v0320 Eq. (1')]
    sum_past_residuals: float = 0.0

    # Current stratum AIPW estimate [Spec §2.1 Eq. (1)]
    R_hat_k: float = 0.0

    # --- Optional: per-item distribution arrays for Eq. (15) [Stage 6.5] ---
    # NOTE [R3]: These are logically IMMUTABLE identity fields (set once at
    # construction by make_strata, never modified afterward). They are placed
    # in the defaulted section solely because Python dataclass ordering
    # requires all fields with defaults to follow all fields without defaults.
    # Do NOT treat these as mutable state. [Spec section 2.4 Eq. (15)]
    target_distributions: Optional[np.ndarray] = None
    surrogate_distributions: Optional[np.ndarray] = None
    # --- Optional: label-free loss proxy for Remark 1 acquisition [Stage 13] ---
    # Set by the loader when surrogate_type starts with "remark1_"; None
    # otherwise. ResidualMagnitudeAcquisition reads this to compute
    # |ell_proxy - surrogate_scores|.
    ell_proxy: Optional[np.ndarray] = None

    def remaining_indices(self) -> np.ndarray:
        """
        Return local (0-indexed within this stratum) indices of unlabeled items.

        Source: [DERIVED — verify] complement of labeled_mask.
        """
        # [DERIVED — verify]: np.where(~mask)[0] gives local indices of False entries
        return np.where(~self.labeled_mask)[0]
