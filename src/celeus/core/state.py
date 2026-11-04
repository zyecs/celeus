"""
CELEUS core state dataclasses: CeleusConfig and StratumState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


@dataclass
class CeleusConfig:
    """
    All CELEUS hyperparameters loaded from YAML config.
    No magic numbers: every field corresponds 1-to-1 with default.yaml.
    """
    # Number of strata
    K: int = 1
    # Importance-weight clipping threshold
    u_max: float = 20.0
    # Loss lower bound L: minimum possible per-item loss. Default 0.
    loss_lower: float = 0.0
    # Loss upper bound: maximum possible per-item loss. Accuracy=1.0, CE=max(losses).
    # Used to rescale to [a, b] and for the Hoeffding population correction.
    loss_bound: float = 1.0
    # Labels per round (the runner enforces B_round = K; one per stratum)
    B_round: int = 1
    # Minimum allocation fraction per stratum
    beta_min: float = 0.05
    # Coverage parameter of the sample-level confidence sequence
    alpha_1: float = 0.025
    # Coverage parameter of the finite-pool-to-population correction
    alpha_2: float = 0.025
    # Grid density for the e-value CS (points per unit original scale)
    cs_grid_size: int = 2000
    # Cap constant for the betting fractions, anytime mode
    c_betting: float = 0.5
    # Cap constant for the betting fractions, fixed-horizon mode; 0 < c_fixed < 1
    c_fixed: float = 0.5
    # Mixture weight of the hedged capital process
    theta: float = 0.5
    # Width threshold for early stopping
    epsilon: float = 0.02
    # Maximum total labels
    T_max: int = 5000
    # Fixed-horizon mode: the CI is valid only at t = T_max
    fixed_horizon: bool = False
    # Surrogate model update cadence (0 = never update)
    surrogate_update_interval: int = 0
    # Reproducibility seed
    seed: int = 42
    # Use per-round adaptive local scaling bounds (a_t, b_t).
    # When False, the runner uses one fixed pair of bounds for every round.
    adaptive_bounds: bool = True
    # When True, CeleusRunner and the baselines continue past the epsilon-width
    # stopping rule up to T_max so that the full trajectory is observable. The
    # crossing time is still recorded in the trajectory.
    monitor_to_T_max: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "CeleusConfig":
        """
        Load CeleusConfig from a YAML file.

        Raises ValueError on unknown keys (strict mapping).
        """
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"CeleusConfig.from_yaml: unknown keys in {path}: {sorted(unknown)}"
            )
        missing = known_fields - set(data.keys())
        if missing:
            raise ValueError(
                f"CeleusConfig.from_yaml: missing keys in {path}: {sorted(missing)}"
            )
        return cls(**data)


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

    Index convention:
      label_order stores LOCAL 0-indexed positions within pool_indices.
      The importance-weight formula uses 1-indexed m (cumulative within-stratum
      label count); callers must convert: m = position_in_label_order + 1.
    """

    # Identity, immutable after construction.

    # Stratum index, 0-based
    k: int
    # Global pool indices for items in this stratum
    pool_indices: np.ndarray          # shape (N_k,) int
    # Stratum size
    N_k: int
    # Stratum weight: N_k / N
    w_k: float
    # Surrogate scores for items in this stratum, in [0,1]
    surrogate_scores: np.ndarray      # shape (N_k,) float
    # Cached surrogate mean: (1/N_k) Σ s_n
    surrogate_mean: float

    # Mutable labeling state.

    # Boolean mask; True where item has been labeled.
    # NOT a dataclass default — must be passed explicitly (depends on N_k).
    labeled_mask: np.ndarray          # shape (N_k,) bool

    # Number of labels acquired so far
    M_k: int = 0

    # Local indices in query order, 0-indexed within pool_indices
    # IS-weight formula uses 1-indexed m; caller converts: m = idx_in_list + 1
    label_order: list = field(default_factory=list)

    # L_m values (ground-truth losses) in query order
    losses: list = field(default_factory=list)

    # q_k(m) sampling probabilities
    q_values: list = field(default_factory=list)

    # Running AIPW state.

    # Σ_{m=1}^{t-1} (L_m - s_m) — running sum of raw residuals
    sum_past_residuals: float = 0.0

    # Current stratum AIPW estimate
    R_hat_k: float = 0.0

    # Optional per-item class-probability arrays.
    # These are logically immutable identity fields: make_strata sets them once
    # at construction and nothing modifies them afterward. They sit in the
    # defaulted section only because Python dataclass ordering requires fields
    # with defaults to follow fields without defaults. Do not treat them as
    # mutable state.
    target_distributions: Optional[np.ndarray] = None
    surrogate_distributions: Optional[np.ndarray] = None
    # Optional label-free loss proxy for score-based acquisition.
    # Set by the loader for the `remark1_*` and `remark2_*` surrogate types;
    # None otherwise. ResidualMagnitudeAcquisition reads it to compute
    # |ell_proxy - surrogate_scores|.
    ell_proxy: Optional[np.ndarray] = None

    def remaining_indices(self) -> np.ndarray:
        """
        Return local (0-indexed within this stratum) indices of unlabeled items.
        """
        return np.where(~self.labeled_mask)[0]
