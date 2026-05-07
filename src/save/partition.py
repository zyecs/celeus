"""
SAVE evaluation pool and partition utilities.

EvaluationPool: holds surrogate scores and (optionally) ground-truth losses.
make_strata: creates list[StratumState] from a pool and partition indices.
make_trivial_partition: trivial K=1 partition (all items in stratum 0).

Source: Spec §2.1 setup (stratum weights), Spec §1.2-§1.3 (pool fields),
        Blueprint §1.1 (O(1) surrogate mean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from save.core.state import StratumState


# ---------------------------------------------------------------------------
# EvaluationPool
# ---------------------------------------------------------------------------

@dataclass
class EvaluationPool:
    """
    Holds all items available for evaluation.

    Fields
    ------
    N : int
        Total pool size. [Spec §1.2]
    surrogate_scores : np.ndarray
        Shape (N,) float in [0, 1]. Surrogate model scores. [Spec §1.3]
    ground_truth_losses : np.ndarray or None
        Shape (N,) float in [0, 1]. None for unlabeled pools. [Spec §1.2]
    item_ids : np.ndarray
        Shape (N,) int or str identifiers. [DERIVED — verify]
    metadata : dict
        Dataset name, model name, etc. [DERIVED — verify]
    """

    N: int
    surrogate_scores: np.ndarray
    ground_truth_losses: Optional[np.ndarray]
    item_ids: np.ndarray
    metadata: dict
    target_distributions: Optional[np.ndarray] = None
    surrogate_distributions: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None  # shape (N, D), L2-normalized
    # --- Stage 13: label-free loss proxy for Remark 1 strategies ---
    # Populated by loader.load_from_config when surrogate_type is one of
    # remark1_strategy{1,2,3}; None for all other surrogate types.
    ell_proxy: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        scores_path: str,
        losses_path: Optional[str] = None,
    ) -> "EvaluationPool":
        """
        Load pool from .npy or .npz files.

        Parameters
        ----------
        scores_path : str
            Path to surrogate scores file (.npy or .npz with 'scores' key).
        losses_path : str or None
            Path to ground-truth losses file. None for unlabeled pools.

        Returns
        -------
        EvaluationPool
        """
        # Load surrogate scores
        if scores_path.endswith(".npz"):
            data = np.load(scores_path)
            surrogate_scores = data["scores"].astype(np.float64)
        else:
            surrogate_scores = np.load(scores_path).astype(np.float64)

        N = len(surrogate_scores)

        # Load ground-truth losses (optional)
        ground_truth_losses: Optional[np.ndarray] = None
        if losses_path is not None:
            if losses_path.endswith(".npz"):
                data = np.load(losses_path)
                loaded_losses = data["losses"].astype(np.float64)
            else:
                loaded_losses = np.load(losses_path).astype(np.float64)

            # Length alignment check [requirements_zzj_cc.md §6.2]
            if loaded_losses.shape[0] != N:
                raise ValueError(
                    f"losses length {loaded_losses.shape[0]} != "
                    f"scores length {N}"
                )
            ground_truth_losses = loaded_losses

        item_ids = np.arange(N)
        metadata = {"scores_path": scores_path, "losses_path": losses_path}

        return cls(
            N=N,
            surrogate_scores=surrogate_scores,
            ground_truth_losses=ground_truth_losses,
            item_ids=item_ids,
            metadata=metadata,
        )

    @classmethod
    def from_synthetic(
        cls,
        N: int,
        R_true: float,
        rho_surr: float,
        rng: np.random.Generator,
    ) -> "EvaluationPool":
        """
        Generate a synthetic evaluation pool for testing.

        Ground-truth losses ~ Beta(a, b) with mean = R_true.
        Surrogate scores = rho_surr * losses + (1 - rho_surr) * Uniform(0, 1),
        clipped to [0, 1].

        When rho_surr == 1.0, surrogate_scores == ground_truth_losses exactly.
        This is REQUIRED for Gate G0-2 (oracle sanity). [Spec §6.1 E1]

        Beta parameterization: a=2, b = a*(1 - R_true) / R_true, mean = a/(a+b) = R_true.
        [DERIVED — verify]: standard Beta moment-matching with a=2.

        Parameters
        ----------
        N : int
            Pool size. [Spec §1.2]
        R_true : float
            Target mean loss (E[L]). [Spec §1.2]
        rho_surr : float
            Surrogate-loss correlation in [0, 1]. rho_surr=1.0 → perfect surrogate.
        rng : np.random.Generator
            Seeded generator (CLAUDE.md Rule 4 — no global state).

        Returns
        -------
        EvaluationPool
        """
        # Beta parameters: mean = a / (a + b) = R_true → b = a*(1-R_true)/R_true
        # [DERIVED — verify]: moment matching for Beta(a, b) with a=2
        a = 2.0
        b = a * (1.0 - R_true) / R_true  # ensures mean = R_true

        # numpy Generator.beta works directly (no scipy needed here)
        # [DERIVED — verify]: np.random.Generator.beta(a, b, size) is valid
        ground_truth_losses = rng.beta(a, b, size=N)

        if rho_surr == 1.0:
            # Perfect surrogate: exact copy (required for Gate G0-2)
            # [Spec §6.1 E1]: oracle sanity requires surrogate_scores == losses
            surrogate_scores = ground_truth_losses.copy()
        else:
            # Noisy surrogate: convex combination with Uniform noise
            # [DERIVED — verify]: matches surrogate degradation model Spec §5 ablation A4
            noise = rng.uniform(0.0, 1.0, size=N)
            surrogate_scores = np.clip(
                rho_surr * ground_truth_losses + (1.0 - rho_surr) * noise,
                0.0,
                1.0,
            )

        item_ids = np.arange(N)
        metadata = {
            "dataset": "synthetic",
            "rho_surr": float(rho_surr),
            "R_true": float(R_true),
        }

        return cls(
            N=N,
            surrogate_scores=surrogate_scores,
            ground_truth_losses=ground_truth_losses,
            item_ids=item_ids,
            metadata=metadata,
        )

    @classmethod
    def from_synthetic_ce(
        cls,
        N: int,
        R_true: float,
        rho_surr: float,
        loss_range: tuple[float, float] = (0.0, 3.0),
        rng: Optional[np.random.Generator] = None,
    ) -> "EvaluationPool":
        """
        Generate a synthetic CE-scale evaluation pool for Stage 13 tests.

        The existing ``from_synthetic`` assumes a Beta mean in (0, 1),
        which cannot represent a CE-scale ``R_true > 1``. This factory
        uses a shifted exponential distribution so ``R_true`` can be
        anywhere strictly inside ``loss_range`` without clip bias, and
        attaches synthetic target distributions so Remark 1 strategies
        (which need ``p_f(.|x)``) can be exercised end-to-end on
        synthetic data.

        Parameters
        ----------
        N : int
            Pool size.
        R_true : float
            Target mean of ``ground_truth_losses``. Must lie strictly
            in ``(loss_range[0], loss_range[1])``; otherwise ``ValueError``.
        rho_surr : float
            Desired correlation between ``ground_truth_losses`` and
            ``surrogate_scores``. Value in ``[0, 1]``.
        loss_range : tuple[float, float]
            ``(lo, hi)`` bounds on CE loss. Defaults to ``(0, 3)`` which
            supports ``R_true = 1.2`` with plenty of headroom.
        rng : np.random.Generator
            Seeded generator (CLAUDE.md Rule 5 — no global state).

        Returns
        -------
        EvaluationPool
            With ``ground_truth_losses`` on CE scale, correlated
            ``surrogate_scores``, synthetic ``target_distributions``
            whose entropy correlates with the loss, and ``ell_proxy=None``
            (the loader populates ``ell_proxy`` for real Remark 1 runs;
            synthetic fixtures can set it directly via the pool's
            attribute).

        """
        if rng is None:
            raise ValueError("from_synthetic_ce requires an explicit rng")
        lo, hi = loss_range
        if not (lo < R_true < hi):
            raise ValueError(
                f"R_true={R_true} must lie strictly in loss_range=({lo}, {hi})"
            )

        # Shifted exponential: samples = lo + Exp(scale), clipped to [lo, hi].
        # With clipping at width = hi - lo, E[min(X, width)] for X ~ Exp(scale)
        # is scale * (1 - exp(-width / scale)); solve this so the clipped
        # distribution targets R_true without systematic clip bias.
        # [Ref: Stage 13 Spec §4.2]
        width = hi - lo

        def _clipped_exp_mean(scale_value: float) -> float:
            return scale_value * (1.0 - np.exp(-width / scale_value))

        target_shifted_mean = R_true - lo
        scale_lo = 1e-12
        scale_hi = max(target_shifted_mean, 1e-6)
        while _clipped_exp_mean(scale_hi) < target_shifted_mean:
            scale_hi *= 2.0
        for _ in range(100):
            scale_mid = 0.5 * (scale_lo + scale_hi)
            if _clipped_exp_mean(scale_mid) < target_shifted_mean:
                scale_lo = scale_mid
            else:
                scale_hi = scale_mid
        scale = scale_hi
        # Use stratified inverse-CDF sampling to reduce finite-N drift while
        # preserving the shifted exponential marginal family. [DERIVED — verify]
        uniforms = (np.arange(N, dtype=np.float64) + rng.random(N)) / float(N)
        rng.shuffle(uniforms)
        raw = -scale * np.log1p(-uniforms)
        losses = np.clip(lo + raw, lo, hi).astype(np.float64)

        sample_mean = float(losses.mean())
        if abs(sample_mean - R_true) / R_true > 0.05:
            raise ValueError(
                f"from_synthetic_ce sampler mean {sample_mean:.4f} "
                f"deviates >5% from R_true={R_true}. "
                f"Increase N or widen loss_range to reduce clip bias."
            )

        # Correlated surrogate scores: linear mix of the true losses with
        # independent zero-mean noise. [DERIVED — verify]
        noise = rng.standard_normal(N) * (hi - lo) / 2.0
        surrogate_scores = (
            rho_surr * losses + (1.0 - rho_surr) * (float(losses.mean()) + noise)
        ).astype(np.float64)

        # Synthetic target distributions over C=4 classes, where low-loss
        # items are peaky (low entropy) and high-loss items are flat
        # (high entropy). This makes all three Remark 1 strategies
        # produce nontrivial scores on synthetic data. [Ref: Stage 13 Spec §4.2]
        C = 4
        target_distributions = np.empty((N, C), dtype=np.float64)
        loss_frac = (losses - lo) / (hi - lo)  # in [0, 1]
        alpha_values = 1.0 + 10.0 * (1.0 - loss_frac)
        for i in range(N):
            target_distributions[i] = rng.dirichlet(
                np.full(C, alpha_values[i])
            )

        item_ids = np.arange(N)
        metadata = {
            "synthetic": True,
            "R_true": R_true,
            "rho_surr": rho_surr,
            "loss_range": loss_range,
            "source": "from_synthetic_ce",
        }

        return cls(
            N=N,
            surrogate_scores=surrogate_scores,
            ground_truth_losses=losses,
            item_ids=item_ids,
            metadata=metadata,
            target_distributions=target_distributions,
            surrogate_distributions=None,
            embeddings=None,
            ell_proxy=None,
        )


# ---------------------------------------------------------------------------
# Partition factories
# ---------------------------------------------------------------------------

def make_strata(
    pool: EvaluationPool,
    partition_indices: np.ndarray,
) -> list[StratumState]:
    """
    Create a list of StratumState objects from pool and partition assignment.

    Parameters
    ----------
    pool : EvaluationPool
        Evaluation pool.
    partition_indices : np.ndarray
        Shape (N,) int in [0, K-1]; maps each item to a stratum.

    Returns
    -------
    list[StratumState]
        One StratumState per unique stratum label, sorted by stratum index.
        Stratum weights w_k = N_k / N sum to 1.0 within float tolerance.
        [Spec §2.1 setup: "Stratum weights w_k = N_k / N"]
    """
    unique_ks = np.unique(partition_indices)
    strata: list[StratumState] = []

    for k in unique_ks:
        # Boolean mask for items in stratum k
        mask = partition_indices == k
        local_pool_indices = np.where(mask)[0]  # global pool indices
        N_k = int(mask.sum())

        # Stratum weight: w_k = N_k / N  [Spec §2.1 setup]
        w_k = N_k / pool.N

        # Surrogate scores for this stratum (copy to avoid aliasing)
        surrogate_scores_k = pool.surrogate_scores[mask].copy()

        # Cache surrogate mean once: O(N_k), not recomputed per call
        # [Blueprint §1.1 note on O(1) surrogate mean during estimation]
        surrogate_mean_k = float(surrogate_scores_k.mean())

        # labeled_mask must be initialized explicitly (depends on N_k)
        # [Implementation Note 6: do NOT use dataclass default_factory here]
        labeled_mask_k = np.zeros(N_k, dtype=bool)

        stratum = StratumState(
            k=int(k),
            pool_indices=local_pool_indices,
            N_k=N_k,
            w_k=w_k,
            surrogate_scores=surrogate_scores_k,
            surrogate_mean=surrogate_mean_k,
            labeled_mask=labeled_mask_k,
        )
        # Forward optional distribution arrays to stratum [Eq. (15); Stage 6.5]
        if pool.target_distributions is not None:
            stratum.target_distributions = pool.target_distributions[mask].copy()
        if pool.surrogate_distributions is not None:
            stratum.surrogate_distributions = pool.surrogate_distributions[mask].copy()
        # Forward Remark 1 label-free loss proxy [Stage 13 §5.2]
        if pool.ell_proxy is not None:
            stratum.ell_proxy = pool.ell_proxy[mask].copy()
        strata.append(stratum)

    return strata


def make_trivial_partition(pool: EvaluationPool) -> list[StratumState]:
    """
    Create a trivial K=1 partition: all items in stratum 0.

    Source: [DERIVED — trivial K=1 special case of make_strata]
    """
    # All items assigned to stratum 0
    partition_indices = np.zeros(pool.N, dtype=int)
    return make_strata(pool, partition_indices)


def make_quantile_partition(
    pool: EvaluationPool,
    K: int,
) -> list[StratumState]:
    """
    Partition pool into K strata via quantile bins of surrogate scores.

    Requires surrogate scores with sufficient variation to produce K distinct
    quantile bins. Raises ValueError if fewer than K non-empty strata result
    (e.g., discrete or low-cardinality surrogates).

    Source: Spec §1.3; Spec §3.6.
    [Spec §3.6]: Var_stratified = Var_unstratified - sum_k w_k (R_bar_k - R_bar)^2
    """
    assert pool.N >= K, f"make_quantile_partition: pool.N={pool.N} < K={K}"
    if K == 1:
        return make_trivial_partition(pool)

    # K-1 quantile boundaries at 1/K, 2/K, ..., (K-1)/K  [Spec §3.6]
    quantile_fractions = np.linspace(1.0 / K, (K - 1.0) / K, K - 1)
    boundaries = np.quantile(pool.surrogate_scores, quantile_fractions)

    # right=False: bins[i-1] <= value < bins[i]; range [0, K-1] giving K bins
    partition_indices = np.digitize(pool.surrogate_scores, boundaries, right=False)

    n_unique = len(np.unique(partition_indices))
    if n_unique < K:
        raise ValueError(
            f"make_quantile_partition: only {n_unique} non-empty strata created "
            f"from {K} requested; surrogate scores may have insufficient variation "
            f"(e.g., discrete or low-cardinality). Require at least {K} distinct "
            f"quantile levels."
        )

    return make_strata(pool, partition_indices)
