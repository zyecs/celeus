"""
CELEUS evaluation pool and partition utilities.

EvaluationPool: holds surrogate scores and (optionally) ground-truth losses.
make_strata: creates list[StratumState] from a pool and partition indices.
make_trivial_partition: trivial K=1 partition (all items in stratum 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from celeus.core.state import StratumState


# EvaluationPool

@dataclass
class EvaluationPool:
    """
    Holds all items available for evaluation.

    Fields
    ------
    N : int
        Total pool size.
    surrogate_scores : np.ndarray
        Shape (N,) float in [0, 1]. Surrogate model scores.
    ground_truth_losses : np.ndarray or None
        Shape (N,) float in [0, 1]. None for unlabeled pools.
    item_ids : np.ndarray
        Shape (N,) int or str identifiers.
    metadata : dict
        Dataset name, model name, etc.
    """

    N: int
    surrogate_scores: np.ndarray
    ground_truth_losses: Optional[np.ndarray]
    item_ids: np.ndarray
    metadata: dict
    target_distributions: Optional[np.ndarray] = None
    surrogate_distributions: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None  # shape (N, D), L2-normalized
    # Label-free loss proxy for the score-based acquisition strategies.
    # loader.load_from_config populates it for the `remark1_*` and `remark2_*`
    # surrogate types; it stays None for every other surrogate type.
    ell_proxy: Optional[np.ndarray] = None

    # Constructors

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

            # Length alignment check
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

        When rho_surr == 1.0, surrogate_scores == ground_truth_losses exactly,
        which the oracle sanity check needs.

        Beta parameterization: a=2, b = a*(1 - R_true) / R_true, so the mean is
        a/(a+b) = R_true.

        Parameters
        ----------
        N : int
            Pool size.
        R_true : float
            Target mean loss (E[L]).
        rho_surr : float
            Surrogate-loss correlation in [0, 1]. rho_surr=1.0 → perfect surrogate.
        rng : np.random.Generator
            Seeded generator; no global random state is used.

        Returns
        -------
        EvaluationPool
        """
        # Beta parameters: mean = a / (a + b) = R_true → b = a*(1-R_true)/R_true
        a = 2.0
        b = a * (1.0 - R_true) / R_true  # ensures mean = R_true

        # numpy Generator.beta works directly (no scipy needed here)
        ground_truth_losses = rng.beta(a, b, size=N)

        if rho_surr == 1.0:
            # Perfect surrogate: exact copy, required by the oracle sanity check.
            surrogate_scores = ground_truth_losses.copy()
        else:
            # Noisy surrogate: convex combination with uniform noise. This is
            # the surrogate degradation model used in the ablation.
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
        Generate a synthetic evaluation pool on the cross-entropy scale.

        The existing ``from_synthetic`` assumes a Beta mean in (0, 1),
        which cannot represent a CE-scale ``R_true > 1``. This factory
        uses a shifted exponential distribution so ``R_true`` can be
        anywhere strictly inside ``loss_range`` without clip bias, and
        attaches synthetic target distributions so the cross-entropy
        strategies (which need ``p_f(.|x)``) can be exercised end-to-end on
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
            Seeded generator; no global random state is used.

        Returns
        -------
        EvaluationPool
            With ``ground_truth_losses`` on CE scale, correlated
            ``surrogate_scores``, synthetic ``target_distributions``
            whose entropy correlates with the loss, and ``ell_proxy=None``.
            The loader populates ``ell_proxy`` for real runs; synthetic
            fixtures can set the attribute directly.
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
        # preserving the shifted exponential marginal family.
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
        # independent zero-mean noise.
        noise = rng.standard_normal(N) * (hi - lo) / 2.0
        surrogate_scores = (
            rho_surr * losses + (1.0 - rho_surr) * (float(losses.mean()) + noise)
        ).astype(np.float64)

        # Synthetic target distributions over C=4 classes, where low-loss
        # items are peaky (low entropy) and high-loss items are flat
        # (high entropy). This makes every cross-entropy strategy produce
        # nontrivial scores on synthetic data.
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


# Partition factories

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
    """
    unique_ks = np.unique(partition_indices)
    strata: list[StratumState] = []

    for k in unique_ks:
        # Boolean mask for items in stratum k
        mask = partition_indices == k
        local_pool_indices = np.where(mask)[0]  # global pool indices
        N_k = int(mask.sum())

        # Stratum weight: w_k = N_k / N
        w_k = N_k / pool.N

        # Surrogate scores for this stratum (copy to avoid aliasing)
        surrogate_scores_k = pool.surrogate_scores[mask].copy()

        # Cache the surrogate mean once: O(N_k) here, O(1) per estimator call.
        surrogate_mean_k = float(surrogate_scores_k.mean())

        # labeled_mask must be initialized explicitly because it depends on
        # N_k; do not use a dataclass default_factory here.
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
        # Forward the optional per-class distribution arrays to the stratum.
        if pool.target_distributions is not None:
            stratum.target_distributions = pool.target_distributions[mask].copy()
        if pool.surrogate_distributions is not None:
            stratum.surrogate_distributions = pool.surrogate_distributions[mask].copy()
        # Forward the label-free loss proxy.
        if pool.ell_proxy is not None:
            stratum.ell_proxy = pool.ell_proxy[mask].copy()
        strata.append(stratum)

    return strata


def make_trivial_partition(pool: EvaluationPool) -> list[StratumState]:
    """
    Create a trivial K=1 partition: all items in stratum 0.
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

    Stratification lowers the variance by sum_k w_k (R_bar_k - R_bar)^2.
    """
    assert pool.N >= K, f"make_quantile_partition: pool.N={pool.N} < K={K}"
    if K == 1:
        return make_trivial_partition(pool)

    # K-1 quantile boundaries at 1/K, 2/K, ..., (K-1)/K
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
