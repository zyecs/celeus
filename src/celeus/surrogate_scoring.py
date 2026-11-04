"""
Pure-math helpers for the label-free surrogate scoring strategies.

Naming: the `remark1_*` helpers handle the cross-entropy loss, which the paper
presents in Remark 2; the `remark2_*` helpers handle the 0-1 loss, which the
paper presents in Remark 1. The config prefixes keep an older numbering, so do
not read them as paper remark numbers.

Each strategy is a label-free instantiation of the oracle proposal of
Theorem 4, q*(j) ∝ |ℓ(f(x_j), y_j) - ℓ̂(f(x_j), x_j)|, with the unknown target
loss replaced by a proxy that needs no label.
"""

from __future__ import annotations

import numpy as np


# Primitives


def predictive_entropy_nats(P: np.ndarray) -> np.ndarray:
    """
    Predictive entropy H(h, x) = -Σ_y p_h(y|x) log p_h(y|x), in nats.

    Natural log, no base-C normalization. The existing `self_entropy`
    surrogate_type normalizes by log(C) to produce a [0,1] score. That is a
    different quantity and must not be used here, because ℓ̂ has to enter the
    estimator in the same units (nats) as the true cross-entropy loss.
    """
    P_safe = np.clip(P, 1e-12, None)
    return -np.sum(P * np.log(P_safe), axis=1)


def mode_loss(P: np.ndarray, loss_type: str) -> np.ndarray:
    """
    Mode loss ℓ_mode(f, x) = ℓ(f(x), argmax_y p_f(y|x)).

    For cross_entropy: -log max_y p_f(y|x).
    For accuracy: identically 0 (argmax matches itself).
    """
    y_hat = P.argmax(axis=1)
    if loss_type == "cross_entropy":
        top1 = P[np.arange(len(P)), y_hat]
        return -np.log(np.clip(top1, 1e-12, None))
    elif loss_type == "accuracy":
        return np.zeros(len(P), dtype=np.float64)
    else:
        raise NotImplementedError(
            f"mode_loss does not support loss_type={loss_type!r}"
        )


# Primitives for the 0-1 loss strategies (paper Remark 1)


def uncertainty_score(P: np.ndarray) -> np.ndarray:
    """
    Predictive uncertainty score ℓ_unc(h, x) = 1 − Σ_y p_h²(y|x).

    Bounded in [0, 1−1/C]. Equals 0 for one-hot (certain) distributions and
    1−1/C for uniform ones. This is the predictive dispersion ℓ_disp of the
    paper's Remark 1.
    """
    return 1.0 - np.sum(P ** 2, axis=1)


def hard_pseudo_label_loss(P_h1: np.ndarray, P_h2: np.ndarray) -> np.ndarray:
    """
    Hard pseudo-label loss ℓ_hard(h1, h2, x) = 𝟙{ŷ_h1(x) ≠ ŷ_h2(x)}.

    Returns 1.0 where the top-1 predictions of h1 and h2 disagree, 0.0 otherwise.
    """
    return (P_h1.argmax(axis=1) != P_h2.argmax(axis=1)).astype(np.float64)


def soft_pseudo_label_loss(P_h1: np.ndarray, P_h2: np.ndarray) -> np.ndarray:
    """
    Soft pseudo-label loss ℓ_soft(h1, h2, x) = 1 − p_h1(ŷ_h2(x)|x).

    Uses h2's top-1 prediction as a proxy label for h1. Returns the
    probability mass h1 does NOT assign to h2's top prediction.
    """
    y_hat_h2 = P_h2.argmax(axis=1)
    p_h1_at_y_hat_h2 = P_h1[np.arange(len(P_h1)), y_hat_h2]
    return 1.0 - p_h1_at_y_hat_h2


# Cross-entropy loss strategies (paper Remark 2)


def remark1_strategy1(
    P_f: np.ndarray, P_g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 1 — target-surrogate entropy gap. Strategy C of Remark 2.

    ell_proxy = H(f, x), hat_ell = H(g, x).
    Acquisition score: |H(f, x) - H(g, x)|.
    """
    ell_proxy = predictive_entropy_nats(P_f)
    hat_ell = predictive_entropy_nats(P_g)
    return ell_proxy, hat_ell


def remark1_strategy2(
    P_f: np.ndarray, loss_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 2 — mode loss as target-loss proxy, self-entropy as surrogate
    score. Strategy A of Remark 2, used in the main cross-entropy experiments.

    ell_proxy = ℓ_mode(f, x), hat_ell = H(f, x).
    Acquisition score: |ℓ_mode(f, x) - H(f, x)|.
    """
    ell_proxy = mode_loss(P_f, loss_type)
    hat_ell = predictive_entropy_nats(P_f)
    return ell_proxy, hat_ell


def remark1_strategy3(
    P_f: np.ndarray, loss_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 3 — self-entropy as target-loss proxy, mode loss as surrogate
    score. Strategy B of Remark 2.

    ell_proxy = H(f, x), hat_ell = ℓ_mode(f, x).
    Acquisition score: |H(f, x) - ℓ_mode(f, x)|. The magnitude equals
    Strategy 2's, but the two quantities swap roles inside the estimator, so
    the correction term and the variance differ.
    """
    ell_proxy = predictive_entropy_nats(P_f)
    hat_ell = mode_loss(P_f, loss_type)
    return ell_proxy, hat_ell


# 0-1 accuracy loss strategies (paper Remark 1)


def remark2_strategy1(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 1 — target-surrogate dispersion gap. Not one of the three
    strategies reported in Remark 1.

    ell_proxy = ℓ_unc(f, x), hat_ell = ℓ_unc(g, x).
    Acquisition score: |ℓ_unc(f, x) - ℓ_unc(g, x)|.
    """
    ell_proxy = uncertainty_score(P_f)
    hat_ell = uncertainty_score(P_g)
    return ell_proxy, hat_ell


def remark2_strategy2(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 2 — dispersion as target-loss proxy, soft pseudo-label loss as
    surrogate score. Strategy B of Remark 1.

    ell_proxy = ℓ_unc(f, x), hat_ell = ℓ_soft(f, g, x).
    Acquisition score: |ℓ_unc(f, x) - ℓ_soft(f, g, x)|.
    """
    ell_proxy = uncertainty_score(P_f)
    hat_ell = soft_pseudo_label_loss(P_f, P_g)
    return ell_proxy, hat_ell


def remark2_strategy3(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 3 — soft pseudo-label loss as target-loss proxy, dispersion as
    surrogate score. Strategy C of Remark 1.

    ell_proxy = ℓ_soft(f, g, x), hat_ell = ℓ_unc(f, x).
    Acquisition score: |ℓ_soft(f, g, x) - ℓ_unc(f, x)|. The magnitude equals
    Strategy 2's, but the two quantities swap roles inside the estimator.
    """
    ell_proxy = soft_pseudo_label_loss(P_f, P_g)
    hat_ell = uncertainty_score(P_f)
    return ell_proxy, hat_ell


def remark2_strategy4(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 4 — hard pseudo-label loss as target-loss proxy, dispersion as
    surrogate score. Strategy A of Remark 1, used in the main 0-1 experiments.

    ell_proxy = ℓ_hard(f, g, x), hat_ell = ℓ_unc(f, x).
    Acquisition score: |ℓ_hard(f, g, x) - ℓ_unc(f, x)|.
    """
    ell_proxy = hard_pseudo_label_loss(P_f, P_g)
    hat_ell = uncertainty_score(P_f)
    return ell_proxy, hat_ell


def remark2_strategy5(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 5 — hard pseudo-label loss as target-loss proxy, soft pseudo-label
    loss as surrogate score. Not one of the three strategies in Remark 1.

    ell_proxy = ℓ_hard(f, g, x), hat_ell = ℓ_soft(f, g, x).
    Acquisition score: |ℓ_hard(f, g, x) - ℓ_soft(f, g, x)|.
    """
    ell_proxy = hard_pseudo_label_loss(P_f, P_g)
    hat_ell = soft_pseudo_label_loss(P_f, P_g)
    return ell_proxy, hat_ell


def remark1_oracle(P_f: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Oracle strategy for the cross-entropy loss (Remark 2): the true
    cross-entropy loss is the proxy and the predictive entropy is the
    surrogate score. Infeasible in practice, used as an experimental
    benchmark.

    ell_proxy = ℓ_ce(f, x, y) = -log p_f(y | x)
    ell_hat   = H(f, x)
    Acquisition score: |ℓ_ce(f, x, y) - H(f, x)|.
    """
    N = P_f.shape[0]
    p_y = P_f[np.arange(N), y]
    ell_proxy = -np.log(p_y.clip(min=1e-12))
    ell_hat = predictive_entropy_nats(P_f)
    return ell_proxy, ell_hat


__all__ = [
    "predictive_entropy_nats",
    "mode_loss",
    "uncertainty_score",
    "hard_pseudo_label_loss",
    "soft_pseudo_label_loss",
    "remark1_strategy1",
    "remark1_strategy2",
    "remark1_strategy3",
    "remark1_oracle",
    "remark2_strategy1",
    "remark2_strategy2",
    "remark2_strategy3",
    "remark2_strategy4",
    "remark2_strategy5",
]
