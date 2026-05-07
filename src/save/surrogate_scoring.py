"""
Pure-math helpers for Remark 1 and Remark 2 surrogate scoring strategies.

Remark 1 (cross-entropy loss): three label-free instantiations of Theorem 5's
oracle proposal q*(j) ∝ |ℓ(f(x_j), y_j) - ℓ̂(f(x_j), x_j)|.
Source: tmp/draft_save_v0410.tex Remark 1 (lines 433-453).

Remark 2 (0-1 accuracy loss): five label-free instantiations using
uncertainty scores, hard/soft pseudo-label losses.
Source: tmp/draft_save_v0413.tex Remark 2 (lines 455-497).

CLAUDE.md G1: every formula cites source.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def predictive_entropy_nats(P: np.ndarray) -> np.ndarray:
    """
    Predictive entropy H(h, x) = -Σ_y p_h(y|x) log p_h(y|x), in nats.

    Natural log, no base-C normalization. The existing `self_entropy`
    surrogate_type normalizes by log(C) to produce a [0,1] score — that
    is a different quantity and must not be used here because ℓ̂ must
    enter the AIPW estimator in the same units (nats) as the true CE loss.

    Source: draft_save_v0410.tex line 437 (unqualified `log`).
    """
    P_safe = np.clip(P, 1e-12, None)
    return -np.sum(P * np.log(P_safe), axis=1)


def mode_loss(P: np.ndarray, loss_type: str) -> np.ndarray:
    """
    Mode loss ℓ_mode(f, x) = ℓ(f(x), argmax_y p_f(y|x)).

    For cross_entropy: -log max_y p_f(y|x).
    For accuracy: identically 0 (argmax matches itself).

    Source: draft_save_v0410.tex line 437.
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


# ---------------------------------------------------------------------------
# Remark 2 primitives (0-1 accuracy loss)
# ---------------------------------------------------------------------------


def uncertainty_score(P: np.ndarray) -> np.ndarray:
    """
    Predictive uncertainty score ℓ_unc(h, x) = 1 − Σ_y p_h²(y|x).

    Bounded in [0, 1−1/C]. Equals 0 for one-hot (certain) distributions,
    1−1/C for uniform. Used in Remark 2 strategies for 0-1 accuracy loss.

    Source: draft_save_v0413.tex Remark 2 line 458.
    """
    return 1.0 - np.sum(P ** 2, axis=1)


def hard_pseudo_label_loss(P_h1: np.ndarray, P_h2: np.ndarray) -> np.ndarray:
    """
    Hard pseudo-label loss ℓ_hard(h1, h2, x) = 𝟙{ŷ_h1(x) ≠ ŷ_h2(x)}.

    Returns 1.0 where the top-1 predictions of h1 and h2 disagree, 0.0 otherwise.

    Source: draft_save_v0413.tex Remark 2 line 458.
    """
    return (P_h1.argmax(axis=1) != P_h2.argmax(axis=1)).astype(np.float64)


def soft_pseudo_label_loss(P_h1: np.ndarray, P_h2: np.ndarray) -> np.ndarray:
    """
    Soft pseudo-label loss ℓ_soft(h1, h2, x) = 1 − p_h1(ŷ_h2(x)|x).

    Uses h2's top-1 prediction as a proxy label for h1. Returns the
    probability mass h1 does NOT assign to h2's top prediction.

    Source: draft_save_v0413.tex Remark 2 line 458.
    """
    y_hat_h2 = P_h2.argmax(axis=1)
    p_h1_at_y_hat_h2 = P_h1[np.arange(len(P_h1)), y_hat_h2]
    return 1.0 - p_h1_at_y_hat_h2


# ---------------------------------------------------------------------------
# Remark 1 strategies
# ---------------------------------------------------------------------------


def remark1_strategy1(
    P_f: np.ndarray, P_g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 1 — target–surrogate entropy gap.

    ell_proxy = H(f, x), hat_ell = H(g, x).
    Acquisition score: |H(f, x) - H(g, x)|.

    Source: draft_save_v0410.tex Remark 1 lines 440-443.
    """
    ell_proxy = predictive_entropy_nats(P_f)
    hat_ell = predictive_entropy_nats(P_g)
    return ell_proxy, hat_ell


def remark1_strategy2(
    P_f: np.ndarray, loss_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 2 — mode loss as true-loss proxy, self-entropy as surrogate.

    ell_proxy = ℓ_mode(f, x), hat_ell = H(f, x).
    Acquisition score: |ℓ_mode(f, x) - H(f, x)|.

    Source: draft_save_v0410.tex Remark 1 lines 444-447.
    """
    ell_proxy = mode_loss(P_f, loss_type)
    hat_ell = predictive_entropy_nats(P_f)
    return ell_proxy, hat_ell


def remark1_strategy3(
    P_f: np.ndarray, loss_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strategy 3 — self-entropy as true-loss proxy, mode loss as surrogate.

    ell_proxy = H(f, x), hat_ell = ℓ_mode(f, x).
    Acquisition score: |H(f, x) - ℓ_mode(f, x)|  —  magnitude equals S2's
    but the role of each proxy inside the AIPW estimator differs.

    Source: draft_save_v0410.tex Remark 1 lines 448-450.
    """
    ell_proxy = predictive_entropy_nats(P_f)
    hat_ell = mode_loss(P_f, loss_type)
    return ell_proxy, hat_ell


# ---------------------------------------------------------------------------
# Remark 2 strategies (0-1 accuracy loss)
# ---------------------------------------------------------------------------


def remark2_strategy1(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remark 2, Strategy 1 — target–surrogate uncertainty gap.

    ell_proxy = ℓ_unc(f, x), hat_ell = ℓ_unc(g, x).
    Acquisition score: |ℓ_unc(f, x) - ℓ_unc(g, x)|.

    Source: draft_save_v0413.tex Remark 2 lines 461-466.
    """
    ell_proxy = uncertainty_score(P_f)
    hat_ell = uncertainty_score(P_g)
    return ell_proxy, hat_ell


def remark2_strategy2(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remark 2, Strategy 2 — uncertainty as true-loss proxy, soft pseudo-label loss as surrogate.

    ell_proxy = ℓ_unc(f, x), hat_ell = ℓ_soft(f, g, x).
    Acquisition score: |ℓ_unc(f, x) - ℓ_soft(f, g, x)|.

    Source: draft_save_v0413.tex Remark 2 lines 468-473.
    """
    ell_proxy = uncertainty_score(P_f)
    hat_ell = soft_pseudo_label_loss(P_f, P_g)
    return ell_proxy, hat_ell


def remark2_strategy3(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remark 2, Strategy 3 — soft pseudo-label loss as true-loss proxy, uncertainty as surrogate.

    ell_proxy = ℓ_soft(f, g, x), hat_ell = ℓ_unc(f, x).
    Acquisition score: |ℓ_soft(f, g, x) - ℓ_unc(f, x)|  —  magnitude
    equals S2's but the role of each proxy inside the AIPW estimator differs.

    Source: draft_save_v0413.tex Remark 2 lines 475-479.
    """
    ell_proxy = soft_pseudo_label_loss(P_f, P_g)
    hat_ell = uncertainty_score(P_f)
    return ell_proxy, hat_ell


def remark2_strategy4(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remark 2, Strategy 4 — hard pseudo-label loss as true-loss proxy, uncertainty as surrogate.

    ell_proxy = ℓ_hard(f, g, x), hat_ell = ℓ_unc(f, x).
    Acquisition score: |ℓ_hard(f, g, x) - ℓ_unc(f, x)|.

    Source: draft_save_v0413.tex Remark 2 lines 481-486.
    """
    ell_proxy = hard_pseudo_label_loss(P_f, P_g)
    hat_ell = uncertainty_score(P_f)
    return ell_proxy, hat_ell


def remark2_strategy5(P_f: np.ndarray, P_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remark 2, Strategy 5 — hard pseudo-label loss as true-loss proxy, soft pseudo-label loss as surrogate.

    ell_proxy = ℓ_hard(f, g, x), hat_ell = ℓ_soft(f, g, x).
    Acquisition score: |ℓ_hard(f, g, x) - ℓ_soft(f, g, x)|.

    Source: draft_save_v0413.tex Remark 2 lines 488-493.
    """
    ell_proxy = hard_pseudo_label_loss(P_f, P_g)
    hat_ell = soft_pseudo_label_loss(P_f, P_g)
    return ell_proxy, hat_ell


def remark1_oracle(P_f: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remark 1 Oracle (cross-entropy) — true CE as ell_proxy, predictive entropy as ell_hat.

    ell_proxy = ℓ_ce(f, x, y) = -log p_f(y | x)        (variance-relevant true loss)
    ell_hat   = H(f, x)                                 (canonical Remark 1 surrogate)
    Acquisition score: |ℓ_ce(f, x, y) - H(f, x)|.

    Source: paper Remark 1 lines 405-407 (Oracle case).
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
