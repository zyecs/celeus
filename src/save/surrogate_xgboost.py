"""
XGBoost-based oracle surrogate scores for SAVE ablation.

Both variants are oracle-assisted — trained on ground_truth_losses.
Results must be interpreted as upper bounds on achievable variance reduction.

[POTENTIAL ISSUE]: Surrogate scores trained on the same ground_truth_losses
used by the AIPW estimator. Conditionally unbiased given fixed training data.
Acceptable for oracle upper-bound ablation, not theoretically rigorous.
"""

from __future__ import annotations

import math

import numpy as np


def _extract_features_surrogate(logits: np.ndarray) -> np.ndarray:
    """
    Extract 3 surrogate-only features from logits.

    Features: surr_max_prob, surr_entropy, surr_top2_gap
    Source: spec §4.2 "xgboost" variant
    """
    # Stable softmax
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    C = logits.shape[1]

    # surr_max_prob
    max_prob = probs.max(axis=1)

    # surr_entropy: H(p) / log(C) — normalized
    log_probs = np.log(np.clip(probs, 1e-12, None))
    entropy = -(probs * log_probs).sum(axis=1) / math.log(C) if C >= 2 else np.zeros(len(logits))

    # surr_top2_gap: top1_prob - top2_prob
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1] if C >= 2 else np.ones(len(logits))

    return np.column_stack([max_prob, entropy, top2_gap])


def _extract_features_oracle(
    surrogate_logits: np.ndarray,
    target_logits: np.ndarray,
) -> np.ndarray:
    """
    Extract ~8 features from both surrogate and target logits.

    Features: surr_max_prob, surr_entropy, surr_top2_gap,
              tgt_max_prob, tgt_entropy, tgt_top2_gap, agreement, kl_divergence
    Source: spec §4.2 "xgboost_oracle" variant
    """
    surr_feats = _extract_features_surrogate(surrogate_logits)

    # Target features (same computation)
    tgt_shifted = target_logits - target_logits.max(axis=1, keepdims=True)
    tgt_exp = np.exp(tgt_shifted)
    tgt_probs = tgt_exp / tgt_exp.sum(axis=1, keepdims=True)

    C = target_logits.shape[1]

    tgt_max_prob = tgt_probs.max(axis=1)
    tgt_log_probs = np.log(np.clip(tgt_probs, 1e-12, None))
    tgt_entropy = -(tgt_probs * tgt_log_probs).sum(axis=1) / math.log(C) if C >= 2 else np.zeros(len(target_logits))
    tgt_sorted = np.sort(tgt_probs, axis=1)[:, ::-1]
    tgt_top2_gap = tgt_sorted[:, 0] - tgt_sorted[:, 1] if C >= 2 else np.ones(len(target_logits))

    # Agreement: 1 if argmax(surr) == argmax(tgt)
    surr_shifted = surrogate_logits - surrogate_logits.max(axis=1, keepdims=True)
    surr_exp = np.exp(surr_shifted)
    surr_probs = surr_exp / surr_exp.sum(axis=1, keepdims=True)
    agreement = (surr_probs.argmax(axis=1) == tgt_probs.argmax(axis=1)).astype(np.float64)

    # KL divergence: KL(p_target || p_surrogate + 1e-10)
    kl = (tgt_probs * (tgt_log_probs - np.log(np.clip(surr_probs, 1e-10, None)))).sum(axis=1)

    return np.column_stack([
        surr_feats,
        tgt_max_prob, tgt_entropy, tgt_top2_gap,
        agreement, kl,
    ])


def train_xgboost_surrogate(
    surrogate_logits: np.ndarray,
    ground_truth_losses: np.ndarray,
    seed: int,
    target_logits: np.ndarray | None = None,
) -> np.ndarray:
    """
    Train XGBoost regressor on logit features to predict losses.

    Parameters
    ----------
    surrogate_logits : np.ndarray
        Shape (N, C) surrogate model logits.
    ground_truth_losses : np.ndarray
        Shape (N,) training labels (oracle access).
    seed : int
        Random seed for reproducibility. CLAUDE.md Rule 4.
    target_logits : np.ndarray or None
        Shape (N, C) target model logits. If provided, use full oracle features.
        If None, use surrogate-only features (3 features).

    Returns
    -------
    np.ndarray
        Shape (N,) predicted surrogate scores.

    Source: spec §4.2; CLAUDE.md Rule 3 (hyperparameters documented).
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "xgboost is required for surrogate_type='xgboost' or 'xgboost_oracle'. "
            "Install with: pip install xgboost"
        )

    if target_logits is not None:
        features = _extract_features_oracle(surrogate_logits, target_logits)
    else:
        features = _extract_features_surrogate(surrogate_logits)

    # Hyperparameters documented per CLAUDE.md Rule 3
    model = XGBRegressor(
        n_estimators=100,    # sufficient for N < 50k
        max_depth=6,         # standard default
        learning_rate=0.1,   # standard default
        random_state=seed,   # reproducibility
    )
    model.fit(features, ground_truth_losses)
    return model.predict(features).astype(np.float64)
