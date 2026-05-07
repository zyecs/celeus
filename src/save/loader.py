"""
SAVE loader — bridges .pt files from data_process to EvaluationPool.

Loads pre-computed LLM evaluation data and constructs EvaluationPool
with surrogate scores derived from a cheaper model's logits.

Path template: {data_root}/{dataset}/{model_name}/{filename}.pt
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from save.partition import EvaluationPool


def compute_surrogate_scores(
    logits: torch.Tensor,
    surrogate_type: str,
) -> np.ndarray:
    """
    Compute inverted surrogate scores from raw logits.

    Bounded types produce scores in [0, 1] with semantics:
    high score = high expected loss (surrogate uncertain).

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C) raw model logits.
    surrogate_type : str
        'inv_max_softmax', 'inv_confidence_gap', or 'self_entropy'.

    Returns
    -------
    np.ndarray
        Shape (N,) float64 surrogate scores in [0, 1].
    """
    probs = F.softmax(logits.float(), dim=1)

    if surrogate_type == "inv_max_softmax":
        max_probs, _ = probs.max(dim=1)
        scores = 1.0 - max_probs
    elif surrogate_type == "inv_confidence_gap":
        top2 = probs.topk(2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
        scores = 1.0 - gap
    elif surrogate_type == "self_entropy":
        # H(p_surr) / log(C) — normalized entropy in [0, 1]
        # Source: spec §4.1a; [DERIVED — verify]: standard normalization
        C = logits.shape[1]
        if C == 1:
            raise ValueError("self_entropy requires C >= 2 (single-class has zero entropy)")
        log_probs = torch.log(torch.clamp(probs, min=1e-12))
        entropy = -(probs * log_probs).sum(dim=1)
        scores = torch.clamp(entropy / math.log(C), min=0.0)
    else:
        raise ValueError(
            f"Unknown surrogate_type '{surrogate_type}'. "
            "Supported: 'inv_max_softmax', 'inv_confidence_gap', 'self_entropy'."
        )

    return scores.detach().cpu().numpy().astype(np.float64)


# Surrogate types whose scores may exceed [0, 1]
_UNBOUNDED_SURROGATES = frozenset({
    "surrogate_ce_loss", "xgboost", "xgboost_oracle", "cross_entropy_proxy",
})


def compute_surrogate_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> np.ndarray:
    """
    Compute surrogate CE loss: s_i = -log(p_surr(y_true_i) + eps).

    Oracle-assisted — requires ground-truth labels.
    Source: spec §4.1b

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C) surrogate model logits.
    targets : torch.Tensor
        Shape (N,) integer class labels in [0, C).

    Returns
    -------
    np.ndarray
        Shape (N,) float64 surrogate CE loss scores (unbounded, >= 0).
    """
    probs = F.softmax(logits.float(), dim=1)
    ce = -torch.log(probs[torch.arange(len(targets)), targets.long()] + 1e-12)
    return ce.detach().cpu().numpy().astype(np.float64)


def compute_cross_entropy_proxy(
    surrogate_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> np.ndarray:
    """
    Compute per-item H(p_surr, p_target) as a label-free loss predictor.

    H(p_surr, p_target) = -Σ_c p_surr(c) · log p_target(c)

    High values indicate items where the surrogate is confident about
    something the target model does not predict well — correlated with
    high target loss. No ground-truth labels needed.

    Source: Berrada et al. (scaling-up-active-testing); Draft Remark 1.
    [DEPRECATED as of 2026-04-11 (Stage 13) — prefer remark1_strategy{1,2,3}
    for Theorem-5-aligned surrogate scoring. This function is kept for
    stage-8..12 reproducibility and remains selectable via
    surrogate_type='cross_entropy_proxy'.]

    Parameters
    ----------
    surrogate_logits : torch.Tensor
        Shape (N, C) surrogate model logits.
    target_logits : torch.Tensor
        Shape (N, C) target model logits.

    Returns
    -------
    np.ndarray
        Shape (N,) float64 cross-entropy proxy scores (>= 0).
    """
    p_surr = F.softmax(surrogate_logits.float(), dim=1)
    p_target = F.softmax(target_logits.float(), dim=1)
    p_target_safe = torch.clamp(p_target, min=1e-12)
    ce = -(p_surr * torch.log(p_target_safe)).sum(dim=1)
    return ce.detach().cpu().numpy().astype(np.float64)


def _load_pt(path: Path) -> torch.Tensor:
    """Load a .pt tensor file with safety check."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected .pt file not found: {path}\n"
            f"Check that data_root, dataset, and model_name are correct."
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def _apply_mask_to_tensor(tensor, N_raw: int, mask, path_for_error):
    """Validate shape[0] against N_raw and slice along dim 0 if mask is given.

    Works uniformly for 1D / 2D / 3D ``torch.Tensor`` and ``np.ndarray``.
    The caller is expected to pass in the fully-materialized tensor after
    any ``.float()`` / ``.squeeze()`` pipeline the particular load site
    already performs — this helper only validates and masks dim 0.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
    N_raw : int
        Expected shape[0] of ``tensor`` (pre-filter). If filter is active,
        ``N_raw`` is ``mask.shape[0]``; otherwise it equals the final pool ``N``.
    mask : np.ndarray[bool] of shape (N_raw,) or None
    path_for_error : Any
        Informational value included in error messages (path or name).
    """
    if mask is not None:
        if mask.dtype != np.bool_ or mask.shape != (N_raw,):
            raise ValueError(
                f"mask must be bool[{N_raw}], got dtype={mask.dtype} shape={mask.shape}"
            )
    if tensor.shape[0] != N_raw:
        raise ValueError(
            f"{path_for_error}: first dim {tensor.shape[0]} != N_raw {N_raw}"
        )
    return tensor[mask] if mask is not None else tensor


def _load_losses(
    target_dir: Path,
    loss_type: str,
    ce_nll_filter: dict | None = None,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """
    Load ground-truth losses from .pt file.

    When ``ce_nll_filter`` is enabled AND ``loss_type == "cross_entropy"``, a
    boolean mask is built from the raw CE losses via
    ``save.filters.build_ce_nll_mask`` and applied to the returned losses.
    The returned ``metadata["loss_range"]`` is computed on the FILTERED array.
    Raises ``ValueError`` when the filter keeps zero samples. Returns the
    raw-length mask so the caller can apply it to other N-axis tensors.

    Parameters
    ----------
    target_dir : Path
        Directory containing the target model's output files.
    loss_type : str
        'accuracy' or 'cross_entropy'.
    ce_nll_filter : dict or None
        When enabled and loss_type == "cross_entropy", filters samples where
        per-sample CE > threshold. Ignored for accuracy. Keys: enabled (bool),
        threshold (float).

    Returns
    -------
    (ground_truth_losses, loss_metadata, mask)
        losses : np.ndarray shape (N,) float64 — filtered if applicable
        loss_metadata : dict with loss_type and optional loss_range (filtered)
        mask : np.ndarray[bool] shape (N_raw,) or None
    """
    if loss_type == "accuracy":
        accuracy = _load_pt(target_dir / "all_set_per_sample_accuracy.pt")
        accuracy = accuracy.float().squeeze()
        assert accuracy.ndim == 1, f"Expected 1D accuracy, got shape {accuracy.shape}"
        ground_truth_losses = (1.0 - accuracy).numpy().astype(np.float64)

        # Validate binary
        unique_vals = np.unique(ground_truth_losses)
        if not np.all(np.isin(unique_vals, [0.0, 1.0])):
            raise ValueError(
                f"Accuracy values are not binary. Unique loss values: {unique_vals}"
            )
        return ground_truth_losses, {"loss_type": "accuracy"}, None

    elif loss_type == "cross_entropy":
        ce_loss = _load_pt(
            target_dir / "all_set_per_sample_cross_entropy_loss.pt"
        )
        ce_loss = ce_loss.float().squeeze()
        assert ce_loss.ndim == 1, f"Expected 1D CE loss, got shape {ce_loss.shape}"
        ground_truth_losses = ce_loss.numpy().astype(np.float64)

        if not np.all(np.isfinite(ground_truth_losses)):
            raise ValueError("CE loss contains NaN or Inf values.")
        if ground_truth_losses.min() < 0:
            raise ValueError(
                f"CE loss contains negative values (min={ground_truth_losses.min():.6g})."
            )

        mask = None
        if ce_nll_filter and ce_nll_filter.get("enabled"):
            from save.filters import build_ce_nll_mask
            mask = build_ce_nll_mask(
                ground_truth_losses, float(ce_nll_filter["threshold"])
            )
            ground_truth_losses = ground_truth_losses[mask]
            if ground_truth_losses.size == 0:
                raise ValueError(
                    f"ce_nll_filter threshold={ce_nll_filter['threshold']} "
                    "kept 0 samples; relax threshold."
                )

        # No [0,1] normalization — raw CE values
        # Coverage IS valid for CE loss: EValueCS rescales [a,b] → [0,1] internally,
        # where [a,b] is set from loss_bound=max(losses) in benchmark.py.
        # The grid covers the full (a,b) range on the original scale.
        return (
            ground_truth_losses,
            {
                "loss_type": "cross_entropy",
                "loss_range": [0.0, float(ground_truth_losses.max())],
            },
            mask,
        )

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Supported: 'accuracy', 'cross_entropy'."
        )


def _load_embeddings(
    data_root: Path,
    dataset: str,
    embedding_model: str,
    N_raw: int,
    mask: np.ndarray | None = None,
) -> Optional[np.ndarray]:
    """
    Load and L2-normalize embeddings for Cer-Eval stratification.


    When ``mask`` is provided, it is applied after shape validation against
    ``N_raw`` (pre-filter count); the returned embeddings have shape
    ``(mask.sum(), D)``.

    Returns
    -------
    np.ndarray shape (N, D) float64, L2-normalized rows, or None if file missing.
    """
    emb_path = data_root / dataset / embedding_model / "all_set_embeddings_layerm2_last.pt"
    if not emb_path.is_file():
        return None

    raw = _load_pt(emb_path).float().squeeze()
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    assert raw.ndim == 2, f"Expected 2D embeddings, got shape {raw.shape}"

    # Validate pre-filter shape, apply mask if given.
    raw = _apply_mask_to_tensor(raw, N_raw, mask, emb_path)

    embeddings = raw.numpy().astype(np.float64)
    # L2-normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings


def load_experiment(
    config: dict, ce_nll_filter: dict | None = None,
) -> EvaluationPool:
    """
    Load a (target, surrogate, dataset) experiment from .pt files.

    Config keys: data_root, dataset, target_model, surrogate_model, surrogate_type.
    Optional: loss_type ('accuracy' default), embedding_model (defaults to surrogate_model).
    Path template: {data_root}/{dataset}/{model}/{file}.pt

    When ``ce_nll_filter`` is enabled AND ``loss_type == "cross_entropy"``, the
    filter is applied at load time inside ``_load_losses``. The returned pool
    is internally self-consistent: every N-axis tensor is sliced by the same
    boolean mask before downstream consumption. Accuracy loss silently
    ignores the filter.

    Returns EvaluationPool with ground_truth_losses, surrogate_scores from
    inverted softmax, and optionally distributions and embeddings.
    """
    data_root = Path(config["data_root"]).resolve()
    dataset = config["dataset"]
    target_model = config["target_model"]
    surrogate_model = config["surrogate_model"]
    surrogate_type = config.get("surrogate_type", "auto")
    loss_type = config.get("loss_type", "accuracy")

    if surrogate_type != "none" and surrogate_model == target_model:
        raise ValueError(
            f"surrogate_model '{surrogate_model}' is the same as target_model. "
            "The surrogate must be a different (cheaper) model."
        )

    target_dir = data_root / dataset / target_model
    surrogate_dir = data_root / dataset / surrogate_model

    # Load ground-truth losses (mask built inside _load_losses when filter active)
    ground_truth_losses, loss_metadata, mask = _load_losses(
        target_dir, loss_type, ce_nll_filter=ce_nll_filter,
    )
    N = len(ground_truth_losses)
    N_raw = mask.shape[0] if mask is not None else N

    target_distributions: Optional[np.ndarray] = None
    surrogate_distributions: Optional[np.ndarray] = None
    ell_proxy: Optional[np.ndarray] = None
    num_classes = 0

    if surrogate_type == "none":
        surrogate_scores = np.zeros(N, dtype=np.float64)
    elif surrogate_type == "remark2_oracle_strategy4":
        if loss_type != "accuracy":
            raise ValueError(
                f"{surrogate_type} requires loss_type='accuracy'. "
                "Remark 2 in the draft (lines 455-497) is designed for "
                "0-1 loss; combining it with loss_type='cross_entropy' "
                "is not supported. Use remark1_strategy{1,2,3} for CE experiments."
            )
        target_logits_path = target_dir / "all_set_scores.pt"
        if not target_logits_path.is_file():
            raise ValueError(
                f"{surrogate_type} requires target logits at {target_logits_path}"
            )
        target_logits_local = _load_pt(target_logits_path).float()
        target_logits_local = _apply_mask_to_tensor(
            target_logits_local, N_raw, mask, target_logits_path,
        )
        num_classes = int(target_logits_local.shape[1])
        target_distributions = (
            F.softmax(target_logits_local, dim=1).detach().cpu().numpy().astype(np.float64)
        )

        from save.surrogate_scoring import uncertainty_score

        ell_proxy = ground_truth_losses.copy()
        surrogate_scores = uncertainty_score(target_distributions)
    elif surrogate_type == "remark1_oracle":
        if loss_type != "cross_entropy":
            raise ValueError(
                f"{surrogate_type} requires loss_type='cross_entropy'. "
                "Remark 1 oracle uses true CE; combining with loss_type='accuracy' "
                "is not supported. Use remark2_oracle_strategy4 for accuracy oracles."
            )
        target_logits_path = target_dir / "all_set_scores.pt"
        if not target_logits_path.is_file():
            raise ValueError(
                f"{surrogate_type} requires target logits at {target_logits_path}"
            )
        target_logits_local = _load_pt(target_logits_path).float()
        target_logits_local = _apply_mask_to_tensor(
            target_logits_local, N_raw, mask, target_logits_path,
        )
        num_classes = int(target_logits_local.shape[1])
        target_distributions = (
            F.softmax(target_logits_local, dim=1).detach().cpu().numpy().astype(np.float64)
        )
        from save.surrogate_scoring import predictive_entropy_nats
        # ground_truth_losses already populated by _load_losses (line ~355) — for
        # loss_type='cross_entropy', that's the true CE -log p_f(y|x).
        ell_proxy = ground_truth_losses.copy()
        surrogate_scores = predictive_entropy_nats(target_distributions)
    else:
        # Load surrogate logits
        surrogate_logits = _load_pt(surrogate_dir / "all_set_scores.pt").float()
        assert surrogate_logits.ndim == 2, f"Expected 2D logits, got {surrogate_logits.shape}"
        surrogate_logits = _apply_mask_to_tensor(
            surrogate_logits, N_raw, mask, surrogate_dir / "all_set_scores.pt",
        )
        N_surr, C = surrogate_logits.shape
        num_classes = int(C)

        if N_surr != N:
            raise ValueError(
                f"N mismatch: target has {N} items, surrogate has {N_surr}. "
                "This may happen when mixing base and ICL (_icl4) model variants."
            )

        # Stage 13: Remark 1 surrogate scoring strategies dispatch.
        # Gate: all three strategies require loss_type='cross_entropy' (spec §3.5).
        # Dispatch lives here because this outer scope has access to distributions.
        if surrogate_type.startswith("remark1_"):
            if loss_type != "cross_entropy":
                raise ValueError(
                    f"{surrogate_type} requires loss_type='cross_entropy'. "
                    "Remark 1 in the draft (lines 433-453) is designed for "
                    "continuous losses; combining it with loss_type='accuracy' "
                    "produces proxies on a different scale than the 0/1 true "
                    "loss and is not supported. Use surrogate_type='xgboost' "
                    "or 'inv_max_softmax' for accuracy experiments."
                )
            target_logits_path = target_dir / "all_set_scores.pt"
            if not target_logits_path.is_file():
                raise ValueError(
                    f"{surrogate_type} requires target logits at {target_logits_path}"
                )
            target_logits_local = _load_pt(target_logits_path).float()
            target_logits_local = _apply_mask_to_tensor(
                target_logits_local, N_raw, mask, target_logits_path,
            )
            target_distributions = (
                F.softmax(target_logits_local, dim=1).detach().cpu().numpy().astype(np.float64)
            )
            surrogate_distributions_local = None
            if surrogate_type == "remark1_strategy1":
                surrogate_distributions_local = (
                    F.softmax(surrogate_logits.float(), dim=1)
                    .detach().cpu().numpy().astype(np.float64)
                )

            from save.surrogate_scoring import (
                remark1_strategy1,
                remark1_strategy2,
                remark1_strategy3,
            )

            if surrogate_type == "remark1_strategy1":
                if surrogate_distributions_local is None:
                    raise ValueError("remark1_strategy1 requires surrogate distributions")
                ell_proxy, surrogate_scores = remark1_strategy1(
                    target_distributions, surrogate_distributions_local
                )
            elif surrogate_type == "remark1_strategy2":
                ell_proxy, surrogate_scores = remark1_strategy2(
                    target_distributions, loss_type=loss_type
                )
            elif surrogate_type == "remark1_strategy3":
                ell_proxy, surrogate_scores = remark1_strategy3(
                    target_distributions, loss_type=loss_type
                )
            else:
                raise ValueError(
                    f"Unknown remark1 strategy: {surrogate_type}. "
                    "Supported: remark1_strategy{1,2,3}"
                )
            if surrogate_distributions_local is not None:
                surrogate_distributions = surrogate_distributions_local
        elif surrogate_type.startswith("remark2_"):
            # Stage 15: Remark 2 surrogate scoring strategies dispatch.
            # Gate: all five strategies require loss_type='accuracy' (draft Remark 2).
            # Source: draft_save_v0413.tex Remark 2 lines 455-497.
            if loss_type != "accuracy":
                raise ValueError(
                    f"{surrogate_type} requires loss_type='accuracy'. "
                    "Remark 2 in the draft (lines 455-497) is designed for "
                    "0-1 loss; combining it with loss_type='cross_entropy' "
                    "is not supported. Use remark1_strategy{1,2,3} for CE experiments."
                )
            target_logits_path = target_dir / "all_set_scores.pt"
            if not target_logits_path.is_file():
                raise ValueError(
                    f"{surrogate_type} requires target logits at {target_logits_path}"
                )
            target_logits_local = _load_pt(target_logits_path).float()
            target_logits_local = _apply_mask_to_tensor(
                target_logits_local, N_raw, mask, target_logits_path,
            )
            target_distributions = (
                F.softmax(target_logits_local, dim=1).detach().cpu().numpy().astype(np.float64)
            )
            surrogate_distributions = (
                F.softmax(surrogate_logits.float(), dim=1)
                .detach().cpu().numpy().astype(np.float64)
            )

            from save.surrogate_scoring import (
                remark2_strategy1,
                remark2_strategy2,
                remark2_strategy3,
                remark2_strategy4,
                remark2_strategy5,
            )

            r2_dispatch = {
                "remark2_strategy1": remark2_strategy1,
                "remark2_strategy2": remark2_strategy2,
                "remark2_strategy3": remark2_strategy3,
                "remark2_strategy4": remark2_strategy4,
                "remark2_strategy5": remark2_strategy5,
            }
            strategy_fn = r2_dispatch.get(surrogate_type)
            if strategy_fn is None:
                raise ValueError(
                    f"Unknown remark2 strategy: {surrogate_type}. "
                    "Supported: remark2_strategy{1,2,3,4,5}, "
                    "remark2_oracle_strategy4"
                )
            ell_proxy, surrogate_scores = strategy_fn(
                target_distributions, surrogate_distributions
            )
        else:
            # Auto-select surrogate type
            if surrogate_type == "auto":
                surrogate_type = "inv_confidence_gap" if C == 2 else "inv_max_softmax"

            # Compute surrogate scores — dispatch by type
            if surrogate_type == "surrogate_ce_loss":
                # Oracle-assisted: needs ground-truth labels
                # Source: spec §4.1b
                targets_path = data_root / dataset / "all_set_targets.pt"
                if not targets_path.is_file():
                    raise FileNotFoundError(
                        f"surrogate_ce_loss requires targets file: {targets_path}\n"
                        "This file contains integer class labels for all items."
                    )
                targets = _load_pt(targets_path)
                # Apply mask along dim 0 BEFORE any argmax/squeeze (works for
                # both 1D integer labels and 2D one-hot encodings).
                targets = _apply_mask_to_tensor(
                    targets, N_raw, mask, targets_path,
                )
                if targets.ndim == 2:
                    # One-hot encoded → convert to integer labels
                    targets = targets.argmax(dim=1)
                targets = targets.squeeze()
                assert targets.ndim == 1, f"Expected 1D targets, got shape {targets.shape}"
                assert len(targets) == N, f"Targets length {len(targets)} != N={N}"
                surrogate_scores = compute_surrogate_ce_loss(surrogate_logits, targets)
                if loss_type == "accuracy":
                    warnings.warn(
                        "surrogate_ce_loss paired with loss_type='accuracy': scale mismatch "
                        "(unbounded surrogate with binary loss).",
                        stacklevel=2,
                    )
            elif surrogate_type == "cross_entropy_proxy":
                # Label-free: H(p_surr, p_target) as loss predictor
                # Source: Berrada et al.; Draft Remark 1
                target_logits_path = target_dir / "all_set_scores.pt"
                if not target_logits_path.is_file():
                    raise FileNotFoundError(
                        f"cross_entropy_proxy requires target logits: {target_logits_path}"
                    )
                target_logits_for_proxy = _load_pt(target_logits_path).float()
                target_logits_for_proxy = _apply_mask_to_tensor(
                    target_logits_for_proxy, N_raw, mask, target_logits_path,
                )
                surrogate_scores = compute_cross_entropy_proxy(
                    surrogate_logits, target_logits_for_proxy
                )
            elif surrogate_type in ("xgboost", "xgboost_oracle"):
                # Oracle-assisted XGBoost surrogate — Source: spec §4.2
                from save.surrogate_xgboost import train_xgboost_surrogate
                surr_logits_np = surrogate_logits.detach().cpu().numpy().astype(np.float64)
                tgt_logits_np = None
                if surrogate_type == "xgboost_oracle":
                    tgt_logits_path = target_dir / "all_set_scores.pt"
                    if tgt_logits_path.is_file():
                        _tgt_raw = _load_pt(tgt_logits_path).float()
                        _tgt_raw = _apply_mask_to_tensor(
                            _tgt_raw, N_raw, mask, tgt_logits_path,
                        )
                        tgt_logits_np = _tgt_raw.detach().cpu().numpy().astype(np.float64)
                    else:
                        warnings.warn(
                            "xgboost_oracle requested but target logits unavailable; "
                            "falling back to xgboost (surrogate-only features).",
                            stacklevel=2,
                        )
                seed = config.get("save_config", {}).get("seed", 42)
                surrogate_scores = train_xgboost_surrogate(
                    surr_logits_np, ground_truth_losses, seed, tgt_logits_np
                )
            else:
                surrogate_scores = compute_surrogate_scores(surrogate_logits, surrogate_type)

        assert np.all(np.isfinite(surrogate_scores)), "Surrogate scores contain NaN/Inf"
        # Remark 1 strategies produce nats-scale scores (H(f,x), ℓ_mode(f,x))
        # that can exceed 1.0 for multi-class tasks (e.g. H(f,x) for C=4 can
        # reach log(4) ≈ 1.39). Treat them as unbounded, same as surrogate_ce_loss.
        is_unbounded = (
            surrogate_type in _UNBOUNDED_SURROGATES
            or surrogate_type.startswith("remark1_")
        )
        if not is_unbounded:
            assert surrogate_scores.min() >= 0.0 and surrogate_scores.max() <= 1.0
        else:
            # Clip unbounded surrogates to [0, L] to satisfy the paper's assumption
            # 0 ≤ ℓ̂ ≤ L. [Draft page 5: "the loss predictor is uniformly bounded"]
            # This ensures |ℓ - ŝ| ≤ L for the AIPW correction term bounds.
            L = float(np.max(ground_truth_losses))
            surrogate_scores = np.clip(surrogate_scores, 0.0, L)

        # Correlation check
        corr = float(np.corrcoef(surrogate_scores, ground_truth_losses)[0, 1])
        if np.isfinite(corr) and abs(corr) < 0.1:
            warnings.warn(
                f"Very low surrogate-loss correlation ({corr:.4f}). "
                "AIPW variance reduction may be minimal.",
                stacklevel=2,
            )

        # Distribution loading (both-or-neither)
        if target_distributions is None:
            target_logits_path = target_dir / "all_set_scores.pt"
            if target_logits_path.is_file():
                target_logits = _load_pt(target_logits_path).float()
                target_logits = _apply_mask_to_tensor(
                    target_logits, N_raw, mask, target_logits_path,
                )
                target_distributions = F.softmax(target_logits, dim=1).detach().cpu().numpy().astype(np.float64)
                surrogate_distributions = F.softmax(surrogate_logits, dim=1).detach().cpu().numpy().astype(np.float64)

    # Embedding loading (optional, for Cer-Eval)
    # Source: spec §3 — default embedding_model = surrogate_model
    embedding_model = config.get("embedding_model", surrogate_model)
    embeddings = _load_embeddings(
        data_root, dataset, embedding_model, N_raw=N_raw, mask=mask,
    )

    metadata = {
        "dataset": dataset,
        "target_model": target_model,
        "surrogate_model": surrogate_model,
        "surrogate_type": surrogate_type,
        "num_classes": int(num_classes),
        **loss_metadata,
    }
    if mask is not None:
        metadata["ce_nll_filter"] = {
            "threshold": float(ce_nll_filter["threshold"]),
            "kept": int(N),
            "original_n": int(N_raw),
        }
    else:
        metadata["ce_nll_filter"] = None

    return EvaluationPool(
        N=N,
        surrogate_scores=surrogate_scores,
        ground_truth_losses=ground_truth_losses,
        item_ids=np.arange(N),
        metadata=metadata,
        ell_proxy=ell_proxy,
        target_distributions=target_distributions,
        surrogate_distributions=surrogate_distributions,
        embeddings=embeddings,
    )
