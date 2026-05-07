# src/save/paper_experiment/cell_paths.py
"""Canonical paths for paper-experiment cell .npz files (spec §4 output schema).

Field separator is ``__`` (double underscore) so that model names containing
single underscores (``llama2_7b``, ``Mixtral_8x7b``, ``deepseek_67b``,
``qwen25_72b``) are unambiguously parseable.
"""
from __future__ import annotations

from pathlib import Path


_VALID_METHODS = frozenset({"M1", "M2", "M3", "M4", "M5"})
_VALID_LOSSES = frozenset({"accuracy", "cross_entropy"})
_SEP = "__"


def _format_beta(beta_min: float) -> str:
    return f"{float(beta_min):g}"


def main_cell_path(
    base: Path,
    method: str,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
) -> Path:
    if method not in _VALID_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join(["cell", method, dataset, surrogate, target, loss]) + ".npz"
    return base / "trajectories" / "main" / name


def main_subcell_path(
    base: Path,
    method: str,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    chunk: int,
) -> Path:
    """Sub-cell path for one seed-chunk of a main cell.

    Sub-cells are intermediate (see Task 13b). The merge stage consolidates
    all sub-cells for a (method,dataset,surrogate,target,loss) key into a
    bundled ``cell__...npz`` and deletes them.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "subcell", method, dataset, surrogate, target, loss,
        f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return base / "_subcells" / "main" / name


def ce_sweep_cell_path(
    base: Path,
    dataset: str,
    surrogate: str,
    target: str,
    surrogate_type: str,
    beta_min: float,
) -> Path:
    name = _SEP.join([
        "cell", dataset, surrogate, target, surrogate_type,
        f"beta{_format_beta(beta_min)}",
    ]) + ".npz"
    return base / "trajectories" / "ce_sweep" / name


def ce_sweep_subcell_path(
    base: Path,
    dataset: str,
    surrogate: str,
    target: str,
    surrogate_type: str,
    beta_min: float,
    chunk: int,
) -> Path:
    name = _SEP.join([
        "subcell", dataset, surrogate, target, surrogate_type,
        f"beta{_format_beta(beta_min)}", f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return base / "_subcells" / "ce_sweep" / name


def beta_sweep_cell_path(
    base: Path,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    beta_min: float,
) -> Path:
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "cell", dataset, surrogate, target, loss,
        f"beta{_format_beta(beta_min)}",
    ]) + ".npz"
    return base / "trajectories" / "beta_sweep" / name


def beta_sweep_subcell_path(
    base: Path,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    beta_min: float,
    chunk: int,
) -> Path:
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "subcell", dataset, surrogate, target, loss,
        f"beta{_format_beta(beta_min)}", f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return base / "_subcells" / "beta_sweep" / name


def oracle_accuracy_cell_path(
    base: Path,
    dataset: str,
    surrogate: str,
    target: str,
    surrogate_type: str,
) -> Path:
    name = _SEP.join([
        "cell", "oracle_accuracy", dataset, surrogate, target, surrogate_type,
    ]) + ".npz"
    return base / "trajectories" / "oracle_accuracy" / name


def oracle_accuracy_subcell_path(
    base: Path,
    dataset: str,
    surrogate: str,
    target: str,
    surrogate_type: str,
    chunk: int,
) -> Path:
    name = _SEP.join([
        "subcell", "oracle_accuracy", dataset, surrogate, target,
        surrogate_type, f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return base / "_subcells" / "oracle_accuracy" / name


def acquisition_sweep_cell_path(
    base: Path,
    *,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    surrogate_type: str,
) -> Path:
    """Final cell .npz path for §6.5 acquisition-sweep trajectories.

    Filename: ``cell__acquisition_sweep__{dataset}__{surrogate}__{target}__{loss}__{surrogate_type}.npz``
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "cell", "acquisition_sweep", dataset, surrogate, target, loss,
        surrogate_type,
    ]) + ".npz"
    return Path(base) / "trajectories" / "acquisition_sweep" / name


def acquisition_sweep_subcell_path(
    base: Path,
    *,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    surrogate_type: str,
    chunk: int,
) -> Path:
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "subcell", "acquisition_sweep", dataset, surrogate, target, loss,
        surrogate_type, f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return Path(base) / "_subcells" / "acquisition_sweep" / name


def wallclock_cell_path(
    base: Path,
    *,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str = "accuracy",
) -> Path:
    """Final cell .npz path for §6.5 wallclock trajectories (Task 9).

    Filename: ``cell__wallclock__{dataset}__{surrogate}__{target}__{loss}.npz``.

    Cer-Eval (M5) is unparameterized w.r.t. surrogate_type / config_name —
    the algorithm uses only embeddings and ground-truth losses — so the
    filename carries only the four core axes.
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "cell", "wallclock", dataset, surrogate, target, loss,
    ]) + ".npz"
    return Path(base) / "trajectories" / "wallclock" / name


def wallclock_subcell_path(
    base: Path,
    *,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    chunk: int,
) -> Path:
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "subcell", "wallclock", dataset, surrogate, target, loss,
        f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return Path(base) / "_subcells" / "wallclock" / name


def hparam_sweep_cell_path(
    base: Path,
    *,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    config_name: str,
) -> Path:
    """Final cell .npz path for §6.5 hparam-sweep trajectories.

    Filename: ``cell__hparam_sweep__{dataset}__{surrogate}__{target}__{loss}__{config_name}.npz``
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "cell", "hparam_sweep", dataset, surrogate, target, loss,
        config_name,
    ]) + ".npz"
    return Path(base) / "trajectories" / "hparam_sweep" / name


def hparam_sweep_subcell_path(
    base: Path,
    *,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    config_name: str,
    chunk: int,
) -> Path:
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    name = _SEP.join([
        "subcell", "hparam_sweep", dataset, surrogate, target, loss,
        config_name, f"chunk{int(chunk):03d}",
    ]) + ".npz"
    return Path(base) / "_subcells" / "hparam_sweep" / name


def _strip(filename: str, prefix: str) -> list[str]:
    if not filename.startswith(prefix) or not filename.endswith(".npz"):
        raise ValueError(f"not a {prefix!r}-prefixed .npz: {filename!r}")
    stem = filename[: -len(".npz")]
    tokens = stem.split(_SEP)
    if tokens[0] != prefix[: -len(_SEP)]:
        raise ValueError(f"unexpected prefix token in {filename!r}: {tokens[0]!r}")
    return tokens[1:]


def parse_main_cell_filename(filename: str) -> dict:
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 5:
        raise ValueError(
            f"main cell filename has {len(tokens)} tokens (expected 5): {filename!r}"
        )
    method, dataset, surrogate, target, loss = tokens
    if method not in _VALID_METHODS:
        raise ValueError(f"unknown method in filename: {method!r}")
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss in filename: {loss!r}")
    return {
        "method": method, "dataset": dataset,
        "surrogate": surrogate, "target": target, "loss": loss,
    }


def _parse_beta_token(tok: str) -> float:
    if not tok.startswith("beta"):
        raise ValueError(f"expected beta token: {tok!r}")
    return float(tok[len("beta"):])


def parse_ce_sweep_cell_filename(filename: str) -> dict:
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 5:
        raise ValueError(
            f"ce_sweep cell filename has {len(tokens)} tokens (expected 5): {filename!r}"
        )
    dataset, surrogate, target, surrogate_type, beta_tok = tokens
    return {
        "dataset": dataset, "surrogate": surrogate, "target": target,
        "surrogate_type": surrogate_type, "beta_min": _parse_beta_token(beta_tok),
    }


def parse_beta_sweep_cell_filename(filename: str) -> dict:
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 5:
        raise ValueError(
            f"beta_sweep cell filename has {len(tokens)} tokens (expected 5): {filename!r}"
        )
    dataset, surrogate, target, loss, beta_tok = tokens
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss in filename: {loss!r}")
    return {
        "dataset": dataset, "surrogate": surrogate, "target": target,
        "loss": loss, "beta_min": _parse_beta_token(beta_tok),
    }


def parse_oracle_accuracy_cell_filename(filename: str) -> dict:
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 5:
        raise ValueError(
            f"oracle_accuracy cell filename has {len(tokens)} tokens (expected 5): {filename!r}"
        )
    stage_tok, dataset, surrogate, target, surrogate_type = tokens
    if stage_tok != "oracle_accuracy":
        raise ValueError(f"unknown oracle stage token in filename: {stage_tok!r}")
    return {
        "dataset": dataset,
        "surrogate": surrogate,
        "target": target,
        "loss": "accuracy",
        "surrogate_type": surrogate_type,
    }


def parse_acquisition_sweep_cell_filename(filename: str) -> dict:
    """Parse ``cell__acquisition_sweep__{dataset}__{surrogate}__{target}__{loss}__{surrogate_type}.npz``."""
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 6 or tokens[0] != "acquisition_sweep":
        raise ValueError(
            f"unrecognised acquisition_sweep filename: {filename!r}"
        )
    _, dataset, surrogate, target, loss, surrogate_type = tokens
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss in filename: {loss!r}")
    return {
        "dataset": dataset,
        "surrogate": surrogate,
        "target": target,
        "loss": loss,
        "surrogate_type": surrogate_type,
    }


def parse_wallclock_cell_filename(filename: str) -> dict:
    """Parse ``cell__wallclock__{dataset}__{surrogate}__{target}__{loss}.npz``."""
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 5 or tokens[0] != "wallclock":
        raise ValueError(
            f"unrecognised wallclock filename: {filename!r}"
        )
    _, dataset, surrogate, target, loss = tokens
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss in filename: {loss!r}")
    return {
        "dataset": dataset,
        "surrogate": surrogate,
        "target": target,
        "loss": loss,
    }


def parse_hparam_sweep_cell_filename(filename: str) -> dict:
    """Parse ``cell__hparam_sweep__{dataset}__{surrogate}__{target}__{loss}__{config_name}.npz``."""
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) != 6 or tokens[0] != "hparam_sweep":
        raise ValueError(
            f"unrecognised hparam_sweep filename: {filename!r}"
        )
    _, dataset, surrogate, target, loss, config_name = tokens
    if loss not in _VALID_LOSSES:
        raise ValueError(f"unknown loss in filename: {loss!r}")
    return {
        "dataset": dataset,
        "surrogate": surrogate,
        "target": target,
        "loss": loss,
        "config_name": config_name,
    }


def classify_cell_filename(filename: str) -> str:
    """Return one of ``"main"`` / ``"ce_sweep"`` / ``"beta_sweep"``.

    Disambiguates by token count + known method prefix. Raises if ambiguous.
    """
    tokens = _strip(filename, "cell" + _SEP)
    if len(tokens) == 5 and tokens[0] in _VALID_METHODS:
        return "main"
    if len(tokens) == 5 and tokens[3] in _VALID_LOSSES:
        return "beta_sweep"
    if len(tokens) == 5 and tokens[4].startswith("beta"):
        return "ce_sweep"
    raise ValueError(f"cannot classify: {filename!r}")
