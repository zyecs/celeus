"""Access to the method-parameter block of an experiment config.

The block is called ``celeus_config``. Configs written before the project
was renamed call it ``save_config``; this module is the only place that
accepts the old name, so old configs still load instead of silently losing
their method parameters.
"""
from __future__ import annotations

import warnings

METHOD_CONFIG_KEY = "celeus_config"
_LEGACY_METHOD_CONFIG_KEY = "save_config"


def method_config(raw_config: dict) -> dict:
    """Return the method-parameter block of ``raw_config`` (empty if absent)."""
    if METHOD_CONFIG_KEY in raw_config:
        if _LEGACY_METHOD_CONFIG_KEY in raw_config:
            raise ValueError(
                f"config sets both {METHOD_CONFIG_KEY!r} and the legacy "
                f"{_LEGACY_METHOD_CONFIG_KEY!r}; keep only {METHOD_CONFIG_KEY!r}"
            )
        return raw_config[METHOD_CONFIG_KEY]
    if _LEGACY_METHOD_CONFIG_KEY in raw_config:
        warnings.warn(
            f"config key {_LEGACY_METHOD_CONFIG_KEY!r} is deprecated; "
            f"rename it to {METHOD_CONFIG_KEY!r}",
            DeprecationWarning,
            stacklevel=2,
        )
        return raw_config[_LEGACY_METHOD_CONFIG_KEY]
    return {}
