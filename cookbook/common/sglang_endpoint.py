"""SGLang endpoint policy shared by serving and profiling entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

_MODEL_LOADER_CONFIG_ARG = "--model-loader-extra-config"
_Endpoint = TypeVar("_Endpoint")


def configure_fastsafetensors_endpoint(
    endpoint: _Endpoint,
    server_args: Mapping[str, str],
) -> _Endpoint:
    """Make the baked fastsafetensors file the only loader configuration.

    autoinference-utils 0.2.3 injects a historical 64-thread loader argument
    unless a recipe supplies its own override. Remove that default on this
    endpoint instance; other endpoint instances and non-fastsafetensors loads
    keep the dependency's defaults unchanged.
    """
    if server_args.get("--load-format") != "fastsafetensors":
        return endpoint
    if _MODEL_LOADER_CONFIG_ARG in server_args:
        raise ValueError(
            f"{_MODEL_LOADER_CONFIG_ARG} conflicts with FASTSAFETENSORS_CONFIG"
        )

    operational_args = dict(endpoint.DEFAULT_OPERATIONAL_ARGS)  # type: ignore[attr-defined]
    operational_args.pop(_MODEL_LOADER_CONFIG_ARG, None)
    endpoint.DEFAULT_OPERATIONAL_ARGS = operational_args  # type: ignore[attr-defined]
    return endpoint
