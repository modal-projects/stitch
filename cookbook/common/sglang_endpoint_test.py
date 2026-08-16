from __future__ import annotations

import pytest

from .sglang_endpoint import configure_fastsafetensors_endpoint


class _Endpoint:
    DEFAULT_OPERATIONAL_ARGS = {
        "--enable-metrics": "",
        "--model-loader-extra-config": (
            '{"enable_multithread_load":true,"num_threads":64}'
        ),
    }


def test_fastsafetensors_uses_only_baked_config() -> None:
    endpoint = _Endpoint()

    configured = configure_fastsafetensors_endpoint(
        endpoint,
        {"--load-format": "fastsafetensors"},
    )

    assert configured is endpoint
    assert configured.DEFAULT_OPERATIONAL_ARGS == {"--enable-metrics": ""}
    assert "--model-loader-extra-config" in _Endpoint.DEFAULT_OPERATIONAL_ARGS


def test_fastsafetensors_rejects_recipe_override() -> None:
    with pytest.raises(ValueError, match="conflicts with FASTSAFETENSORS_CONFIG"):
        configure_fastsafetensors_endpoint(
            _Endpoint(),
            {
                "--load-format": "fastsafetensors",
                "--model-loader-extra-config": '{"enable_gds":false}',
            },
        )


def test_other_load_format_preserves_endpoint_defaults() -> None:
    endpoint = _Endpoint()

    configure_fastsafetensors_endpoint(endpoint, {"--load-format": "auto"})

    assert endpoint.DEFAULT_OPERATIONAL_ARGS is _Endpoint.DEFAULT_OPERATIONAL_ARGS
