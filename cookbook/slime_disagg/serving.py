"""Dedicated B200 native-INT4 SGLang serving image for the slime rollout pool.

A thin wrapper over :func:`cookbook.serving.build_b200_serving_image`. The image
itself is trainer-agnostic (see that module): INT4 vs NVFP4 is driven by the
served checkpoint's own ``compressed-tensors`` config, not by this builder, and
the delta apply lives in the engine (``/pull_weights``) so no trainer package
is installed.
"""

from __future__ import annotations

import modal

from cookbook.serving import build_b200_serving_image


def build_int4_b200_serving_image(
    *,
    hf_cache_path: str,
    experiment: str,
    delta_volume_name: str = "",
) -> modal.Image:
    return build_b200_serving_image(
        hf_cache_path=hf_cache_path,
        experiment=experiment,
        delta_volume_name=delta_volume_name,
    )
