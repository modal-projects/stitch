"""Miles SGLang serving image for the disaggregated rollout pool.

The Modal GPU type is selected by each experiment's ``ModalConfig`` in
``modal_train.py``; this image wrapper does not force B200 vs H200. The shared
builder still has a historical B200 name because it started with the Blackwell
rollout path, but the image itself is trainer-agnostic: quantization is driven
by the served checkpoint's config, and the delta apply lives in the engine
(``/pull_weights``), so no trainer package is installed.

This wrapper clears the SGLang kernel cache as the final filesystem step
(modal_train mounts a kernel-cache volume at /root/.cache/sglang, which can't
mount over a non-empty path).
"""

from __future__ import annotations

import modal

from cookbook.serving import build_b200_serving_image as _build_shared_serving_image


def build_miles_serving_image(
    *,
    hf_cache_path: str,
    experiment: str,
    delta_volume_name: str,
) -> modal.Image:
    return _build_shared_serving_image(
        hf_cache_path=hf_cache_path,
        experiment=experiment,
        delta_volume_name=delta_volume_name,
        clear_sglang_cache_at_end=True,
    )


def build_nvfp4_b200_serving_image(**kwargs) -> modal.Image:
    """Backward-compatible name for existing NVFP4/B200 experiment configs."""
    return build_miles_serving_image(**kwargs)
