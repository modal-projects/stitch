"""Miles SGLang serving image for the disaggregated rollout pool.

The Modal GPU type is selected by each experiment's ``ModalConfig`` in
``modal_train.py``; this image wrapper does not force B200 vs H200. The shared
builder still has a historical B200 name because it started with the Blackwell
rollout path, but the image itself is trainer-agnostic: quantization is driven by
the served checkpoint's config, not by this builder.

This wrapper pins miles as the ``--no-deps`` decoder package, does a full clone
(the miles ref is not a branch tip), and clears the SGLang kernel cache as the
final filesystem step (modal_train mounts a kernel-cache volume at
/root/.cache/sglang, which can't mount over a non-empty path).
"""

from __future__ import annotations

import modal

from cookbook.serving import build_b200_serving_image as _build_shared_serving_image


def build_miles_serving_image(
    *,
    trainer_repo_url: str,
    trainer_repo_ref: str,
    trainer_root: str,
    hf_cache_path: str,
    experiment: str,
) -> modal.Image:
    return _build_shared_serving_image(
        trainer_repo_url=trainer_repo_url,
        trainer_repo_ref=trainer_repo_ref,
        trainer_root=trainer_root,
        hf_cache_path=hf_cache_path,
        experiment=experiment,
        shallow_clone=False,
        clear_sglang_cache_at_end=True,
    )


def build_nvfp4_b200_serving_image(**kwargs) -> modal.Image:
    """Backward-compatible name for existing NVFP4/B200 experiment configs."""
    return build_miles_serving_image(**kwargs)
