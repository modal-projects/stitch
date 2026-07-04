"""Dedicated B200 native-INT4 SGLang serving image for the slime rollout pool.

A thin wrapper over :func:`cookbook.serving.build_b200_serving_image`. The image
itself is trainer-agnostic (see that module): INT4 vs NVFP4 is driven by the
served checkpoint's own ``compressed-tensors`` config, not by this builder. This
wrapper only pins slime as the ``--no-deps`` decoder package and does a shallow
clone (the slime ref is a branch-tip commit).
"""

from __future__ import annotations

import modal

from cookbook.serving import build_b200_serving_image


def build_int4_b200_serving_image(
    *,
    trainer_repo_url: str,
    trainer_repo_ref: str,
    trainer_root: str,
    hf_cache_path: str,
    experiment: str,
) -> modal.Image:
    return build_b200_serving_image(
        trainer_repo_url=trainer_repo_url,
        trainer_repo_ref=trainer_repo_ref,
        trainer_root=trainer_root,
        hf_cache_path=hf_cache_path,
        experiment=experiment,
        shallow_clone=True,
    )
