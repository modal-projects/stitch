"""Dedicated B200 NVFP4 SGLang serving image for the miles rollout pool.

A thin wrapper over :func:`cookbook.serving.build_b200_serving_image` — the miles
twin of cookbook/slime_disagg/serving.py. The image itself is trainer-agnostic
(see that module): NVFP4 vs INT4 is driven by the served checkpoint's own quant
config, not by this builder. This wrapper pins miles as the ``--no-deps`` decoder
package, does a full clone (the miles ref is not a branch tip), and clears the
SGLang kernel cache as the final filesystem step (modal_train mounts a
kernel-cache volume at /root/.cache/sglang, which can't mount over a non-empty
path).
"""

from __future__ import annotations

import modal

from cookbook.serving import build_b200_serving_image


def build_nvfp4_b200_serving_image(
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
        shallow_clone=False,
        clear_sglang_cache_at_end=True,
    )
