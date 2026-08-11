"""Trainer-launch infrastructure shared by every recipe."""

from __future__ import annotations

import os
from typing import Any


def resolve_config(
    cfg: Any,
    tmpdir: str,
    *,
    checkpoint_fields: tuple[str, ...],
    yaml_fields: tuple[str, ...],
) -> None:
    """Resolve HF repo-id checkpoint fields to local paths and materialize inline YAML
    config dicts to files the trainer reads. Absolute paths are left untouched."""
    import yaml
    from huggingface_hub import snapshot_download

    for attr in checkpoint_fields:
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))
    for field in yaml_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field, path)


def materialize_node_local_yaml(
    cfg: Any, field: str, dest_dir: str = "/root/.node_yaml"
) -> None:
    """Write an inline-dict config field to a deterministic node-local YAML path, so every
    worker re-reads identical content at an identical path — unlike ``resolve_config``'s
    per-launch tmpdir. Call on every node before the rank gate. No-op unless the field is
    a dict; mutates ``cfg`` in place."""
    import yaml

    if isinstance(val := getattr(cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(cfg, field, path)


def deploy_pool_and_spawn(run: Any) -> Any:
    """The shared body of a recipe's ``launch`` entrypoint: deploy this run's pool, block until its
    gateway is healthy, then spawn training. ``run`` is the recipe's ``app`` module (its ``app`` /
    ``APP_NAME`` / ``spawn_train``). Readiness rides the session-routing LB gateway, so the
    check validates the whole traffic path (router + pool), not just the pool. Returns the
    trainer call so the recipe can either detach or supervise it."""
    from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool
    from stitch.service import await_pool_ready

    run.app.deploy()
    await_pool_ready(ModalFlashLBPool(run.APP_NAME, "Server"))
    return run.spawn_train()
