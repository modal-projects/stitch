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


def spawn_on_pool(run: Any) -> Any:
    """Spawn a run's trainer on its deployed pool. The launch path never deploys
    or redeploys — the pool's whole lifecycle belongs to explicit ``modal
    deploy`` invocations of the run's app — so a missing pool fails fast with
    that instruction instead."""
    from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool
    from stitch.service import await_pool_ready

    pool = ModalFlashLBPool(run.APP_NAME, "Server")
    if not _pool_reachable(pool):
        raise SystemExit(
            f"No deployed pool for {run.APP_NAME!r}. Deploy it first:\n"
            f"  EXPERIMENT_CONFIG={os.environ.get('EXPERIMENT_CONFIG', '<experiment>')} "
            f"RUN_ID={os.environ['RUN_ID']} "
            f"uv run --extra modal modal deploy -m {run.__name__}"
        )
    await_pool_ready(pool, replica_floor=run.modal_cfg.rollout_min_containers)
    return run.spawn_train()


def _pool_reachable(pool: Any) -> bool:
    """Whether the pool's gateway resolves; a stopped or never-deployed app does
    not. Only those two outcomes fail the launch — an unexpected error propagates
    rather than second-guess a live pool."""
    from modal.exception import NotFoundError

    try:
        pool.gateway_url()
    except (NotFoundError, RuntimeError):
        return False
    return True
