"""Trainer-launch infrastructure shared by every recipe."""

from __future__ import annotations

import os
from typing import Any

from cookbook.common.config import ModalConfig
from stitch.types import VersionRef


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


def deploy_pool_and_spawn(run: Any, *, skip_rollout_ready_check: bool = False) -> Any:
    """Deploy a run's pool, wait for its floor, then spawn its trainer."""
    run.app.deploy()
    return _await_floor_and_spawn(run, skip_rollout_ready_check=skip_rollout_ready_check)


def spawn_on_pool(run: Any, *, skip_rollout_ready_check: bool = False) -> Any:
    """Spawn a run's trainer on its already-deployed pool. Never deploys: a
    missing pool fails fast with the deploy command rather than silently
    replace a live one."""
    if not pool_reachable(run):
        raise SystemExit(
            f"No deployed pool for {run.APP_NAME!r}. Deploy it first:\n"
            f"  EXPERIMENT_CONFIG={os.environ.get('EXPERIMENT_CONFIG', '<experiment>')} "
            f"RUN_ID={os.environ['RUN_ID']} "
            f"uv run --extra modal modal deploy -m {run.__name__}"
        )
    return _await_floor_and_spawn(run, skip_rollout_ready_check=skip_rollout_ready_check)


def await_rollout_ready(
    app_name: str,
    config: ModalConfig,
    *,
    latest: VersionRef | None = None,
    skip: bool = False,
) -> None:
    """Apply the same readiness policy before spawning and inside the trainer."""
    if skip:
        print(f"Skipping rollout readiness check for {app_name}", flush=True)
        return

    from stitch.pools.modal_flash import ModalFlashPool
    from stitch.service import await_pool_ready

    await_pool_ready(
        ModalFlashPool(app_name, "Server"),
        replica_floor=config.rollout_min_containers,
        min_ready=config.rollout_min_ready,
        latest=latest,
    )


def _await_floor_and_spawn(run: Any, *, skip_rollout_ready_check: bool) -> Any:
    await_rollout_ready(
        run.APP_NAME, run.modal_cfg, skip=skip_rollout_ready_check
    )
    if skip_rollout_ready_check:
        return run.spawn_train(skip_rollout_ready_check=True)
    return run.spawn_train()


def pool_reachable(run: Any) -> bool:
    """Whether the run's pool gateway resolves. Only a stopped or never-deployed
    app counts as unreachable; anything else propagates."""
    from modal.exception import NotFoundError

    from stitch.pools.modal_flash import ModalFlashPool

    try:
        ModalFlashPool(run.APP_NAME, "Server").gateway_url()
    except (NotFoundError, RuntimeError):
        return False
    return True
