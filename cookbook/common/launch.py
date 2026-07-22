"""Trainer-launch helpers shared by every recipe: build the train command (sourcing the
model-arch MODEL_ARGS script) and resolve/materialize a config before launch. Both are
framework-agnostic given the framework's root + model-script attribute + YAML fields.
"""

from __future__ import annotations

import os
import shlex
from typing import Any


def build_train_cmd(cfg: Any, root: str, model_script_attr: str) -> str:
    """The train command. ``train_async.py`` / ``train.py`` live at the framework root
    and consume the ``MODEL_ARGS`` bash array defined by the sourced model script."""
    train_script = f"{root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    model_script = getattr(cfg, model_script_attr, "")
    if model_script:
        inner = (
            f"source {root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cfg.cli_args())}"


def resolve_config(cfg: Any, tmpdir: str, yaml_fields: tuple[str, ...]) -> None:
    """Resolve HF repo-id checkpoint fields to local paths and materialize inline YAML
    config dicts to files the trainer reads. Absolute paths are left untouched."""
    from huggingface_hub import snapshot_download
    import yaml

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))
    for field in yaml_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field, path)


def materialize_node_local_yaml(cfg: Any, field: str, dest_dir: str = "/root/.node_yaml") -> None:
    """Write an inline-dict config field to a deterministic node-local YAML path, so every Ray
    actor across nodes re-reads identical content at an identical path — unlike ``resolve_config``'s
    per-launch tmpdir. Call on every node (before the rank gate). No-op unless the field is a dict;
    mutates ``cfg`` in place. (e.g. miles' ``te_precision_config_file``.)"""
    import yaml

    if isinstance(val := getattr(cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(cfg, field, path)
