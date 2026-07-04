"""Shared trainer-side launch helpers for the disagg cookbook trainers.

slime and miles drive the same launch spine: resolve HF repo ids + materialize
inline YAML configs, build the ``train.py`` command (optionally sourcing a model
arch script), and smoke the deployed Flash rollout pool. The per-trainer
``helpers.py`` modules are thin wrappers that supply the only axes that actually
differ: which config-field tuple to materialize, the model-script attribute
name, and whether the rollout pool has a warm floor or scales from zero.
"""

from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

from stitch.providers.modal import discover_flash_targets, resolve_flash_gateway_url


# ── Config preparation ────────────────────────────────────────────────────────


def prepare_config(cfg: Any, tmpdir: str, yaml_config_fields: Iterable[str]) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs.

    Repo-id-shaped checkpoint fields are snapshot-downloaded from the HF cache;
    absolute paths (already-prepared checkpoints) are left untouched. Inline dict
    configs in ``yaml_config_fields`` are written to ``tmpdir`` and the field is
    repointed at the file the trainer reads.
    """
    from huggingface_hub import snapshot_download
    import yaml

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))

    for field in yaml_config_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field, path)


def build_train_cmd(cfg: Any, trainer_root: str, *, model_script_attr: str) -> str:
    """Build the training command, sourcing model arch args if needed.

    slime/miles ``train.py`` / ``train_async.py`` live at the repo root and consume
    the ``MODEL_ARGS`` bash array defined by the sourced model script.
    ``model_script_attr`` is the config attribute naming that script
    (``slime_model_script`` / ``miles_model_script``).
    """
    train_script = f"{trainer_root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    model_script = getattr(cfg, model_script_attr)
    if model_script:
        inner = (
            f"source {trainer_root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cfg.cli_args())}"


# ── Flash pool smoke check ────────────────────────────────────────────────────


class VersionAheadError(RuntimeError):
    """Raised when a monotonic rollout pool has already advanced past a smoke version."""


def smoke_flash_pool(
    *,
    app_name: str,
    cls_name: str,
    model_name: str,
    weight_version: int,
    expect_min_containers: int,
    timeout_seconds: int,
    wake_on_demand: bool,
) -> None:
    """Poll the Flash gateway until the pool serves completions at the expected
    weight version, via the gateway and each container directly.

    ``wake_on_demand`` chooses the readiness model. False (warm floor): require
    ``expect_min_containers`` live containers, then check each, then complete via
    the gateway. True (scale-from-zero): a completion sent to the gateway is what
    scales the pool 0->1 (Flash holds the request through the cold start, so the
    timeout must be generous); ``expect_min_containers`` is advisory and the
    direct-container check confirms the warmed pool afterward.
    """
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            if wake_on_demand:
                _smoke_wake_on_demand(app_name, cls_name, model_name, weight_version)
            else:
                _smoke_warm_floor(app_name, cls_name, model_name, weight_version, expect_min_containers)
            return
        except VersionAheadError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(f"Flash pool smoke did not pass before timeout: {last_error}")
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def _smoke_warm_floor(
    app_name: str, cls_name: str, model_name: str, expected: int, expect_min_containers: int
) -> None:
    gateway = resolve_flash_gateway_url(app_name, cls_name)
    targets = discover_flash_targets(app_name, cls_name)
    if len(targets) < expect_min_containers:
        raise RuntimeError(
            f"expected at least {expect_min_containers} containers, found {len(targets)}: {targets}"
        )
    print(f"Gateway URL: {gateway}")
    print(f"Direct container URLs ({len(targets)}):")
    for target in targets:
        print(f"  {target}")
    _assert_containers_at_version([gateway, *targets], expected)
    _assert_gateway_completion_exact(gateway, model_name, expected)


def _smoke_wake_on_demand(app_name: str, cls_name: str, model_name: str, expected: int) -> None:
    gateway = resolve_flash_gateway_url(app_name, cls_name)
    print(f"Gateway URL: {gateway}")
    # Wake the (scaled-to-zero) pool and wait for a container to serve. Flash
    # holds the request through the cold start, so the timeout must exceed it.
    data = _post_json(f"{gateway}/v1/chat/completions", _completion_payload(model_name, expected), timeout=900)
    print(f"Gateway completion: {data}")
    start, end = int(data.get("weight_version_start", -1)), int(data.get("weight_version_end", -1))
    if start > expected or end > expected:
        raise VersionAheadError(f"gateway served version {start}->{end}, already past expected {expected}")
    if start != expected or end != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")
    # The pool is warm now; confirm each live container reports the version.
    targets = discover_flash_targets(app_name, cls_name)
    print(f"Direct container URLs ({len(targets)}):")
    _assert_containers_at_version([gateway, *targets], expected)


def _assert_containers_at_version(targets: list[str], expected: int) -> None:
    for target in targets:
        info = _get_json(f"{target}/server_info", timeout=30)
        print(f"{target} server_info={info}")
        current = int(info["current_version"])
        if current > expected:
            raise VersionAheadError(f"{target} current_version={current} already passed expected {expected}")
        if current != expected:
            raise RuntimeError(f"{target} current_version={current} expected {expected}")


def _assert_gateway_completion_exact(gateway: str, model_name: str, expected: int) -> None:
    data = _post_json(f"{gateway}/v1/chat/completions", _completion_payload(model_name, expected), timeout=180)
    print(f"Gateway completion: {data}")
    if int(data.get("weight_version_start", -1)) != expected or int(data.get("weight_version_end", -1)) != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _completion_payload(model_name: str, expected: int) -> dict:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "weight_version": {"exact_version": expected},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)
