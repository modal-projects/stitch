"""Optional SLIME trainer-side hooks for a provider hot-load API shim.

These hooks run in SLIME, not in the rollout provider:

1. Copy the completed SLIME disk-delta version directory to the mounted S3 transport.
2. Announce the new snapshot identity to the provider.
3. Poll the provider's pool readiness endpoint until enough replicas report
   the target identity.

They assume the provider can consume the files written by SLIME, or that a
provider-specific materialization step has been inserted before upload.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.protocol import RolloutPoolState, weight_identity


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShimConfig:
    transport_root: Path
    api_base_url: str
    api_key: str | None = None
    provider_model: str | None = None
    provider_deployment: str | None = None
    base_snapshot_identity: str = "base"
    compression_format: str = "deltas_zstd"
    checksum_format: str = "adler32"
    reset_prompt_cache: str = "new_session"
    readiness_threshold: float = 1.0
    poll_timeout_seconds: float = 30 * 60
    poll_interval_seconds: float = 5.0

    @classmethod
    def from_env(cls, args: Any | None = None) -> "ShimConfig":
        return cls(
            transport_root=Path(
                _setting(
                    args,
                    "api_shim_transport_root",
                    "STITCH_SHIM_TRANSPORT_ROOT",
                    required=True,
                )
            ),
            api_base_url=_setting(
                args,
                "api_shim_base_url",
                "STITCH_SHIM_API_BASE_URL",
                default=getattr(args, "rollout_http_endpoint_url", None),
                required=True,
            ).rstrip("/"),
            api_key=_setting(args, "api_shim_api_key", "STITCH_SHIM_API_KEY"),
            provider_model=_setting(
                args, "api_shim_provider_model", "STITCH_SHIM_PROVIDER_MODEL"
            ),
            provider_deployment=_setting(
                args,
                "api_shim_provider_deployment",
                "STITCH_SHIM_PROVIDER_DEPLOYMENT",
            ),
            base_snapshot_identity=_setting(
                args,
                "api_shim_base_snapshot_identity",
                "STITCH_SHIM_BASE_SNAPSHOT_IDENTITY",
                default="base",
            ),
            compression_format=_setting(
                args,
                "api_shim_compression_format",
                "STITCH_SHIM_COMPRESSION_FORMAT",
                default=str(getattr(args, "update_weight_encoding", "deltas_zstd")),
            ),
            checksum_format=_setting(
                args,
                "api_shim_checksum_format",
                "STITCH_SHIM_CHECKSUM_FORMAT",
                default="adler32",
            ),
            reset_prompt_cache=_setting(
                args,
                "api_shim_reset_prompt_cache",
                "STITCH_SHIM_RESET_PROMPT_CACHE",
                default="new_session",
            ),
            readiness_threshold=float(
                _setting(
                    args,
                    "api_shim_readiness_threshold",
                    "STITCH_SHIM_READINESS_THRESHOLD",
                    default="1.0",
                )
            ),
            poll_timeout_seconds=float(
                _setting(
                    args,
                    "api_shim_poll_timeout_seconds",
                    "STITCH_SHIM_POLL_TIMEOUT_SECONDS",
                    default="1800",
                )
            ),
            poll_interval_seconds=float(
                _setting(
                    args,
                    "api_shim_poll_interval_seconds",
                    "STITCH_SHIM_POLL_INTERVAL_SECONDS",
                    default="5",
                )
            ),
        )

    def identity_for_version(self, version: int) -> str:
        return weight_identity(version)

    def previous_identity_for_version(self, version: int) -> str:
        if int(version) <= 1:
            return self.base_snapshot_identity
        return self.identity_for_version(int(version) - 1)

    def transport_path_for_identity(self, identity: str) -> Path:
        return self.transport_root / identity


def copy_delta_to_transport(
    args: Any, version_dir: str, rollout_engines: list[Any]
) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook.

    SLIME calls this on every training rank after that rank has finished
    writing its sparse delta files. Each rank copies the rank-prefixed files it
    can see into the mounted S3 transport path. On non-distributed local runs,
    the hook copies every file under
    ``version_dir``.
    """

    del rollout_engines
    cfg = ShimConfig.from_env(args)
    identity = Path(version_dir).name
    files = _uploadable_files(Path(version_dir), rank=_distributed_rank())
    if not files:
        logger.info("No local delta files to copy for %s", version_dir)
        return

    destination = cfg.transport_path_for_identity(identity)
    for path in files:
        rel = path.relative_to(version_dir).as_posix()
        target = destination / rel
        _replace_transport_file(path, target)
        logger.info("Copied %s to %s", path, target)


# Backward-compatible name for configs written before the transport moved from
# direct boto3 uploads to Modal's CloudBucketMount filesystem.
upload_delta_to_s3 = copy_delta_to_transport


def _replace_transport_file(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` on transports that reject overwrites."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def publish_delta_to_hot_load(
    args: Any,
    version_dir: str,
    files: list[str],
    weight_version: str | int,
    rollout_engines: list[Any],
) -> list[Any]:
    """SLIME ``custom_delta_publish_path`` hook.

    Rank 0 calls this once per published SLIME version after every rank's
    ``copy_delta_to_transport`` hook has completed. The hook blocks until the
    provider reports enough replicas on the announced snapshot identity.
    """

    del version_dir, files, rollout_engines
    version = int(weight_version)
    cfg = ShimConfig.from_env(args)
    identity = cfg.identity_for_version(version)
    _post_hot_load(
        cfg,
        identity=identity,
        previous_identity=cfg.previous_identity_for_version(version),
    )
    state = wait_until_ready(cfg, identity)
    logger.info(
        "Provider hot-load ready for %s: %s/%s replicas",
        identity,
        state.ready_count(target_snapshot_identity=identity),
        len(state.replicas),
    )
    return []


def rollout_request_weight_version_hook(
    args: Any, sample: Any, request: dict[str, Any]
) -> None:
    """SLIME ``custom_rollout_request_hook_path`` hook.

    The publish hook has already decided which versions are usable by polling the
    provider pool. This request hook turns that trainer decision into provider
    admission control, so requests routed to lagging replicas fail with a
    retryable 409 instead of producing unusable samples.
    """

    mode = str(
        getattr(
            args,
            "api_shim_rollout_request_weight_version_mode",
            os.environ.get("STITCH_SHIM_ROLLOUT_REQUEST_WEIGHT_VERSION_MODE", "exact"),
        )
    )
    if mode == "none":
        return

    target_version = _rollout_request_target_version(args, request)
    if mode == "exact":
        request["payload"]["weight_version"] = {"exact_version": target_version}
    elif mode == "min":
        request["payload"]["weight_version"] = {"min_required_version": target_version}
    else:
        raise ValueError(
            f"Unsupported api_shim_rollout_request_weight_version_mode: {mode!r}"
        )

    request["max_retries"] = int(
        _setting(
            args,
            "api_shim_rollout_request_retry_attempts",
            "STITCH_SHIM_ROLLOUT_REQUEST_RETRY_ATTEMPTS",
            default="60",
        )
    )
    request["retry_sleep"] = float(
        _setting(
            args,
            "api_shim_rollout_request_retry_sleep",
            "STITCH_SHIM_ROLLOUT_REQUEST_RETRY_SLEEP",
            default="1.0",
        )
    )

    if getattr(sample, "session_id", None):
        # Neutral by default. External clients send this through the front-door
        # relabel proxy, which maps it to Modal-Session-ID before the gateway.
        affinity_header = _setting(
            args,
            "api_shim_session_affinity_header",
            "STITCH_SHIM_SESSION_AFFINITY_HEADER",
            default="x-session-affinity",
        )
        headers = dict(request.get("headers") or {})
        headers.setdefault(affinity_header, sample.session_id)
        request["headers"] = headers


def _rollout_request_target_version(args: Any, request: dict[str, Any]) -> int:
    rollout_id = int(request["rollout_id"])
    lag = int(
        _setting(
            args,
            "api_shim_rollout_request_version_lag",
            "STITCH_SHIM_ROLLOUT_REQUEST_VERSION_LAG",
            default="0",
        )
    )
    if bool(request.get("evaluation", False)):
        lag = 0
    return max(0, rollout_id - lag)


def wait_until_ready(cfg: ShimConfig, identity: str) -> RolloutPoolState:
    deadline = time.monotonic() + cfg.poll_timeout_seconds
    while True:
        state = _get_hot_load_state(cfg)
        if state.is_ready(
            threshold=cfg.readiness_threshold,
            target_snapshot_identity=identity,
        ):
            return state
        if time.monotonic() >= deadline:
            ready = state.ready_count(target_snapshot_identity=identity)
            raise TimeoutError(
                f"Timed out waiting for {identity}: {ready}/{len(state.replicas)} "
                f"replicas ready at threshold {cfg.readiness_threshold}; last_state={state.to_dict()}"
            )
        logger.info(
            "Waiting for %s readiness: %.3f < %.3f",
            identity,
            state.readiness_fraction(target_snapshot_identity=identity),
            cfg.readiness_threshold,
        )
        time.sleep(cfg.poll_interval_seconds)


def _post_hot_load(cfg: ShimConfig, *, identity: str, previous_identity: str) -> None:
    payload = {
        "identity": identity,
        "incremental_snapshot_metadata": {
            "previous_snapshot_identity": previous_identity,
            "compression_format": cfg.compression_format,
            "checksum_format": cfg.checksum_format,
        },
        "reset_prompt_cache": cfg.reset_prompt_cache,
    }
    _request_json(
        f"{cfg.api_base_url}/hot_load/v1/models/hot_load",
        method="POST",
        headers=_headers(cfg),
        payload=payload,
    )


def _get_hot_load_state(cfg: ShimConfig) -> RolloutPoolState:
    payload = _request_json(
        f"{cfg.api_base_url}/hot_load/v1/models/hot_load",
        method="GET",
        headers=_headers(cfg),
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"hot-load readiness response must be a JSON object, got {type(payload).__name__}"
        )
    return RolloutPoolState.from_dict(payload)


def _request_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} failed with HTTP {exc.code}: {detail}"
        ) from exc
    if not content:
        return {}
    return json.loads(content.decode("utf-8"))


def _headers(cfg: ShimConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.provider_model:
        headers["Provider-Model"] = cfg.provider_model
    if cfg.provider_deployment:
        headers["Provider-Deployment"] = cfg.provider_deployment
    return headers


def _uploadable_files(version_dir: Path, *, rank: int | None) -> list[Path]:
    if not version_dir.exists():
        return []
    files = sorted(
        path
        for path in version_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".tmp")
    )
    if rank is None:
        return files
    rank_prefix = f"rank{rank:04d}_"
    return [path for path in files if path.name.startswith(rank_prefix)]


def _distributed_rank() -> int | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def _setting(
    args: Any | None,
    attr: str,
    env: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    value = getattr(args, attr, None) if args is not None else None
    if value is None:
        value = os.environ.get(env)
    if value is None:
        value = default
    if value is None and required:
        raise RuntimeError(
            f"Missing required setting {attr!r} or environment variable {env}"
        )
    return str(value) if value is not None else ""
