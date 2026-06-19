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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.protocol import RolloutPoolState, parse_weight_identity, weight_identity


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShimConfig:
    api_base_url: str
    api_key: str | None = None
    provider_model: str | None = None
    provider_deployment: str | None = None
    base_snapshot_identity: str = "base"
    compression_format: str = "zstd"
    checksum_format: str = "xxh3-128"
    reset_prompt_cache: str = "new_session"
    readiness_threshold: float = 1.0
    poll_timeout_seconds: float = 30 * 60
    poll_interval_seconds: float = 5.0

    @classmethod
    def from_env(cls, args: Any | None = None) -> "ShimConfig":
        return cls(
            api_base_url=_setting(
                args,
                "api_shim_base_url",
                "STITCH_SHIM_API_BASE_URL",
                default=getattr(args, "rollout_endpoint_url", None),
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
                default="zstd",
            ),
            checksum_format=_setting(
                args,
                "api_shim_checksum_format",
                "STITCH_SHIM_CHECKSUM_FORMAT",
                default=str(getattr(args, "update_weight_delta_checksum", "xxh3-128")),
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


def announce_and_wait(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook (publish-only mode).

    Disk-delta publish-only writes ``weight_v{N}/`` straight to the mounted S3
    transport, so there is nothing to copy. Rank 0 signals the provider's
    customer hot-load API for the just-written version and blocks until the
    elastic pool reports it ready, so the next rollout only runs once enough
    replicas serve the new weights. (slime also advances its own ``latest``
    pointer; the POST drives the front door — which owns the canonical pointer —
    and gives us the readiness gate.)
    """

    del rollout_engines
    if _distributed_rank() not in (None, 0):
        return
    identity = Path(version_dir).name
    version = parse_weight_identity(identity)
    if version is None:
        # _capture_baseline (the first update_weights) calls the pre-push hook
        # with the disk-dir root, not a published version dir — nothing to
        # announce yet. Only weight_v{N} dirs are hot-loaded.
        return
    cfg = ShimConfig.from_env(args)
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
    # PR #5's apply_rollout_request_hook builds request={url,payload,headers,
    # max_retries,retry_sleep} with NO rollout_id — that is per-rollout context,
    # not per-request (sample.index is the batch position, not the trainer step).
    # So the step-based version pin only runs if a caller supplies a rollout_id.
    # announce_and_wait already blocks the next rollout until the pool serves the
    # target, so skipping the pin is safe; it is belt-and-suspenders admission
    # control for lagging replicas. TODO: re-derive a per-request target under
    # the PR #5 contract (e.g. the latest published version) if pinning is wanted.
    rollout_id = request.get("rollout_id")
    if mode != "none" and rollout_id is not None:
        target_version = _rollout_request_target_version(
            args, int(rollout_id), bool(request.get("evaluation", False))
        )
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


def _rollout_request_target_version(args: Any, rollout_id: int, evaluation: bool) -> int:
    lag = int(
        _setting(
            args,
            "api_shim_rollout_request_version_lag",
            "STITCH_SHIM_ROLLOUT_REQUEST_VERSION_LAG",
            default="0",
        )
    )
    if evaluation:
        lag = 0
    return max(0, int(rollout_id) - lag)


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
