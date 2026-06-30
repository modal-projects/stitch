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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cookbook.rollout_control import apply_session_affinity
from cookbook.rollout_control import distributed_rank as _distributed_rank
from cookbook.rollout_control import read_setting as _setting
from stitch.protocol import RolloutPoolState, parse_weight_identity, weight_identity


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShimConfig:
    api_base_url: str
    transport_root: Path | None = None
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
    # Run id: partitions the transport (`<run_id>/weight_v{N}/`) and is folded
    # into the single self-identifying pointer, so a fresh run's chain is isolated
    # and a finished run's pointer can't fast-forward a new run's cold start.
    # None = the run-less customer flat layout.
    run_id: str | None = None

    @classmethod
    def from_env(cls, args: Any | None = None) -> "ShimConfig":
        transport = _setting(
            args, "api_shim_transport_root", "STITCH_SHIM_TRANSPORT_ROOT", default=""
        )
        return cls(
            api_base_url=_setting(
                args,
                "api_shim_base_url",
                "STITCH_SHIM_API_BASE_URL",
                default=getattr(args, "rollout_endpoint_url", None),
                required=True,
            ).rstrip("/"),
            transport_root=Path(transport) if transport else None,
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
            # run_id is NOT read here: a per-launch value can't ride an --api-shim-*
            # arg (dropped by slime's parser) or a late env var (doesn't reach the
            # Ray actor this hook runs in). announce_and_wait derives it from the
            # version dir slime wrote (update_weight_disk_dir = <local>/<run_id>).
        )

    def identity_for_version(self, version: int) -> str:
        return weight_identity(version)

    def previous_identity_for_version(self, version: int) -> str:
        if int(version) <= 1:
            return self.base_snapshot_identity
        return self.identity_for_version(int(version) - 1)

    def transport_path_for_identity(self, identity: str) -> Path:
        if self.transport_root is None:
            raise RuntimeError("transport_root is not configured (api_shim_transport_root)")
        base = self.transport_root / self.run_id if self.run_id else self.transport_root
        return base / identity

    def readiness_identity(self, identity: str) -> str:
        """The run-scoped snapshot identity the pool reports for readiness matching
        (``<run_id>/weight_vN``), so a replica on a finished run's same-numbered
        version isn't miscounted as ready for this run."""
        return f"{self.run_id}/{identity}" if self.run_id else identity


def announce_and_wait(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook (publish-only mode).

    slime publishes ``weight_v{N}/`` to a LOCAL disk dir (its disk-delta writer
    uses atomic rename, which the S3 CloudBucketMount does not support). Rank 0
    copies that version dir to the shared transport the provider pool pulls from,
    then signals the provider's customer hot-load API and blocks until the
    elastic pool reports the version ready — so the next rollout only runs once
    enough replicas serve the new weights. The POST drives the front door, which
    owns the canonical ``latest`` pointer.
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
    # The run id is the partition dir slime wrote into
    # (update_weight_disk_dir = <local>/<run_id>): derive it from the version dir
    # rather than an env var or --api-shim-* arg, neither of which crosses into the
    # Ray training actor this hook runs in. update_weight_disk_dir is a known slime
    # arg, so the partition reliably reaches here.
    run_id = Path(version_dir).parent.name or None
    if run_id:
        cfg = replace(cfg, run_id=run_id)
    # slime's atomic-rename writer can't target the S3 mount (ENOSYS), so the
    # version dir lives on local disk; copy it to the transport (PutObject) the
    # provider pool reads before signalling the hot-load.
    _copy_version_to_transport(Path(version_dir), cfg.transport_path_for_identity(identity))
    _post_hot_load(
        cfg,
        identity=identity,
        previous_identity=cfg.previous_identity_for_version(version),
    )
    target = cfg.readiness_identity(identity)
    state = wait_until_ready(cfg, identity)
    logger.info(
        "Provider hot-load ready for %s: %s/%s replicas",
        target,
        state.ready_count(target_snapshot_identity=target),
        len(state.replicas),
    )


def announce_claim(args: Any, *, run_id: str) -> None:
    """Trainer launch hook (rank 0): claim the rollout pool for this run via the
    front door, the standalone counterpart to :func:`cookbook.bulletin_hooks.claim_pool`.

    POST the empty pointer (``weight_v000000``) under a fresh ``run_id`` so the
    front door's cross-run :func:`decide_pointer_move` resets ``latest`` to base
    *before* the first delta — every replica (cold, or warm on a finished run)
    reconciles to base up front instead of inferring the reset from the first
    publish's run mismatch. Best-effort and non-blocking: the pool may be scaled
    to zero at launch, and the first :func:`announce_and_wait` already blocks on
    readiness, so a transient claim failure costs only the early reset, not
    correctness. ``run_id`` must be fresh per launch (the epoch/fence token).
    """
    if _distributed_rank() not in (None, 0):
        return
    cfg = replace(ShimConfig.from_env(args), run_id=run_id)
    try:
        _post_hot_load(
            cfg,
            identity=weight_identity(0),
            previous_identity=cfg.base_snapshot_identity,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Best-effort pool claim failed for run %r; the first publish's "
            "cross-run reset will still establish base",
            run_id,
            exc_info=True,
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

    # Generous retries on every rollout request so a cold/scaling/lagging replica
    # (a transient 503, or a 409 weight-version reject) is retried rather than
    # failing the rollout — the point of the elastic-pool readiness model. This
    # also rides out a pool that is still warming when rollouts begin.
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

    # The customer authenticates every request, and the provider front door
    # enforces it on all routes (inference included), so the trainer must attach
    # the same auth headers it sends to the hot-load API.
    cfg = ShimConfig.from_env(args)
    headers = dict(request.get("headers") or {})
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.provider_model:
        headers["Provider-Model"] = cfg.provider_model
    if cfg.provider_deployment:
        headers["Provider-Deployment"] = cfg.provider_deployment
    request["headers"] = headers
    # Neutral by default. The front-door relabel proxy maps this to
    # Modal-Session-ID before the gateway.
    affinity_header = _setting(
        args,
        "api_shim_session_affinity_header",
        "STITCH_SHIM_SESSION_AFFINITY_HEADER",
        default="x-session-affinity",
    )
    apply_session_affinity(request, getattr(sample, "session_id", None), affinity_header)


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
    # The pool reports current_snapshot_identity run-scoped (`<run_id>/weight_vN`),
    # so match against the same composite — otherwise a replica still serving a
    # finished run's same-numbered version would falsely count as ready.
    target = cfg.readiness_identity(identity)
    deadline = time.monotonic() + cfg.poll_timeout_seconds
    while True:
        state = _get_hot_load_state(cfg)
        if state.is_ready(
            threshold=cfg.readiness_threshold,
            target_snapshot_identity=target,
        ):
            return state
        if time.monotonic() >= deadline:
            ready = state.ready_count(target_snapshot_identity=target)
            raise TimeoutError(
                f"Timed out waiting for {target}: {ready}/{len(state.replicas)} "
                f"replicas ready at threshold {cfg.readiness_threshold}; last_state={state.to_dict()}"
            )
        logger.info(
            "Waiting for %s readiness: %.3f < %.3f",
            target,
            state.readiness_fraction(target_snapshot_identity=target),
            cfg.readiness_threshold,
        )
        time.sleep(cfg.poll_interval_seconds)


def _post_hot_load(cfg: ShimConfig, *, identity: str, previous_identity: str) -> None:
    payload = {
        "identity": identity,
        # The run id tells the front door which (possibly new) run this belongs to;
        # a new run restarts the version space instead of being a rewind.
        "run_id": cfg.run_id,
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


def _copy_version_to_transport(version_dir: Path, destination: Path) -> None:
    """Copy a locally-published version dir to the (S3-mounted) transport.

    Uses plain writes, not rename: PutObject works on the CloudBucketMount while
    rename does not. On a single-node trainer all ranks share the local FS, so
    rank 0 copies the whole dir (every rank's shard + the index).
    """
    files = sorted(
        path
        for path in version_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".tmp")
    )
    for path in files:
        target = destination / path.relative_to(version_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()  # the transport may reject overwrites
        with path.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)



