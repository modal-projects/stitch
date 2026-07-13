"""Optional SLIME trainer-side hooks for a provider hot-load API shim.

These hooks run in SLIME, not in the rollout provider:

1. Copy the completed SLIME disk-delta version directory to the mounted S3 transport.
2. Announce the new snapshot identity to the provider.
3. Poll the provider's pool readiness endpoint until every live replica reports
   the target identity.

They assume the provider can consume the files written by SLIME, or that a
provider-specific materialization step has been inserted before upload.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookbook.rollout_control import apply_session_affinity
from cookbook.rollout_control import distributed_rank as _distributed_rank
from cookbook.rollout_control import read_setting as _setting
from cookbook.standalone_rollouts.ledger import (
    LEDGER_FILENAME,
    IdentityLedger,
    LedgerCorruption,
    load_ledger_data,
)
from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import RolloutPoolState, parse_weight_identity, weight_identity


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShimConfig:
    api_base_url: str
    base_snapshot_identity: str
    transport_root: Path | None = None
    api_key: str | None = None
    provider_model: str | None = None
    provider_deployment: str | None = None
    checksum_format: str = "xxh3-128"
    poll_timeout_seconds: float = 30 * 60
    poll_interval_seconds: float = 5.0

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
                required=True,
            ),
            checksum_format=_setting(
                args,
                "api_shim_checksum_format",
                "STITCH_SHIM_CHECKSUM_FORMAT",
                default=str(getattr(args, "update_weight_delta_checksum", "xxh3-128")),
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
        if self.transport_root is None:
            raise RuntimeError(
                "transport_root is not configured (api_shim_transport_root)"
            )
        return self.transport_root / identity


def assert_clean_transport(cfg: ShimConfig) -> None:
    """Require the base-only state used to start one append-only training run."""
    if cfg.transport_root is None:
        raise RuntimeError("transport_root is not configured (api_shim_transport_root)")
    persisted = load_ledger_data(cfg.transport_root)
    if persisted is None:
        raise RuntimeError(
            "identity ledger is not initialized; deploy the front door first"
        )
    try:
        ledger = IdentityLedger.from_dict(
            persisted, expected_base_identity=cfg.base_snapshot_identity
        )
        run_id, pointer_version = FilesystemBulletinBoard(
            cfg.transport_root, layout="slime"
        ).read_latest()
    except (LedgerCorruption, ValueError) as exc:
        raise RuntimeError(f"transport control state is invalid: {exc}") from exc
    if run_id is not None or ledger.head_version != 0 or pointer_version != 0:
        raise RuntimeError(
            "transport prefix already contains a training chain; use a fresh prefix "
            "and redeploy for an independent run"
        )

    allowed = {LEDGER_FILENAME, "latest"}
    leftovers = sorted(
        path.name for path in cfg.transport_root.iterdir() if path.name not in allowed
    )
    if leftovers:
        raise RuntimeError(
            f"transport prefix contains leftover uploads {leftovers!r}; use a fresh prefix"
        )


def announce_and_wait(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook (publish-only mode).

    slime publishes ``weight_v{N}/`` to a LOCAL disk dir (its disk-delta writer
    uses atomic rename, which the S3 CloudBucketMount does not support). Rank 0
    copies that version dir to the shared transport the provider pool pulls from,
    then signals the provider's customer hot-load API and blocks until the
    elastic pool reports the version ready — so the next rollout only runs once
    every live replica serves the new weights. The POST drives the front door, which
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
    if version == 0:
        raise RuntimeError(
            "SLIME v0 is the configured base and must not be uploaded as a delta"
        )
    cfg = ShimConfig.from_env(args)
    # slime's atomic-rename writer can't target the S3 mount (ENOSYS), so the
    # version dir lives on local disk; copy it to the transport (PutObject) the
    # provider pool reads before signalling the hot-load.
    _copy_version_to_transport(
        Path(version_dir), cfg.transport_path_for_identity(identity)
    )
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
    """SLIME ``custom_rollout_request_hook_path`` hook: retry budget, provider
    auth headers, and session affinity for one rollout request.

    Version admission needs no per-request pin here: announce_and_wait blocks
    the next rollout until every live replica serves the target, while a newly
    started replica synchronizes to the latest version before becoming ready.
    """
    # Generous retries on every rollout request so transient cold-start or
    # scaling failures do not fail the rollout. This also rides out a pool that
    # is still warming when rollouts begin.
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
    apply_session_affinity(
        request, getattr(sample, "session_id", None), affinity_header
    )


def wait_until_ready(cfg: ShimConfig, identity: str) -> RolloutPoolState:
    target = identity
    deadline = time.monotonic() + cfg.poll_timeout_seconds
    while True:
        state = _get_hot_load_state(cfg)
        if state.is_ready(
            threshold=1.0,
            target_snapshot_identity=target,
        ):
            return state
        if time.monotonic() >= deadline:
            ready = state.ready_count(target_snapshot_identity=target)
            raise TimeoutError(
                f"Timed out waiting for {target}: {ready}/{len(state.replicas)} "
                f"replicas ready; last_state={state.to_dict()}"
            )
        logger.info(
            "Waiting for %s readiness: %.3f < %.3f",
            target,
            state.readiness_fraction(target_snapshot_identity=target),
            1.0,
        )
        time.sleep(cfg.poll_interval_seconds)


def _post_hot_load(cfg: ShimConfig, *, identity: str, previous_identity: str) -> None:
    payload = {
        "identity": identity,
        "incremental_snapshot_metadata": {
            "previous_snapshot_identity": previous_identity,
            "compression_format": "zstd",
            "checksum_format": cfg.checksum_format,
        },
        "reset_prompt_cache": "new_session",
    }
    response = _request_json(
        f"{cfg.api_base_url}/hot_load/v1/models/hot_load",
        method="POST",
        headers=_headers(cfg),
        payload=payload,
    )
    if (
        not isinstance(response, dict)
        or response.get("accepted") is not True
        or response.get("identity") != identity
    ):
        raise RuntimeError(
            f"provider did not accept identity {identity!r}: {response!r}"
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
        try:
            identical = _files_equal(path, target)
        except FileNotFoundError:
            identical = False
        else:
            if not identical:
                raise FileExistsError(
                    f"immutable upload {target} already exists with different contents"
                )
            continue
        with path.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _files_equal(left: Path, right: Path) -> bool:
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(1024 * 1024)
            right_chunk = right_file.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True
