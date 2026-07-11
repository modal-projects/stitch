"""Modal provider helpers for disaggregated rollout."""

from __future__ import annotations

from collections.abc import Callable


def commit_volume(volume_name: str) -> None:
    import modal

    modal.Volume.from_name(volume_name, version=2, create_if_missing=True).commit()


def reload_volume(volume_name: str) -> None:
    import modal

    modal.Volume.from_name(volume_name, version=2, create_if_missing=True).reload()


def volume_reloader(volume_name: str) -> Callable[[], None]:
    return lambda: reload_volume(volume_name)


def _version_materialized(version_dir: str) -> tuple[bool, str]:
    """Is the published version at ``version_dir`` fully present on this host?

    A version is materialized once its index and every blob the index references
    are present and non-empty. The trainer writes each file atomically (temp +
    rename), so a file is either fully there or absent — presence + nonzero size
    is a sufficient readiness signal (the strong per-tensor check is the engine's
    checksum after apply). Returns ``(ready, detail)``; ``detail`` names the first
    thing still missing, for the timeout message.
    """
    import json
    import os

    if not os.path.isdir(version_dir):
        return False, f"version dir absent: {version_dir}"
    index_path = os.path.join(version_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False, f"index absent: {index_path}"
    try:
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, ValueError) as e:
        return False, f"index unreadable: {e}"
    blobs = sorted(set(weight_map.values()))
    for blob in blobs:
        try:
            if os.path.getsize(os.path.join(version_dir, blob)) == 0:
                return False, f"blob empty: {blob}"
        except OSError:
            return False, f"blob absent: {blob}"
    return True, f"materialized ({len(blobs)} blobs)"


def pull_weights_pre_read_hook(source_dir: str, target_version: int) -> None:
    """Engine-side ``--custom-pull-weights-pre-read-hook`` target: make the
    published ``target_version`` fully readable on THIS host before the engine
    reads it. The engine's pull assumes the bytes are already on disk; this hook
    is what makes that precondition true, so the readiness/retry concern lives
    here and not in the engine's apply logic.

    Object-store-backed volumes have read-after-write propagation lag — seconds
    for the small index, longer for multi-GB blobs. A single reload fired right
    after the publisher's wake routinely lands *before* the commit has propagated
    to this host, which is why an unconditional one-shot reload saw the version
    dir as missing. So reload-and-verify in a loop: reload the volume, check the
    version dir + index + every referenced blob are present, and retry with
    backoff until the version is fully materialized or a deadline elapses.
    Verifying completeness (not merely that the dir exists) is what stops the
    engine from applying a half-propagated delta and failing its checksum.

    The engine imports this by dotted path inside its scheduler process, so the
    volume name travels via the ``DELTA_VOLUME_NAME`` env var (already set on the
    serving container for the sidecar's bulletin refresh).
    """
    import os
    import time

    volume_name = os.environ.get("DELTA_VOLUME_NAME", "")
    if not volume_name or target_version <= 0:
        return
    version_dir = os.path.join(source_dir, f"weight_v{target_version:06d}")
    deadline_s = float(os.environ.get("STITCH_PULL_READY_TIMEOUT_S", "120"))
    poll_s = float(os.environ.get("STITCH_PULL_READY_POLL_S", "0.5"))
    started = time.monotonic()
    attempt = 0
    while True:
        reload_volume(volume_name)
        ready, detail = _version_materialized(version_dir)
        if ready:
            return
        elapsed = time.monotonic() - started
        if elapsed >= deadline_s:
            raise TimeoutError(
                f"weight version {target_version} not materialized after "
                f"{elapsed:.1f}s / {attempt} reloads: {detail}"
            )
        attempt += 1
        time.sleep(min(poll_s * attempt, 5.0))


def discover_flash_targets(app_name: str, cls_name: str) -> list[str]:
    import modal
    import modal.experimental

    # Resolve the Cls by name for its Modal-client-side side effect; the returned
    # handle is intentionally unused (flash_get_containers queries by name).
    modal.Cls.from_name(app_name, cls_name)
    containers = modal.experimental.flash_get_containers(app_name, cls_name)
    return _flash_targets_from_containers(containers)


def _flash_targets_from_containers(containers) -> list[str]:
    targets: list[str] = []
    for container in containers:
        host = (
            container.get("host")
            if isinstance(container, dict)
            else getattr(container, "host", None)
        )
        if host:
            targets.append(normalize_base_url(str(host)))
    return targets


def resolve_flash_gateway_url(app_name: str, cls_name: str) -> str:
    import modal

    cls = modal.Cls.from_name(app_name, cls_name)
    urls = cls._experimental_get_flash_urls()
    if not urls:
        raise RuntimeError(
            f"No Flash gateway URL found for {app_name}.{cls_name}. "
            "Deploy the app first so Modal starts the Flash pool."
        )
    return str(urls[0]).rstrip("/")


def wake_targets(targets: list[str], version: int, *, timeout: float = 5.0) -> None:
    import logging
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    logger = logging.getLogger(__name__)

    if not targets:
        return

    # Wakes run in the trainer publish hot path; fan out instead of paying one
    # round-trip per container serially. One httpx.Client is shared across the
    # pool (Clients are thread-safe) so connections and keep-alive are reused.
    # trust_env=False keeps proxy env vars from rerouting localhost/gateway hops.
    with httpx.Client(timeout=timeout, trust_env=False) as client:

        def wake_one(target: str) -> None:
            url = f"{target}/rpc_sync_from_bulletin_board"
            try:
                resp = client.post(url, json={"target_version": int(version)})
                resp.raise_for_status()
                logger.info("Wake sync accepted by %s: %s", target, resp.text[:200])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to wake %s for version %s: %s", target, version, exc)

        with ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
            list(pool.map(wake_one, targets))


def normalize_base_url(host: str) -> str:
    host = host.rstrip("/")
    if host.startswith(("http://", "https://")):
        return host
    return f"https://{host}"
