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


def pull_weights_pre_read_hook(source_dir: str, target_version: int) -> None:
    """Engine-side ``--custom-pull-weights-pre-read-hook`` target: fetch the
    latest committed volume snapshot onto THIS host so the engine's pull can read
    the published version.

    Exactly ONE reload. The caller (sidecar) only pulls after it has seen
    ``target_version`` at the bulletin pointer, so a single reload surfaces that
    commit; there is no need — and it is actively harmful — to reload in a loop.
    Repeated ``reload()`` calls thrash the Modal-v2 mount and stall the very
    materialization a readiness loop would poll for: in practice a reload loop
    turned ~3s delta pulls into 100–500s ones and tripped the engine's 300s
    watchdog, killing replicas mid-run.

    Completeness is handled downstream, not here: the engine stages each delta to
    local disk and size-verifies it against its safetensors header before
    applying, and a not-yet-materialized (short/absent) blob fails fast so the
    sidecar retries with a fresh single reload. That keeps the fast path fast
    while still never applying a partial delta.

    The engine imports this by dotted path inside its scheduler process, so the
    volume name travels via the ``DELTA_VOLUME_NAME`` env var (already set on the
    serving container for the sidecar's bulletin refresh).
    """
    import os

    volume_name = os.environ.get("DELTA_VOLUME_NAME", "")
    if not volume_name or target_version <= 0:
        return
    reload_volume(volume_name)


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
