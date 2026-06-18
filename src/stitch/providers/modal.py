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

    import requests

    logger = logging.getLogger(__name__)

    def wake_one(target: str) -> None:
        url = f"{target}/rpc_sync_from_bulletin_board"
        try:
            resp = requests.post(
                url, json={"target_version": int(version)}, timeout=timeout
            )
            resp.raise_for_status()
            logger.info("Wake sync accepted by %s: %s", target, resp.text[:200])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to wake %s for version %s: %s", target, version, exc)

    if not targets:
        return
    # Wakes run in the trainer publish hot path; fan out instead of paying
    # one round-trip per container serially.
    with ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
        list(pool.map(wake_one, targets))


def normalize_base_url(host: str) -> str:
    host = host.rstrip("/")
    if host.startswith(("http://", "https://")):
        return host
    return f"https://{host}"
