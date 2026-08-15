"""The Modal-agnostic sidecar entrypoint: a ``SidecarConfig`` (one value for every
serving knob), a lossless CLI flag round-trip (``to_argv``/``from_argv``), and
``main``, which builds the Store + Engine from a config and calls
:func:`stitch.service.serve`.

A programmatic deployment constructs these instances itself and calls ``serve``
directly; this module exists so a container can launch the sidecar as its own
subprocess (``python -m stitch.sidecar``) with one flag vocabulary shared by
every recipe.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Literal

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.stores.factory import MODAL_VOLUME, S3, StoreBackend, create_store
from stitch.sync import CommitMode

logger = logging.getLogger(__name__)

DeltaUpdateMode = Literal["disk", "cpu"]


@dataclass(frozen=True, kw_only=True)
class SidecarConfig:
    """Everything one replica's sidecar needs: the ``serve`` serving knobs plus
    the wiring to build its Store and local SGLang Engine. Defaults describe a
    fresh boot beside a freshly started engine; a container launch overrides
    only what its recipe sets."""

    host: str = "0.0.0.0"
    port: int = 8000
    upstream: str = "http://127.0.0.1:8001"
    bulletin_root: str
    base_checkpoint_dir: str
    local_checkpoint_dir: str | None = None
    delta_update_mode: DeltaUpdateMode
    disk_load_format: str = "auto"
    store_backend: StoreBackend = MODAL_VOLUME
    volume_name: str | None = None
    s3_root: str | None = None
    s3_endpoint_url: str | None = None
    commit_mode: CommitMode = "in_place"
    flush_cache_on_commit: bool = False
    run_id: str
    boot_version: int = 0
    debug_requests: bool = False
    reconcile_interval: float = 5.0
    watchdog_interval: float = 5.0
    watchdog_failure_threshold: int = 3

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.boot_version < 0:
            raise ValueError("boot_version must be non-negative")
        if self.delta_update_mode == "disk" and not self.local_checkpoint_dir:
            raise ValueError("local_checkpoint_dir is required in disk mode")

    def to_argv(self) -> list[str]:
        """Render as CLI flags. ``from_argv(self.to_argv()) == self``, including
        defaults: optional values are flagged only when set, toggles only when on."""
        argv = [
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--upstream",
            self.upstream,
            "--bulletin-root",
            self.bulletin_root,
            "--base-checkpoint-dir",
            self.base_checkpoint_dir,
            "--delta-update-mode",
            self.delta_update_mode,
            "--disk-load-format",
            self.disk_load_format,
            "--store-backend",
            self.store_backend,
            "--commit-mode",
            self.commit_mode,
            "--run-id",
            self.run_id,
            "--boot-version",
            str(self.boot_version),
            "--reconcile-interval",
            str(self.reconcile_interval),
            "--watchdog-interval",
            str(self.watchdog_interval),
            "--watchdog-failure-threshold",
            str(self.watchdog_failure_threshold),
        ]
        if self.local_checkpoint_dir is not None:
            argv += ["--local-checkpoint-dir", self.local_checkpoint_dir]
        if self.volume_name is not None:
            argv += ["--volume-name", self.volume_name]
        if self.s3_root is not None:
            argv += ["--s3-root", self.s3_root]
        if self.s3_endpoint_url is not None:
            argv += ["--s3-endpoint-url", self.s3_endpoint_url]
        if self.flush_cache_on_commit:
            argv.append("--flush-cache-on-commit")
        if self.debug_requests:
            argv.append("--debug-requests")
        return argv

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> SidecarConfig:
        """Parse CLI flags into a config (``to_argv``'s round-trip partner)."""
        parser = _sidecar_parser()
        args = parser.parse_args(argv)
        if args.delta_update_mode == "disk" and not args.local_checkpoint_dir:
            parser.error("--local-checkpoint-dir is required in disk mode")
        return cls(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            bulletin_root=args.bulletin_root,
            base_checkpoint_dir=args.base_checkpoint_dir,
            local_checkpoint_dir=args.local_checkpoint_dir,
            delta_update_mode=args.delta_update_mode,
            disk_load_format=args.disk_load_format,
            store_backend=args.store_backend,
            volume_name=args.volume_name,
            s3_root=args.s3_root,
            s3_endpoint_url=args.s3_endpoint_url,
            commit_mode=args.commit_mode,
            flush_cache_on_commit=args.flush_cache_on_commit,
            run_id=args.run_id,
            boot_version=args.boot_version,
            debug_requests=args.debug_requests,
            reconcile_interval=args.reconcile_interval,
            watchdog_interval=args.watchdog_interval,
            watchdog_failure_threshold=args.watchdog_failure_threshold,
        )


def _sidecar_parser() -> argparse.ArgumentParser:
    """One flag per ``SidecarConfig`` field, in ``to_argv`` order."""
    p = argparse.ArgumentParser(prog="stitch.sidecar")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upstream", default="http://127.0.0.1:8001")
    p.add_argument("--bulletin-root", required=True)
    p.add_argument("--base-checkpoint-dir", required=True)
    p.add_argument("--delta-update-mode", choices=["disk", "cpu"], required=True)
    p.add_argument("--disk-load-format", default="auto")
    p.add_argument(
        "--store-backend",
        choices=[MODAL_VOLUME, S3],
        default=MODAL_VOLUME,
    )
    p.add_argument("--commit-mode", choices=["in_place", "quiesce"], default="in_place")
    p.add_argument("--run-id", required=True)
    p.add_argument("--boot-version", type=int, default=0)
    p.add_argument(
        "--reconcile-interval", type=float, default=5.0
    )  # 0 disables the periodic re-check
    p.add_argument("--watchdog-interval", type=float, default=5.0)
    p.add_argument("--watchdog-failure-threshold", type=int, default=3)
    p.add_argument("--local-checkpoint-dir")
    p.add_argument("--volume-name")
    p.add_argument("--s3-root")
    p.add_argument("--s3-endpoint-url")
    p.add_argument("--flush-cache-on-commit", action="store_true")
    p.add_argument("--debug-requests", action="store_true")
    return p


def _configure_logging() -> None:
    """Emit INFO logs to stdout (uvicorn configures only its own loggers)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> None:
    """The sidecar subprocess entrypoint: parse flags, build the Store + SGLang
    Engine, and serve the versioned rollout proxy beside the local engine."""
    _configure_logging()
    config = SidecarConfig.from_argv(argv)
    store = create_store(
        config.store_backend,
        local_root=config.bulletin_root,
        # Empty flags normalize to unset, as the stores' own validation treats them.
        volume_name=config.volume_name or None,
        run_id=config.run_id,
        s3_root=config.s3_root or None,
        s3_endpoint_url=config.s3_endpoint_url or None,
    )
    engine = SGLangEngine(
        config.upstream,
        config.base_checkpoint_dir,
        config.local_checkpoint_dir,
        delta_update_mode=config.delta_update_mode,
        disk_load_format=config.disk_load_format,
    )
    serve(
        store,
        engine,
        run_id=config.run_id,
        boot_version=config.boot_version,
        commit_mode=config.commit_mode,
        flush_cache_on_commit=config.flush_cache_on_commit,
        host=config.host,
        port=config.port,
        debug_requests=config.debug_requests,
        reconcile_interval=config.reconcile_interval,
        watchdog_interval=config.watchdog_interval,
        watchdog_failure_threshold=config.watchdog_failure_threshold,
    )


if __name__ == "__main__":
    main()
