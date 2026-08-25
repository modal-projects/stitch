"""The Modal-agnostic sidecar entrypoint: a ``SidecarConfig`` (one value for every
serving knob), a lossless CLI flag round-trip (``to_argv``/``from_argv``), and
``main`` (``python -m stitch.sidecar``), which builds the Store and local SGLang
Engine from a config and calls :func:`stitch.service.serve`.

Core deliberately does not choose Store backends. ``Store`` is a port consumers
implement, so any name -> class dispatch frozen here would be useless to a
package that subclasses ``Store`` for its own runs. Instead the config names a
``store_factory`` (an importable ``package.module:callable``) and carries its
keyword arguments as opaque ``store_options``; ``main`` resolves the factory and
calls ``factory(local_root=..., run_id=..., **store_options)``. The launching
package owns the dispatch behind that callable — this repo's recipes point at
``cookbook.common.storage:create_store``. A programmatic deployment skips argv
entirely and passes its own Store to :func:`run` (or calls ``serve`` directly).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.stores.base import Store
from stitch.sync import CommitMode

DeltaUpdateMode = Literal["disk", "cpu"]

# Keyword arguments ``main`` always passes to the store factory; ``store_options``
# supplies everything else, so these names are reserved.
_FACTORY_KWARGS = frozenset({"local_root", "run_id"})


@dataclass(frozen=True, kw_only=True)
class SidecarConfig:
    """Everything one replica's sidecar needs: the ``serve`` serving knobs plus
    the wiring to build its Store and local SGLang Engine. The store wiring is
    open: ``store_factory`` names the callable that builds the Store, and
    ``store_options`` are its extra keyword arguments, round-tripped verbatim —
    core never enumerates any backend's parameters. Defaults describe a fresh
    boot beside a freshly started engine; a container launch overrides only
    what its recipe sets."""

    host: str = "0.0.0.0"
    port: int = 8000
    upstream: str = "http://127.0.0.1:8001"
    bulletin_root: str
    base_checkpoint_dir: str
    local_checkpoint_dir: str | None = None
    delta_update_mode: DeltaUpdateMode
    disk_load_format: str = "auto"
    store_factory: str
    store_options: dict[str, str] = field(default_factory=dict)
    commit_mode: CommitMode = "in_place"
    flush_cache_on_commit: bool = False
    version_lease_ttl: float = 0.0  # seconds; 0 disables version leases
    lease_header: str = "Modal-Session-ID"
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
        if ":" not in self.store_factory:
            raise ValueError(
                "store_factory must be an importable 'package.module:callable' "
                f"reference, got {self.store_factory!r}"
            )
        reserved = _FACTORY_KWARGS.intersection(self.store_options)
        if reserved:
            raise ValueError(
                f"store_options may not set {sorted(reserved)}; "
                "main passes local_root and run_id to the factory itself"
            )

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
            "--store-factory",
            self.store_factory,
            "--commit-mode",
            self.commit_mode,
            "--version-lease-ttl",
            str(self.version_lease_ttl),
            "--lease-header",
            self.lease_header,
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
        for key, value in self.store_options.items():
            argv += ["--store-opt", f"{key}={value}"]
        if self.local_checkpoint_dir is not None:
            argv += ["--local-checkpoint-dir", self.local_checkpoint_dir]
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
        store_options: dict[str, str] = {}
        for option in args.store_opt:
            key, separator, value = option.partition("=")
            if not separator or not key:
                parser.error(f"--store-opt expects KEY=VALUE, got {option!r}")
            if key in store_options:
                parser.error(f"--store-opt {key!r} given more than once")
            store_options[key] = value
        return cls(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            bulletin_root=args.bulletin_root,
            base_checkpoint_dir=args.base_checkpoint_dir,
            local_checkpoint_dir=args.local_checkpoint_dir,
            delta_update_mode=args.delta_update_mode,
            disk_load_format=args.disk_load_format,
            store_factory=args.store_factory,
            store_options=store_options,
            commit_mode=args.commit_mode,
            flush_cache_on_commit=args.flush_cache_on_commit,
            version_lease_ttl=args.version_lease_ttl,
            lease_header=args.lease_header,
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
    p.add_argument("--store-factory", required=True, metavar="MODULE:CALLABLE")
    p.add_argument(
        "--store-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra keyword argument for the store factory; repeatable",
    )
    p.add_argument("--commit-mode", choices=["in_place", "quiesce"], default="in_place")
    p.add_argument(
        "--version-lease-ttl",
        type=float,
        default=0.0,
        help="seconds a pinned session's version lease outlives its last request; 0 disables leases",
    )
    p.add_argument("--lease-header", default="Modal-Session-ID")
    p.add_argument("--run-id", required=True)
    p.add_argument("--boot-version", type=int, default=0)
    p.add_argument(
        "--reconcile-interval", type=float, default=5.0
    )  # 0 disables the periodic re-check
    p.add_argument("--watchdog-interval", type=float, default=5.0)
    p.add_argument("--watchdog-failure-threshold", type=int, default=3)
    p.add_argument("--local-checkpoint-dir")
    p.add_argument("--flush-cache-on-commit", action="store_true")
    p.add_argument("--debug-requests", action="store_true")
    return p


def _load_store_factory(spec: str) -> Callable[..., Store]:
    """Resolve a ``package.module:callable`` reference to the store factory."""
    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute)
    except AttributeError:
        raise ImportError(
            f"store factory {spec!r}: module {module_name!r} "
            f"has no attribute {attribute!r}"
        ) from None
    if not callable(factory):
        raise TypeError(f"store factory {spec!r} is not callable")
    return factory


def run(config: SidecarConfig, store: Store) -> None:
    """Build the local SGLang Engine from ``config`` and serve the versioned
    rollout proxy beside it. The caller owns the Store — a programmatic
    deployment constructs its own instance instead of going through argv."""
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
        version_lease_ttl=config.version_lease_ttl,
        lease_header=config.lease_header,
        host=config.host,
        port=config.port,
        debug_requests=config.debug_requests,
        reconcile_interval=config.reconcile_interval,
        watchdog_interval=config.watchdog_interval,
        watchdog_failure_threshold=config.watchdog_failure_threshold,
    )


def _configure_logging() -> None:
    """Emit INFO logs to stdout (uvicorn configures only its own loggers)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> None:
    """The sidecar subprocess entrypoint: parse flags, build the Store via the
    configured factory, and serve the versioned rollout proxy beside the local
    engine. The factory contract is
    ``factory(*, local_root, run_id, **store_options) -> Store``, every option
    value the flag-provided string."""
    _configure_logging()
    config = SidecarConfig.from_argv(argv)
    factory = _load_store_factory(config.store_factory)
    store = factory(
        local_root=config.bulletin_root,
        run_id=config.run_id,
        **config.store_options,
    )
    if not isinstance(store, Store):
        raise TypeError(
            f"store factory {config.store_factory!r} returned "
            f"{type(store).__name__}, which is not a stitch Store"
        )
    run(config, store)


if __name__ == "__main__":
    main()
