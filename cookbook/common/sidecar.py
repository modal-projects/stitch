"""The cookbook's sidecar subprocess entrypoint: build the Store from the
recipe-chosen backend and hand it to :func:`stitch.sidecar.run`, which builds
the local Engine and serves the versioned rollout proxy.

The flag vocabulary and arg parsing live in ``stitch.sidecar.SidecarConfig``;
choosing the Store backend is the consuming package's decision (core's
``Store`` port stays open to backends this repo doesn't ship), so it lives
here with the recipes. The Server container launches this as a subprocess via
``cookbook.common.process.start_sidecar`` (``python -m cookbook.common.sidecar``).
"""

from __future__ import annotations

import logging
import sys

from stitch.sidecar import SidecarConfig, run

from . import storage

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Emit INFO logs to stdout (uvicorn configures only its own loggers)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the shared flag vocabulary, build the recipe's Store, and serve."""
    _configure_logging()
    config = SidecarConfig.from_argv(argv)
    store = storage.create_store(
        config.store_backend,
        local_root=config.bulletin_root,
        # Empty flags normalize to unset, as the stores' own validation treats them.
        volume_name=config.volume_name or None,
        run_id=config.run_id,
        s3_root=config.s3_root or None,
        s3_endpoint_url=config.s3_endpoint_url or None,
    )
    run(config, store)


if __name__ == "__main__":
    main()
