"""Choose and configure a ``Store`` by backend name.

The sidecar and training hooks take the backend as a CLI flag / config string
and need one local layout regardless of the backing service; keep the mapping
here so every consumer shares it. Heavy client libraries (``boto3``, ``modal``)
are imported lazily by the store implementations, so this module is always
cheap to import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from stitch.stores.base import Store
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.stores.s3 import S3Store

StoreBackend = Literal["modal-volume", "s3"]

MODAL_VOLUME: StoreBackend = "modal-volume"
S3: StoreBackend = "s3"


def create_store(
    backend: str,
    *,
    local_root: str | Path,
    run_id: str,
    volume_name: str | None = None,
    s3_root: str | None = None,
    s3_endpoint_url: str | None = None,
) -> Store:
    """Create a Store with one local layout regardless of its backing service."""
    if backend == MODAL_VOLUME:
        return ModalVolumeStore(local_root, volume_name=volume_name, run_id=run_id)
    if backend == S3:
        if not s3_root:
            raise ValueError("s3_root is required for the S3 store")
        return S3Store(
            s3_root,
            cache_dir=local_root,
            endpoint_url=s3_endpoint_url,
            run_id=run_id,
        )
    raise ValueError(f"unsupported store backend: {backend!r}")
