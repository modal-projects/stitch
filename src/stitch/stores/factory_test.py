"""``create_store`` maps a CLI/config backend string onto one local layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from stitch.stores.factory import MODAL_VOLUME, S3, create_store
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.stores.s3 import S3Store


def test_create_store_selects_the_backend(tmp_path: Path) -> None:
    volume = create_store(
        MODAL_VOLUME,
        local_root=tmp_path / "volume",
        run_id="run-a",
    )
    s3 = create_store(
        S3,
        local_root=tmp_path / "cache",
        run_id="run-a",
        s3_root="s3://bucket/experiment/run-a",
    )

    assert isinstance(volume, ModalVolumeStore)
    assert isinstance(s3, S3Store)
    assert s3.cache_dir == tmp_path / "cache"
    assert s3.run_id == "run-a"


def test_s3_store_requires_s3_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="s3_root is required"):
        create_store(S3, local_root=tmp_path / "cache", run_id="run-a")


def test_create_store_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported store backend"):
        create_store("gcs", local_root=tmp_path / "cache", run_id="run-a")
