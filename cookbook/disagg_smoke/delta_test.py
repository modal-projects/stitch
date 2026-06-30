"""Structural tests for the synthetic delta encoder.

These need numpy/zstandard/xxhash (the decoder's deps), which are present on the
Modal smoke image but not necessarily in the bare dev env, so they importorskip.
The full byte-for-byte round-trip against slime's real decoder is asserted by
``control_plane_test`` in ``app.py`` (it needs slime installed)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("zstandard")
pytest.importorskip("xxhash")

from cookbook.disagg_smoke import delta  # noqa: E402


def _write_base_checkpoint(directory: Path) -> dict[str, np.ndarray]:
    """A minimal single-shard safetensors file with a couple of uint8 tensors."""
    tensors = {
        "a.weight": np.arange(64, dtype=np.uint8),
        "b.weight": np.full(32, 7, dtype=np.uint8),
    }
    header: dict[str, object] = {}
    offset = 0
    for name, arr in tensors.items():
        header[name] = {"dtype": "U8", "shape": [arr.size], "data_offsets": [offset, offset + arr.size]}
        offset += arr.size
    header_json = json.dumps(header).encode()
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for arr in tensors.values():
            f.write(arr.tobytes())
    return tensors


def test_select_delta_tensors_is_deterministic_and_smallest_first(tmp_path):
    _write_base_checkpoint(tmp_path / "base")
    names = delta.select_delta_tensors(str(tmp_path / "base"), count=2)
    assert names == ["b.weight", "a.weight"]  # 32 bytes before 64 bytes


def test_publish_next_writes_valid_slime_version_dir(tmp_path):
    base = tmp_path / "base"
    _write_base_checkpoint(base)
    names = delta.select_delta_tensors(str(base), count=2)
    publisher = delta.DeltaPublisher(str(base), str(tmp_path / "run"), names)

    assert publisher.publish_next() == 1
    version_dir = tmp_path / "run" / "weight_v000001"
    index = json.loads((version_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"] == {
        "version": "000001",
        "base_version": "000000",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "xxh3-128",
    }
    assert set(index["weight_map"]) == set(names)
    assert (version_dir / delta.DELTA_FILENAME).exists()

    # v2 chains on v1.
    assert publisher.publish_next() == 2
    index2 = json.loads(
        (tmp_path / "run" / "weight_v000002" / "model.safetensors.index.json").read_text()
    )
    assert index2["metadata"]["version"] == "000002"
    assert index2["metadata"]["base_version"] == "000001"


def test_xor_delta_reconstructs_new_bytes_and_checksum(tmp_path):
    """xor(old, new) recorded in the file + checksum(new) is self-consistent: the
    decoder applies old ^ delta == new, then checksums the patched region."""
    base = tmp_path / "base"
    base_tensors = _write_base_checkpoint(base)
    names = ["a.weight"]
    publisher = delta.DeltaPublisher(str(base), str(tmp_path / "run"), names)
    publisher.publish_next()
    new_bytes = publisher.expected_tensor("a.weight")

    version_dir = tmp_path / "run" / "weight_v000001"
    raw = (version_dir / delta.DELTA_FILENAME).read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    data = raw[8 + header_len :]

    import zstandard

    begin, end = header["a.weight"]["data_offsets"]
    decoded_delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(data[begin:end]), dtype=np.uint8)
    reconstructed = np.bitwise_xor(base_tensors["a.weight"], decoded_delta)
    assert np.array_equal(reconstructed, new_bytes)
    assert header["__metadata__"]["a.weight"] == delta._checksum("xxh3-128", new_bytes.tobytes())
