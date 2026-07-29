"""Construct bounded-memory XOR deltas for weight-update validation."""

from __future__ import annotations

import json
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_STREAM_BYTES = 8 * 1024 * 1024
_CHANGE_STRIDE_BYTES = 1024 * 1024
_ZERO_CHUNK = bytes(_STREAM_BYTES)
_XOR_CHUNK = bytearray(_ZERO_CHUNK)
_XOR_CHUNK[::_CHANGE_STRIDE_BYTES] = b"\x01" * (_STREAM_BYTES // _CHANGE_STRIDE_BYTES)
_XOR_CHUNK = bytes(_XOR_CHUNK)


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        (header_bytes,) = struct.unpack("<Q", handle.read(8))
        return 8 + header_bytes, json.loads(handle.read(header_bytes))


def _compress_xor(byte_count: int) -> bytes:
    import zstandard

    compressor = zstandard.ZstdCompressor(level=1).compressobj()
    output = []
    remaining = byte_count
    while remaining:
        size = min(remaining, len(_XOR_CHUNK))
        output.append(compressor.compress(memoryview(_XOR_CHUNK)[:size]))
        remaining -= size
    output.append(compressor.flush())
    return b"".join(output)


def _target_checksum(
    handle: Any,
    *,
    offset: int,
    byte_count: int,
) -> str:
    import xxhash

    checksum = xxhash.xxh3_128()
    handle.seek(offset)
    remaining = byte_count
    position = 0
    while remaining:
        chunk = handle.read(min(remaining, _STREAM_BYTES))
        if not chunk:
            raise RuntimeError("checkpoint tensor ended before its declared size")
        first_change = (-position) % _CHANGE_STRIDE_BYTES
        if first_change < len(chunk):
            changed = bytearray(chunk)
            for index in range(
                first_change,
                len(changed),
                _CHANGE_STRIDE_BYTES,
            ):
                changed[index] ^= 1
            chunk = changed
        checksum.update(chunk)
        remaining -= len(chunk)
        position += len(chunk)
    return checksum.hexdigest()


def write_full_coverage_delta(
    checkpoint_dir: str,
    source_dir: str,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Write one DELTA with an element-level change in every tensor."""

    import numpy as np
    import safetensors.numpy

    started = time.perf_counter()
    checkpoint = Path(checkpoint_dir)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names_by_shard: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        names_by_shard.setdefault(filename, []).append(name)

    target_dir = Path(source_dir) / "weight_v000001"
    target_dir.mkdir(parents=True, exist_ok=False)

    def write_shard(item: tuple[str, list[str]]) -> dict[str, int | str]:
        filename, names = item
        source_path = checkpoint / filename
        data_start, header = _safetensors_header(source_path)
        names.sort(key=lambda name: header[name]["data_offsets"][0])
        compressed_tensors: dict[str, np.ndarray] = {}
        checksums: dict[str, str] = {}
        tensor_bytes = 0
        changed_tensors = 0
        changed_bytes = 0
        with source_path.open("rb") as handle:
            for name in names:
                begin, end = header[name]["data_offsets"]
                byte_count = end - begin
                checksums[name] = _target_checksum(
                    handle,
                    offset=data_start + begin,
                    byte_count=byte_count,
                )
                compressed_tensors[name] = np.frombuffer(
                    _compress_xor(byte_count),
                    dtype=np.uint8,
                )
                tensor_bytes += byte_count
                changed_tensors += int(byte_count > 0)
                changed_bytes += math.ceil(byte_count / _CHANGE_STRIDE_BYTES)
            if hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(
                        handle.fileno(),
                        0,
                        0,
                        os.POSIX_FADV_DONTNEED,
                    )
                except OSError:
                    pass

        target_path = target_dir / filename
        safetensors.numpy.save_file(
            compressed_tensors,
            target_path,
            metadata=checksums,
        )
        result: dict[str, int | str] = {
            "filename": filename,
            "tensors": len(names),
            "tensor_bytes": tensor_bytes,
            "changed_tensors": changed_tensors,
            "changed_bytes": changed_bytes,
            "wire_bytes": target_path.stat().st_size,
        }
        print(f"SYNTHETIC_DELTA_SHARD={json.dumps(result, sort_keys=True)}")
        return result

    with ThreadPoolExecutor(
        max_workers=min(workers, len(names_by_shard)),
    ) as executor:
        shards = list(executor.map(write_shard, names_by_shard.items()))

    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "xxh3-128",
                },
                "weight_map": weight_map,
            }
        )
    )
    return {
        "scope": "all_tensors",
        "tensors": len(weight_map),
        "changed_tensors": sum(int(shard["changed_tensors"]) for shard in shards),
        "tensor_bytes": sum(int(shard["tensor_bytes"]) for shard in shards),
        "changed_bytes": sum(int(shard["changed_bytes"]) for shard in shards),
        "change_stride_bytes": _CHANGE_STRIDE_BYTES,
        "wire_bytes": sum(int(shard["wire_bytes"]) for shard in shards),
        "generation_s": round(time.perf_counter() - started, 6),
    }
