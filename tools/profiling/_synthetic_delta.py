"""Construct bounded-memory XOR deltas for weight-update validation."""

from __future__ import annotations

import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_STREAM_BYTES = 8 * 1024 * 1024
_ZERO_CHUNK = bytes(_STREAM_BYTES)


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        (header_bytes,) = struct.unpack("<Q", handle.read(8))
        return 8 + header_bytes, json.loads(handle.read(header_bytes))


def _choose_bfloat16_tensor(
    checkpoint_dir: Path,
    weight_map: dict[str, str],
) -> str:
    headers: dict[Path, dict[str, Any]] = {}
    candidates: list[tuple[int, int, str]] = []
    for name, filename in weight_map.items():
        path = checkpoint_dir / filename
        if path not in headers:
            _, headers[path] = _safetensors_header(path)
        info = headers[path][name]
        begin, end = info["data_offsets"]
        if info["dtype"] != "BF16" or end <= begin:
            continue
        priority = (
            0
            if name.endswith(("model.norm.weight", "language_model.norm.weight"))
            else 1
            if "norm" in name
            else 2
        )
        candidates.append((priority, end - begin, name))
    if not candidates:
        raise RuntimeError("checkpoint has no BF16 tensor for a controlled delta")
    return min(candidates)[2]


def _compress_xor(byte_count: int, *, flip_first_byte: bool) -> bytes:
    import zstandard

    compressor = zstandard.ZstdCompressor(level=1).compressobj()
    output = []
    remaining = byte_count
    if flip_first_byte:
        output.append(compressor.compress(b"\x01"))
        remaining -= 1
    while remaining:
        size = min(remaining, len(_ZERO_CHUNK))
        output.append(compressor.compress(memoryview(_ZERO_CHUNK)[:size]))
        remaining -= size
    output.append(compressor.flush())
    return b"".join(output)


def _target_checksum(
    handle: Any,
    *,
    offset: int,
    byte_count: int,
    flip_first_byte: bool,
) -> str:
    import xxhash

    checksum = xxhash.xxh3_128()
    handle.seek(offset)
    remaining = byte_count
    first = True
    while remaining:
        chunk = handle.read(min(remaining, _STREAM_BYTES))
        if not chunk:
            raise RuntimeError("checkpoint tensor ended before its declared size")
        if first and flip_first_byte:
            changed = bytearray(chunk)
            changed[0] ^= 1
            chunk = changed
        checksum.update(chunk)
        remaining -= len(chunk)
        first = False
    return checksum.hexdigest()


def write_full_coverage_delta(
    checkpoint_dir: str,
    source_dir: str,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Write one DELTA containing every tensor and one safe changed byte."""

    import numpy as np
    import safetensors.numpy

    started = time.perf_counter()
    checkpoint = Path(checkpoint_dir)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    primary_name = _choose_bfloat16_tensor(checkpoint, weight_map)
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
        changed_bytes = 0
        with source_path.open("rb") as handle:
            for name in names:
                begin, end = header[name]["data_offsets"]
                byte_count = end - begin
                flip_first_byte = name == primary_name
                checksums[name] = _target_checksum(
                    handle,
                    offset=data_start + begin,
                    byte_count=byte_count,
                    flip_first_byte=flip_first_byte,
                )
                compressed_tensors[name] = np.frombuffer(
                    _compress_xor(
                        byte_count,
                        flip_first_byte=flip_first_byte,
                    ),
                    dtype=np.uint8,
                )
                tensor_bytes += byte_count
                changed_bytes += int(flip_first_byte)
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
        "changed_tensors": 1,
        "tensor_bytes": sum(int(shard["tensor_bytes"]) for shard in shards),
        "changed_bytes": sum(int(shard["changed_bytes"]) for shard in shards),
        "wire_bytes": sum(int(shard["wire_bytes"]) for shard in shards),
        "primary_tensor": primary_name,
        "generation_s": round(time.perf_counter() - started, 6),
    }
