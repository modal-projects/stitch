"""Construct deterministic rollout-checkpoint XOR deltas for profiling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CheckpointFormat = Literal["nvfp4", "mxfp4", "fp8"]

_STREAM_BYTES = 8 * 1024 * 1024
_PATTERN_COUNT = 2
_GENERATOR_VERSION = "rollout-value-density-v5"
_DEFAULT_QUANTIZED_VALUE_DENSITY = {
    "nvfp4": 0.003,
    "mxfp4": 0.003,
    "fp8": 0.006,
}
_STATIC_INPUT_SCALE_SUFFIX = ".input_scale"


@dataclass(frozen=True)
class SyntheticDeltaSpec:
    """Mutable rollout-visible logical values in one synthetic RL step."""

    checkpoint_format: CheckpointFormat
    quantized_value_density: float | None = None
    high_precision_value_density: float | None = None
    output_shards: int | None = None
    output_shard_layout: str | None = None
    immutable_prefixes: tuple[str, ...] = ()
    immutable_suffixes: tuple[str, ...] = ()

    @property
    def effective_quantized_value_density(self) -> float:
        if self.quantized_value_density is not None:
            return self.quantized_value_density
        return _DEFAULT_QUANTIZED_VALUE_DENSITY[self.checkpoint_format]

    @property
    def effective_high_precision_value_density(self) -> float:
        if self.high_precision_value_density is not None:
            return self.high_precision_value_density
        return self.effective_quantized_value_density

    def validate(self) -> None:
        for name, density in (
            ("quantized_value_density", self.effective_quantized_value_density),
            (
                "high_precision_value_density",
                self.effective_high_precision_value_density,
            ),
        ):
            if not 0.0 < density < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.output_shards is not None and self.output_shards < 1:
            raise ValueError("output_shards must be positive")
        if self.output_shard_layout is not None and self.output_shards is None:
            raise ValueError("output_shard_layout requires output_shards")

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_format": self.checkpoint_format,
            "quantized_value_density": self.effective_quantized_value_density,
            "high_precision_value_density": (
                self.effective_high_precision_value_density
            ),
            "output_shards": self.output_shards,
            "output_shard_layout": self.output_shard_layout,
            "immutable_prefixes": list(self.immutable_prefixes),
            "immutable_suffixes": list(self.immutable_suffixes),
        }


@dataclass(frozen=True)
class _Encoding:
    name: str
    logical_values_per_byte: float
    density: float
    alignment: int
    transform: Literal["bf16", "fp16", "fp32", "packed_fp4", "fp8", "e8m0"]


@dataclass(frozen=True)
class _EncodedDelta:
    compressed: bytes
    target_checksum: str
    changed_values: int
    changed_bytes: int


def synthetic_delta_profile_id(spec: SyntheticDeltaSpec, *, seed: int = 42) -> str:
    """Return a stable cache key for a generator configuration."""

    spec.validate()
    config = json.dumps(spec.as_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(config.encode()).hexdigest()[:10]
    return f"{_GENERATOR_VERSION}-{spec.checkpoint_format}-{digest}-seed{seed}"


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        (header_bytes,) = struct.unpack("<Q", raw_size)
        header = json.loads(handle.read(header_bytes))
    return 8 + header_bytes, header


def _immutable_reason(name: str, spec: SyntheticDeltaSpec) -> str | None:
    if name.endswith(_STATIC_INPUT_SCALE_SUFFIX):
        return "static_input_scale"
    if any(name.startswith(prefix) for prefix in spec.immutable_prefixes):
        return "configured_prefix"
    if any(name.endswith(suffix) for suffix in spec.immutable_suffixes):
        return "configured_suffix"
    return None


def _encoding_for(
    *,
    name: str,
    dtype: str,
    spec: SyntheticDeltaSpec,
) -> _Encoding | None:
    if _immutable_reason(name, spec) is not None:
        return None

    quantized_density = spec.effective_quantized_value_density
    high_precision_density = spec.effective_high_precision_value_density
    checkpoint_format = spec.checkpoint_format

    if checkpoint_format == "nvfp4":
        if name.endswith(".weight_scale_2") and dtype == "F32":
            return _Encoding(
                "nvfp4_global_scale",
                0.25,
                quantized_density,
                4,
                "fp32",
            )
        if name.endswith(".weight_scale") and dtype == "F8_E4M3":
            return _Encoding(
                "nvfp4_block_scale",
                1.0,
                quantized_density,
                1,
                "fp8",
            )
        if name.endswith(".weight") and dtype == "U8":
            return _Encoding(
                "nvfp4_packed_weight",
                2.0,
                quantized_density,
                1,
                "packed_fp4",
            )
    elif checkpoint_format == "mxfp4":
        if name.endswith(".weight_packed") and dtype == "U8":
            return _Encoding(
                "mxfp4_packed_weight",
                2.0,
                quantized_density,
                1,
                "packed_fp4",
            )
        if name.endswith(".weight_scale") and dtype == "U8":
            return _Encoding(
                "mxfp4_block_scale",
                1.0,
                quantized_density,
                1,
                "e8m0",
            )
    elif checkpoint_format == "fp8":
        if name.endswith(".weight") and dtype == "F8_E4M3":
            return _Encoding(
                "fp8_weight",
                1.0,
                quantized_density,
                1,
                "fp8",
            )
        if name.endswith((".weight_scale", ".weight_scale_inv")):
            if dtype == "F32":
                return _Encoding(
                    "fp8_scale",
                    0.25,
                    quantized_density,
                    4,
                    "fp32",
                )
            if dtype == "U8":
                return _Encoding(
                    "fp8_e8m0_scale",
                    1.0,
                    quantized_density,
                    1,
                    "e8m0",
                )

    if dtype == "BF16":
        return _Encoding("bfloat16", 0.5, high_precision_density, 2, "bf16")
    if dtype == "F16":
        return _Encoding("float16", 0.5, high_precision_density, 2, "fp16")
    if dtype == "F32":
        return _Encoding("float32", 0.25, high_precision_density, 4, "fp32")
    raise ValueError(
        f"no safe synthetic transform for tensor {name!r} with dtype={dtype!r} "
        f"in a {checkpoint_format!r} checkpoint"
    )


def _selection_identity(name: str, encoding: _Encoding) -> str:
    """Keep the two halves of a fused gate/up global scale compatible."""

    if encoding.name != "nvfp4_global_scale":
        return name
    return name.replace(".gate_proj.", ".gate_up_proj.").replace(
        ".up_proj.",
        ".gate_up_proj.",
    )


class _PatternBank:
    """Reusable deterministic selection masks at a logical-value density."""

    def __init__(self, seed: int):
        import numpy as np

        self.np = np
        self.seed = seed
        self.patterns: dict[tuple[Any, ...], list[Any]] = {}
        self.lock = threading.Lock()

    def _key(self, encoding: _Encoding) -> tuple[Any, ...]:
        return (
            encoding.transform,
            encoding.logical_values_per_byte,
            encoding.density,
            encoding.alignment,
        )

    def _patterns_for(self, encoding: _Encoding) -> list[Any]:
        key = self._key(encoding)
        patterns = self.patterns.get(key)
        if patterns is not None:
            return patterns
        with self.lock:
            patterns = self.patterns.get(key)
            if patterns is None:
                patterns = [
                    self._build_pattern(encoding, index)
                    for index in range(_PATTERN_COUNT)
                ]
                self.patterns[key] = patterns
            return patterns

    def _build_pattern(self, encoding: _Encoding, index: int) -> Any:
        np = self.np
        seed_material = f"{self.seed}:{self._key(encoding)!r}:{index}".encode()
        pattern_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(pattern_seed)
        mask = np.zeros(_STREAM_BYTES, dtype=np.uint8)
        logical_values = round(_STREAM_BYTES * encoding.logical_values_per_byte)
        changed = round(logical_values * encoding.density)
        positions = rng.choice(logical_values, size=changed, replace=False)
        if encoding.transform in {"bf16", "fp16"}:
            mask[positions * 2] = 0x01
        elif encoding.transform == "fp32":
            mask[positions * 4] = 0x01
        elif encoding.transform == "packed_fp4":
            np.bitwise_or.at(
                mask,
                positions // 2,
                np.where(positions % 2, 0x10, 0x01).astype(np.uint8),
            )
        else:
            mask[positions] = 0x01
        return mask

    def mask(
        self,
        *,
        name: str,
        encoding: _Encoding,
        byte_offset: int,
        byte_count: int,
    ) -> Any:
        np = self.np
        identity = _selection_identity(name, encoding)
        digest = hashlib.sha256(
            f"{self.seed}:{self._key(encoding)!r}:{identity}".encode()
        ).digest()
        patterns = self._patterns_for(encoding)
        pattern = patterns[digest[0] % len(patterns)]
        phase_slots = _STREAM_BYTES // encoding.alignment
        phase = (
            int.from_bytes(digest[1:9], "little") % phase_slots
        ) * encoding.alignment
        start = (phase + byte_offset) % _STREAM_BYTES
        mask = np.empty(byte_count, dtype=np.uint8)
        first = min(byte_count, _STREAM_BYTES - start)
        mask[:first] = pattern[start : start + first]
        if first < byte_count:
            mask[first:] = pattern[: byte_count - first]
        return mask


def _changed_values(mask: Any, encoding: _Encoding) -> int:
    import numpy as np

    if encoding.transform == "packed_fp4":
        return int(np.count_nonzero(mask & 0x0F) + np.count_nonzero(mask >> 4))
    return int(np.count_nonzero(mask))


def _make_delta(
    base: Any,
    mask: Any,
    encoding: _Encoding,
    *,
    tensor_name: str,
) -> Any:
    """Advance selected values by one safe stored-code step."""

    import numpy as np

    delta = mask
    if encoding.transform == "packed_fp4":
        low_selected = np.flatnonzero(mask & 0x0F)
        high_selected = np.flatnonzero(mask >> 4)
        delta.fill(0)
        if low_selected.size:
            values = base[low_selected] & 0x0F
            magnitude = values & 0x07
            target = np.where(magnitude < 0x07, magnitude + 1, magnitude - 1)
            delta[low_selected] |= magnitude ^ target
        if high_selected.size:
            values = base[high_selected] >> 4
            magnitude = values & 0x07
            target = np.where(magnitude < 0x07, magnitude + 1, magnitude - 1)
            delta[high_selected] |= (magnitude ^ target) << 4
        return delta

    selected = np.flatnonzero(mask)
    if encoding.transform == "bf16" and selected.size:
        high = base[selected + 1]
        if np.any((high & 0x7F) == 0x7F):
            raise ValueError(
                f"{tensor_name!r} contains selected non-finite bfloat16 values"
            )
        fraction = base[selected] & 0x7F
        target = np.where(fraction < 0x7F, fraction + 1, fraction - 1)
        delta[selected] = fraction ^ target
    elif encoding.transform == "fp16" and selected.size:
        high = base[selected + 1]
        if np.any((high & 0x7C) == 0x7C):
            raise ValueError(
                f"{tensor_name!r} contains selected non-finite float16 values"
            )
        fraction = base[selected]
        target = np.where(fraction < 0xFF, fraction + 1, fraction - 1)
        delta[selected] = fraction ^ target
    elif encoding.transform == "fp32" and selected.size:
        values = base.view("<u4")
        selected_values = values[selected // 4]
        if np.any((selected_values & 0x7F800000) == 0x7F800000):
            raise ValueError(
                f"{tensor_name!r} contains selected non-finite float32 values"
            )
        fraction = base[selected]
        target = np.where(fraction < 0xFF, fraction + 1, fraction - 1)
        delta[selected] = fraction ^ target
    elif encoding.transform == "fp8" and selected.size:
        values = base[selected]
        if np.any((values & 0x7F) == 0x7F):
            raise ValueError(f"{tensor_name!r} contains selected NaN E4M3 values")
        fraction = values & 0x07
        maximum = np.where((values & 0x78) == 0x78, 0x06, 0x07)
        target = np.where(fraction < maximum, fraction + 1, fraction - 1)
        delta[selected] = fraction ^ target
    elif encoding.transform == "e8m0" and selected.size:
        values = base[selected]
        if np.any(values == 0xFF):
            raise ValueError(f"{tensor_name!r} contains selected NaN E8M0 values")
        target = np.where(values < 0xFE, values + 1, values - 1)
        delta[selected] = values ^ target
    return delta


def _encode_tensor_delta(
    handle: Any,
    *,
    name: str,
    offset: int,
    byte_count: int,
    encoding: _Encoding,
    patterns: _PatternBank,
    compressor_context: Any,
) -> _EncodedDelta | None:
    import numpy as np
    import xxhash

    checksum = xxhash.xxh3_128()
    compressor = compressor_context.compressobj()
    compressed = bytearray()
    changed_values = 0
    changed_bytes = 0
    handle.seek(offset)
    position = 0
    while position < byte_count:
        chunk = handle.read(min(_STREAM_BYTES, byte_count - position))
        if not chunk:
            raise RuntimeError(f"{name!r} ended before its declared size")
        base = np.frombuffer(chunk, dtype=np.uint8)
        mask = patterns.mask(
            name=name,
            encoding=encoding,
            byte_offset=position,
            byte_count=len(chunk),
        )
        delta = _make_delta(base, mask, encoding, tensor_name=name)
        checksum.update(np.bitwise_xor(base, delta))
        changed_values += _changed_values(delta, encoding)
        changed_bytes += int(np.count_nonzero(delta))
        compressed.extend(compressor.compress(delta))
        position += len(chunk)
    compressed.extend(compressor.flush())
    if changed_values == 0:
        return None
    return _EncodedDelta(
        compressed=bytes(compressed),
        target_checksum=checksum.hexdigest(),
        changed_values=changed_values,
        changed_bytes=changed_bytes,
    )


def _empty_encoding_stats() -> dict[str, int]:
    return {
        "eligible_tensors": 0,
        "changed_tensors": 0,
        "tensor_bytes": 0,
        "logical_values": 0,
        "changed_values": 0,
        "changed_bytes": 0,
        "compressed_bytes": 0,
    }


def write_standard_delta(
    checkpoint_dir: str,
    source_dir: str,
    *,
    spec: SyntheticDeltaSpec,
    seed: int = 42,
    workers: int | None = None,
    output_shard_for_tensor: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Write a checksummed XOR delta directly in rollout-checkpoint space."""

    import numpy as np
    import safetensors.numpy
    import zstandard

    spec.validate()
    if output_shard_for_tensor is not None and spec.output_shards is None:
        raise ValueError("output_shard_for_tensor requires output_shards")
    started = time.perf_counter()
    checkpoint = Path(checkpoint_dir)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint has no non-empty weight map: {index_path}")
    names_by_shard: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        names_by_shard.setdefault(filename, []).append(name)

    target_dir = Path(source_dir) / "weight_v000001"
    target_dir.mkdir(parents=True, exist_ok=False)
    patterns = _PatternBank(seed)

    def encode_source_shard(item: tuple[str, list[str]]) -> dict[str, Any]:
        filename, names = item
        source_path = checkpoint / filename
        data_start, header = _safetensors_header(source_path)
        names.sort(key=lambda tensor_name: header[tensor_name]["data_offsets"][0])
        encoded_tensors: dict[str, Any] = {}
        checksums: dict[str, str] = {}
        immutable: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tensors": 0, "tensor_bytes": 0}
        )
        by_encoding: dict[str, dict[str, int]] = defaultdict(_empty_encoding_stats)
        compressor = zstandard.ZstdCompressor(level=1)
        with source_path.open("rb") as handle:
            for name in names:
                tensor = header[name]
                begin, end = tensor["data_offsets"]
                byte_count = end - begin
                reason = _immutable_reason(name, spec)
                if reason is not None:
                    immutable[reason]["tensors"] += 1
                    immutable[reason]["tensor_bytes"] += byte_count
                    continue
                encoding = _encoding_for(
                    name=name,
                    dtype=tensor["dtype"],
                    spec=spec,
                )
                if encoding is None:
                    raise AssertionError(f"{name!r} was not classified")
                stats = by_encoding[encoding.name]
                stats["eligible_tensors"] += 1
                stats["tensor_bytes"] += byte_count
                stats["logical_values"] += round(
                    byte_count * encoding.logical_values_per_byte
                )
                encoded = _encode_tensor_delta(
                    handle,
                    name=name,
                    offset=data_start + begin,
                    byte_count=byte_count,
                    encoding=encoding,
                    patterns=patterns,
                    compressor_context=compressor,
                )
                if encoded is None:
                    continue
                encoded_tensors[name] = np.frombuffer(
                    encoded.compressed,
                    dtype=np.uint8,
                )
                checksums[name] = encoded.target_checksum
                stats["changed_tensors"] += 1
                stats["changed_values"] += encoded.changed_values
                stats["changed_bytes"] += encoded.changed_bytes
                stats["compressed_bytes"] += len(encoded.compressed)
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

        result = {
            "filename": filename,
            "encoded_tensors": encoded_tensors,
            "checksums": checksums,
            "immutable": dict(immutable),
            "by_encoding": dict(by_encoding),
        }
        print(
            "SYNTHETIC_DELTA_SOURCE="
            + json.dumps(
                {
                    "filename": filename,
                    "changed_tensors": len(encoded_tensors),
                    "compressed_bytes": sum(
                        tensor.nbytes for tensor in encoded_tensors.values()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    worker_count = min(
        workers or os.cpu_count() or 1,
        len(names_by_shard),
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        source_shards = list(executor.map(encode_source_shard, names_by_shard.items()))

    aggregate: dict[str, dict[str, int | float]] = defaultdict(_empty_encoding_stats)
    immutable: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensors": 0, "tensor_bytes": 0}
    )
    entries: list[tuple[str, Any, str, str]] = []
    for shard in source_shards:
        for name, tensor in shard["encoded_tensors"].items():
            entries.append(
                (
                    name,
                    tensor,
                    shard["checksums"][name],
                    shard["filename"],
                )
            )
        for encoding, values in shard["by_encoding"].items():
            for key, value in values.items():
                aggregate[encoding][key] += value
        for reason, values in shard["immutable"].items():
            for key, value in values.items():
                immutable[reason][key] += value
    if not entries:
        raise RuntimeError("synthetic delta did not change any checkpoint value")

    output: list[dict[str, Any]]
    if spec.output_shards is None:
        output_by_name: dict[str, dict[str, Any]] = {}
        for name, tensor, checksum, source_filename in entries:
            shard = output_by_name.setdefault(
                source_filename,
                {"filename": source_filename, "tensors": {}, "checksums": {}},
            )
            shard["tensors"][name] = tensor
            shard["checksums"][name] = checksum
        output = list(output_by_name.values())
    else:
        shard_count = min(spec.output_shards, len(entries))
        output = [
            {
                "filename": (f"model-{index:05d}-of-{shard_count:05d}.safetensors"),
                "tensors": {},
                "checksums": {},
                "payload_bytes": 0,
            }
            for index in range(shard_count)
        ]
        if output_shard_for_tensor is None:
            ordered_entries = sorted(
                entries,
                key=lambda item: (-item[1].nbytes, item[0]),
            )
        else:
            ordered_entries = sorted(entries, key=lambda item: item[0])
        for name, tensor, checksum, _ in ordered_entries:
            if output_shard_for_tensor is None:
                shard = min(output, key=lambda item: item["payload_bytes"])
            else:
                shard_index = output_shard_for_tensor(name)
                if not 0 <= shard_index < shard_count:
                    raise ValueError(
                        f"output shard {shard_index} for {name!r} is outside "
                        f"[0, {shard_count})"
                    )
                shard = output[shard_index]
            shard["tensors"][name] = tensor
            shard["checksums"][name] = checksum
            shard["payload_bytes"] += tensor.nbytes

    def save_output_shard(shard: dict[str, Any]) -> dict[str, Any]:
        target_path = target_dir / shard["filename"]
        safetensors.numpy.save_file(
            shard["tensors"],
            target_path,
            metadata=shard["checksums"],
        )
        result = {
            "filename": shard["filename"],
            "tensors": len(shard["tensors"]),
            "compressed_bytes": sum(
                tensor.nbytes for tensor in shard["tensors"].values()
            ),
            "wire_bytes": target_path.stat().st_size,
        }
        print(
            f"SYNTHETIC_DELTA_SHARD={json.dumps(result, sort_keys=True)}",
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=min(worker_count, len(output))) as executor:
        output_stats = list(executor.map(save_output_shard, output))

    delta_weight_map = {
        name: shard["filename"] for shard in output for name in shard["tensors"]
    }
    for values in aggregate.values():
        logical_values = int(values["logical_values"])
        tensor_bytes = int(values["tensor_bytes"])
        values["changed_value_density"] = int(values["changed_values"]) / logical_values
        values["changed_byte_density"] = int(values["changed_bytes"]) / tensor_bytes

    metadata = {
        "version": "000001",
        "base_version": "000000",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "xxh3-128",
        "synthetic_generator": _GENERATOR_VERSION,
        "synthetic_spec": json.dumps(
            spec.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "synthetic_seed": str(seed),
    }
    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": metadata, "weight_map": delta_weight_map})
    )

    eligible_tensors = sum(
        int(values["eligible_tensors"]) for values in aggregate.values()
    )
    changed_tensors = sum(
        int(values["changed_tensors"]) for values in aggregate.values()
    )
    tensor_bytes = sum(int(values["tensor_bytes"]) for values in aggregate.values())
    compressed_bytes = sum(
        int(values["compressed_bytes"]) for values in aggregate.values()
    )
    changed_bytes = sum(int(values["changed_bytes"]) for values in aggregate.values())
    wire_bytes = sum(int(shard["wire_bytes"]) for shard in output_stats)
    return {
        "generator": _GENERATOR_VERSION,
        "scope": "mutable_rollout_values",
        "spec": spec.as_dict(),
        "seed": seed,
        "checkpoint_shards": len(names_by_shard),
        "delta_shards": len(output_stats),
        "checkpoint_tensors": len(weight_map),
        "eligible_tensors": eligible_tensors,
        "changed_tensors": changed_tensors,
        "eligible_tensor_bytes": tensor_bytes,
        "changed_bytes": changed_bytes,
        "changed_byte_density": changed_bytes / tensor_bytes,
        "compressed_bytes": compressed_bytes,
        "wire_bytes": wire_bytes,
        "wire_ratio": wire_bytes / tensor_bytes,
        "immutable": dict(immutable),
        "by_encoding": dict(aggregate),
        "generation_s": round(time.perf_counter() - started, 6),
    }


def prepare_standard_delta(
    checkpoint_dir: str,
    source_dir: str,
    *,
    spec: SyntheticDeltaSpec,
    commit: Callable[[], None],
    seed: int = 42,
    workers: int | None = None,
    output_shard_for_tensor: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Reuse or build one durable standardized profiling delta."""

    metadata_path = Path(source_dir) / "delta_profile.json"
    index_path = Path(source_dir) / "weight_v000001" / "model.safetensors.index.json"
    expected = {
        "generator": _GENERATOR_VERSION,
        "spec": spec.as_dict(),
        "seed": seed,
    }
    if metadata_path.is_file() and index_path.is_file():
        result = json.loads(metadata_path.read_text())
        if all(result.get(key) == value for key, value in expected.items()):
            print(f"Reusing synthetic delta at {source_dir}")
            print(f"SYNTHETIC_DELTA={json.dumps(result, sort_keys=True)}")
            return result

    shutil.rmtree(source_dir, ignore_errors=True)
    Path(source_dir).mkdir(parents=True)
    result = write_standard_delta(
        checkpoint_dir,
        source_dir,
        spec=spec,
        seed=seed,
        workers=workers,
        output_shard_for_tensor=output_shard_for_tensor,
    )
    metadata_path.write_text(json.dumps(result, sort_keys=True))
    commit()
    print(f"Committed synthetic delta at {source_dir}")
    print(f"SYNTHETIC_DELTA={json.dumps(result, sort_keys=True)}")
    return result


_FULL_COVERAGE_CHANGE_STRIDE_BYTES = 1024 * 1024
_FULL_COVERAGE_ZERO_CHUNK = bytes(_STREAM_BYTES)
_FULL_COVERAGE_XOR_CHUNK = bytearray(_FULL_COVERAGE_ZERO_CHUNK)
_FULL_COVERAGE_XOR_CHUNK[::_FULL_COVERAGE_CHANGE_STRIDE_BYTES] = b"\x01" * (
    _STREAM_BYTES // _FULL_COVERAGE_CHANGE_STRIDE_BYTES
)
_FULL_COVERAGE_XOR_CHUNK = bytes(_FULL_COVERAGE_XOR_CHUNK)


def _compress_full_coverage_xor(byte_count: int) -> bytes:
    import zstandard

    compressor = zstandard.ZstdCompressor(level=1).compressobj()
    output = []
    remaining = byte_count
    while remaining:
        size = min(remaining, len(_FULL_COVERAGE_XOR_CHUNK))
        output.append(compressor.compress(memoryview(_FULL_COVERAGE_XOR_CHUNK)[:size]))
        remaining -= size
    output.append(compressor.flush())
    return b"".join(output)


def _full_coverage_target_checksum(
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
        first_change = (-position) % _FULL_COVERAGE_CHANGE_STRIDE_BYTES
        if first_change < len(chunk):
            changed = bytearray(chunk)
            for index in range(
                first_change,
                len(changed),
                _FULL_COVERAGE_CHANGE_STRIDE_BYTES,
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
    """Write one sparse DELTA with an element-level change in every tensor."""

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
                checksums[name] = _full_coverage_target_checksum(
                    handle,
                    offset=data_start + begin,
                    byte_count=byte_count,
                )
                compressed_tensors[name] = np.frombuffer(
                    _compress_full_coverage_xor(byte_count),
                    dtype=np.uint8,
                )
                tensor_bytes += byte_count
                changed_tensors += int(byte_count > 0)
                changed_bytes += math.ceil(
                    byte_count / _FULL_COVERAGE_CHANGE_STRIDE_BYTES
                )
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
        print(
            f"SYNTHETIC_DELTA_SHARD={json.dumps(result, sort_keys=True)}",
            flush=True,
        )
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
        "change_stride_bytes": _FULL_COVERAGE_CHANGE_STRIDE_BYTES,
        "wire_bytes": sum(int(shard["wire_bytes"]) for shard in shards),
        "generation_s": round(time.perf_counter() - started, 6),
    }
