"""Synthesize slime-format disk-delta version dirs from a base HF checkpoint.

The smoke test publishes real weight versions *without* running a Megatron
trainer: :class:`DeltaPublisher` perturbs a handful of tensors and writes
byte-for-byte valid ``weight_v{N:06d}/`` delta dirs (xor + zstd + xxh3-128 —
the qwen delta config's wire format) that slime's real
``slime.utils.disk_delta.apply_deltas`` decoder accepts and checksum-verifies.

This lets the control-plane test drive the true host-side delta-apply +
reconcile path on CPU, and lets the serving smoke reload real (slightly
perturbed) weights into a live SGLang engine.

Only stdlib + numpy + zstandard + xxhash are used (the same deps the serving
image installs for the decoder), so this module imports with no slime present.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from pathlib import Path

# Wire-format constants, mirroring slime.utils.disk_delta + the qwen delta
# config the trainer publishes with. The decoder reads these from each
# version's model.safetensors.index.json `metadata` block.
DELTA_ENCODING = "xor"
COMPRESSION_FORMAT = "zstd"
CHECKSUM_FORMAT = "xxh3-128"
DELTA_FILENAME = "delta-00001-of-00001.safetensors"


def _checksum(checksum_format: str, buf: bytes) -> str:
    """Per-tensor checksum of the full *new* tensor bytes.

    Identical to slime.utils.disk_delta's hasher choice, so the checksum the
    decoder recomputes over the patched region matches what we record here.
    """
    if checksum_format == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128(buf).hexdigest()
    if checksum_format == "blake3":
        import blake3

        return blake3.blake3(buf).hexdigest()
    if checksum_format == "adler32":
        import zlib

        return f"{zlib.adler32(buf) & 0xFFFFFFFF:08x}"
    raise KeyError(f"unsupported checksum_format {checksum_format!r}")


def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int]]:
    """``name -> (shard_path, byte_offset, nbytes)`` for every tensor.

    Reads each shard's safetensors header exactly the way the decoder locates
    tensors in the local checkpoint, so the names we publish deltas for line up
    with the regions the decoder patches.
    """
    locations: dict[str, tuple[str, int, int]] = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, data_start + begin, end - begin)
    return locations


def _read_tensor_bytes(location: tuple[str, int, int]):
    import numpy as np

    path, offset, nbytes = location
    with open(path, "rb") as f:
        f.seek(offset)
        return np.frombuffer(f.read(nbytes), dtype=np.uint8).copy()


def select_delta_tensors(ckpt_dir: str, count: int = 3) -> list[str]:
    """Pick a few small tensors to perturb (deterministic, model-agnostic).

    Smallest-first keeps the synthetic delta cheap and the round-trip fast while
    still exercising the multi-tensor apply + checksum path.
    """
    locations = _tensor_locations(ckpt_dir)
    if not locations:
        raise FileNotFoundError(f"no *.safetensors tensors under {ckpt_dir!r}")
    by_size = sorted(locations, key=lambda n: (locations[n][2], n))
    return by_size[: max(1, count)]


def _encode_xor_delta_file(out_path: Path, tensors: dict[str, tuple]) -> None:
    """Write one safetensors-format xor delta file.

    ``tensors`` maps ``name -> (old_bytes, new_bytes)`` (uint8 arrays). Each
    entry's data section holds ``zstd(old ^ new)``; ``__metadata__[name]`` is the
    checksum of the full *new* bytes — what the decoder recomputes over the
    region after applying the xor.
    """
    import numpy as np
    import zstandard

    compressor = zstandard.ZstdCompressor()
    header: dict[str, object] = {}
    metadata: dict[str, str] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (old, new) in tensors.items():
        if old.shape != new.shape:
            raise ValueError(f"tensor {name!r} shape changed: {old.shape} -> {new.shape}")
        delta = np.bitwise_xor(old, new)
        compressed = compressor.compress(delta.tobytes())
        header[name] = {
            "dtype": "U8",
            "shape": [len(compressed)],
            "data_offsets": [offset, offset + len(compressed)],
        }
        metadata[name] = _checksum(CHECKSUM_FORMAT, new.tobytes())
        blobs.append(compressed)
        offset += len(compressed)
    header["__metadata__"] = metadata
    header_json = json.dumps(header).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for blob in blobs:
            f.write(blob)


def _write_index(
    version_dir: Path, *, version: int, base_version: int, names: list[str]
) -> None:
    index = {
        "metadata": {
            # Zero-padded strings: the decoder's applied-version state file and
            # base-version precondition compare these as strings.
            "version": f"{version:06d}",
            "base_version": f"{base_version:06d}",
            "delta_encoding": DELTA_ENCODING,
            "compression_format": COMPRESSION_FORMAT,
            "checksum_format": CHECKSUM_FORMAT,
        },
        "weight_map": {name: DELTA_FILENAME for name in names},
    }
    with open(version_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


class DeltaPublisher:
    """Publishes a chain of synthetic delta versions for one run.

    Each :meth:`publish_next` perturbs the tracked tensors a little further and
    writes the next ``weight_v{N:06d}/`` dir under ``run_dir``. ``current`` tracks
    the cumulative *new* bytes per tensor, so :meth:`expected_tensor` is the exact
    byte content the decoder must produce in the local checkpoint after applying
    up to that version — the round-trip assertion the test pins.
    """

    def __init__(self, base_checkpoint_dir: str, run_dir: str, tensor_names: list[str]) -> None:
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_dir = Path(run_dir)
        self.names = list(tensor_names)
        self._locations = _tensor_locations(base_checkpoint_dir)
        missing = [n for n in self.names if n not in self._locations]
        if missing:
            raise KeyError(f"tensors not in base checkpoint: {missing}")
        # Start from the base bytes — a new run forks at base.
        self.current = {n: _read_tensor_bytes(self._locations[n]) for n in self.names}
        self.version = 0

    def expected_tensor(self, name: str):
        return self.current[name]

    def publish_next(self, *, changed_elems: int = 16) -> int:
        """Perturb the tracked tensors and write the next delta version dir.

        Only the low byte of a few 2-byte elements is bumped, so bf16/fp16
        weights stay finite (the exponent byte is untouched) and the perturbed
        model still generates — but the bytes change, so the round-trip is a real
        check. Returns the new version number.
        """
        import numpy as np

        base_version = self.version
        self.version += 1
        rng = np.random.default_rng(self.version)
        old = {n: self.current[n].copy() for n in self.names}
        for name in self.names:
            buf = self.current[name].copy()
            low_bytes = np.arange(0, buf.size - 1, 2)  # low (mantissa) byte of each 2-byte element
            if low_bytes.size:
                picks = rng.choice(low_bytes, size=min(changed_elems, low_bytes.size), replace=False)
                buf[picks] = (buf[picks].astype(np.uint16) + 1).astype(np.uint8)
            self.current[name] = buf

        version_dir = self.run_dir / f"weight_v{self.version:06d}"
        version_dir.mkdir(parents=True, exist_ok=True)
        tensors = {n: (old[n], self.current[n]) for n in self.names}
        _encode_xor_delta_file(version_dir / DELTA_FILENAME, tensors)
        _write_index(version_dir, version=self.version, base_version=base_version, names=self.names)
        return self.version


def read_local_tensor(local_checkpoint_dir: str, name: str):
    """Read one tensor's raw bytes from a (patched) local checkpoint."""
    return _read_tensor_bytes(_tensor_locations(local_checkpoint_dir)[name])
