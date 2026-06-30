"""SGLang weight-sync sidecar entry point for the miles_disagg example.

A thin adapter over :mod:`cookbook.sidecar`: the shared spine owns every knob;
this only names miles' host-side delta decoder and asks for it to be injected.
``helpers.start_sglang_sidecar`` launches it via
``python3 -m cookbook.miles_disagg.sidecar``.

miles' ``disk_delta`` is byte-identical to slime's (same XOR/overwrite + zstd +
xxh3/blake3/adler32 wire format), but the rollout pool image ships miles
``--no-deps``, not slime, so the decoder must be injected explicitly rather than
relying on the engine's slime default.
"""

from __future__ import annotations

from cookbook.sidecar import run_sidecar


DISK_DELTA_MODULE = "miles.utils.disk_delta"


def main() -> None:
    run_sidecar(disk_delta_module=DISK_DELTA_MODULE, inject_apply_deltas=True)


if __name__ == "__main__":
    main()
