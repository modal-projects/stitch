"""SGLang weight-sync sidecar entry point for the slime_disagg example.

A thin adapter over :mod:`cookbook.sidecar`: the shared spine owns every knob;
this only names slime's host-side delta decoder. ``helpers.start_sglang_sidecar``
launches it via ``python3 -m cookbook.slime_disagg.sidecar``.
"""

from __future__ import annotations

from cookbook.sidecar import run_sidecar


# slime's decoder is the engine's lazy default, so it need not be injected.
DISK_DELTA_MODULE = "slime.utils.disk_delta"


def main() -> None:
    run_sidecar(disk_delta_module=DISK_DELTA_MODULE, inject_apply_deltas=False)


if __name__ == "__main__":
    main()
