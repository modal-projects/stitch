"""SGLang weight-sync sidecar entry point for the miles_disagg example.

A thin entrypoint over :mod:`cookbook.sidecar`: the shared spine owns every
knob. ``helpers.start_sglang_sidecar`` launches it via
``python3 -m cookbook.miles_disagg.sidecar``.

The delta apply lives in the engine (``/pull_weights``), so no trainer decoder
is selected here — miles and slime publish the same wire format.
"""

from __future__ import annotations

from cookbook.sidecar import run_sidecar


def main() -> None:
    run_sidecar()


if __name__ == "__main__":
    main()
