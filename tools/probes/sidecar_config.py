"""CPU-only sanity check that the sidecar's config surface works inside a bare
image (debian_slim + the stitch source, no fastapi/sglang/modal deps):
``SidecarConfig.from_argv`` must parse a valid invocation, and a disk-mode
invocation without --local-checkpoint-dir must fail with the validation
message. Run by ``uv run modal run -e stitch-dev -m tools.probes.sidecar_config``;
the result is the single PROBE_RESULT verdict line.
"""

from __future__ import annotations

import modal

app = modal.App("stitch-sidecar-config-probe")

image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("stitch")

_GOOD_SNIPPET = (
    "from stitch.sidecar import SidecarConfig; "
    "c = SidecarConfig.from_argv(['--bulletin-root', '/cache', "
    "'--base-checkpoint-dir', '/model', '--delta-update-mode', 'cpu', "
    "'--store-backend', 'modal-volume', '--run-id', 'probe']); "
    "assert c.store_backend == 'modal-volume'"
)
# Missing --local-checkpoint-dir while in disk mode: expect the parse error.
_BAD_SNIPPET = (
    "from stitch.sidecar import SidecarConfig; "
    "SidecarConfig.from_argv(['--bulletin-root', '/cache', "
    "'--base-checkpoint-dir', '/model', '--delta-update-mode', 'disk', "
    "'--store-backend', 'modal-volume', '--run-id', 'probe'])"
)


@app.function(image=image, cpu=1.0, memory=512)
def probe() -> str:
    import subprocess
    import sys

    good_run = subprocess.run(
        [sys.executable, "-c", _GOOD_SNIPPET],
        capture_output=True,
        text=True,
    )
    bad_run = subprocess.run(
        [sys.executable, "-c", _BAD_SNIPPET],
        capture_output=True,
        text=True,
    )
    ok = (
        good_run.returncode == 0
        and bad_run.returncode != 0
        and "--local-checkpoint-dir is required in disk mode" in bad_run.stderr
    )
    verdict = (
        f"PROBE_RESULT ok={str(ok).lower()} "
        f"detail=good_rc={good_run.returncode} bad_rc={bad_run.returncode}"
    )
    print(verdict)
    if not ok:
        print("--- good-args stderr ---\n", good_run.stderr)
        print("--- bad-args stderr ---\n", bad_run.stderr)
    return verdict
