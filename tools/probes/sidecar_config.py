"""CPU-only sanity check that ``python -m stitch.sidecar`` works inside a
bare image (debian_slim + the stitch source, no fastapi/sglang/modal deps):
--help must exit 0, and a disk-mode invocation without --local-checkpoint-dir
must fail with the validation message. Run by
``uv run modal run -e stitch-dev -m tools.probes.sidecar_config``; the result
is the single PROBE_RESULT verdict line.
"""

from __future__ import annotations

import modal

app = modal.App("stitch-sidecar-config-probe")

image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("stitch")

_HELP = ["--help"]
# Missing --local-checkpoint-dir while in disk mode: expect the parse error.
_BAD = [
    "--bulletin-root",
    "/cache",
    "--base-checkpoint-dir",
    "/model",
    "--delta-update-mode",
    "disk",
    "--run-id",
    "probe",
]


@app.function(image=image, cpu=1.0, memory=512)
def probe() -> str:
    import subprocess
    import sys

    help_run = subprocess.run(
        [sys.executable, "-m", "stitch.sidecar", *_HELP],
        capture_output=True,
        text=True,
    )
    bad_run = subprocess.run(
        [sys.executable, "-m", "stitch.sidecar", *_BAD],
        capture_output=True,
        text=True,
    )
    ok = (
        help_run.returncode == 0
        and bad_run.returncode != 0
        and "--local-checkpoint-dir is required in disk mode" in bad_run.stderr
    )
    verdict = (
        f"PROBE_RESULT ok={str(ok).lower()} "
        f"detail=help_rc={help_run.returncode} bad_rc={bad_run.returncode}"
    )
    print(verdict)
    if not ok:
        print("--- --help stdout ---\n", help_run.stdout)
        print("--- --help stderr ---\n", help_run.stderr)
        print("--- bad-args stderr ---\n", bad_run.stderr)
    return verdict
