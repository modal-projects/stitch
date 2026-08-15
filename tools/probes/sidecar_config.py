"""CPU-only sanity check that ``python -m stitch.sidecar`` works inside a bare
image (debian_slim + the stitch source, no fastapi/sglang/modal deps):
--help must exit 0, a disk-mode invocation without --local-checkpoint-dir must
fail with the validation message, and an unresolvable --store-factory must fail
with core's factory-resolution error. Run by
``uv run --extra modal modal run -e stitch-dev -m tools.probes.sidecar_config``;
the result is the single PROBE_RESULT verdict line.
"""

from __future__ import annotations

import modal

app = modal.App("stitch-sidecar-config-probe")

image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("stitch")

_HELP = ["--help"]
# Missing --local-checkpoint-dir while in disk mode: expect the parse error
# (raised before the factory reference is ever resolved).
_DISK_BAD = [
    "--bulletin-root",
    "/cache",
    "--base-checkpoint-dir",
    "/model",
    "--delta-update-mode",
    "disk",
    "--store-factory",
    "stitch.sidecar:no_such_factory",
    "--run-id",
    "probe",
]
# A parseable invocation whose factory reference does not resolve: expect
# core's factory-resolution error, not a Store construction attempt.
_FACTORY_BAD = [
    "--bulletin-root",
    "/cache",
    "--base-checkpoint-dir",
    "/model",
    "--delta-update-mode",
    "cpu",
    "--store-factory",
    "stitch.sidecar:no_such_factory",
    "--run-id",
    "probe",
]


@app.function(image=image, cpu=1.0, memory=512)
def probe() -> str:
    import subprocess
    import sys

    def run_sidecar(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "stitch.sidecar", *argv],
            capture_output=True,
            text=True,
        )

    help_run = run_sidecar(_HELP)
    disk_run = run_sidecar(_DISK_BAD)
    factory_run = run_sidecar(_FACTORY_BAD)
    ok = (
        help_run.returncode == 0
        and disk_run.returncode != 0
        and "--local-checkpoint-dir is required in disk mode" in disk_run.stderr
        and factory_run.returncode != 0
        and "has no attribute 'no_such_factory'" in factory_run.stderr
    )
    verdict = (
        f"PROBE_RESULT ok={str(ok).lower()} "
        f"detail=help_rc={help_run.returncode} disk_rc={disk_run.returncode} "
        f"factory_rc={factory_run.returncode}"
    )
    print(verdict)
    if not ok:
        print("--- help stderr ---\n", help_run.stderr)
        print("--- disk-mode stderr ---\n", disk_run.stderr)
        print("--- factory stderr ---\n", factory_run.stderr)
    return verdict
