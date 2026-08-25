"""Shared deployment constants — container mount points, ports, and timeouts."""

from __future__ import annotations

from pathlib import Path

# Container mount points (Modal Volumes attach here).
HF_CACHE_PATH = Path("/root/.cache/huggingface")
CHECKPOINTS_PATH = Path("/checkpoints")
DATA_PATH = Path("/data")
STITCH_PATH = Path("/stitch")
DRAFT_PATH = Path("/draft")
SGLANG_CACHE_PATH = (
    "/root/.cache/sglang"  # sglang kernel/JIT cache; survives cold starts
)

# Ports.
SIDECAR_PORT = 8000  # the container's public port
SGLANG_PORT = 8001  # the private sglang server behind the sidecar
RAY_PORT = 6379

# Modal's native sticky-routing header. The same session ID is also used by the
# cookbook router when selecting a rollout replica.
MODAL_SESSION_ID_HEADER = "Modal-Session-ID"

# Internal lease-identity header. Modal-Session-ID is platform-reserved and
# does not survive the *.modal.direct edge intact, so the router strips it
# and re-injects the session value as Stitch-Lease-Key upstream.
STITCH_LEASE_HEADER = "Stitch-Lease-Key"

# Timeouts.
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 60 * MINUTES
