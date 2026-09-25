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
KERNEL_CACHE_PATH = Path("/root/.cache/kernel-cache")

# Ports.
SIDECAR_PORT = 8000  # the container's public port
SGLANG_PORT = 8001  # the private sglang server behind the sidecar
RAY_PORT = 6379

# Stable trajectory affinity at the rollout front door and within a Modal pool.
MODAL_SESSION_ID_HEADER = "Modal-Session-ID"
STITCH_SESSION_ID_HEADER = "X-Stitch-Session-ID"

# Timeouts.
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 60 * MINUTES
