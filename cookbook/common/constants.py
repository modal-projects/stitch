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

# Timeouts.
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 60 * MINUTES
