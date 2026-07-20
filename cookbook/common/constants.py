"""Shared deployment constants — container mount points, ports, and timeouts, general
across every recipe. The per-experiment names/values (APP_NAME, DELTA_VOLUME_NAME, the
disk layout) live in the config modules.
"""

from __future__ import annotations

from pathlib import Path

# Container mount points (Modal Volumes attach here).
HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")
PREP_PATH = Path("/prep")  # <PREP>/<tag>/{bf16 masters, served base, torch_dist ref_load}
SGLANG_CACHE_PATH = "/root/.cache/sglang"  # sglang kernel/JIT cache; survives cold starts

# Ports.
SIDECAR_PORT = 8000  # the container's public port
SGLANG_PORT = 8001   # the private sglang server behind the sidecar
RAY_PORT = 6379

# Timeouts.
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
