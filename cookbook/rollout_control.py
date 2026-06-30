"""Trainer-side primitives shared by both rollout-control planes.

The cookbook has two *intended* control planes for advancing the rollout pool's
``latest`` pointer:

- **trainer-as-writer** (slime_disagg / miles_disagg): the trainer's rank-0
  publish hook writes the Modal-Volume bulletin board directly
  (:mod:`cookbook.bulletin_hooks`).
- **frontdoor-as-writer** (standalone_rollouts): the trainer POSTs a hot-load
  signal to an external front door that owns the pointer
  (:mod:`cookbook.standalone_rollouts.slime.hooks`).

Those two planes are the real "disagg vs standalone" axis and stay distinct. But
the trainer-side *plumbing* around them was copy-forked: an identical torch rank
probe, an identical args-or-env settings reader, and identical session-affinity
header stamping. This module owns that plumbing once so the two hook stacks only
encode their genuine difference (who writes the pointer), not boilerplate.
"""

from __future__ import annotations

import os
from typing import Any


def distributed_rank() -> int | None:
    """Return this process's torch-distributed rank, or ``None`` if torch
    distributed isn't initialized (single-process / pre-init).

    Used to gate rank-0-only side effects (pointer writes, transport copies,
    pool wakes) so only one writer acts per publish.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def read_setting(
    args: Any | None,
    attr: str,
    env: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    """Read a string setting from the trainer ``args`` namespace, falling back to
    an environment variable, then a default.

    The trainer's ``--custom-config-path`` setattr's every config key onto
    ``args``; an env var is the deploy-time fallback for values that can't ride a
    per-launch arg. Returns ``""`` for an absent, non-required setting.
    """
    value = getattr(args, attr, None) if args is not None else None
    if value is None:
        value = os.environ.get(env)
    if value is None:
        value = default
    if value is None and required:
        raise RuntimeError(
            f"Missing required setting {attr!r} or environment variable {env}"
        )
    return str(value) if value is not None else ""


def apply_session_affinity(request: dict[str, Any], session_id: Any, header: str) -> None:
    """Stamp ``header: session_id`` onto a rollout request's headers (idempotent).

    Sticky-session routing: a sample carrying a ``session_id`` should land on the
    same replica across turns. ``setdefault`` so an explicit caller header wins.
    """
    if not session_id:
        return
    headers = dict(request.get("headers") or {})
    headers.setdefault(header, str(session_id))
    request["headers"] = headers
