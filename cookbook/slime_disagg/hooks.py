"""Modal publish + rollout-gating hooks for the slime_disagg example.

Thin wrappers around :mod:`cookbook.bulletin_hooks` with the slime-specific
env-var fallbacks for the Flash app / server class name.
"""

from __future__ import annotations

from typing import Any

from cookbook.bulletin_hooks import (
    CachedLatestPointer,
    bulletin_root,
    commit_and_wake as _commit_and_wake,
    distributed_rank,
    gated_rollout_request_hook,
)


def commit_and_wake(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook (publish-only, bulletin board)."""
    _commit_and_wake(
        args,
        version_dir,
        rollout_engines,
        app_name_env="SLIME_DELTA_APP_NAME",
        cls_name_env="SLIME_DELTA_SERVER_CLS_NAME",
    )


# Re-export for the trainer's custom_rollout_request_hook_path.
__all__ = ["commit_and_wake", "gated_rollout_request_hook"]
