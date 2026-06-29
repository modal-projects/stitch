"""Modal publish + rollout-gating hooks for the miles_disagg example.

Thin wrappers around :mod:`cookbook.bulletin_hooks` with the miles-specific
env-var fallbacks for the Flash app / server class name.
"""

from __future__ import annotations

from typing import Any

from cookbook.bulletin_hooks import (
    commit_and_wake as _commit_and_wake,
    gated_rollout_request_hook,
)


def commit_and_wake(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """miles ``custom_delta_pre_push_path`` hook (publish-only, bulletin board)."""
    _commit_and_wake(
        args,
        version_dir,
        rollout_engines,
        app_name_env="MILES_DELTA_APP_NAME",
        cls_name_env="MILES_DELTA_SERVER_CLS_NAME",
    )


# Re-export for the trainer's custom_rollout_request_hook_path.
__all__ = ["commit_and_wake", "gated_rollout_request_hook"]
