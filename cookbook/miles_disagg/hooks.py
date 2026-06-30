"""Modal publish + rollout-gating hooks for the miles_disagg example.

Thin wrappers around :mod:`cookbook.bulletin_hooks` with the miles-specific
env-var fallbacks for the Flash app / server class name.
"""

from __future__ import annotations

from typing import Any

from cookbook.bulletin_hooks import (
    claim_pool as _claim_pool,
    commit_and_wake as _commit_and_wake,
    gated_rollout_request_hook,
)


_APP_NAME_ENV = "MILES_DELTA_APP_NAME"
_CLS_NAME_ENV = "MILES_DELTA_SERVER_CLS_NAME"


def claim_pool(args: Any) -> None:
    """Claim the rollout pool for this run at launch (resets every replica to base)."""
    _claim_pool(args, app_name_env=_APP_NAME_ENV, cls_name_env=_CLS_NAME_ENV)


def commit_and_wake(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """miles ``custom_delta_pre_push_path`` hook (publish-only, bulletin board)."""
    _commit_and_wake(
        args,
        version_dir,
        rollout_engines,
        app_name_env=_APP_NAME_ENV,
        cls_name_env=_CLS_NAME_ENV,
    )


# Re-export for the trainer's custom_rollout_request_hook_path.
__all__ = ["claim_pool", "commit_and_wake", "gated_rollout_request_hook"]
