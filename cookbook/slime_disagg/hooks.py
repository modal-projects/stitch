"""Hook entry points for the slime_disagg example.

This module exists because the experiment configs reference these symbols by
dotted string (``custom_delta_pre_push_path = "cookbook.slime_disagg.hooks.
commit_and_wake"``, ``custom_rollout_request_hook_path = "...gated_rollout_
request_hook"``), resolved inside the trainer process. The implementations
live in :mod:`cookbook.bulletin_hooks`.
"""

from cookbook.bulletin_hooks import claim_pool, commit_and_wake, gated_rollout_request_hook


__all__ = ["claim_pool", "commit_and_wake", "gated_rollout_request_hook"]
