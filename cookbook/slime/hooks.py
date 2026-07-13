"""slime framework-hook shims: thin re-exports of the shared logic (common/hooks.py).

A slime run config points its dotted hook paths at these names, e.g.
``custom_delta_pre_push_path = "cookbook.slime.hooks.commit_and_wake"`` (slime's publish
hook) and ``custom_rollout_request_hook_path = "cookbook.slime.hooks.gated_rollout_request_hook"``.
"""

from cookbook.common.hooks import claim_pool, commit_and_wake, gated_rollout_request_hook

__all__ = ["claim_pool", "commit_and_wake", "gated_rollout_request_hook"]
