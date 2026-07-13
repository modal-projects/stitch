"""miles framework-hook shims: thin re-exports of the shared logic (common/hooks.py).

A miles run config points its dotted hook paths at these names, e.g.
``custom_update_weight_post_write_path = "cookbook.miles.hooks.commit_and_wake"``.
The only per-framework difference is this re-export (the signatures already match miles').
"""

from cookbook.common.hooks import claim_pool, commit_and_wake, gated_rollout_request_hook

__all__ = ["claim_pool", "commit_and_wake", "gated_rollout_request_hook"]
