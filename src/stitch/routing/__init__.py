"""Rollout-request routing: a policy-driven load balancer over a Pool's replicas.

The policies (gorgo, session-affinity, least-request, ...) come from the
``gorgo`` package; this package owns the stitch-side glue — prompt token
extraction (``tokens``), per-replica routing state and candidate filtering
(``state``), and the HTTP proxy service (``service``).
"""

from stitch.routing.service import create_router_app, serve_router

__all__ = ["create_router_app", "serve_router"]
