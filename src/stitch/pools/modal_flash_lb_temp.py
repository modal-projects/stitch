"""``ModalFlashLBPool`` — the ``Pool`` instance for a Modal Flash service whose traffic
enters through the session-routing LB deployed alongside it.

With a per-container queue bound (sglang ``--max-queued-requests``), Flash's own sticky
routing turns the first saturated replicas into 503 attractors: the sessions already stuck
on a full replica keep retrying it, so it stays full while the rest of the pool starves.
The cookbook router (``cookbook/common/router.py``, a fork of the flash-smart-router folded
into the run's app as the ``Router`` + ``RouterRegistry`` classes) routes sessions by *live
container load* and retries a 503 on a healthier replica, so load spreads instead of
sticking.

Only the gateway moves: rollout traffic enters through the router's URL, while replica
discovery, ``wake``, and ``scale`` stay on the upstream ``Server`` class (the router holds
no weights), inherited from ``ModalFlashPool``. Like that pool, this is a client to a
running deployment, and every Modal call is import-lazy so the module loads without Modal.

This module is temporary: it exists until smart routing is productized on the Flash side
and the router classes can be dropped from the recipes.
"""

from __future__ import annotations

from stitch.pools.modal_flash import ModalFlashPool


class ModalFlashLBPool(ModalFlashPool):
    def __init__(
        self,
        app_name: str,
        cls_name: str,
        lb_app_name: str | None = None,
        lb_cls_name: str = "Router",
    ) -> None:
        super().__init__(app_name, cls_name)
        # The router is a class in the same app, deployed alongside the upstream.
        self.lb_app_name = lb_app_name or app_name
        self.lb_cls_name = lb_cls_name

    def _lb_server(self):
        import modal

        return modal.Server.from_name(self.lb_app_name, self.lb_cls_name)

    def gateway_url(self) -> str:
        return self._require_gateway(self._lb_server().get_url())

    async def gateway_url_async(self) -> str:
        return self._require_gateway(await self._lb_server().get_url.aio())

    def _require_gateway(self, url: str | None) -> str:
        if not url:
            raise RuntimeError(
                f"no gateway URL for {self.lb_app_name}.{self.lb_cls_name} — deploy the "
                "router classes with the pool (the cookbook recipe apps do this: see "
                "cookbook/common/router.py)"
            )
        return str(url).rstrip("/")
