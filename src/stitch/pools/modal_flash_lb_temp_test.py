"""ModalFlashLBPool harness: same-app gateway derivation and the gateway-vs-upstream
split, against a stubbed ``modal`` module (the real Modal calls are validated e2e)."""

from __future__ import annotations

import asyncio
import sys
import types

from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool


class _SyncAsync:
    """Mimics a Modal SDK synchronicity-wrapped method: callable, with an ``.aio``."""

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    async def aio(self):
        return self.value


def _run_with_fake_modal(urls: dict, containers: list, fn) -> None:
    """Patch ``sys.modules['modal']`` to resolve staged URLs/containers, then run ``fn``."""
    lookups: list[tuple[str, str]] = []
    modal_stub = types.ModuleType("modal")
    modal_stub.Server = types.SimpleNamespace(
        from_name=lambda app, cls: (
            lookups.append((app, cls)),
            types.SimpleNamespace(get_url=_SyncAsync(urls.get((app, cls)))),
        )[1]
    )
    experimental_stub = types.ModuleType("modal.experimental")
    experimental_stub.flash_get_containers = lambda app, cls: containers
    modal_stub.experimental = experimental_stub
    originals = {
        name: sys.modules.get(name) for name in ("modal", "modal.experimental")
    }
    sys.modules["modal"] = modal_stub
    sys.modules["modal.experimental"] = experimental_stub
    try:
        fn(lookups)
    finally:
        for name, original in originals.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


def test_gateway_defaults_to_router_cls_in_same_app() -> None:
    pool = ModalFlashLBPool("pool", "Server")
    assert (pool.lb_app_name, pool.lb_cls_name) == ("pool", "Router")
    pool = ModalFlashLBPool(
        "pool", "Server", lb_app_name="custom-lb", lb_cls_name="Proxy"
    )
    assert (pool.lb_app_name, pool.lb_cls_name) == ("custom-lb", "Proxy")


def test_gateway_url_resolves_router_cls_not_upstream() -> None:
    pool = ModalFlashLBPool("pool", "Server")

    def check(lookups):
        url = pool.gateway_url()
        assert url == "https://router.example"
        assert lookups == [("pool", "Router")]

    _run_with_fake_modal({("pool", "Router"): "https://router.example/"}, [], check)


def test_gateway_url_async_resolves_router_cls() -> None:
    pool = ModalFlashLBPool("pool", "Server")

    def check(lookups):
        url = asyncio.run(pool.gateway_url_async())
        assert url == "https://router.example"
        assert lookups == [("pool", "Router")]

    _run_with_fake_modal({("pool", "Router"): "https://router.example/"}, [], check)


def test_gateway_url_requires_deployed_router() -> None:
    pool = ModalFlashLBPool("pool", "Server")

    def check(_lookups):
        try:
            pool.gateway_url()
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected RuntimeError")
        assert "pool.Router" in message
        assert "router classes" in message

    _run_with_fake_modal({}, [], check)


def test_replicas_and_scale_stay_on_upstream_app() -> None:
    pool = ModalFlashLBPool("pool", "Server")

    def check(lookups):
        assert pool.discover_replicas() == ["https://h1:8000"]
        assert pool._server().get_url() == "https://upstream.example"
        assert lookups == [("pool", "Server")]

    # ModalFlashLBPool inherits ``discover_replicas`` and ``_server`` untouched; the
    # stubbed lookup proves they still name the upstream app, not the LB.
    _run_with_fake_modal(
        {("pool", "Server"): "https://upstream.example"},
        [{"host": "h1:8000"}],
        check,
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"modal_flash_lb_temp harness: {len(tests)} PASS")
