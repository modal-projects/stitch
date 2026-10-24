import asyncio

import httpx
import pytest

from stitch.pools.modal_flash import ModalFlashPool, proxy_auth_headers
from stitch.types import VersionRef


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    return {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}


def test_authenticated_gateway_and_replica_routing(credentials):
    pool = ModalFlashPool("app", "Server")
    pool._upstream_url_cache = "https://pool"
    assert pool.gateway_headers() == credentials
    url, headers = pool.replica_request("https://replica", "/wake")
    assert url == "https://pool/wake"
    assert headers == {**credentials, "modal-flash-upstream": "replica:443"}


@pytest.mark.parametrize(
    "present", [None, "MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]
)
def test_required_credentials_reject_missing_or_partial_token(monkeypatch, present):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    if present:
        monkeypatch.setenv(present, "test")
    with pytest.raises(ValueError, match="Set both"):
        proxy_auth_headers(required=True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wake_sends_auth_and_replica_routing(credentials, monkeypatch, asynchronous):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200)

    transport = httpx.MockTransport(handle)
    sync_client, async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: sync_client(transport=transport, **kw)
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: async_client(transport=transport, **kw)
    )
    pool = ModalFlashPool("app", "Server")
    pool._upstream_url_cache = "https://pool"
    if asynchronous:
        asyncio.run(pool.wake_async(["https://replica"], VersionRef("run", 1)))
    else:
        pool.wake(["https://replica"], VersionRef("run", 1))
    assert len(requests) == 1
    assert requests[0].headers["Modal-Key"] == credentials["Modal-Key"]
    assert requests[0].headers["Modal-Secret"] == credentials["Modal-Secret"]
    assert requests[0].headers["modal-flash-upstream"] == "replica:443"
