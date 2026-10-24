import httpx

from stitch import service
from stitch.pools.base import Pool
from stitch.types import PoolState, ReplicaState


def test_pool_readiness_authenticates_gateway_health(monkeypatch):
    class AuthenticatedPool(Pool):
        def gateway_url(self):
            return "https://gateway"

        def gateway_headers(self):
            return {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}

    async def ready(pool):
        return PoolState([ReplicaState(ready=True)])

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr(service, "readiness", ready)
    assert service.await_pool_ready(AuthenticatedPool(), replica_floor=1, timeout=1)
    assert len(requests) == 1
    assert requests[0].headers["Modal-Key"] == "wk-test"
    assert requests[0].headers["Modal-Secret"] == "ws-test"
