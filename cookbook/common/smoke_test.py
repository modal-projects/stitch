import json

import pytest

from cookbook.common import smoke
from stitch.pools.base import Pool


@pytest.mark.parametrize("claimed", [False, True])
def test_smoke_authenticates_gateway_and_replica_requests(monkeypatch, claimed):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    calls = []

    class FakePool(Pool):
        def __init__(self, *args):
            pass

        def gateway_url(self):
            return "https://gateway"

        def discover_replicas(self):
            return ["https://replica"]

        def replica_request(self, replica, path):
            return "https://gateway" + path, {
                "Modal-Key": "wk-test",
                "Modal-Secret": "ws-test",
                "modal-flash-upstream": "replica:443",
            }

    def urlopen(request, **kwargs):
        calls.append(request)
        assert request.get_header("Modal-key") == "wk-test"
        assert request.get_header("Modal-secret") == "ws-test"
        if request.data:
            result = {
                "choices": [{"message": {"content": "OK"}}],
                "weight_version_start": 0,
                "weight_version_end": 0,
            }
        else:
            result = {
                "run_id": "run" if claimed else None,
                "applied": "run/weight_v000000",
            }
        import io

        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(smoke, "ModalFlashPool", FakePool)
    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)
    smoke.smoke_flash_pool(
        app_name="app",
        cls_name="Server",
        model_name="model",
        weight_version=0,
        timeout_seconds=0,
    )
    assert len(calls) == (3 if claimed else 2)
    if claimed:
        assert calls[-1].get_header("Modal-flash-upstream") == "replica:443"
