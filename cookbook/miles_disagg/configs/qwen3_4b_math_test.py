import asyncio
import json

import httpx
import pytest

from cookbook.miles_disagg.configs import qwen3_4b_math


@pytest.mark.parametrize("status", [200, 503])
def test_math_agent_preserves_session_requests_and_propagates_failure(
    monkeypatch, status
):
    requests = []
    client = httpx.AsyncClient

    def upstream(request):
        requests.append(request)
        return httpx.Response(status, json={"choices": []})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upstream), **kwargs),
    )

    async def run():
        await qwen3_4b_math.generate_math_answer(
            base_url="http://session/sessions/example",
            prompt=[{"role": "user", "content": "What is 1+1?"}],
            request_kwargs={"temperature": 0, "max_tokens": 16},
            metadata={},
        )

    if status == 200:
        asyncio.run(run())
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(run())
    assert len(requests) == 1
    assert str(requests[0].url) == "http://session/sessions/example/v1/chat/completions"
    assert json.loads(requests[0].content) == {
        "model": qwen3_4b_math.SOURCE_MODEL,
        "messages": [{"role": "user", "content": "What is 1+1?"}],
        "temperature": 0,
        "max_tokens": 16,
    }
