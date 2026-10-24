import pytest

from tools.check_endpoint_auth import violations


@pytest.mark.parametrize(
    "source",
    [
        "@app.server(unauthenticated = True)\nclass Server: pass",
        "@app.server(\n unauthenticated=True,\n)\nclass Server: pass",
        'options = {"unauthenticated": True}\n@app.server(**options)\nclass Server: pass',
        "@app.server(unauthenticated=public)\nclass Server: pass",
        "@modal.web_server(8000)\ndef serve(): pass",
        "@modal.asgi_app(requires_proxy_auth=False)\ndef serve(): pass",
        "from modal import fastapi_endpoint as endpoint\n@endpoint()\ndef serve(): pass",
    ],
)
def test_rejects_public_endpoint(source):
    assert violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "@app.server()\nclass Server: pass",
        "@app.server(unauthenticated=False)\nclass Server: pass",
        "@modal.fastapi_endpoint(requires_proxy_auth=True)\ndef serve(): pass",
        '# unauthenticated=True\nexample = "unauthenticated=True"',
    ],
)
def test_accepts_authenticated_endpoint_and_documentation(source):
    assert violations(source) == []
