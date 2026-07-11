"""Front-door authorization policy.

Pure, Modal-free so it is unit-testable; ``modal_serve.py`` supplies the env
values and turns a rejection into a FastAPI response. The Modal server is
deployed ``unauthenticated=True`` (auth is enforced in-app), so this check is the
*only* gate in front of the hot-load API and the inference proxy.
"""

from __future__ import annotations

from collections.abc import Mapping


def authorize(
    headers: Mapping[str, str],
    *,
    api_key: str | None,
    provider_model: str | None,
    provider_deployment: str | None,
) -> tuple[int, str] | None:
    """Return ``(status, message)`` to reject a request, or ``None`` to allow it.

    The API key is mandatory. When it is unset the front door rejects *every*
    request (fail closed) instead of serving an open public endpoint — a missing
    or mis-named secret key must not silently disable the only auth gate.
    Provider-Model / Provider-Deployment are optional secondary checks, enforced
    only when configured. Header lookups are case-insensitive when ``headers`` is
    a Starlette ``Headers`` (the production caller); tests pass lowercase keys.
    """
    if not api_key:
        return (503, "provider auth is not configured")
    if headers.get("authorization") != f"Bearer {api_key}":
        return (401, "unauthorized")
    if provider_model and headers.get("provider-model") != provider_model:
        return (400, "Provider-Model header does not match this deployment")
    if provider_deployment and headers.get("provider-deployment") != provider_deployment:
        return (400, "Provider-Deployment header does not match this deployment")
    return None
