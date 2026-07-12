"""Pure authorization policy for the public standalone front door."""

from __future__ import annotations

from collections.abc import Mapping


def authorize(
    headers: Mapping[str, str],
    *,
    api_key: str | None,
    provider_model: str | None,
    provider_deployment: str | None,
) -> tuple[int, str] | None:
    """Return an HTTP rejection or ``None`` when the request is authorized."""
    if not api_key:
        return (503, "provider auth is not configured")
    if headers.get("authorization") != f"Bearer {api_key}":
        return (401, "unauthorized")
    if provider_model and headers.get("provider-model") != provider_model:
        return (400, "Provider-Model header does not match this deployment")
    if (
        provider_deployment
        and headers.get("provider-deployment") != provider_deployment
    ):
        return (400, "Provider-Deployment header does not match this deployment")
    return None
