"""Proxy credentials for cookbook clients of authenticated Modal Servers."""

from __future__ import annotations

import os


def proxy_auth_secret():
    import modal

    return modal.Secret.from_name(
        os.environ.get("STITCH_PROXY_AUTH_SECRET", "stitch-proxy-auth"),
        required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
    )
