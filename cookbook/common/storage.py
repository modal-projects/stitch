"""Select and configure the checkpoint store used by cookbook deployments."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from stitch.stores.base import Store
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.stores.s3 import S3Store

StoreBackend = Literal["modal-volume", "s3"]

MODAL_VOLUME: StoreBackend = "modal-volume"
S3: StoreBackend = "s3"

_BACKEND_ENV = "STITCH_STORE_BACKEND"
_S3_SECRET_ENV = "STITCH_S3_SECRET_NAME"
_S3_ROOT_ENV = "S3_ROOT"
_S3_ENDPOINT_ENV = "S3_ENDPOINT_URL"
_AWS_TOKEN_PATH = Path("/tmp/stitch-aws-web-identity-token")


@dataclass(frozen=True)
class StoreDeployment:
    """Store settings known while a cookbook app and its images are defined.

    S3 credentials and the bucket root remain in a Modal Secret. Only the
    backend and Secret name are baked into images so remote module imports
    reconstruct the same app definition.
    """

    backend: StoreBackend
    s3_secret_name: str | None = None

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> StoreDeployment:
        values = os.environ if environ is None else environ
        backend = values.get(_BACKEND_ENV, MODAL_VOLUME)
        if backend not in (MODAL_VOLUME, S3):
            raise ValueError(
                f"{_BACKEND_ENV} must be {MODAL_VOLUME!r} or {S3!r}, got {backend!r}"
            )
        configured_secret = values.get(_S3_SECRET_ENV) or None
        if backend == S3 and configured_secret is None:
            raise ValueError(f"{_S3_SECRET_ENV} is required for the S3 store")
        secret_name = configured_secret if backend == S3 else None
        return cls(backend=backend, s3_secret_name=secret_name)

    @property
    def image_environment(self) -> dict[str, str]:
        values = {_BACKEND_ENV: self.backend}
        if self.s3_secret_name is not None:
            values[_S3_SECRET_ENV] = self.s3_secret_name
        return values

    @property
    def extra_packages(self) -> tuple[str, ...]:
        return ("boto3",) if self.backend == S3 else ()

    def modal_secrets(self) -> list[Any]:
        if self.s3_secret_name is None:
            return []
        import modal

        return [modal.Secret.from_name(self.s3_secret_name)]

    def bootstrap_credentials(
        self,
        environ: MutableMapping[str, str] | None = None,
        token_path: Path = _AWS_TOKEN_PATH,
    ) -> None:
        """Expose Modal OIDC credentials through boto3's web-identity chain.

        Static AWS credentials continue to work unchanged. A Secret containing
        ``AWS_ROLE_ARN`` can instead rely on the short-lived identity token Modal
        injects into each container.
        """
        if self.backend != S3:
            return
        values = os.environ if environ is None else environ
        token = values.get("MODAL_IDENTITY_TOKEN")
        role = values.get("AWS_ROLE_ARN")
        if (
            not token
            or not role
            or values.get("AWS_ACCESS_KEY_ID")
            or values.get("AWS_WEB_IDENTITY_TOKEN_FILE")
        ):
            return
        descriptor = os.open(
            token_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
            token_file.write(token)
        values["AWS_WEB_IDENTITY_TOKEN_FILE"] = str(token_path)

    def hook_config(
        self,
        namespace: str,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Build the trainer-hook settings after its Secret is available."""
        config = {"stitch_store_backend": self.backend}
        if self.backend == MODAL_VOLUME:
            return config

        values = os.environ if environ is None else environ
        root = values.get(_S3_ROOT_ENV, "").rstrip("/")
        if not root:
            raise ValueError(f"{_S3_ROOT_ENV} is required for the S3 store")
        namespace = namespace.strip("/")
        if not namespace:
            raise ValueError("an S3 store namespace is required")
        config["stitch_s3_root"] = f"{root}/{namespace}"
        if endpoint_url := values.get(_S3_ENDPOINT_ENV):
            config["stitch_s3_endpoint_url"] = endpoint_url
        return config


def create_store(
    backend: str,
    *,
    local_root: str | Path,
    run_id: str,
    volume_name: str | None = None,
    s3_root: str | None = None,
    s3_endpoint_url: str | None = None,
) -> Store:
    """Create a Store with one local layout regardless of its backing service."""
    if backend == MODAL_VOLUME:
        return ModalVolumeStore(local_root, volume_name=volume_name, run_id=run_id)
    if backend == S3:
        if not s3_root:
            raise ValueError("s3_root is required for the S3 store")
        return S3Store(
            s3_root,
            cache_dir=local_root,
            endpoint_url=s3_endpoint_url,
            run_id=run_id,
        )
    raise ValueError(f"unsupported store backend: {backend!r}")
