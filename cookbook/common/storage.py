"""Select and configure the checkpoint store used by cookbook deployments."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.stores import factory as _factory

# The store factory + backend names live in stitch core; recipes and trainer
# hooks keep importing them from here.
MODAL_VOLUME = _factory.MODAL_VOLUME
S3 = _factory.S3
StoreBackend = _factory.StoreBackend
create_store = _factory.create_store

_BACKEND_ENV = "STITCH_STORE_BACKEND"
_S3_SECRET_ENV = "STITCH_S3_SECRET_NAME"
_S3_ROOT_ENV = "S3_ROOT"
_S3_ENDPOINT_ENV = "S3_ENDPOINT_URL"
_AWS_TOKEN_PATH = Path("/tmp/stitch-aws-web-identity-token")
_LOCAL_PUBLICATION_ROOT = Path("/tmp/stitch-publications")


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

    def updates_dir(self, run_dir: str | Path) -> Path:
        """Return the trainer-visible update directory for this backend.

        Volume publications are written on the shared mount. S3 publications
        stay on each trainer host until that host uploads its files.
        """
        run_dir = Path(run_dir)
        if self.backend == S3:
            return _LOCAL_PUBLICATION_ROOT / run_dir.name / "updates"
        return run_dir / "updates"

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
