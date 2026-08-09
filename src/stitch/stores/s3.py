"""``S3Store`` — an S3-backed counterpart to ``ModalVolumeStore``.

``root`` identifies one run's directory in an S3 bucket. Stitch owns the
``<root>/latest`` pointer, while published versions live under
``<root>/updates/weight_vNNNNNN/``. This is the same layout exposed by a Modal
Volume store; only the persistence and visibility operations differ.

S3 is strongly read-after-write consistent, so ``refresh`` and ``commit`` are
no-ops. Because S3 is not a mounted filesystem, ``materialize`` mirrors the
``updates`` directory into ``cache_dir`` and returns the requested local version
directory.

``boto3`` is imported only when needed. Credentials and region use its standard
resolution chain, while ``endpoint_url`` supports S3-compatible services.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from stitch.stores.base import Store
from stitch.types import VersionManifest, VersionRef

_POINTER = "latest"
_UPDATES = "updates"
_INDEX = "model.safetensors.index.json"


class S3Store(Store):
    """Store one run's version chain in S3 and materialize it into a local cache."""

    def __init__(
        self,
        root: str,
        *,
        cache_dir: str | Path,
        run_id: str,
        endpoint_url: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        parsed = urlparse(root if "://" in root else f"s3://{root}")
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"invalid S3 root {root!r}; expected s3://<bucket>[/<prefix>]"
            )

        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.cache_dir = Path(cache_dir)
        self.run_id = run_id
        self.endpoint_url = endpoint_url
        self._client = None

    def refresh(self) -> None:
        """Do nothing because S3 reads immediately observe completed writes."""

    def read_pointer(self) -> VersionRef | None:
        text = self._read_text(self._key(_POINTER))
        return VersionRef.parse(text) if text else None

    def advance_pointer(self, ref: VersionRef) -> None:
        self._check_ref(ref)
        self._s3().put_object(
            Bucket=self.bucket,
            Key=self._key(_POINTER),
            Body=ref.identity.encode("utf-8"),
        )

    def claim(self, boot: VersionRef) -> None:
        if not boot.run_id:
            raise ValueError(
                "claim requires a run_id (the run's per-launch epoch token)"
            )
        self.advance_pointer(boot)

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        version_dir = self._version_dir(ref)
        self._download(self._version_key(ref, _INDEX), version_dir / _INDEX)
        return VersionManifest.from_hf_index(version_dir, run_id=ref.run_id)

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        self._check_ref(manifest.ref)
        source = Path(files_dir)
        for path in sorted(path for path in source.rglob("*") if path.is_file()):
            relative_path = path.relative_to(source).as_posix()
            self._s3().upload_file(
                str(path),
                self.bucket,
                self._version_key(manifest.ref, relative_path),
            )

    def materialize(self, ref: VersionRef) -> str:
        self._check_ref(ref)
        self._sync(self._key(_UPDATES), self.cache_dir / _UPDATES)
        return str(self._version_dir(ref))

    def _check_ref(self, ref: VersionRef) -> None:
        if ref.run_id != self.run_id:
            raise ValueError(
                f"store is scoped to run {self.run_id!r}, got {ref.run_id!r}"
            )

    def _version_dir(self, ref: VersionRef) -> Path:
        self._check_ref(ref)
        return self.cache_dir / _UPDATES / Path(ref.identity).name

    def _version_key(self, ref: VersionRef, *parts: str) -> str:
        self._check_ref(ref)
        return self._key(_UPDATES, Path(ref.identity).name, *parts)

    def _key(self, *parts: str) -> str:
        return "/".join(
            part.strip("/") for part in (self.prefix, *parts) if part.strip("/")
        )

    def _s3(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    def _read_text(self, key: str) -> str | None:
        client = self._s3()
        try:
            body = client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except client.exceptions.NoSuchKey:
            return None
        return body.read().decode("utf-8").strip()

    def _download(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._s3().download_file(self.bucket, key, str(destination))

    def _sync(self, key_prefix: str, destination_root: Path) -> None:
        """Mirror objects under ``key_prefix`` while retaining unchanged cache files."""
        client = self._s3()
        object_prefix = f"{key_prefix}/"
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=object_prefix
        )
        for page in pages:
            for obj in page.get("Contents", []):
                relative_key = obj["Key"][len(object_prefix) :]
                if not relative_key:
                    continue
                relative_path = Path(relative_key)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"unsafe S3 object key {obj['Key']!r}")
                destination = destination_root / relative_path
                if destination.exists() and destination.stat().st_size == obj["Size"]:
                    continue
                self._download(obj["Key"], destination)
