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

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from stitch.stores.base import Store
from stitch.types import PointerConflict, VersionManifest, VersionRef

_POINTER = "latest"
_UPDATES = "updates"
_INDEX = "model.safetensors.index.json"
_SHA256_METADATA = "stitch-sha256"


@dataclass(frozen=True)
class UploadedObject:
    """One local file observed after S3 accepted its upload."""

    relative_path: str
    size: int
    checksum_sha256: str
    s3_checksum_sha256: str | None


@dataclass(frozen=True)
class UploadReceipt:
    """The objects uploaded by one trainer host for one immutable version."""

    ref: VersionRef
    objects: tuple[UploadedObject, ...]


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
        ref, _etag = self._read_pointer_with_etag()
        return ref

    def advance_pointer(self, ref: VersionRef) -> None:
        expected, etag = self._read_pointer_with_etag()
        self._put_pointer(ref, expected=expected, etag=etag)

    def compare_and_advance_pointer(
        self, expected: VersionRef | None, ref: VersionRef
    ) -> None:
        actual, etag = self._read_pointer_with_etag()
        if actual != expected:
            raise PointerConflict(expected, actual, ref)
        self._put_pointer(ref, expected=expected, etag=etag)

    def _put_pointer(
        self,
        ref: VersionRef,
        *,
        expected: VersionRef | None,
        etag: str | None,
    ) -> None:
        self._check_ref(ref)
        condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
        try:
            self._s3().put_object(
                Bucket=self.bucket,
                Key=self._key(_POINTER),
                Body=ref.identity.encode("utf-8"),
                **condition,
            )
        except Exception as exc:  # noqa: BLE001 - botocore is an optional dependency
            if not _is_conditional_write_failure(exc):
                raise
            actual = self.read_pointer()
            raise PointerConflict(expected, actual, ref) from exc

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
        """Upload and verify a complete version from one local directory."""
        self._check_ref(manifest.ref)
        receipt = self.upload_version_files(manifest.ref, files_dir)
        verified = self.verify_version(manifest.ref, [receipt])
        if verified != manifest:
            raise ValueError(
                f"verified manifest for {manifest.ref.identity} differs from source"
            )

    def upload_version_files(
        self, ref: VersionRef, files_dir: str | Path
    ) -> UploadReceipt:
        """Upload this host's local files directly to ``ref``'s final prefix.

        The SHA-256 stored with each object identifies the trainer bytes, while
        S3's additional SHA-256 checksum validates the transfer itself. The
        returned receipt is small enough to gather across distributed ranks.
        """
        self._check_ref(ref)
        source = Path(files_dir)
        expected_name = Path(ref.identity).name
        if source.name != expected_name:
            raise ValueError(
                f"expected version directory {expected_name!r}, got {source.name!r}"
            )
        files = sorted(path for path in source.rglob("*") if path.is_file())
        if not files:
            raise FileNotFoundError(f"version directory has no files: {source}")

        uploaded: list[UploadedObject] = []
        for path in files:
            relative_path = path.relative_to(source).as_posix()
            checksum = _sha256(path)
            self._s3().upload_file(
                str(path),
                self.bucket,
                self._version_key(ref, relative_path),
                ExtraArgs={
                    "ChecksumAlgorithm": "SHA256",
                    "Metadata": {_SHA256_METADATA: checksum},
                },
            )
            head = self._head_version_object(ref, relative_path)
            s3_checksum = head.get("ChecksumSHA256")
            uploaded_object = UploadedObject(
                relative_path=relative_path,
                size=path.stat().st_size,
                checksum_sha256=checksum,
                s3_checksum_sha256=(
                    s3_checksum if isinstance(s3_checksum, str) else None
                ),
            )
            self._validate_head(ref, uploaded_object, head)
            uploaded.append(uploaded_object)
        return UploadReceipt(ref=ref, objects=tuple(uploaded))

    def verify_version(
        self, ref: VersionRef, receipts: Iterable[UploadReceipt]
    ) -> VersionManifest:
        """Verify gathered host uploads and return ``ref``'s S3-backed manifest."""
        self._check_ref(ref)
        expected: dict[str, UploadedObject] = {}
        for receipt in receipts:
            if receipt.ref != ref:
                raise ValueError(
                    f"receipt is for {receipt.ref.identity}, expected {ref.identity}"
                )
            for uploaded in receipt.objects:
                _validate_relative_path(uploaded.relative_path)
                previous = expected.get(uploaded.relative_path)
                if previous is not None and (
                    previous.size != uploaded.size
                    or previous.checksum_sha256 != uploaded.checksum_sha256
                    or (
                        previous.s3_checksum_sha256 is not None
                        and uploaded.s3_checksum_sha256 is not None
                        and previous.s3_checksum_sha256 != uploaded.s3_checksum_sha256
                    )
                ):
                    raise ValueError(
                        f"conflicting upload receipts for {uploaded.relative_path!r}"
                    )
                expected[uploaded.relative_path] = uploaded

        if _INDEX not in expected:
            raise FileNotFoundError(
                f"incomplete S3 version {ref.identity}: missing {_INDEX} upload receipt"
            )
        for relative_path, uploaded in sorted(expected.items()):
            head = self._head_version_object(ref, relative_path)
            self._validate_head(ref, uploaded, head)

        manifest = self.read_manifest(ref)
        if manifest.ref != ref:
            raise ValueError(
                f"S3 index identifies {manifest.ref.identity}, expected {ref.identity}"
            )
        missing = sorted(set(manifest.files) - expected.keys())
        if missing:
            raise FileNotFoundError(
                f"incomplete S3 version {ref.identity}: missing verified uploads for "
                + ", ".join(missing)
            )
        return manifest

    def _head_version_object(
        self, ref: VersionRef, relative_path: str
    ) -> dict[str, object]:
        client = self._s3()
        try:
            return client.head_object(
                Bucket=self.bucket,
                Key=self._version_key(ref, relative_path),
                ChecksumMode="ENABLED",
            )
        except client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(
                f"missing S3 object for {ref.identity}: {relative_path}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - botocore is an optional dependency
            if not _is_missing_object(exc):
                raise
            raise FileNotFoundError(
                f"missing S3 object for {ref.identity}: {relative_path}"
            ) from exc

    def _validate_head(
        self,
        ref: VersionRef,
        uploaded: UploadedObject,
        head: dict[str, object],
    ) -> None:
        path = uploaded.relative_path
        if int(head.get("ContentLength", -1)) != uploaded.size:
            raise ValueError(f"S3 size mismatch for {ref.identity}/{path}")
        metadata = head.get("Metadata") or {}
        if not isinstance(metadata, dict) or metadata.get(_SHA256_METADATA) != (
            uploaded.checksum_sha256
        ):
            raise ValueError(f"S3 SHA-256 checksum mismatch for {ref.identity}/{path}")
        actual_s3_checksum = head.get("ChecksumSHA256")
        if (
            uploaded.s3_checksum_sha256 is not None
            and actual_s3_checksum != uploaded.s3_checksum_sha256
        ):
            raise ValueError(f"S3 transfer checksum mismatch for {ref.identity}/{path}")

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

    def _read_pointer_with_etag(self) -> tuple[VersionRef | None, str | None]:
        client = self._s3()
        try:
            response = client.get_object(Bucket=self.bucket, Key=self._key(_POINTER))
        except client.exceptions.NoSuchKey:
            return None, None
        text = response["Body"].read().decode("utf-8").strip()
        ref = VersionRef.parse(text) if text else None
        etag = str(response.get("ETag") or "") or None
        return ref, etag

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(path: str) -> None:
    candidate = Path(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe S3 object path {path!r}")


def _error_details(exc: Exception) -> tuple[str | None, int | None]:
    response = getattr(exc, "response", {})
    if not isinstance(response, dict):
        return None, None
    error = response.get("Error") or {}
    metadata = response.get("ResponseMetadata") or {}
    code = error.get("Code") if isinstance(error, dict) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return str(code) if code is not None else None, int(status) if status else None


def _is_missing_object(exc: Exception) -> bool:
    code, status = _error_details(exc)
    return code in {"NoSuchKey", "NotFound", "404"} or status == 404


def _is_conditional_write_failure(exc: Exception) -> bool:
    code, status = _error_details(exc)
    return code in {
        "ConditionalRequestConflict",
        "PreconditionFailed",
        "409",
        "412",
    } or status in {409, 412}
