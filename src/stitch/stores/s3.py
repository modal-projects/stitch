"""``S3Store`` — the ``Store`` instance backed by an S3 (or S3-compatible) bucket.

Each run's chain lives under ``<prefix>/<run_id>/weight_vNNNNNN/`` as HF-safetensors +
delta metadata, and a ``<prefix>/latest`` object holds the self-identifying pointer
identity. S3 is strongly read-after-write consistent, so ``refresh`` is a no-op and a
single ``latest`` PUT is the durable, immediately-visible pointer move (no rename). Unlike
a mounted Volume there is no shared filesystem, so ``materialize`` downloads a version's
chain into ``cache_dir`` — the local path the engine's pull then reads.

``boto3`` is imported lazily (only when the store is used), so the module loads without it;
credentials/region come from the standard AWS resolution chain, and ``endpoint_url`` targets
S3-compatible stores (MinIO, R2, ...).
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from stitch.stores.base import Store
from stitch.types import VersionManifest, VersionRef

_POINTER = "latest"
_INDEX = "model.safetensors.index.json"


class S3Store(Store):
    def __init__(self, root: str, *, cache_dir: str | Path, endpoint_url: str | None = None) -> None:
        # root is an ``s3://bucket/prefix`` URI (bare ``bucket/prefix`` is accepted too);
        # cache_dir is the local directory materialize downloads into (what the engine reads).
        parsed = urlparse(root if "://" in root else f"s3://{root}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.cache_dir = Path(cache_dir)
        self.endpoint_url = endpoint_url
        self._client = None

    def refresh(self) -> None:
        pass  # S3 is strongly read-after-write consistent; nothing to make visible

    def read_pointer(self) -> VersionRef | None:
        text = self._read_text(self._key(_POINTER))
        return VersionRef.parse(text) if text else None

    def advance_pointer(self, ref: VersionRef) -> None:
        # The caller has already run decide_pointer_move; a single-object PUT is the atomic,
        # immediately-visible durable write. Monotonicity is the single-writer caller's job.
        self._s3().put_object(Bucket=self.bucket, Key=self._key(_POINTER), Body=ref.identity.encode("utf-8"))

    def claim(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("claim requires a run_id (the run's per-launch epoch token)")
        self.advance_pointer(VersionRef(run_id, 0))

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        # from_hf_index reads a local dir, so land just the small index where materialize
        # would put the version and parse it (the heavy shards download in materialize).
        index_dir = self.cache_dir / ref.identity
        self._download(self._key(ref.identity, _INDEX), index_dir / _INDEX)
        return VersionManifest.from_hf_index(index_dir, run_id=ref.run_id)

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        # Upload every file the trainer wrote under files_dir to the version's key prefix.
        # Objects land before the caller moves the pointer; read-after-write makes them visible.
        src = Path(files_dir)
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            key = self._key(manifest.ref.identity, path.relative_to(src).as_posix())
            self._s3().upload_file(str(path), self.bucket, key)

    def materialize(self, ref: VersionRef) -> str:
        # No shared mount: sync the run's published versions (the delta chain back to the
        # nearest anchor) into the local cache so the engine's pull can read them, then return
        # the version dir. Its parent is the run dir the pull walks back over.
        self._sync(self._key(ref.run_id), self.cache_dir / ref.run_id)
        return str(self.cache_dir / ref.identity)

    # ── S3 helpers ───────────────────────────────────────────────────────────────
    def _key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p.strip("/"))

    def _s3(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    def _read_text(self, key: str) -> str | None:
        s3 = self._s3()
        try:
            return s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8").strip()
        except s3.exceptions.NoSuchKey:
            return None

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3().download_file(self.bucket, key, str(dest))

    def _sync(self, key_prefix: str, dest_root: Path) -> None:
        """Download every object under ``key_prefix`` into ``dest_root``, skipping any file
        already present at the same size — so re-materializing a chain only fetches new versions."""
        s3 = self._s3()
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=key_prefix + "/"):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(key_prefix) + 1:]
                if not rel:  # the prefix "directory" placeholder, if any
                    continue
                dest = dest_root / rel
                if dest.exists() and dest.stat().st_size == obj["Size"]:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(self.bucket, obj["Key"], str(dest))


def pull_weights_pre_read_hook(source_dir: str, target_version: int) -> None:
    """Engine-side ``--custom-pull-weights-pre-read-hook``: download the run's published
    versions onto THIS host's ``source_dir`` so the engine's pull can read them — the S3
    analogue of the Modal-Volume reload, for engines that span hosts not sharing the cache.

    ``source_dir`` is the run dir (``<cache>/<run_id>``); its basename is the run. The bucket/
    prefix travel via ``DELTA_S3_URI`` (and optional ``DELTA_S3_ENDPOINT_URL``) on the serving
    container. Guarded on ``target_version > 0`` — version 0 is the engine's own served base."""
    uri = os.environ.get("DELTA_S3_URI", "")
    if not uri or target_version <= 0:
        return
    run_id = Path(source_dir).name
    store = S3Store(uri, cache_dir=Path(source_dir).parent, endpoint_url=os.environ.get("DELTA_S3_ENDPOINT_URL") or None)
    store.materialize(VersionRef(run_id, target_version))
