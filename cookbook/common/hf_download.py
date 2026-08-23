"""Parallel Hugging Face repository downloads onto Modal Volumes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

_MANIFEST_SUFFIX = ".stitch-hf-snapshot.json"
DOWNLOAD_MAX_CONTAINERS = 32


@dataclass(frozen=True)
class CachedRepoFile:
    """One immutable safetensors shard stored in the default HF cache."""

    repo_id: str
    revision: str
    filename: str


@dataclass(frozen=True)
class LocalRepoFile:
    """One immutable safetensors shard stored in an explicit local directory."""

    repo_id: str
    revision: str
    filename: str
    destination_dir: str


@dataclass(frozen=True)
class _SnapshotManifest:
    repo_id: str
    revision: str


class _Volume(Protocol):
    def reload(self) -> None: ...

    def commit(self) -> None: ...


class _DownloadFunction(Protocol):
    def spawn(self, repo_file: Any) -> Any: ...


def download_cached_safetensors_file(
    repo_file: CachedRepoFile, *, commit: Callable[[], None]
) -> str:
    """Download one shard into Hugging Face's default cache and commit it."""

    _require_safetensors(repo_file.filename)
    with _isolated_xet_cache():
        from huggingface_hub import hf_hub_download

        # Deliberately omit cache_dir and local_dir. A Volume mounted at
        # ~/.cache/huggingface therefore gets the standard hub cache layout.
        hf_hub_download(
            repo_id=repo_file.repo_id,
            revision=repo_file.revision,
            filename=repo_file.filename,
        )
    commit()
    print(f"Downloaded {repo_file.filename}", flush=True)
    return repo_file.filename


def download_local_safetensors_file(
    repo_file: LocalRepoFile, *, commit: Callable[[], None]
) -> str:
    """Download one shard into an explicit repository directory and commit it."""

    _require_safetensors(repo_file.filename)
    destination_dir = Path(repo_file.destination_dir)
    scratch_root = destination_dir.with_name(f".{destination_dir.name}.hf-downloads")
    scratch = scratch_root / sha256(repo_file.filename.encode()).hexdigest()
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        with _isolated_xet_cache():
            from huggingface_hub import hf_hub_download

            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_file.repo_id,
                    revision=repo_file.revision,
                    filename=repo_file.filename,
                    cache_dir=scratch / "hub",
                    local_dir=scratch / "repository",
                )
            )
        destination = _repo_path(destination_dir, repo_file.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.partial")
        shutil.copy2(downloaded, partial, follow_symlinks=True)
        os.replace(partial, destination)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    commit()
    print(f"Downloaded {repo_file.filename}", flush=True)
    return repo_file.filename


def download_cached_snapshot(
    download_file: _DownloadFunction,
    repo_id: str,
    revision: str | None,
    *,
    volume: _Volume,
) -> str:
    """Populate the default HF cache, spawning one call per safetensors shard.

    Neither this coordinator nor its shard workers pass ``cache_dir`` or
    ``local_dir``. Mounting a Volume at ``~/.cache/huggingface`` is therefore
    sufficient for ordinary Hugging Face clients to reuse the result.
    """

    resolved_revision, filenames = _repo_manifest(repo_id, revision)
    volume.reload()
    if revision != resolved_revision:
        _cache_requested_revision(repo_id, revision, resolved_revision, filenames)
        volume.commit()
    missing = _missing_cached_files(repo_id, resolved_revision, filenames)

    other_files = tuple(name for name in missing if not name.endswith(".safetensors"))
    if other_files:
        print(
            f"Downloading {len(other_files)} non-safetensors files for "
            f"{repo_id}@{resolved_revision} in one batch",
            flush=True,
        )
        _download_cached_files(repo_id, resolved_revision, other_files)
        volume.commit()

    safetensors = tuple(name for name in missing if name.endswith(".safetensors"))
    if safetensors:
        print(
            f"Spawning {len(safetensors)} safetensors downloads for "
            f"{repo_id}@{resolved_revision}",
            flush=True,
        )
        calls = [
            download_file.spawn(CachedRepoFile(repo_id, resolved_revision, filename))
            for filename in safetensors
        ]
        _verify_results(_gather_calls(calls), safetensors)
        volume.reload()

    snapshot = Path(local_cached_snapshot(repo_id, revision))
    missing = [name for name in filenames if not _repo_path(snapshot, name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Hugging Face download finished with {len(missing)} missing files: {preview}"
        )

    print(
        f"HF_DOWNLOAD_VERDICT status=complete repo_id={repo_id!r} "
        f"revision={resolved_revision} files={len(filenames)} destination={snapshot}",
        flush=True,
    )
    return str(snapshot)


def local_cached_snapshot(repo_id: str, revision: str | None) -> str:
    """Resolve a snapshot using Hugging Face's unconfigured, local-only client."""

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=True,
    )


def download_local_snapshot(
    download_file: _DownloadFunction,
    repo_id: str,
    revision: str | None,
    destination: str | Path,
    *,
    volume: _Volume,
) -> str:
    """Populate an explicit directory, spawning one call per safetensors shard."""

    resolved_revision, filenames = _repo_manifest(repo_id, revision)
    manifest = _SnapshotManifest(repo_id, resolved_revision)
    destination = Path(destination)
    staging = destination.with_name(f"{destination.name}.partial")

    volume.reload()
    if _is_complete(destination, filenames, manifest):
        if not _has_manifest(destination, manifest):
            _write_manifest(destination, manifest)
            volume.commit()
        print(f"Reusing {repo_id}@{resolved_revision} at {destination}", flush=True)
        return str(destination)

    if not _has_manifest(staging, manifest):
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    _write_manifest(staging, manifest)
    missing = [name for name in filenames if not _repo_path(staging, name).is_file()]
    other_files = tuple(name for name in missing if not name.endswith(".safetensors"))
    if other_files:
        print(
            f"Downloading {len(other_files)} non-safetensors files for "
            f"{repo_id}@{resolved_revision} in one batch",
            flush=True,
        )
        _download_local_files(repo_id, resolved_revision, other_files, staging)
    volume.commit()

    safetensors = tuple(
        name
        for name in filenames
        if name.endswith(".safetensors") and not _repo_path(staging, name).is_file()
    )
    if safetensors:
        print(
            f"Spawning {len(safetensors)} safetensors downloads for "
            f"{repo_id}@{resolved_revision}",
            flush=True,
        )
        calls = [
            download_file.spawn(
                LocalRepoFile(repo_id, resolved_revision, filename, str(staging))
            )
            for filename in safetensors
        ]
        _verify_results(_gather_calls(calls), safetensors)
        volume.reload()

    missing = [name for name in filenames if not _repo_path(staging, name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Hugging Face download finished with {len(missing)} missing files: {preview}"
        )

    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)
    _manifest_path(staging).unlink(missing_ok=True)
    _write_manifest(destination, manifest)
    volume.commit()
    print(
        f"HF_DOWNLOAD_VERDICT status=complete repo_id={repo_id!r} "
        f"revision={resolved_revision} files={len(filenames)} destination={destination}",
        flush=True,
    )
    return str(destination)


def _repo_manifest(repo_id: str, revision: str | None) -> tuple[str, tuple[str, ...]]:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id=repo_id, revision=revision, files_metadata=False)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not resolve a commit for {repo_id!r}")
    filenames = tuple(sorted(sibling.rfilename for sibling in info.siblings))
    return info.sha, filenames


def _missing_cached_files(
    repo_id: str, revision: str, filenames: Sequence[str]
) -> tuple[str, ...]:
    from huggingface_hub import try_to_load_from_cache

    missing = []
    for filename in filenames:
        cached = try_to_load_from_cache(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
        )
        if not isinstance(cached, str) or not Path(cached).is_file():
            missing.append(filename)
    return tuple(missing)


def _cache_requested_revision(
    repo_id: str,
    requested_revision: str | None,
    resolved_revision: str,
    filenames: Sequence[str],
) -> None:
    """Create the standard HF ref for a branch or tag without custom metadata."""

    filename = next(
        (name for name in filenames if not name.endswith(".safetensors")),
        filenames[0],
    )
    with _isolated_xet_cache():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo_id,
            revision=requested_revision,
            filename=filename,
        )
    cached_revision = Path(local_cached_snapshot(repo_id, requested_revision)).name
    if cached_revision != resolved_revision:
        raise RuntimeError(
            f"{repo_id}@{requested_revision or 'main'} changed from "
            f"{resolved_revision} to {cached_revision} during download; retry"
        )


def _download_cached_files(
    repo_id: str,
    revision: str,
    filenames: Sequence[str],
) -> None:
    with _isolated_xet_cache():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(filenames),
            max_workers=4,
        )


def _download_local_files(
    repo_id: str,
    revision: str,
    filenames: Sequence[str],
    destination: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="stitch-hf-") as cache_dir:
        with _isolated_xet_cache():
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=Path(cache_dir) / "hub",
                local_dir=destination,
                allow_patterns=list(filenames),
                max_workers=4,
            )
    shutil.rmtree(destination / ".cache", ignore_errors=True)


@contextmanager
def _isolated_xet_cache() -> Iterator[None]:
    """Keep Xet's long-lived log handle off a mounted Modal Volume."""

    from huggingface_hub import constants

    previous_constant = constants.HF_XET_CACHE
    previous_environment = os.environ.get("HF_XET_CACHE")
    with tempfile.TemporaryDirectory(prefix="stitch-hf-xet-") as cache_dir:
        constants.HF_XET_CACHE = cache_dir
        os.environ["HF_XET_CACHE"] = cache_dir
        try:
            yield
        finally:
            constants.HF_XET_CACHE = previous_constant
            if previous_environment is None:
                os.environ.pop("HF_XET_CACHE", None)
            else:
                os.environ["HF_XET_CACHE"] = previous_environment


def _gather_calls(calls: Sequence[Any]) -> Sequence[str]:
    from modal import FunctionCall

    return FunctionCall.gather(*calls)


def _verify_results(results: Sequence[str], expected: tuple[str, ...]) -> None:
    if tuple(results) != expected:
        raise RuntimeError(
            f"safetensors downloads returned unexpected files: {results!r}"
        )


def _require_safetensors(filename: str) -> None:
    if not filename.endswith(".safetensors"):
        raise ValueError(f"expected a safetensors file, got {filename!r}")


def _is_complete(
    destination: Path,
    filenames: tuple[str, ...],
    manifest: _SnapshotManifest,
) -> bool:
    if not destination.is_dir():
        return False
    if not all(_repo_path(destination, name).is_file() for name in filenames):
        return False
    marker = _manifest_path(destination)
    # Adopt complete snapshots created by the former snapshot_download helpers.
    return not marker.exists() or _has_manifest(destination, manifest)


def _has_manifest(directory: Path, expected: _SnapshotManifest) -> bool:
    try:
        values = json.loads(_manifest_path(directory).read_text())
        return values == asdict(expected)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _write_manifest(directory: Path, manifest: _SnapshotManifest) -> None:
    _manifest_path(directory).write_text(
        json.dumps(asdict(manifest), sort_keys=True) + "\n"
    )


def _manifest_path(directory: Path) -> Path:
    return directory.with_name(f".{directory.name}{_MANIFEST_SUFFIX}")


def _repo_path(directory: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"repository filename escapes its destination: {filename!r}")
    return directory / relative
