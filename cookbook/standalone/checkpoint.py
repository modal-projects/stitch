"""Checkpoint and boot-pointer invariants for standalone rollout pools."""

from __future__ import annotations

import json
import time
from pathlib import Path

from stitch.publish import claim_run
from stitch.stores.base import Store
from stitch.types import VersionRef


def require_checkpoint(model_path: str | Path) -> Path:
    """Return a complete Hugging Face safetensors checkpoint or fail closed."""

    target = Path(model_path)
    if not target.exists():
        raise FileNotFoundError(
            f"missing model checkpoint {target}; "
            "run cookbook.standalone.prep_app::download_base"
        )

    if not (target / "config.json").is_file():
        raise RuntimeError(
            f"incomplete model checkpoint {target}: missing config.json; "
            "run cookbook.standalone.prep_app::download_base"
        )

    index_path = target / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"corrupted safetensors index in {target}") from exc
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise RuntimeError(f"malformed safetensors index in {target}")
        weight_files = {
            _checkpoint_file(target, filename)
            for filename in index["weight_map"].values()
        }
    else:
        weight_files = set(target.glob("*.safetensors"))

    if not weight_files:
        raise RuntimeError(
            f"incomplete model checkpoint {target}: no safetensors files"
        )
    missing = sorted(
        str(path.relative_to(target))
        for path in weight_files
        if not path.is_file() or path.stat().st_size == 0
    )
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"incomplete model checkpoint {target}: {len(missing)} missing or empty "
            f"safetensors files ({preview})"
        )
    return target


def claim_boot_pointer(
    store: Store, run_id: str, *, boot_version: int = 0
) -> VersionRef:
    """Claim an absent boot pointer without rewinding an existing run."""

    store.refresh()
    pointer = store.read_pointer()
    if pointer is None:
        claim_run(store, None, run_id, boot_version=boot_version)
        store.refresh()
        pointer = store.read_pointer()
    return _require_run_pointer(pointer, run_id, boot_version)


def wait_for_boot_pointer(
    store: Store,
    run_id: str,
    *,
    boot_version: int = 0,
    timeout_seconds: float = 60,
) -> VersionRef:
    """Wait for the launcher's claim and reject a pointer from another run."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        store.refresh()
        if (pointer := store.read_pointer()) is not None:
            return _require_run_pointer(pointer, run_id, boot_version)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.5, remaining))
    raise RuntimeError(
        f"boot pointer for {run_id!r} was not claimed within {timeout_seconds:g}s"
    )


def _require_run_pointer(
    pointer: VersionRef | None, run_id: str, boot_version: int
) -> VersionRef:
    if pointer is None:
        raise RuntimeError(f"boot pointer for {run_id!r} is missing after claim")
    if pointer.run_id != run_id:
        raise RuntimeError(
            f"checkpoint store for {run_id!r} points at another run: "
            f"{pointer.identity!r}"
        )
    if pointer.version < boot_version:
        raise RuntimeError(
            f"checkpoint store for {run_id!r} points below boot version "
            f"{boot_version}: {pointer.identity!r}"
        )
    return pointer


def _checkpoint_file(target: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise RuntimeError(f"malformed safetensors filename in {target}: {filename!r}")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"safetensors filename escapes checkpoint {target}: {filename!r}"
        )
    return target / relative
