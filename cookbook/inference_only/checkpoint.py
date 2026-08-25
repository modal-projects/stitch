"""Boot checks shared by prep and the inference Server."""

from pathlib import Path
from typing import Any


def require_checkpoint(model_path: str | Path) -> Path:
    """Return ``model_path`` when it is a complete HF checkpoint.

    A missing directory is a FileNotFoundError so launch can point at prep.
    An existing incomplete directory fails closed instead of becoming a boot
    input.
    """
    import json

    target = Path(model_path)
    if not target.exists():
        raise FileNotFoundError(
            f"missing model checkpoint {target}; "
            "run cookbook.inference_only.prep_app::download_model"
        )

    # Require config.json in all cases
    config_path = target / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"incomplete model checkpoint {target}: missing config.json; "
            "run cookbook.inference_only.prep_app::download_model"
        )

    # If model.safetensors.index.json exists, verify all named shards are present and non-empty
    index_path = target / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(
                f"corrupted index.json in {target}; "
                "run cookbook.inference_only.prep_app::download_model"
            ) from e
        if not isinstance(index, dict):
            raise RuntimeError(
                f"malformed index.json in {target}: top level is not a JSON object; "
                "run cookbook.inference_only.prep_app::download_model"
            )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(
                f"malformed index.json in {target}: missing weight_map; "
                "run cookbook.inference_only.prep_app::download_model"
            )
        shard_files = set(weight_map.values())
        missing = [s for s in shard_files if not (target / s).is_file()]
        if missing:
            raise RuntimeError(
                f"incomplete model checkpoint {target}: missing shards "
                + ", ".join(missing)
                + "; run cookbook.inference_only.prep_app::download_model"
            )
        # Verify shards are non-empty
        empty = [s for s in shard_files if (target / s).stat().st_size == 0]
        if empty:
            raise RuntimeError(
                f"incomplete model checkpoint {target}: empty shards "
                + ", ".join(empty)
                + "; run cookbook.inference_only.prep_app::download_model"
            )
    else:
        # Single-file model: require at least one *.safetensors file
        safetensors_files = list(target.glob("*.safetensors"))
        if not safetensors_files:
            raise RuntimeError(
                f"incomplete model checkpoint {target}: no safetensors files found; "
                "run cookbook.inference_only.prep_app::download_model"
            )
        # Verify safetensors are non-empty
        empty = [f.name for f in safetensors_files if f.stat().st_size == 0]
        if empty:
            raise RuntimeError(
                f"incomplete model checkpoint {target}: empty safetensors "
                + ", ".join(empty)
                + "; run cookbook.inference_only.prep_app::download_model"
            )

    return target


def claim_boot_pointer(store: Any, run_id: str, *, boot_version: int = 0) -> None:
    """Claim the boot pointer iff no pointer exists yet (single-writer, never rewind).

    Only the launcher's one-shot remote claim calls this. An existing pointer —
    the boot claim or any later publish — is left untouched, so a relaunched run
    can never rewind a replica to the boot checkpoint.
    """
    from stitch.publish import claim_run

    store.refresh()
    if store.read_pointer() is not None:
        return
    claim_run(store, None, run_id, boot_version=boot_version)


def ensure_boot_pointer(store: Any, run_id: str, *, boot_version: int = 0, timeout_seconds: int = 60) -> None:
    """Wait for the boot pointer to be claimed by the launcher.

    Replicas only refresh + read and wait briefly, then fail fast if missing.
    This prevents any rewind race since only the launcher writes the pointer.
    """
    import time

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        store.refresh()
        pointer = store.read_pointer()
        if pointer is not None:
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"boot pointer for {run_id} was not claimed by the launcher; "
        "run cookbook.inference_only.launch so its remote claim completes "
        "before starting replicas"
    )
