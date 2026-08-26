"""Checkpoint materialization: target validity, version-dir preparation, index generation.

The pure dir-layout/copy/index unit behind the publisher app
(``cookbook.inference_only.publish_app``) and its spawned jobs.
"""

import importlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPERIMENT = os.environ["EXPERIMENT_CONFIG"]
exp = importlib.import_module(f"cookbook.inference_only.configs.{EXPERIMENT}")


def _updates_dir(run_dir: Path) -> Path:
    return run_dir / "updates"


def _version_dir(run_dir: Path, version: int) -> Path:
    return _updates_dir(run_dir) / f"weight_v{version:06d}"


def _delta_index_metadata(version_dir: Path) -> dict[str, Any] | None:
    """Index metadata when ``version_dir`` is a delta (``delta_encoding`` set), else None."""
    index_path = version_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None
    meta = index.get("metadata") or {}
    return meta if meta.get("delta_encoding") else None


def _target_is_valid(version_dir: Path, source_dir: Path) -> bool:
    """True when ``version_dir`` is a complete materialization of ``source_dir``.

    Deltas are checked against their own weight_map: a full source's shard
    superset would condemn every staged delta.
    """
    index_path = version_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    if _delta_index_metadata(version_dir) is not None:
        index = json.loads(index_path.read_text())
        delta_shards = {str(f) for f in (index.get("weight_map") or {}).values()}
        if not delta_shards:
            return False
        target_shards = {p.name for p in version_dir.glob("*.safetensors")}
        return delta_shards <= target_shards
    source_shards = {p.name: p for p in source_dir.glob("*.safetensors")}
    target_shards = {p.name: p for p in version_dir.glob("*.safetensors")}
    if not source_shards.keys() <= target_shards.keys():
        return False
    # Names alone are not enough: a container recycled mid-copy leaves a
    # TRUNCATED shard with the right name. Compare sizes against the source.
    return all(
        target_shards[name].stat().st_size == shard.stat().st_size
        for name, shard in source_shards.items()
    )


def _prepare_version_dir(version_dir: Path, source_dir: Path) -> str:
    """Materialize ``source_dir`` into ``version_dir``. A partial dir is removed,
    not resumed (copytree resume is not safe). Returns ``kept`` or ``copied``.
    """
    if version_dir.exists():
        if _target_is_valid(version_dir, source_dir):
            logger.info("version dir %s already fully materialized; keeping", version_dir)
            return "kept"
        logger.warning(
            "removing invalid partial version dir %s (missing index or shards)",
            version_dir,
        )
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True)
    for item in source_dir.iterdir():
        dest = version_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return "copied"


def _ensure_safetensors_index(model_dir: Path, version: int) -> None:
    """Ensure model.safetensors.index.json exists with metadata.version=version.

    If the directory has safetensors files but no index, generate one from tensor names.
    Small single-file models won't have an index; create a minimal one. Any failure
    raises — a missing/wrong index makes stitch.publish derive a wrong manifest, so
    the request must fail loudly and never publish unversioned weights.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text())
        data.setdefault("metadata", {})["version"] = version
        index_path.write_text(json.dumps(data, indent=2))
        return

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"no safetensors files found in {model_dir}; cannot generate index"
        )

    from safetensors import safe_open

    main_file = next(
        (f for f in safetensors_files if f.name == "model.safetensors"),
        safetensors_files[0],
    )
    # framework="numpy": only the header (tensor names) is read, and the publisher
    # image ships numpy but not torch.
    with safe_open(str(main_file), framework="numpy") as f:
        keys = list(f.keys())

    index = {
        "metadata": {"version": version},
        "weight_map": {key: main_file.name for key in keys},
    }
    index_path.write_text(json.dumps(index, indent=2))
    logger.info("generated safetensors index for %s (version=%d)", model_dir, version)


def _touched_delta_tensors(run_dir: Path, through_version: int) -> set[str]:
    """Union of tensor names changed by staged delta versions 1..through_version.

    A chained delta (base_version > 0) is generated against the BASE checkpoint's
    bytes, which equal version-N bytes exactly for tensors no earlier delta
    touched — so selection must exclude everything any earlier delta changed.
    """
    touched: set[str] = set()
    for version in range(1, through_version + 1):
        index_path = _version_dir(run_dir, version) / "model.safetensors.index.json"
        if _delta_index_metadata(index_path.parent) is None:
            continue
        index = json.loads(index_path.read_text())
        touched.update((index.get("weight_map") or {}).keys())
    return touched



def _delta_anchor_dir(run_dir: Path, base_version: int) -> Path:
    """Raw-weight anchor for a delta on base_version.

    v0 anchors at the config's base checkpoint. A staged FULL dir anchors at
    itself. A staged DELTA dir holds XOR-encoded blobs, not raw weights, so the
    anchor falls back to the base checkpoint and tensor selection excludes
    everything earlier deltas touched (see _touched_delta_tensors).
    """
    if base_version > 0:
        staged = _version_dir(run_dir, base_version)
        if _delta_index_metadata(staged) is None:
            return staged
    anchor = getattr(exp, "ROLLOUT_CHECKPOINT_PATH", None)
    if anchor is None:
        raise ValueError(
            f"config {EXPERIMENT!r} has no ROLLOUT_CHECKPOINT_PATH; cannot anchor "
            f"a delta on base_version={base_version}"
        )
    return Path(anchor)
