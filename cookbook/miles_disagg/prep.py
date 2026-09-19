"""Prepare pinned BF16 masters, BF16 or NVFP4 served weights, and TorchDist references."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.trainer_image import (
    MEGATRON_PATH,
    MILES_ROOT,
    TORCH_DIST_CONVERT_WRAPPER,
)

_COMPLETION_FILE = ".stitch-prep.json"


def pinned_miles_revision(exp) -> str:
    """Preparation receipts require an immutable converter, including on reuse."""
    if os.environ.get("MILES_LOCAL_DIR"):
        raise ValueError(
            "checkpoint preparation does not support MILES_LOCAL_DIR; "
            "pin the tested Miles changes with a full MILES_REPO_REF commit hash"
        )
    revision = getattr(exp, "MILES_REPO_REF", trainer_image.MILES_REPO_REF)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(
            "checkpoint preparation requires a full MILES_REPO_REF commit hash"
        )
    return revision


def _preparation_identity(exp, output_format: str) -> dict[str, Any]:
    miles_revision = pinned_miles_revision(exp)
    revision = getattr(exp, "SOURCE_REVISION", None)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("checkpoint preparation requires a pinned SOURCE_REVISION")
    identity = {
        "schema": 1,
        "source_model": exp.SOURCE_MODEL,
        "source_revision": revision,
        "output_format": output_format,
        "unpack_fused_experts": getattr(exp, "UNPACK_FUSED_EXPERTS", False),
        "bf16_carveouts": {
            name: getattr(exp.miles, name, None)
            for name in (
                "num_layers_at_start_in_bf16",
                "num_layers_at_end_in_bf16",
                "extra_high_precision_layers_hf",
            )
        },
        "converter": {
            "image": trainer_image.MILES_IMAGE_TAG,
            "miles_revision": miles_revision,
            "extra_packages": list(getattr(exp, "TRAINER_EXTRA_PIP_PACKAGES", ())),
            "image_commands": list(getattr(exp, "TRAINER_IMAGE_RUN_COMMANDS", ())),
            "environment": dict(getattr(exp, "PREP_ENV", {})),
        },
    }
    if output_format == "torch_dist":
        identity["topology"] = {
            "model_type": exp.miles.megatron_model_type,
            "nodes": exp.modal.torch_dist_prep_nodes,
            "gpus_per_node": exp.modal.torch_dist_prep_gpus_per_node,
            "arguments": shlex.split(exp.modal.torch_dist_convert_extra_args),
            "modal_wrapper": getattr(exp, "USE_MODAL_TORCH_DIST_WRAPPER", False),
        }
    # Compare tuples and lists in the representation persisted by the receipt.
    return json.loads(json.dumps(identity))


def _artifact_file(directory: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"checkpoint filename escapes {directory}: {filename!r}")
    return directory / relative


def _file_sizes(directory: Path, filenames) -> dict[str, int]:
    sizes = {}
    for name in sorted(filenames):
        path = _artifact_file(directory, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"checkpoint file is missing or empty: {path}")
        sizes[name] = path.stat().st_size
    return sizes


def _hf_checkpoint_files(directory: str | Path) -> dict[str, int]:
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError(f"invalid checkpoint config: {root / 'config.json'}")
    filenames = {"config.json"}
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint index has no weight map: {index_path}")
        shards = set(weight_map.values())
        filenames.add(index_path.name)
    else:
        shards = {path.name for path in root.glob("*.safetensors")}
    if not shards or any(not isinstance(name, str) or not name for name in shards):
        raise ValueError(f"checkpoint has no safetensors shards: {root}")
    return _file_sizes(root, filenames | shards)


def _existing_artifact_error(directory: Path, detail: str) -> RuntimeError:
    return RuntimeError(
        f"cannot reuse checkpoint {directory}: {detail}; "
        "choose a new checkpoint path or inspect and remove this artifact before retrying"
    )


def _require_completion(
    directory: Path, identity: dict[str, Any], *, name: str = _COMPLETION_FILE
) -> dict[str, int]:
    try:
        manifest = json.loads((directory / name).read_text())
    except (OSError, ValueError) as exc:
        raise _existing_artifact_error(directory, f"missing or invalid {name}") from exc
    if not isinstance(manifest, dict) or manifest.get("identity") != identity:
        raise _existing_artifact_error(directory, f"identity differs in {name}")
    files = manifest.get("files")
    if not isinstance(files, dict) or _file_sizes(directory, files) != files:
        raise _existing_artifact_error(directory, f"file inventory differs in {name}")
    return files


def _write_completion(
    directory: Path,
    identity: dict[str, Any],
    files: dict[str, int],
    *,
    name: str = _COMPLETION_FILE,
) -> None:
    (directory / name).write_text(
        json.dumps({"identity": identity, "files": files}, sort_keys=True) + "\n"
    )


def _has_content(directory: Path) -> bool:
    return directory.exists() and (
        not directory.is_dir() or next(directory.iterdir(), None) is not None
    )


def _torch_dist_reusable(directory: Path, identity: dict[str, Any]) -> bool:
    """Each node's receipt certifies that its checkpoint files were committed."""
    if not _has_content(directory):
        return False
    for rank in range(identity["topology"]["nodes"]):
        _require_completion(directory, identity, name=f".stitch-prep-node-{rank}.json")
    tracker = (directory / "latest_checkpointed_iteration.txt").read_text().strip()
    if tracker == "release":
        checkpoint = directory / "release"
    elif tracker.isdigit():
        checkpoint = directory / f"iter_{int(tracker):07d}"
    else:
        raise _existing_artifact_error(directory, f"invalid tracker {tracker!r}")
    shards = list(checkpoint.glob("*.distcp"))
    if not shards:
        raise _existing_artifact_error(directory, f"no distcp shards in {checkpoint}")
    _file_sizes(checkpoint, {".metadata", "common.pt", *(p.name for p in shards)})
    return True


def apply_prep_environment(exp) -> None:
    """Apply download toggles and the experiment's checkpoint quantizer contract."""
    if getattr(exp, "DISABLE_HF_XET", False):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    if getattr(exp, "DISABLE_HF_TRANSFER", False):
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    # The served base and live trainer export must use an identical quantizer contract.
    os.environ.update(getattr(exp, "PREP_ENV", {}))


def _bf16_masters(exp, source_snapshot: str) -> str:
    """Resolve the immutable BF16 source used by both checkpoint converters."""
    if getattr(exp, "MATERIALIZE_BF16_MASTERS", True):
        return str(exp.BF16_CHECKPOINT_PATH)
    return source_snapshot


def prepare_checkpoints(
    exp,
    checkpoint_volume,
    *,
    source_snapshot: str,
) -> None:
    """Prepare BF16 masters and the BF16 or NVFP4 serving checkpoint."""
    checkpoint_volume.reload()
    materialized_bf16_dir = str(exp.BF16_CHECKPOINT_PATH)
    served_dir = str(exp.miles.hf_checkpoint)
    served_format = getattr(exp, "SERVED_CHECKPOINT_FORMAT", "nvfp4")
    if served_format not in {"bf16", "nvfp4"}:
        raise SystemExit(f"unsupported SERVED_CHECKPOINT_FORMAT={served_format!r}")
    tools = f"{MILES_ROOT}/tools"

    src = source_snapshot
    _hf_checkpoint_files(src)
    if _quantization_config(src):
        raise ValueError(
            f"source checkpoint {src} is quantized; an explicit BF16 conversion is required"
        )

    def _build_bf16(out: str) -> None:
        _copy_tree("bf16 masters", src, out)
        if getattr(exp, "UNPACK_FUSED_EXPERTS", False):
            from cookbook.miles_disagg.unpack_experts import unpack_fused_experts

            unpack_fused_experts(out)

    if getattr(exp, "MATERIALIZE_BF16_MASTERS", True):
        _staged(
            materialized_bf16_dir,
            _build_bf16,
            identity=_preparation_identity(exp, "bf16"),
        )
        bf16_dir = materialized_bf16_dir
    else:
        bf16_dir = src
        print(f"using pinned HF snapshot as bf16 masters: {bf16_dir}", flush=True)

    if served_format == "bf16":
        if served_dir != bf16_dir:
            raise ValueError(
                "BF16 served checkpoint must match BF16_CHECKPOINT_PATH: "
                f"{served_dir} != {bf16_dir}"
            )
        checkpoint_volume.commit()
        print(f"Prepared masters={bf16_dir} served_base={bf16_dir}")
        return

    # nvfp4: miles' TE-direct quantizer. bf16 carve-outs must match the trainer's
    # --num-layers-at-start/end-in-bf16 so the served base == the export layout.
    carveouts: list[str] = []
    if (n := getattr(exp.miles, "num_layers_at_start_in_bf16", None)) is not None:
        carveouts += ["--num-layers-at-start-in-bf16", str(n)]
    if (n := getattr(exp.miles, "num_layers_at_end_in_bf16", None)) is not None:
        carveouts += ["--num-layers-at-end-in-bf16", str(n)]
    if layers := getattr(exp.miles, "extra_high_precision_layers_hf", None):
        carveouts += ["--extra-high-precision-layers-hf", *layers]

    def _build_nvfp4(out: str) -> None:
        print(
            "building nvfp4 served base from bf16 masters (GPU conversion)...",
            flush=True,
        )
        subprocess.run(
            [
                "python",
                f"{tools}/convert_hf_to_nvfp4.py",
                "--model-dir",
                bf16_dir,
                "--save-dir",
                out,
                *carveouts,
            ],
            check=True,
            env={**os.environ, **getattr(exp, "PREP_ENV", {})},
        )

    _staged(served_dir, _build_nvfp4, identity=_preparation_identity(exp, "nvfp4"))
    checkpoint_volume.commit()
    print(f"Prepared masters={bf16_dir} served_base={served_dir}")


def prepare_torch_dist(
    exp,
    checkpoint_volume,
    *,
    rank: int,
    master_addr: str,
    source_snapshot: str,
) -> None:
    """Build the raw-mode torch_dist reference checkpoint from the BF16 source."""
    torch_dist_path = getattr(exp, "TORCH_DIST_CHECKPOINT_PATH", None)
    if torch_dist_path is None:
        raise SystemExit("this config does not use a torch_dist trainer checkpoint")
    checkpoint_volume.reload()
    bf16_dir = _bf16_masters(exp, source_snapshot)
    _hf_checkpoint_files(bf16_dir)
    if bf16_dir != source_snapshot:
        _require_completion(Path(bf16_dir), _preparation_identity(exp, "bf16"))
    if _quantization_config(bf16_dir):
        raise ValueError(f"TorchDist conversion requires BF16 masters: {bf16_dir}")
    torch_dist_dir = str(torch_dist_path)
    identity = _preparation_identity(exp, "torch_dist")
    if _torch_dist_reusable(Path(torch_dist_dir), identity):
        print(f"reusing existing torch_dist {torch_dist_dir}")
        return
    if not exp.miles.megatron_model_type:
        raise SystemExit("prepare_torch_dist requires megatron_model_type")
    from miles.utils.external_utils.model_args_utils import load_model_args

    model_args = shlex.split(load_model_args(exp.miles.megatron_model_type))
    nodes = exp.modal.torch_dist_prep_nodes
    use_wrapper = nodes > 1 and getattr(exp, "USE_MODAL_TORCH_DIST_WRAPPER", False)
    convert = (
        TORCH_DIST_CONVERT_WRAPPER
        if use_wrapper
        else f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
    )
    command = [
        "torchrun",
        "--nnodes",
        str(nodes),
        "--node-rank",
        str(rank),
        "--master-addr",
        master_addr,
        "--master-port",
        "29500",
        "--nproc-per-node",
        str(exp.modal.torch_dist_prep_gpus_per_node),
        convert,
        *model_args,
        "--hf-checkpoint",
        bf16_dir,
        "--save",
        torch_dist_dir,
        "--megatron-to-hf-mode",
        "raw",
        *shlex.split(exp.modal.torch_dist_convert_extra_args),
    ]
    env = {**os.environ, **getattr(exp, "PREP_ENV", {})}
    if use_wrapper:
        env["SKIP_RELEASE_RENAME"] = "1"
    print(
        f"converting bf16 masters -> torch_dist ref_load ({nodes}-node torchrun, rank {rank})...",
        flush=True,
    )
    env["PYTHONPATH"] = f"{MEGATRON_PATH}:{env.get('PYTHONPATH', '')}"
    subprocess.run(command, check=True, env=env)
    # Every node commits its own distcp shards (disjoint files merge on the Volume);
    # a rank-0-only commit would drop the other nodes' shards.
    directory = Path(torch_dist_dir)
    files = _file_sizes(
        directory,
        (
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and not path.name.startswith(".stitch-prep-")
        ),
    )
    checkpoint_volume.commit()
    _write_completion(directory, identity, files, name=f".stitch-prep-node-{rank}.json")
    checkpoint_volume.commit()
    print(f"Prepared torch_dist={torch_dist_dir} node={rank}")


_COPY_WORKERS = int(os.environ.get("PREP_COPY_WORKERS", "8"))
_COPY_CHUNK = 16 << 20
_COPY_LOG_STEP_GB = 50


def _copy_tree(label: str, src: str, dst: str) -> None:
    """Copy cached HF blobs into independent files with bounded parallel reads."""
    src_files = [
        p for p in Path(src).rglob("*") if p.is_file()
    ]  # is_file() follows symlinks
    total_gb = sum(p.stat().st_size for p in src_files) / 1e9
    progress = {"done_gb": 0.0, "next_log_gb": _COPY_LOG_STEP_GB}
    lock = threading.Lock()
    start = time.monotonic()

    def copy_one(p: Path) -> None:
        out = Path(dst) / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(p, "rb") as fsrc,
            open(out, "wb") as fdst,
        ):  # open() follows the symlink to the real blob
            while chunk := fsrc.read(_COPY_CHUNK):
                fdst.write(chunk)
        with lock:
            progress["done_gb"] += p.stat().st_size / 1e9
            done = progress["done_gb"]
            if done >= progress["next_log_gb"] or done >= total_gb:
                rate = done / max(time.monotonic() - start, 1e-3)
                print(
                    f"copying {label}: {done:.0f}/{total_gb:.0f} GB ({100 * done / max(total_gb, 1e-9):.0f}%), {rate:.1f} GB/s",
                    flush=True,
                )
                progress["next_log_gb"] += _COPY_LOG_STEP_GB

    os.makedirs(dst, exist_ok=True)
    with ThreadPoolExecutor(
        max_workers=min(_COPY_WORKERS, len(src_files) or 1)
    ) as pool:
        list(pool.map(copy_one, src_files))
    print(f"copied {label}: {total_gb:.0f} GB", flush=True)


def _staged(final_dir: str, build, *, identity: dict[str, Any]) -> None:
    """Publish a validated HF artifact; existing bytes require matching provenance."""
    final = Path(final_dir)
    if _has_content(final):
        recorded = _require_completion(final, identity)
        if _hf_checkpoint_files(final) != recorded:
            raise _existing_artifact_error(final, "HF shard inventory changed")
        print(f"reusing existing {final_dir}")
        return
    partial = Path(f"{final_dir}.partial")
    if _has_content(partial):
        raise _existing_artifact_error(partial, "unfinished preparation")
    partial.mkdir(parents=True, exist_ok=True)
    build(str(partial))
    files = _hf_checkpoint_files(partial)
    _write_completion(partial, identity, files)
    os.rename(partial, final)


def _quantization_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        cfg = json.load(f) or {}
    return (
        (cfg.get("text_config") or {}).get("quantization_config")
        or cfg.get("quantization_config")
        or {}
    )
