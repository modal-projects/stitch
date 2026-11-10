"""The checkpoints an e2e config needs, made from a Hugging Face release with Miles' own tools.

A rollout checkpoint holds what the trainer exports, under its names and in its dtypes. In BF16
that is the release's bytes, with each fused expert tensor (all experts of a projection in one
3-D tensor) split into the per-expert tensors Miles exports, and the tensors the trainer keeps in
FP32 upcast; its fresh-launch check proves the result is its own first export. In FP8 or NVFP4 it
is Miles' converter run on the BF16 one, on the GPU type that serves it and in the config's
environment, as the trainer quantizes. The trainer loads Miles' torch_dist conversion of the
release. A cut keeps some decoder layers of a BF16 checkpoint, renumbered, for a model too large
to test whole.

    modal run tests/e2e/checkpoints.py::torch_dist \
        --model-type qwen3.6-35B-A3B \
        --source hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
        --target qwen3-6-35b-a3b_torch_dist
    modal run tests/e2e/checkpoints.py::bf16 \
        --release hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
        --target qwen3-6-35b-a3b-bf16 --fp32 linear_attn.A_log
    modal run tests/e2e/checkpoints.py::quantize \
        --tool fp8 --gpu H200 --source qwen3-6-35b-a3b-bf16 --target qwen3-6-35b-a3b-fp8-miles \
        --tool-args "--strategy block --block-size 128 128 --extra-high-precision-layers-hf ..."
    modal run tests/e2e/checkpoints.py::cut \
        --source deepseek-v4-1-flash-1017-bf16 --target deepseek-v4-1-flash-bf16-7layer \
        --layer-prefix layers. --layers 0,1,2,3,20,21,24 --drop mtp. --config '{...}' \
        --fp32 engram.q_weight,engram.k_weight,compressor.norm.weight,indexer.k_norm.weight
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal
from configs import BF16_TRAINER_NVFP4_ENV, DSV41_NVFP4_ENV, NVFP4_ENV
from harness import (
    CHECKPOINTS,
    MEGATRON_PATH,
    MILES,
    checkpoints,
    image,
    install_overlays,
)

HF = "/hf"
INDEX = "model.safetensors.index.json"
SHARD_BYTES = 5 << 30
FUSED_EXPERTS = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$")

app = modal.App(
    "weight-delta-checkpoints", image=image.add_local_python_source("harness")
)
hf_cache = modal.Volume.from_name("huggingface-cache")
VOLUMES = {CHECKPOINTS: checkpoints, HF: hf_cache}


def _path(name: str) -> Path:
    """A release in the Hugging Face cache (hub/...) or a checkpoint on the checkpoints Volume."""
    return Path(HF, name) if name.startswith("hub/") else Path(CHECKPOINTS, name)


@app.function(gpu="H200", cpu=16, memory=524288, timeout=4 * 3600, volumes=VOLUMES)
def torch_dist(model_type: str, source: str, target: str, extra_args: str = "") -> None:
    """Miles' conversion on one GPU, every parallel size 1 (the trainer reshards at load)."""
    install_overlays()
    from miles.utils.external_utils.model_args_utils import shell_safe_model_args

    subprocess.run(
        f"PYTHONPATH={MILES}:{MEGATRON_PATH} torchrun --nproc-per-node 1 "
        f"tools/convert_hf_to_torch_dist.py {shell_safe_model_args(model_type)} "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        f"--hf-checkpoint {_path(source)} --save {_path(target)} {extra_args}",
        shell=True,
        cwd=MILES,
        check=True,
    )
    checkpoints.commit()


def _as_exported(name: str, tensor, fp32: tuple[str, ...]):
    """The trainer's tensors for one release tensor: fused experts one tensor per expert, and the
    tensors it keeps in FP32 upcast (exact from BF16)."""
    if FUSED_EXPERTS.search(name) is None:
        return [(name, tensor.float() if name.endswith(fp32) else tensor)]
    module = name.rsplit(".", 1)[0]
    if name.endswith("down_proj"):
        return [(f"{module}.{i}.down_proj.weight", w) for i, w in enumerate(tensor)]
    gate, up = tensor.chunk(2, dim=-2)
    return [
        (f"{module}.{i}.{projection}.weight", w)
        for i in range(tensor.shape[0])
        for projection, w in (("gate_proj", gate[i]), ("up_proj", up[i]))
    ]


def _save_sharded(target_dir: Path, tensors) -> dict:
    """Writes (name, tensor) pairs as SHARD_BYTES safetensors shards plus their index."""
    from safetensors.torch import save_file

    weight_map, shard, shard_bytes, total_bytes = {}, {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes
        name = f"model-{len(set(weight_map.values())) + 1:05d}.safetensors"
        save_file(shard, target_dir / name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, name))
        shard, shard_bytes = {}, 0

    for name, tensor in tensors:
        shard[name] = tensor.contiguous()
        shard_bytes += tensor.nbytes
        total_bytes += tensor.nbytes
        if shard_bytes >= SHARD_BYTES:
            flush()
    if shard:
        flush()
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (target_dir / INDEX).write_text(json.dumps(index, indent=2))
    return {"tensors": len(weight_map), "bytes": total_bytes}


def _copy_other_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True)
    for path in source_dir.iterdir():
        if path.suffix != ".safetensors" and path.name != INDEX:
            shutil.copy(path, target_dir / path.name)


def _release_tensors(source_dir: Path, rename=lambda name: name):
    """Each tensor under the name `rename` gives it; None skips it unread."""
    from safetensors import safe_open

    for path in sorted(source_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as reader:
            for name in reader.keys():
                if (new_name := rename(name)) is not None:
                    yield new_name, reader.get_tensor(name)


@app.function(cpu=16, memory=262144, timeout=4 * 3600, volumes=VOLUMES)
def bf16(release: str, target: str, fp32: str = "") -> dict:
    """`fp32`: comma-separated name suffixes of the tensors the trainer keeps in FP32 (Miles marks
    them on the model definition)."""
    release_dir, target_dir = _path(release), _path(target)
    _copy_other_files(release_dir, target_dir)
    suffixes = tuple(filter(None, fp32.split(",")))
    stats = _save_sharded(
        target_dir,
        (
            pair
            for name, tensor in _release_tensors(release_dir)
            for pair in _as_exported(name, tensor, suffixes)
        ),
    )
    checkpoints.commit()
    return stats


@app.function(cpu=16, memory=262144, timeout=4 * 3600, volumes=VOLUMES)
def cut(
    source: str,
    target: str,
    layer_prefix: str,
    layers: str,
    drop: str = "",
    config: str = "{}",
    fp32: str = "",
) -> dict:
    """Keeps the decoder layers `layers` lists (comma-separated, renumbered from 0 in that order) and
    every other tensor not under a `drop` prefix, `fp32` upcast as `bf16` does; `config` is JSON of
    config.json fields to set (null removes one), a dotted key for a nested one."""
    source_dir, target_dir = _path(source), _path(target)
    _copy_other_files(source_dir, target_dir)
    new_index = {old: new for new, old in enumerate(map(int, layers.split(",")))}
    dropped = tuple(filter(None, drop.split(",")))
    suffixes = tuple(filter(None, fp32.split(",")))
    layer = re.compile(rf"^{re.escape(layer_prefix)}(\d+)\.")

    def rename(name: str) -> str | None:
        match = layer.match(name)
        if match is None:
            return None if name.startswith(dropped) else name
        if int(match.group(1)) not in new_index:
            return None
        return f"{layer_prefix}{new_index[int(match.group(1))]}.{name[match.end() :]}"

    stats = _save_sharded(
        target_dir,
        (
            pair
            for name, tensor in _release_tensors(source_dir, rename)
            for pair in _as_exported(name, tensor, suffixes)
        ),
    )
    config_path = target_dir / "config.json"
    values = json.loads(config_path.read_text())
    for key, value in json.loads(config).items():
        *parents, leaf = key.split(".")
        node = values
        for parent in parents:
            node = node[parent]
        if value is None:
            del node[leaf]
        else:
            node[leaf] = value
    config_path.write_text(json.dumps(values, indent=2))
    checkpoints.commit()
    return stats


@app.function(cpu=16, memory=262144, timeout=4 * 3600, volumes=VOLUMES)
def convert(
    tool: str, source: str, target: str, tool_args: str, env: dict[str, str]
) -> None:
    install_overlays()
    subprocess.run(
        f"PYTHONPATH={MILES} python3 tools/convert_hf_to_{tool}.py "
        f"--model-dir {_path(source)} --save-dir {_path(target)} {tool_args}",
        shell=True,
        cwd=MILES,
        env={**os.environ, **env},
        check=True,
    )
    checkpoints.commit()


@app.local_entrypoint()
def quantize(
    tool: str, gpu: str, source: str, target: str, tool_args: str = "", env: str = ""
):
    """Miles' tools/convert_hf_to_<tool>.py on the GPU type that serves the result, in the
    environment that sets the trainer's quantizer too (block-FP8 scale format; `--env nvfp4`,
    `--env dsv41-nvfp4` or `--env glm53-flash-nvfp4` for an NVFP4 recipe)."""
    envs = {
        "": {},
        "nvfp4": NVFP4_ENV,
        "dsv41-nvfp4": DSV41_NVFP4_ENV,
        "glm53-flash-nvfp4": BF16_TRAINER_NVFP4_ENV,
    }
    convert.with_options(gpu=gpu).remote(tool, source, target, tool_args, envs[env])
