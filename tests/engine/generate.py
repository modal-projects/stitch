"""A delta set for the engine test, written as a trainer publishes it: a simulated trainer steps
every exported tensor of the full model, Miles' quantizers export it in the rollout checkpoint's
format, and Miles' weight-delta encoder writes each version.

Each worker is one GPU and one owner: it takes every Nth group of tensors (an NVFP4 expert's gate
and up share one global scale, so they go together), steps the group through every version, and
adds it to its owner file of each. Version 0 must export to the rollout checkpoint's bytes, the
rule the trainer's fresh-launch check enforces. The last version's tensors, with the rollout
checkpoint's unexported ones, make the expected checkpoint the engine test compares against.

A perf set is what a production trainer writes. A correctness set also carries each changed tensor's
checksum, which SGLang checks as it applies, and the expected checkpoint.

    modal run tests/engine/generate.py \
        --delta-set qwen3.6-35b-a3b-fp8 --mode correctness --versions 10 --workers 4

Writes engine/<delta set>/<run>/ on the gpu-delta-tests Volume: deltas/weight_vNNNNNN/, report.json
and, for correctness, expected/ (a checkpoint).
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))
import harness  # noqa: E402
from engine_configs import DELTA_SETS  # noqa: E402

CODEC = "gap-zstd"
GRAD_SCALE = 1e-7
MODES = ("perf", "correctness")
FP8_BLOCK = [128, 128]
SWIGLU = re.compile(
    r"^(?P<mlp>.+)\.(?P<role>gate_proj|up_proj|down_proj|w1|w3|w2)\.weight$"
)
ROLES = {"gate_proj": 0, "w1": 0, "up_proj": 1, "w3": 1, "down_proj": 2, "w2": 2}
# A SwiGLU neuron whose gate and up rows both lie at the bottom of the float range is dead: its
# output and every gradient through it are zero, so a real trainer leaves its weights as they are.
DEAD_ROW_AMAX = 1e-30
# --move-global-scale: in each layer this expert's largest weight grows every version, so its NVFP4
# global scale (weight_scale_2), and what SGLang derives from it, changes with every version.
MOVED_EXPERT = ".experts.0."
INDEX = "model.safetensors.index.json"
SHARD_BYTES = 5 << 30
MAX_PENDING_NBYTES = 1 << 28

image = harness.image.add_local_python_source(
    "harness", "engine_configs", "simulated_trainer"
)
app = modal.App("weight-delta-engine-generate", image=image)
HF = "/hf"
hf_cache = modal.Volume.from_name("huggingface-cache")
VOLUMES = {
    harness.CHECKPOINTS: harness.checkpoints,
    harness.DATA: harness.data,
    HF: hf_cache,
}


def _checkpoint(name: str) -> Path:
    """A release in the Hugging Face cache (hub/...) or a checkpoint on the checkpoints Volume."""
    return (
        Path(HF, name) if name.startswith("hub/") else Path(harness.CHECKPOINTS, name)
    )


def _run_dir(name: str, run: str) -> Path:
    return Path(harness.DATA, "engine", name, run)


def _headers(directory: Path) -> dict[str, tuple[str, list[int], Path]]:
    from safetensors import safe_open

    headers = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as reader:
            for name in reader.keys():
                view = reader.get_slice(name)
                headers[name] = (view.get_dtype(), view.get_shape(), path)
    return headers


def _format(name: str, rollout: dict) -> str:
    dtype = rollout[name][0]
    if dtype == "F8_E4M3":
        return "fp8"
    if dtype == "U8" and name.removesuffix("weight") + "weight_scale_2" in rollout:
        return "nvfp4"
    return "plain"


def _groups(config, rollout: dict, bf16: dict) -> list[tuple[str, ...]]:
    """The tensors the trainer exports, grouped by SwiGLU MLP (gate, up, down) where one is whole."""
    names = [
        name
        for name in sorted(bf16)
        if not any(pattern in name for pattern in config.frozen)
    ]
    missing = [name for name in names if name not in rollout]
    if missing:
        raise ValueError(
            f"{len(missing)} tensors missing from the rollout, e.g. {missing[:3]}"
        )
    mlps, groups = {}, []
    for name in names:
        if match := SWIGLU.match(name):
            mlps.setdefault(match["mlp"], {})[ROLES[match["role"]]] = name
        else:
            groups.append((name,))
    for roles in mlps.values():
        if len(roles) == 3 or set(roles) == {0, 1}:
            groups.append(tuple(roles[role] for role in sorted(roles)))
        else:
            groups += [(name,) for name in roles.values()]
    return groups


@app.function(cpu=16, memory=262144, timeout=12 * 3600, volumes=VOLUMES)
def generate(
    name: str,
    run: str,
    mode: str,
    versions: int,
    workers: int,
    worker: int,
    move_global_scale: bool,
) -> dict:
    config = DELTA_SETS[name]
    correctness = mode == "correctness"
    os.environ.update(config.env)
    harness.install_overlays()
    sys.path.insert(0, f"{harness.MILES}/tools")
    import torch
    from convert_hf_to_nvfp4 import quantize_nvfp4
    from miles.backends.training_utils.weight_update.protocols.weight_delta.encoder import (
        DeltaEncoder,
        EncodedTensor,
    )
    from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import (
        OwnerWriter,
        is_raw,
        version_dir,
    )
    from miles.utils.fp8 import quantize_fp8_block
    from miles.utils.nvfp4 import nvfp4_quantize_1d_pair
    from safetensors import safe_open
    from safetensors.torch import save_file
    from simulated_trainer import SimulatedTrainer

    device = torch.device("cuda", 0)
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "U8": torch.uint8,
    }
    rollout = _headers(_checkpoint(config.rollout_checkpoint))
    bf16 = _headers(_checkpoint(config.bf16_checkpoint))
    trainer = SimulatedTrainer(grad_scale=GRAD_SCALE)

    def read(headers, tensor_name):
        with safe_open(str(headers[tensor_name][2]), framework="pt") as reader:
            return reader.get_tensor(tensor_name)

    def export(
        names: tuple[str, ...], formats: list[str], tensors: list
    ) -> list[tuple]:
        """Gate and up quantize together in NVFP4, sharing one global scale; the rest alone."""
        if len(names) >= 2 and formats[0] == formats[1] == "nvfp4":
            units = [(0, 1), *((i,) for i in range(2, len(names)))]
        else:
            units = [(i,) for i in range(len(names))]
        return [
            output
            for unit in units
            for output in export_unit(
                formats[unit[0]], [names[i] for i in unit], [tensors[i] for i in unit]
            )
        ]

    def export_unit(fmt: str, names: list[str], tensors: list) -> list[tuple]:
        if fmt == "plain":
            return list(zip(names, tensors, strict=True))
        if fmt == "fp8":
            weight, scale = quantize_fp8_block(tensors[0], FP8_BLOCK)
            return [(names[0], weight), (f"{names[0]}_scale_inv", scale)]
        outputs = (
            nvfp4_quantize_1d_pair(*tensors)
            if len(names) == 2
            else [quantize_nvfp4(tensors[0])]
        )
        return [
            (f"{tensor_name}{suffix}", value)
            for tensor_name, output in zip(names, outputs, strict=True)
            for suffix, value in zip(("", "_scale", "_scale_2"), output, strict=True)
        ]

    def as_bytes(tensor):
        return tensor.contiguous().reshape(-1).view(torch.uint8)

    run_dir = _run_dir(name, run)
    expected_dir = run_dir / "expected"
    if correctness:
        expected_dir.mkdir(parents=True, exist_ok=True)
    encoders = [
        DeltaEncoder(CODEC, device, MAX_PENDING_NBYTES, checksums=correctness)
        for _ in range(versions)
    ]
    writes: list[list[tuple]] = [[] for _ in range(versions)]
    changed_nbytes, exported_nbytes = [0] * versions, 0
    expected, expected_nbytes, expected_map = {}, 0, {}

    def flush_expected():
        nonlocal expected, expected_nbytes
        file_name = (
            f"worker-{worker:03d}-{len(set(expected_map.values())):04d}.safetensors"
        )
        save_file(expected, str(expected_dir / file_name), metadata={"format": "pt"})
        expected_map.update(dict.fromkeys(expected, file_name))
        expected, expected_nbytes = {}, 0

    all_groups = _groups(config, rollout, bf16)
    dead_neurons, moved_global_scales = 0, 0
    for names in all_groups[worker::workers]:
        formats = [_format(tensor_name, rollout) for tensor_name in names]
        initial = [
            read(bf16 if fmt != "plain" else rollout, tensor_name).to(device)
            for tensor_name, fmt in zip(names, formats, strict=True)
        ]
        no_gradient = [[] for _ in names]
        if len(names) == 3:
            gate, up, _ = initial
            dead = (gate.abs().amax(dim=1) < DEAD_ROW_AMAX) & (
                up.abs().amax(dim=1) < DEAD_ROW_AMAX
            )
            if dead.any():
                no_gradient = [[dead[:, None]], [dead[:, None]], [dead[None, :]]]
                dead_neurons += int(dead.sum())
        previous = {}
        for output_name, value in export(names, formats, initial):
            if not torch.equal(
                as_bytes(value).cpu(), as_bytes(read(rollout, output_name))
            ):
                raise ValueError(
                    f"{output_name}: version 0 does not export to the rollout bytes"
                )
            previous[output_name] = as_bytes(value).cpu().pin_memory()
            exported_nbytes += value.nbytes
        trajectories = [
            trainer.trajectory(tensor_name, tensor, mask)
            for tensor_name, tensor, mask in zip(
                names, initial, no_gradient, strict=True
            )
        ]
        moved = move_global_scale and MOVED_EXPERT in names[0] and "nvfp4" in formats
        moved_global_scales += moved * len(names)
        for version in range(versions):
            values = [next(path) for path in trajectories]
            if moved:
                for value in values:
                    value.view(-1)[value.abs().argmax()] *= 1 + (version + 1) / 16
            outputs = export(names, formats, values)
            pairs, xor_tensors = [], []
            for output_name, value in outputs:
                dtype, shape, _ = rollout[output_name]
                current = as_bytes(value)
                if is_raw(dtype, shape):
                    host = current.cpu()
                    changed = int(torch.count_nonzero(host != previous[output_name]))
                    if changed:
                        previous[output_name].copy_(host)
                        writes[version].append((output_name, dtype, shape, host))
                        changed_nbytes[version] += changed
                else:
                    pairs.append((previous[output_name], current))
                    xor_tensors.append((output_name, dtype, shape))
            for tensor_meta, result in zip(
                xor_tensors,
                encoders[version].encode(pairs) if pairs else [],
                strict=True,
            ):
                if result is not None:
                    writes[version].append((*tensor_meta, result))
                    changed_nbytes[version] += result.changed_nbytes
        # The encoder's copies of the last version into previous run on the GPU's streams.
        torch.cuda.current_stream().synchronize()
        if not correctness:
            continue
        for output_name, last in previous.items():
            dtype, shape, _ = rollout[output_name]
            expected[output_name] = last.clone().view(dtypes[dtype]).reshape(shape)
            expected_nbytes += last.numel()
            if expected_nbytes >= SHARD_BYTES:
                flush_expected()
    if expected:
        flush_expected()

    records = []
    for version in range(versions):
        writer = OwnerWriter(
            version_dir(run_dir / "deltas", version + 1), worker, checksums=correctness
        )
        compressed = iter(zip(*encoders[version].finish(), strict=True))
        for output_name, dtype, shape, written in writes[version]:
            if isinstance(written, EncodedTensor):
                writer.add_xor(
                    output_name,
                    dtype,
                    shape,
                    written.frames,
                    *next(compressed),
                    written.checksum,
                )
            else:
                writer.add_raw(output_name, dtype, shape, memoryview(written.numpy()))
        records.append(writer.finish())
    harness.data.commit()
    return {
        "records": records,
        "expected_map": expected_map,
        "exported_nbytes": exported_nbytes,
        "changed_nbytes": changed_nbytes,
        "dead_neurons": dead_neurons,
        "moved_global_scales": moved_global_scales,
        "groups": len(all_groups[worker::workers]),
    }


@app.function(cpu=16, memory=262144, timeout=4 * 3600, volumes=VOLUMES)
def seal_run(
    name: str,
    run: str,
    mode: str,
    versions: int,
    results: list[dict],
) -> dict:
    """Seals each version's manifest and, for correctness, the expected checkpoint."""
    harness.install_overlays()
    from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import (
        is_published,
        seal,
        version_dir,
    )

    config = DELTA_SETS[name]
    run_dir = _run_dir(name, run)
    for version in range(1, versions + 1):
        directory = version_dir(run_dir / "deltas", version)
        # Modal restarts a preempted call with the same input: the versions it sealed stay as they are.
        if is_published(directory):
            continue
        seal(
            directory,
            [result["records"][version - 1] for result in results],
            codec=CODEC,
            base_version=version - 1,
            target_version=version,
        )
    exported = sum(result["exported_nbytes"] for result in results)
    report = {
        "delta_set": name,
        "run": run,
        "mode": mode,
        "versions": versions,
        "dead_neurons": sum(result["dead_neurons"] for result in results),
        "moved_global_scales": sum(result["moved_global_scales"] for result in results),
        "exported_nbytes": exported,
        "density": [
            sum(result["changed_nbytes"][v] for result in results) / exported
            for v in range(versions)
        ],
        "wire_nbytes": [
            sum((result["records"][v] or {}).get("nbytes", 0) for result in results)
            for v in range(versions)
        ],
    }
    if mode == "correctness":
        report["unexported"] = _seal_expected(config, run_dir, results)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))
    harness.data.commit()
    return report


def _seal_expected(config, run_dir: Path, results: list[dict]) -> int:
    """Completes the expected checkpoint with the rollout's unexported tensors and files; returns their count."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    rollout_dir = _checkpoint(config.rollout_checkpoint)
    expected_dir = run_dir / "expected"
    weight_map = {
        tensor_name: file_name
        for result in results
        for tensor_name, file_name in result["expected_map"].items()
    }
    unexported = {}
    for path in sorted(rollout_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as reader:
            for tensor_name in reader.keys():
                if tensor_name not in weight_map:
                    unexported[tensor_name] = reader.get_tensor(tensor_name)
    if unexported:
        save_file(unexported, str(expected_dir / "unexported.safetensors"))
        weight_map |= dict.fromkeys(unexported, "unexported.safetensors")
    (expected_dir / INDEX).write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    for path in rollout_dir.iterdir():
        if path.suffix != ".safetensors" and path.name != INDEX:
            shutil.copy(path, expected_dir / path.name)
    return len(unexported)


@app.local_entrypoint()
def main(
    delta_set: str,
    mode: str,
    versions: int = 10,
    workers: int = 4,
    move_global_scale: bool = False,
):
    if mode not in MODES:
        raise SystemExit(f"--mode is one of {MODES}")
    run = datetime.now().strftime("%m%d-%H%M%S")
    workers_call = generate.with_options(gpu=DELTA_SETS[delta_set].gpu)
    results = list(
        workers_call.starmap(
            [
                (delta_set, run, mode, versions, workers, w, move_global_scale)
                for w in range(workers)
            ]
        )
    )
    print(
        json.dumps(
            seal_run.remote(delta_set, run, mode, versions, results),
            indent=1,
        )
    )
