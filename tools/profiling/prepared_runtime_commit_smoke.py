"""GPU smoke test for the bounded-memory prepared runtime commit primitive."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket

import modal


SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "stitch-sglang-v0.5.15-post1-prepared-runtime"
SGLANG_FORK_COMMIT = "1cb181c8886139c19e545ca6667645c7545f4f4b"
GIB = 1 << 30


app = modal.App("prepared-runtime-commit-smoke")
image = modal.Image.from_registry("lmsysorg/sglang:v0.5.15.post1").run_commands(
    f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
    f" && git fetch modal-fork {SGLANG_FORK_BRANCH}"
    f" && git checkout {SGLANG_FORK_COMMIT} -- python/"
)


def _broadcast_worker(rank: int, port: int, result_queue) -> None:
    import torch

    from sglang.srt.distributed.parallel_state import (
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.weight_sync.runtime_state import PreparedRuntimeState

    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": "4",
            "LOCAL_RANK": str(rank),
            "SGLANG_PREPARED_BROADCAST_CHECKPOINT": "1",
            "SGLANG_PREPARED_PINNED_GB": "0",
            "SGLANG_PREPARED_TAIL_BUFFER_COUNT": "4",
            "SGLANG_PREPARED_TAIL_CHUNK_MIB": "64",
        }
    )
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=4,
        rank=rank,
        local_rank=rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=4)
    try:
        # Exclude one-time NCCL communicator setup from the measured batches.
        warmup = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        get_tp_group().broadcast(warmup, src=0)
        torch.cuda.synchronize()

        model = torch.nn.Module()
        model.language_model = torch.nn.Module()
        model.language_model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        )
        model.language_model.model.layers = torch.nn.ModuleList([layer])
        state = PreparedRuntimeState(model)
        stats = {
            "source_bytes": 0,
            "batches": 0,
            "pack_s": 0.0,
            "h2d_s": 0.0,
            "broadcast_s": 0.0,
        }
        fill = (0x11, 0x22, 0x33) if rank == 0 else (0xEE, 0xEE, 0xEE)
        source_weights = [
            ("small_a", torch.full((24 << 20,), fill[0], dtype=torch.uint8)),
            ("small_b", torch.full((32 << 20,), fill[1], dtype=torch.uint8)),
            ("large", torch.full((80 << 20,), fill[2], dtype=torch.uint8)),
        ]
        copies = {
            name: tensor.clone()
            for name, tensor in state._iter_batched_cuda_weights(
                source_weights,
                torch.device("cuda"),
                stats,
            )
        }
        torch.cuda.synchronize()
        mismatches = {
            "small_a": int(torch.count_nonzero(copies["small_a"] != 0x11).item()),
            "small_b": int(torch.count_nonzero(copies["small_b"] != 0x22).item()),
            "large": int(torch.count_nonzero(copies["large"] != 0x33).item()),
        }
        result_queue.put(
            {
                "rank": rank,
                "stats": stats,
                "mismatches": mismatches,
            }
        )
    finally:
        destroy_model_parallel()
        torch.distributed.destroy_process_group()


@app.function(image=image, gpu="B300:4", memory=32 * 1024, timeout=20 * 60)
def broadcast_smoke() -> dict:
    """Verify that only TP0 host bytes reach every rank through device broadcast."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_broadcast_worker,
            args=(rank, port, result_queue),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    reports = [result_queue.get(timeout=15 * 60) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        if process.exitcode != 0:
            raise RuntimeError(
                f"broadcast worker failed: pid={process.pid} exit={process.exitcode}"
            )
    reports.sort(key=lambda item: item["rank"])
    if any(any(report["mismatches"].values()) for report in reports):
        raise RuntimeError(reports)
    if any(report["stats"]["pack_s"] != 0 for report in reports[1:]):
        raise RuntimeError(f"non-source TP rank packed checkpoint bytes: {reports}")
    result = {
        "sglang_commit": SGLANG_FORK_COMMIT,
        "reports": reports,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, gpu="B300:1", memory=16 * 1024, timeout=20 * 60)
def smoke() -> dict:
    import torch

    from sglang.srt.weight_sync.runtime_state import (
        PreparedRuntimeState,
        clone_module_tensors,
    )

    os.environ["SGLANG_PREPARED_PINNED_GB"] = "0"
    os.environ["SGLANG_PREPARED_TAIL_BUFFER_COUNT"] = "4"
    os.environ["SGLANG_PREPARED_TAIL_CHUNK_MIB"] = "64"
    os.environ["SGLANG_PREPARED_GPU_STAGING_GB"] = "1"
    os.environ["SGLANG_PREPARED_GPU_RESERVE_GB"] = "1"

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    layer = torch.nn.Module()
    tensor_bytes = GIB + 512 * (1 << 20)
    layer.weight = torch.nn.Parameter(
        torch.zeros(tensor_bytes, dtype=torch.uint8, device="cuda"),
        requires_grad=False,
    )
    model.language_model.model.layers = torch.nn.ModuleList([layer])

    address_before = layer.weight.data_ptr()
    state = PreparedRuntimeState(model)
    preallocate = state.preallocate_transfer_buffers()
    prepared = state.begin_preparation("smoke|1")
    prepared.bytes.fill_(0x5A)
    stage = state.stage_prepared()
    commit = state.commit()
    torch.cuda.synchronize()

    mismatch_count = int(torch.count_nonzero(layer.weight != 0x5A).item())

    packed_stats = {
        "source_bytes": 0,
        "batches": 0,
        "pack_s": 0.0,
        "h2d_s": 0.0,
    }
    source_weights = [
        ("small_a", torch.full((24 << 20,), 0x11, dtype=torch.uint8)),
        ("small_b", torch.full((32 << 20,), 0x22, dtype=torch.uint8)),
        # Larger than the 64 MiB reusable buffer exercises chunked H2D.
        ("large", torch.full((80 << 20,), 0x33, dtype=torch.uint8)),
    ]
    packed_copies = {
        name: tensor.clone()
        for name, tensor in state._iter_batched_cuda_weights(
            source_weights,
            torch.device("cuda"),
            packed_stats,
        )
    }
    torch.cuda.synchronize()
    packed_mismatches = {
        "small_a": int(torch.count_nonzero(packed_copies["small_a"] != 0x11).item()),
        "small_b": int(torch.count_nonzero(packed_copies["small_b"] != 0x22).item()),
        "large": int(torch.count_nonzero(packed_copies["large"] != 0x33).item()),
    }

    shadow = clone_module_tensors(layer)
    with torch.no_grad():
        shadow.weight.fill_(0xA5)
    prepared = state.begin_preparation("smoke|d2h")
    copied_bytes = state._copy_shadow_module(
        "language_model.model.layers.0",
        shadow,
    )
    d2h_samples_match = bool(
        torch.all(prepared.bytes[:4096] == 0xA5)
        and torch.all(prepared.bytes[-4096:] == 0xA5)
    )
    d2h_stage = state.stage_prepared()
    d2h_commit = state.commit()
    torch.cuda.synchronize()
    d2h_mismatch_count = int(torch.count_nonzero(layer.weight != 0xA5).item())

    report = {
        "sglang_commit": SGLANG_FORK_COMMIT,
        "tensor_bytes": tensor_bytes,
        "image_bytes": state.image_nbytes,
        "preallocate": preallocate,
        "stage": stage,
        "commit": commit,
        "address_preserved": layer.weight.data_ptr() == address_before,
        "mismatch_count": mismatch_count,
        "packed_stats": packed_stats,
        "packed_mismatches": packed_mismatches,
        "d2h_copied_bytes": copied_bytes,
        "d2h_samples_match": d2h_samples_match,
        "d2h_stage": d2h_stage,
        "d2h_commit": d2h_commit,
        "d2h_mismatch_count": d2h_mismatch_count,
    }
    if (
        not report["address_preserved"]
        or mismatch_count
        or any(packed_mismatches.values())
        or copied_bytes != tensor_bytes
        or not d2h_samples_match
        or d2h_mismatch_count
    ):
        raise RuntimeError(report)
    if stage["gpu_stage_bytes"] != GIB:
        raise RuntimeError(f"unexpected GPU staging size: {report}")
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(smoke.remote(), indent=2))
