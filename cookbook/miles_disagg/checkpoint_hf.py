"""Distribute HF shard files across the hosts staging a local checkpoint."""

from __future__ import annotations

from pathlib import Path


def writer_ranks(args, path: Path) -> tuple[int, ...]:
    root = getattr(args, "stitch_checkpoint_local_root", None)
    if root and path.resolve().is_relative_to(Path(root).resolve()):
        ranks = tuple(args.stitch_checkpoint_host_ranks)
        if not ranks or ranks[0] != 0 or tuple(sorted(set(ranks))) != ranks:
            raise ValueError(
                "HF writers must be sorted, unique host leaders including rank 0"
            )
        return ranks
    return (0,)


def write_shards(
    chunks, path: Path, *, rank: int, writers: tuple[int, ...], gather, save
):
    """Collectively write each shard once; all ranks derive the same global index.

    The iterator performs collectives. A local disk failure must therefore drain
    the iterator before propagating to peers; otherwise the next gather hangs.
    """
    loads = dict.fromkeys(writers, 0)
    weight_map = {}
    error = None
    if rank in writers:
        try:
            path.mkdir(parents=True, exist_ok=True)
            (path / ".complete").unlink(missing_ok=True)
        except Exception as exc:
            error = f"rank {rank}: {type(exc).__name__}: {exc}"
    for index, tensors in enumerate(chunks, start=1):
        tensors = list(tensors)
        owner = min(loads, key=lambda writer: (loads[writer], writer))
        name = f"model-{index:05d}.safetensors"
        size = sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)
        loads[owner] += size
        for tensor_name, _ in tensors:
            if tensor_name in weight_map:
                error = f"duplicate HF tensor: {tensor_name}"
            weight_map[tensor_name] = name
        if rank == owner and error is None:
            try:
                save(
                    {
                        key: value.detach().to("cpu").contiguous()
                        for key, value in tensors
                    },
                    path / name,
                )
            except Exception as exc:
                error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [message for message in gather(error) if message]
    if errors:
        raise RuntimeError("HF shard export failed:\n" + "\n".join(errors))
    if not weight_map:
        raise RuntimeError(f"HF export to {path} produced no weights")
    return weight_map, sum(loads.values()), loads
