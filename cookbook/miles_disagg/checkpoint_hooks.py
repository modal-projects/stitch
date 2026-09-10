"""Miles integration for local snapshots and background checkpoint persistence.

All collective calls run on the training thread, in the same order on every
rank. Only elected host leaders own uploaders. An in-flight upload skips periodic
saves across the actor group; explicit/final saves wait and then save.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from cookbook.common.constants import TRAINING_CHECKPOINTS_PATH
from cookbook.miles_disagg.checkpoint import CheckpointUpload, CheckpointUploader
from cookbook.miles_disagg.checkpoint_priority import watch_serving

logger = logging.getLogger(__name__)


def configure_checkpointing(
    cfg: Any,
    *,
    run_id: str,
    attempt_id: str,
    volume_name: str,
    upload_mib_per_second: int = 256,
    min_free_disk_mib: int = 65536,
    timeout_seconds: int = 21600,
    delta_quiesce_seconds: float = 5,
    delta_serving_seconds: float = 300,
    delta_min_ready: int = 1,
) -> dict:
    """Set launcher-owned local paths; return configuration for Miles hooks."""
    if getattr(cfg, "save_interval", None) is None:
        cfg.save = cfg.save_hf = None
        return {}
    if getattr(cfg, "train_backend", "megatron") != "megatron" or getattr(
        cfg, "use_critic", False
    ):
        raise ValueError(
            "local checkpoint staging currently requires a Megatron actor-only recipe"
        )
    if getattr(cfg, "async_save", False):
        raise ValueError(
            "local checkpoint staging requires async_save=False; persistence is asynchronous"
        )
    if upload_mib_per_second < 0 or min_free_disk_mib < 0 or timeout_seconds <= 0:
        raise ValueError("invalid checkpoint staging limits")
    if delta_quiesce_seconds < 0 or delta_serving_seconds <= 0 or delta_min_ready < 1:
        raise ValueError("invalid checkpoint delta priority limits")
    from cookbook.miles_disagg.checkpoint import relative_path

    relative_path(run_id)
    relative_path(attempt_id)
    root = Path("/tmp/stitch-checkpoints") / run_id / attempt_id
    hf = getattr(cfg, "save_hf", None)
    if hf:
        from cookbook.miles_disagg.resume import _validate_save_hf_template

        _validate_save_hf_template(hf)
        cfg.save_hf = str(root / hf)
    cfg.save = str(root / "checkpoints")
    # Local snapshots must be closed and immutable before the uploader sees them.
    cfg.async_save = False
    return {
        "stitch_checkpoint_local_root": str(root),
        "stitch_checkpoint_attempt_id": attempt_id,
        "stitch_checkpoint_volume": volume_name,
        "custom_checkpoint_persistence_path": "cookbook.miles_disagg.checkpoint_hooks.CheckpointSession",
        "stitch_checkpoint_upload_mib_per_second": upload_mib_per_second,
        "stitch_checkpoint_min_free_disk_mib": min_free_disk_mib,
        "stitch_checkpoint_timeout_seconds": timeout_seconds,
        "stitch_checkpoint_delta_quiesce_seconds": delta_quiesce_seconds,
        "stitch_checkpoint_delta_serving_seconds": delta_serving_seconds,
        "stitch_checkpoint_delta_min_ready": delta_min_ready,
    }


class CheckpointSession:
    def __init__(self, args: Any):
        import torch.distributed as dist
        from miles.utils.distributed_utils import get_gloo_group

        self.args = args
        self.rank = dist.get_rank()
        self.group = get_gloo_group()
        self.local_root = Path(args.stitch_checkpoint_local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        # Container identity describes shared disk, unlike CUDA local_rank when
        # Ray places independent actor processes on the same trainer host.
        host = os.environ.get("MODAL_TASK_ID")
        if not host:
            raise RuntimeError("checkpoint host election requires MODAL_TASK_ID")
        hosts = self.gather(host)
        self.host_ranks = tuple(
            i for i, value in enumerate(hosts) if value not in hosts[:i]
        )
        args.stitch_checkpoint_host_ranks = self.host_ranks
        self._serving_token = None
        self.uploader = None
        if self.rank in self.host_ranks:
            import modal

            self.uploader = CheckpointUploader(
                TRAINING_CHECKPOINTS_PATH,
                modal.Volume.from_name(args.stitch_checkpoint_volume),
                bytes_per_second=args.stitch_checkpoint_upload_mib_per_second * 1024**2,
                timeout_seconds=args.stitch_checkpoint_timeout_seconds,
            )

    @contextmanager
    def prioritize_update(self, version: int):
        """Quiesce checkpoint I/O before encoding; resume after serving, asynchronously."""
        token = None
        started = time.monotonic()
        try:

            def pause():
                nonlocal token
                if self.uploader:
                    token = self.uploader.io.pause()
                    # Acquire the new lease before releasing an older serving wait.
                    if self._serving_token is not None:
                        self.uploader.io.resume(self._serving_token)
                        self._serving_token = None

            self.collectively(pause)
            deadline = started + self.args.stitch_checkpoint_delta_quiesce_seconds
            while True:
                idle = self.collectively(
                    lambda: self.uploader.io.quiescent if self.uploader else True
                )
                if all(idle) or time.monotonic() >= deadline:
                    if self.rank == 0:
                        logger.log(
                            logging.INFO if all(idle) else logging.WARNING,
                            "CHECKPOINT %s",
                            json.dumps(
                                dict(
                                    phase="delta_quiesced"
                                    if all(idle)
                                    else "delta_quiesce_timeout",
                                    version=version,
                                    seconds=time.monotonic() - started,
                                    busy_ranks=[
                                        rank
                                        for rank, ready in enumerate(idle)
                                        if not ready
                                    ],
                                )
                            ),
                        )
                    break
                time.sleep(0.05)
            yield
            if self.uploader:
                try:
                    from cookbook.common.hooks import _pool
                    from stitch.types import VersionRef

                    watch_serving(
                        self.uploader.io,
                        token,
                        _pool(self.args),
                        VersionRef(self.args.run_id, version),
                        host_rank=self.rank,
                        min_ready=self.args.stitch_checkpoint_delta_min_ready,
                        timeout=self.args.stitch_checkpoint_delta_serving_seconds,
                    )
                    self._serving_token = token
                    token = None
                except Exception:
                    logger.exception(
                        "Could not monitor delta %s on host rank %s; resuming checkpoint I/O",
                        version,
                        self.rank,
                    )
        finally:
            if token is not None:
                self.uploader.io.resume(token)

    def gather(self, value):
        import torch.distributed as dist

        values = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(values, value, group=self.group)
        return values

    def collectively(self, operation: Callable[[], Any]) -> list[Any]:
        result, error = None, None
        try:
            result = operation()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        states = self.gather((result, error))
        errors = [error for _, error in states if error]
        if errors:
            raise RuntimeError("checkpoint persistence failed:\n" + "\n".join(errors))
        return [result for result, _ in states]

    def wait(self) -> None:
        deadline = time.monotonic() + self.args.stitch_checkpoint_timeout_seconds + 60

        def pending_on_host():
            if time.monotonic() >= deadline:
                raise TimeoutError("checkpoint drain deadline expired")
            return self.uploader.pending if self.uploader else 0

        # Keep every rank participating in short collectives. Blocking on a
        # host's upload Future would strand its peers in all_gather long enough
        # to hit the process-group timeout during a slow final checkpoint.
        while True:
            pending = self.collectively(pending_on_host)
            if not any(pending):
                return
            time.sleep(0.5)

    def prepare(self, iteration: int, *, force_sync: bool) -> bool:
        if force_sync:
            self.wait()
        available = self.collectively(
            lambda: self.uploader.has_capacity if self.uploader else True
        )
        if not all(available):
            if self.rank == 0:
                logger.warning(
                    "CHECKPOINT %s",
                    json.dumps(
                        {
                            "phase": "skipped",
                            "iteration": iteration,
                            "reason": "upload_in_progress",
                        }
                    ),
                )
            # The rollout manager saved this small file before entering the actor.
            (
                self.local_root
                / f"checkpoints/rollout/global_dataset_state_dict_{iteration}.pt"
            ).unlink(missing_ok=True)
            return False

        def prepare():
            free = shutil.disk_usage(self.local_root).free
            if free < self.args.stitch_checkpoint_min_free_disk_mib * 1024**2:
                raise RuntimeError(
                    f"insufficient local checkpoint disk: {free} bytes free"
                )
            # Megatron creates the directory on global rank 0. Node-local disk
            # requires that each host creates its own directory before saving.
            (self.local_root / f"checkpoints/iter_{iteration:07d}").mkdir(
                parents=True, exist_ok=True
            )

        self.collectively(prepare)
        self._snapshot_started = time.monotonic()
        return True

    def persist(self, iteration: int) -> None:
        if self.rank == 0:
            logger.info(
                "CHECKPOINT %s",
                json.dumps(
                    {
                        "phase": "local_snapshot_complete",
                        "iteration": iteration,
                        "seconds": time.monotonic() - self._snapshot_started,
                    }
                ),
            )

        def enqueue():
            if not self.uploader:
                return
            hf = self.args.save_hf
            hf_directory = (
                Path(hf.format(rollout_id=iteration))
                .relative_to(self.local_root)
                .as_posix()
                if hf
                else None
            )
            required = ()
            if self.rank == 0:
                required = required_snapshot_files(
                    self.local_root,
                    iteration,
                    hf_directory,
                    include_rollout=bool(self.args.rollout_global_dataset),
                )
            self.uploader.submit(
                CheckpointUpload(
                    local_root=self.local_root,
                    run_id=self.args.run_id,
                    attempt_id=self.args.stitch_checkpoint_attempt_id,
                    iteration=iteration,
                    host_rank=self.rank,
                    host_ranks=self.host_ranks,
                    hf_directory=hf_directory,
                    required_files=required,
                )
            )

        self.collectively(enqueue)


def required_snapshot_files(
    local_root: Path, iteration: int, hf_directory: str | None, *, include_rollout: bool
) -> tuple[str, ...]:
    """Read writer metadata to verify every referenced shard has a durable receipt."""
    checkpoint = f"checkpoints/iter_{iteration:07d}"
    metadata_path = local_root / checkpoint / ".metadata"
    # Only unpickle the checkpoint just produced by our own trainer.
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file)
    required = {
        f"{checkpoint}/.metadata",
        f"{checkpoint}/common.pt",
        f"{checkpoint}/metadata.json",
    }
    required.update(
        f"{checkpoint}/{entry.relative_path}"
        for entry in metadata.storage_data.values()
    )
    if hf_directory:
        hf = local_root / hf_directory
        if not (hf / ".complete").is_file():
            raise RuntimeError("HF export failed: local completion marker is missing")
        index_path = hf / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        if not index["weight_map"]:
            raise RuntimeError("HF export produced no weights")
        required.add(f"{hf_directory}/model.safetensors.index.json")
        required.update(
            f"{hf_directory}/{name}" for name in index["weight_map"].values()
        )
    if include_rollout:
        required.add(f"checkpoints/rollout/global_dataset_state_dict_{iteration}.pt")
    return tuple(sorted(required))
