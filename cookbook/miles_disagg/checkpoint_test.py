from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from cookbook.miles_disagg.checkpoint import CheckpointUpload, CheckpointUploader


class MemoryVolume:
    """Distinct mounts see files only after the writing mount commits."""

    def __init__(self, mount: Path, durable: dict[str, bytes]):
        self.mount = mount
        self.durable = durable
        self.commit_started = threading.Event()
        self.allow_commit = threading.Event()
        self.allow_commit.set()
        self.fail_commit = False

    def commit(self):
        self.commit_started.set()
        assert self.allow_commit.wait(5), "test commit was never released"
        if self.fail_commit:
            raise OSError("injected commit failure")
        for path in self.mount.rglob("*"):
            if path.is_file():
                self.durable[path.relative_to(self.mount).as_posix()] = (
                    path.read_bytes()
                )

    def read_file(self, path):
        try:
            yield self.durable[path]
        except KeyError:
            raise FileNotFoundError(path) from None


def make_upload(tmp_path, host, *, iteration=9, hosts=(0, 8)):
    local = tmp_path / f"local-{host}"
    checkpoint = local / f"checkpoints/iter_{iteration:07d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / f"__{host}_0.distcp").write_bytes(f"weights-{host}".encode())
    hf_dir = f"hf_checkpoints/weight_v{iteration:06d}"
    if host == 0:
        (checkpoint / ".metadata").write_bytes(b"metadata")
        (checkpoint / "common.pt").write_bytes(b"optimizer-scheduler-rng")
        hf = local / hf_dir
        hf.mkdir(parents=True)
        (hf / "model.safetensors").write_bytes(b"hf-weights")
        (hf / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}})
        )
        (hf / ".complete").touch()
    return CheckpointUpload(
        local_root=local,
        run_id="run",
        attempt_id="attempt",
        iteration=iteration,
        host_rank=host,
        host_ranks=hosts,
        hf_directory=hf_dir,
    )


def uploader(tmp_path, host, durable, **kwargs):
    mount = tmp_path / f"mount-{host}"
    mount.mkdir()
    volume = MemoryVolume(mount, durable)
    return CheckpointUploader(
        mount,
        volume,
        bytes_per_second=0,
        poll_seconds=0.01,
        timeout_seconds=4,
        **kwargs,
    ), volume


def wait_for_upload(worker):
    deadline = time.monotonic() + 5
    while worker.pending:
        assert time.monotonic() < deadline, "upload did not finish"
        time.sleep(0.01)


def test_slow_commit_does_not_block_submit_or_publish_partial_checkpoint(tmp_path):
    durable = {}
    first, volume0 = uploader(tmp_path, 0, durable)
    second, _ = uploader(tmp_path, 8, durable)
    upload0 = make_upload(tmp_path, 0)
    upload8 = make_upload(tmp_path, 8)
    volume0.allow_commit.clear()
    try:
        first.submit(upload0)
        assert volume0.commit_started.wait(2)
        second.submit(upload8)
        assert first.pending == 1
        assert not first.has_capacity
        assert not any(path.startswith("run/completed/") for path in durable)
        assert upload0.local_root.exists()
        volume0.allow_commit.set()
        wait_for_upload(first)
        wait_for_upload(second)
        manifest = json.loads(durable[upload0.manifest_path])
        assert manifest["version"] == 10
        assert set(manifest["host_ranks"]) == {0, 8}
        root = "run/attempts/attempt/snapshots/0000009"
        for rank in (0, 8):
            assert (
                durable[f"{root}/checkpoints/iter_0000009/__{rank}_0.distcp"]
                == f"weights-{rank}".encode()
            )
        assert durable[f"{root}/checkpoints/latest_checkpointed_iteration.txt"] == b"9"
        assert not (upload0.local_root / "checkpoints/iter_0000009").exists()
        assert not (upload8.local_root / "checkpoints/iter_0000009").exists()
    finally:
        volume0.allow_commit.set()
        first.close()
        second.close()


def test_failed_upload_retains_local_snapshot_and_never_advertises_completion(tmp_path):
    worker, volume = uploader(tmp_path, 0, {})
    upload = make_upload(tmp_path, 0, hosts=(0,))
    volume.fail_commit = True
    try:
        worker.submit(upload)
        with pytest.raises(RuntimeError, match="injected commit failure"):
            wait_for_upload(worker)
        assert (upload.local_root / "checkpoints/iter_0000009").exists()
        assert upload.manifest_path not in volume.durable
    finally:
        worker.close()


def test_only_one_snapshot_is_accepted_at_a_time(tmp_path):
    worker, volume = uploader(tmp_path, 0, {})
    volume.allow_commit.clear()
    try:
        worker.submit(make_upload(tmp_path, 0, hosts=(0,)))
        assert volume.commit_started.wait(2)
        with pytest.raises(RuntimeError, match="capacity"):
            worker.submit(make_upload(tmp_path, 0, iteration=19, hosts=(0,)))
    finally:
        volume.allow_commit.set()
        worker.close()



def test_missing_shard_receipt_prevents_completion(tmp_path):
    from dataclasses import replace

    worker, volume = uploader(tmp_path, 0, {})
    upload = replace(
        make_upload(tmp_path, 0, hosts=(0,)),
        required_files=("checkpoints/iter_0000009/missing.distcp",),
    )
    try:
        worker.submit(upload)
        with pytest.raises(RuntimeError, match="missing referenced files"):
            wait_for_upload(worker)
        assert upload.manifest_path not in volume.durable
        assert (upload.local_root / upload.checkpoint_directory).is_dir()
    finally:
        worker.close()


def test_completed_upload_releases_capacity_for_the_next_snapshot(tmp_path):
    worker, volume = uploader(tmp_path, 0, {})
    try:
        for iteration in (9, 19):
            upload = make_upload(tmp_path, 0, iteration=iteration, hosts=(0,))
            worker.submit(upload)
            wait_for_upload(worker)
            assert worker.has_capacity
            assert upload.manifest_path in volume.durable
    finally:
        worker.close()
