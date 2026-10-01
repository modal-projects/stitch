"""Background upload of node-local checkpoints: the contract readers rely on.

Ranks run in real threads and exchange collectives through a barrier: host A holds ranks
0 (its leader) and 1, host B holds rank 2 (its leader). Each host has its own local
root; the fake Volume makes each upload batch visible only when it finishes, as the
Volume API does, and records every change in order.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path, PurePosixPath

import pytest

from cookbook.common.checkpoint_upload import CheckpointUploader
from stitch.publisher import TrainerComms

RUN = "run"
TRACKER = "checkpoints/latest_checkpointed_iteration.txt"


class _Exchange:
    """An in-process all-gather across rank threads."""

    def __init__(self, world: int) -> None:
        self._barrier = threading.Barrier(world, timeout=30)
        self._values: list = [None] * world

    def gather(self, rank: int, value):
        self._values[rank] = value
        self._barrier.wait()
        gathered = list(self._values)
        self._barrier.wait()
        return gathered


class _Comms(TrainerComms):
    def __init__(self, rank: int, leader: bool, exchange: _Exchange) -> None:
        self._rank, self._leader, self._exchange = rank, leader, exchange

    def rank(self):
        return self._rank

    def all_gather_object(self, value):
        return self._exchange.gather(self._rank, value)

    def is_host_leader(self):
        return self._leader


class _Volume:
    """Batches become visible, all at once, when their upload finishes."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.events: list[tuple[str, str, bytes]] = []
        self.gates: dict[str, threading.Event] = {}
        # Path -> how many more uploads of it fail.
        self.failing: dict[str, int] = {}
        self._lock = threading.Lock()

    def batch_upload(self, force: bool = False):
        assert force
        return _Batch(self)

    def remove_file(self, path: str) -> None:
        with self._lock:
            if path not in self.files:
                raise FileNotFoundError(path)
            del self.files[path]
            self.events.append(("remove", path, b""))


class _Batch:
    def __init__(self, volume: _Volume) -> None:
        self._volume = volume
        self._puts: list[tuple[str, bytes]] = []

    def __enter__(self):
        return self

    def put_file(self, source, path: str) -> None:
        data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        self._puts.append((path, data))

    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            return False
        for path, _ in self._puts:
            if (gate := self._volume.gates.get(path)) is not None:
                assert gate.wait(timeout=30), f"gate for {path} never opened"
            if self._volume.failing.get(path, 0) > 0:
                self._volume.failing[path] -= 1
                raise OSError(f"upload of {path} failed")
        with self._volume._lock:
            for path, data in self._puts:
                self._volume.files[path] = data
                self._volume.events.append(("put", path, data))
        return False


class _Cluster:
    def __init__(self, tmp_path: Path, volume: _Volume) -> None:
        self.roots = {
            host: tmp_path / host / RUN / "attempt-current" for host in ("A", "B")
        }
        exchange = _Exchange(3)
        placement = [("A", True), ("A", False), ("B", True)]
        self.uploaders = self.collective(
            lambda rank: CheckpointUploader(
                self.roots[placement[rank][0]],
                volume=volume,
                volume_root=PurePosixPath(RUN),
                comms=_Comms(rank, placement[rank][1], exchange),
                streams=2,
                retry_delay_seconds=0,
            )
        )

    @staticmethod
    def collective(call):
        """Run ``call(rank)`` on every rank at once; re-raise the first failure."""
        results: list = [None] * 3
        errors: list = [None] * 3

        def run(rank):
            try:
                results[rank] = call(rank)
            except BaseException as error:  # noqa: BLE001
                errors[rank] = error

        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "a rank hung"
        if any(errors):
            raise next(error for error in errors if error is not None)
        return results

    def step(self):
        self.collective(lambda rank: self.uploaders[rank].step())

    def drain(self):
        self.collective(lambda rank: self.uploaders[rank].drain(poll_seconds=0.01))

    def save(self, iteration: int) -> None:
        """What Miles writes at one save: every rank its Megatron shard, rank 0 the
        shared files, the tracker and the HF export, the rollout manager its state."""
        a, b = self.roots["A"], self.roots["B"]
        ckpt = f"checkpoints/iter_{iteration:07d}"
        hf = f"hf_checkpoints/weight_v{iteration:06d}"
        for root, name in (
            (a, "__0_0.distcp"),
            (a, "__1_0.distcp"),
            (b, "__2_0.distcp"),
        ):
            _write(root / ckpt / name, f"shard {name} of {iteration}")
        _write(a / ckpt / ".metadata", f"plan of {iteration}")
        _write(
            a / f"checkpoints/rollout/global_dataset_state_dict_{iteration}.pt", "data"
        )
        _write(a / TRACKER, str(iteration))
        _write(a / hf / "bf16/model-00001.safetensors", f"weights of {iteration}")
        _write(a / hf / "bf16/model.safetensors.index.json", "{}")
        _write(a / hf / "bf16/.complete", "")
        _write(a / hf / ".complete", "")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _volume_path(relative: str) -> str:
    return f"{RUN}/{relative}"


def _eventually(condition, cluster=None, timeout: float = 30.0) -> None:
    """Wait for background uploads; step the cluster meanwhile when given one."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        if cluster is not None:
            cluster.step()
        time.sleep(0.01)


def _order(volume: _Volume, path: str) -> int:
    return next(
        index
        for index, (kind, event_path, _) in enumerate(volume.events)
        if kind == "put" and event_path == path
    )


@pytest.fixture
def volume() -> _Volume:
    return _Volume()


@pytest.fixture
def cluster(tmp_path: Path, volume: _Volume) -> _Cluster:
    return _Cluster(tmp_path, volume)


def test_a_save_lands_at_run_directory_paths_with_markers_last(cluster, volume):
    cluster.save(9)
    cluster.step()
    cluster.drain()

    markers = {
        _volume_path(TRACKER),
        _volume_path("hf_checkpoints/weight_v000009/.complete"),
        _volume_path("hf_checkpoints/weight_v000009/bf16/.complete"),
    }
    data = {
        _volume_path(f"checkpoints/iter_0000009/{name}")
        for name in ("__0_0.distcp", "__1_0.distcp", "__2_0.distcp", ".metadata")
    } | {
        _volume_path("checkpoints/rollout/global_dataset_state_dict_9.pt"),
        _volume_path("hf_checkpoints/weight_v000009/bf16/model-00001.safetensors"),
        _volume_path("hf_checkpoints/weight_v000009/bf16/model.safetensors.index.json"),
    }
    assert set(volume.files) == data | markers
    assert volume.files[_volume_path(TRACKER)] == b"9"
    # Each file is uploaded exactly once, by its own host, and every data file is
    # durable before the first marker.
    puts = [path for kind, path, _ in volume.events if kind == "put"]
    assert sorted(puts) == sorted(data | markers)
    assert max(_order(volume, path) for path in data) < min(
        _order(volume, path) for path in markers
    )
    # Uploaded files leave local disk; Megatron's tracker stays where it wrote it.
    for root in cluster.roots.values():
        left = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        assert left == ({TRACKER} if root == cluster.roots["A"] else set())


def test_markers_wait_for_every_host(cluster, volume):
    held = _volume_path("checkpoints/iter_0000009/__2_0.distcp")
    volume.gates[held] = threading.Event()
    cluster.save(9)
    cluster.step()
    _eventually(
        lambda: (
            _volume_path("hf_checkpoints/weight_v000009/bf16/model-00001.safetensors")
            in volume.files
        ),
        cluster,
    )
    for _ in range(3):
        cluster.step()

    assert _volume_path("checkpoints/iter_0000009/__0_0.distcp") in volume.files
    assert held not in volume.files
    assert _volume_path(TRACKER) not in volume.files
    assert _volume_path("hf_checkpoints/weight_v000009/.complete") not in volume.files

    volume.gates[held].set()
    cluster.drain()
    assert _order(volume, held) < _order(volume, _volume_path(TRACKER))


def test_a_rewritten_tracker_goes_up_only_with_its_own_save(cluster, volume):
    """Miles rewrites the tracker in place at every save; a held marker keeps the
    content of the save it belongs to."""
    first = _volume_path("checkpoints/iter_0000009/__2_0.distcp")
    second = _volume_path("checkpoints/iter_0000019/__2_0.distcp")
    volume.gates[first] = threading.Event()
    volume.gates[second] = threading.Event()
    cluster.save(9)
    cluster.step()
    cluster.save(19)
    cluster.step()

    volume.gates[first].set()
    _eventually(lambda: volume.files.get(_volume_path(TRACKER)) == b"9", cluster)
    assert second not in volume.files

    volume.gates[second].set()
    cluster.drain()
    trackers = [
        data for kind, path, data in volume.events if path == _volume_path(TRACKER)
    ]
    assert trackers == [b"9", b"19"]
    assert _order(volume, second) < max(
        index
        for index, (_, path, _) in enumerate(volume.events)
        if path == _volume_path(TRACKER)
    )


def test_a_re_save_retracts_old_export_markers_before_its_publish(cluster, volume):
    """After a resume, a re-saved step overwrites an export whose old markers are on
    the Volume; they are gone before the publish that lets readers pick that version."""
    old_markers = [
        _volume_path("hf_checkpoints/weight_v000009/.complete"),
        _volume_path("hf_checkpoints/weight_v000009/bf16/.complete"),
    ]
    for path in old_markers:
        volume.files[path] = b""
    volume.files[_volume_path(TRACKER)] = b"4"
    shard = _volume_path("hf_checkpoints/weight_v000009/bf16/model-00001.safetensors")
    volume.gates[shard] = threading.Event()
    cluster.save(9)

    cluster.step()  # returns before the publish would run

    assert not any(path in volume.files for path in old_markers)
    # The tracker only ever moves forward: the resumed run's still stands.
    assert volume.files[_volume_path(TRACKER)] == b"4"

    volume.gates[shard].set()
    cluster.drain()
    assert all(path in volume.files for path in old_markers)
    assert volume.files[_volume_path(TRACKER)] == b"9"


def test_a_failed_upload_fails_every_rank_and_publishes_no_marker(cluster, volume):
    volume.failing[_volume_path("checkpoints/iter_0000009/__2_0.distcp")] = 10**6
    cluster.save(9)
    cluster.step()

    with pytest.raises(RuntimeError, match="checkpoint upload failed"):
        cluster.drain()
    assert _volume_path(TRACKER) not in volume.files
    assert _volume_path("hf_checkpoints/weight_v000009/.complete") not in volume.files


def test_a_transient_failure_is_retried(cluster, volume):
    target = _volume_path("checkpoints/iter_0000009/__2_0.distcp")
    volume.failing[target] = 1
    cluster.save(9)
    cluster.step()
    cluster.drain()

    assert volume.failing[target] == 0  # it did fail once
    assert target in volume.files
    assert volume.files[_volume_path(TRACKER)] == b"9"


def test_steps_with_nothing_new_upload_nothing(cluster, volume):
    cluster.save(9)
    cluster.step()
    cluster.drain()
    uploads = len(volume.events)

    for _ in range(3):
        cluster.step()

    assert len(volume.events) == uploads


def test_an_earlier_attempts_local_files_are_removed(tmp_path, volume):
    stale = tmp_path / "A" / RUN / "attempt-old" / "checkpoints" / "iter_0000009"
    stale.mkdir(parents=True)
    (stale / "__0_0.distcp").write_text("stale")

    cluster = _Cluster(tmp_path, volume)

    assert not (tmp_path / "A" / RUN / "attempt-old").exists()
    assert cluster.roots["A"].is_dir()
