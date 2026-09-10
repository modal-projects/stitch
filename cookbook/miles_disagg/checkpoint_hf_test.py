from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.checkpoint_hf import write_shards, writer_ranks


class Tensor:
    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size

    def element_size(self):
        return 2

    def detach(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def contiguous(self):
        return self


def test_distributed_writers_produce_one_complete_balanced_index(tmp_path):
    shards = [
        (f"tensor-{i}", Tensor(size))
        for i, size in enumerate([9, 8, 7, 6, 5, 4, 3, 2, 1])
    ]
    owners = {}
    indices = []
    writers = (0, 8, 16, 24)
    for rank in (0, 1, 8, 9, 16, 17, 24, 25):

        def save(tensors, path, rank=rank):
            assert path.name not in owners
            owners[path.name] = rank

        index, total, loads = write_shards(
            ([tensor] for tensor in shards),
            tmp_path / str(rank),
            rank=rank,
            writers=writers,
            gather=lambda error: [error],
            save=save,
        )
        indices.append(index)
    assert all(index == indices[0] for index in indices)
    assert set(indices[0]) == {key for key, _ in shards}
    assert set(owners) == set(indices[0].values())
    assert set(owners.values()) == set(writers)
    assert total == 90
    assert max(loads.values()) - min(loads.values()) <= 18
    assert not (tmp_path / "1").exists()


def test_writer_failure_drains_collective_iterator_before_raising(tmp_path):
    consumed = []

    def chunks():
        for i in range(4):
            consumed.append(i)
            yield [(f"weight-{i}", Tensor(1))]

    def fail(tensors, path):
        raise OSError("no space left")

    with pytest.raises(RuntimeError, match="no space left"):
        write_shards(
            chunks(),
            tmp_path,
            rank=0,
            writers=(0, 8),
            gather=lambda error: [error],
            save=fail,
        )
    assert consumed == [0, 1, 2, 3]


def test_peer_failure_invalidates_the_local_export(tmp_path):
    with pytest.raises(RuntimeError, match="rank 8"):
        write_shards(
            [[("weight", Tensor(1))]],
            tmp_path,
            rank=0,
            writers=(0, 8),
            gather=lambda error: [error, "rank 8: disk failure"],
            save=lambda *args: None,
        )


def test_only_staged_exports_use_host_writers():
    args = SimpleNamespace(
        stitch_checkpoint_local_root="/tmp/checkpoints/run",
        stitch_checkpoint_host_ranks=(0, 8),
    )
    assert writer_ranks(args, Path("/tmp/checkpoints/run/hf/9")) == (0, 8)
    assert writer_ranks(args, Path("/tmp/eval/hf")) == (0,)
    assert writer_ranks(args, Path("/tmp/checkpoints/run/../../eval")) == (0,)
    assert writer_ranks(SimpleNamespace(), Path("/shared/hf")) == (0,)
