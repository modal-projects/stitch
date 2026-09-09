from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import checkpoint_hooks
from cookbook.miles_disagg.checkpoint_hooks import configure_checkpointing


def test_full_checkpoints_use_ephemeral_disk_and_a_separate_volume():
    cfg = SimpleNamespace(
        save_interval=10, save_hf="hf_checkpoints/weight_v{rollout_id:06d}"
    )
    hooks = configure_checkpointing(
        cfg, run_id="run", attempt_id="attempt", volume_name="run-checkpoints"
    )
    assert cfg.save == "/tmp/stitch-checkpoints/run/attempt/checkpoints"
    assert (
        cfg.save_hf
        == "/tmp/stitch-checkpoints/run/attempt/hf_checkpoints/weight_v{rollout_id:06d}"
    )
    assert not cfg.async_save
    assert hooks["stitch_checkpoint_volume"] == "run-checkpoints"


def test_disabled_checkpoints_do_not_start_a_persistence_session():
    cfg = SimpleNamespace(save_interval=None, save_hf="unused")
    assert (
        configure_checkpointing(
            cfg, run_id="run", attempt_id="attempt", volume_name="checkpoints"
        )
        == {}
    )
    assert cfg.save is None and cfg.save_hf is None


def test_async_serialization_cannot_publish_an_unfinished_local_snapshot():
    cfg = SimpleNamespace(save_interval=10, async_save=True)
    with pytest.raises(ValueError, match="async_save=False"):
        configure_checkpointing(
            cfg, run_id="run", attempt_id="attempt", volume_name="checkpoints"
        )



def test_final_drain_keeps_collective_participants_polling(monkeypatch):
    """A host must not wait on network I/O while its peers enter a collective."""
    from unittest.mock import Mock, PropertyMock

    worker = Mock()
    type(worker).pending = PropertyMock(side_effect=[1, 0])
    session = checkpoint_hooks.CheckpointSession.__new__(
        checkpoint_hooks.CheckpointSession
    )
    session.args = SimpleNamespace(stitch_checkpoint_timeout_seconds=1)
    session.rank = 0
    session.uploader = worker
    states = []
    session.gather = lambda value: states.append(value) or [value]
    monkeypatch.setattr(checkpoint_hooks.time, "sleep", lambda seconds: None)
    session.wait()
    assert states == [(1, None), (0, None)]
