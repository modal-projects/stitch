from types import SimpleNamespace

import pytest

from cookbook.common import process
from cookbook.miles_disagg import checkpoint_hooks


@pytest.mark.parametrize("leader", [False, True])
def test_checkpoint_marker_follows_every_host_commit(tmp_path, monkeypatch, leader):
    events = []
    checkpoint = tmp_path / "iter_0000009"
    checkpoint.mkdir()
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / ".complete").touch()
    marker = checkpoint / ".stitch-complete"

    class Volume:
        def commit(self):
            events.append(("commit", marker.exists()))

    def gather(error):
        assert error is None
        assert not marker.exists()
        events.append(("all_hosts", False))
        return [None, None]

    monkeypatch.setattr(checkpoint_hooks, "_volume", lambda args: Volume())
    monkeypatch.setattr(process, "dist_is_container_leader", lambda: leader)
    monkeypatch.setattr(process, "dist_rank", lambda: 0 if leader else 1)
    monkeypatch.setattr(process, "dist_all_gather_object", gather)
    checkpoint_hooks.commit_checkpoint(SimpleNamespace(), 9, str(checkpoint), str(hf))
    assert events == (
        [("commit", False), ("all_hosts", False), ("commit", True)]
        if leader
        else [("all_hosts", False)]
    )
    assert marker.exists() is leader


def test_failed_peer_commit_cannot_publish_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        checkpoint_hooks, "_volume", lambda args: SimpleNamespace(commit=lambda: None)
    )
    monkeypatch.setattr(process, "dist_is_container_leader", lambda: True)
    monkeypatch.setattr(process, "dist_rank", lambda: 0)
    monkeypatch.setattr(
        process, "dist_all_gather_object", lambda error: [None, "peer commit failed"]
    )
    with pytest.raises(RuntimeError, match="peer commit failed"):
        checkpoint_hooks.commit_checkpoint(
            SimpleNamespace(), 9, str(tmp_path), str(tmp_path)
        )
    assert not (tmp_path / ".stitch-complete").exists()


def test_local_commit_failure_is_reported_collectively(tmp_path, monkeypatch):
    def fail():
        raise OSError("mount commit failed")

    def gather(error):
        assert "mount commit failed" in error
        return [error, None]

    monkeypatch.setattr(
        checkpoint_hooks, "_volume", lambda args: SimpleNamespace(commit=fail)
    )
    monkeypatch.setattr(process, "dist_is_container_leader", lambda: True)
    monkeypatch.setattr(process, "dist_all_gather_object", gather)
    with pytest.raises(RuntimeError, match="mount commit failed"):
        checkpoint_hooks.commit_checkpoint(
            SimpleNamespace(), 9, str(tmp_path), str(tmp_path)
        )
    assert not (tmp_path / ".stitch-complete").exists()
