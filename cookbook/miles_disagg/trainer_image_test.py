from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("miles") is None,
    reason="requires the Miles trainer image",
)


@pytest.mark.parametrize("source_rank", [False, True])
def test_delta_refresh_failure_prevents_file_creation(
    tmp_path, monkeypatch, source_rank
):
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import (
        UpdateWeightFromDiskDelta,
    )

    from cookbook.common import hooks

    updates = tmp_path / "updates"
    updater = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    updater.args = SimpleNamespace(
        custom_update_weight_pre_write_path="cookbook.common.hooks.refresh_before_update"
    )
    updater.delta_dir = str(updates)
    updater.weight_version = 1
    calls = []

    def refresh(args, path):
        calls.append((args, path))
        raise OSError("mount reload failed")

    monkeypatch.setattr(hooks, "refresh_before_update", refresh)
    monkeypatch.setattr(
        UpdateWeightFromDiskDelta, "_is_source", property(lambda self: source_rank)
    )

    with pytest.raises(OSError, match="mount reload failed"):
        updater._encode_delta()

    assert calls == [(updater.args, str(updates))]
    assert not updates.exists()
