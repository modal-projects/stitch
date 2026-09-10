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


def test_runtime_patch_stack_can_be_rechecked_without_changing_sources(tmp_path):
    import re
    from pathlib import Path

    import miles

    from cookbook.common.process import apply_git_patches
    from cookbook.miles_disagg.trainer_image import MILES_RUNTIME_PATCHES

    root = Path(miles.__file__).resolve().parent.parent
    paths = {
        path
        for patch in MILES_RUNTIME_PATCHES
        for path in re.findall(r"^\+\+\+ b/(.+)$", Path(patch).read_text(), re.MULTILINE)
    }
    original = {path: (root / path).read_bytes() for path in paths}
    for path, data in original.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    apply_git_patches(list(MILES_RUNTIME_PATCHES), str(tmp_path), "Miles patches")
    apply_git_patches(list(MILES_RUNTIME_PATCHES), str(tmp_path), "Miles patches")
    assert {path: (tmp_path / path).read_bytes() for path in paths} == original
