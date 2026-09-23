from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import prep


def _checkpoint(directory):
    root = Path(directory)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"weights")


def test_matching_checkpoint_is_reused(tmp_path):
    output = str(tmp_path / "checkpoint")
    identity = {"source_revision": "a" * 40}
    prep._staged(output, _checkpoint, identity=identity)

    def must_not_rebuild(_):
        pytest.fail("matching checkpoint should be reused")

    prep._staged(output, must_not_rebuild, identity=identity)


def test_changed_identity_does_not_overwrite_checkpoint(tmp_path):
    output = tmp_path / "checkpoint"
    prep._staged(str(output), _checkpoint, identity={"source_revision": "a" * 40})

    with pytest.raises(RuntimeError, match="identity differs"):
        prep._staged(str(output), _checkpoint, identity={"source_revision": "b" * 40})

    assert (output / "model.safetensors").read_bytes() == b"weights"


def test_incomplete_checkpoint_is_not_published(tmp_path):
    output = tmp_path / "checkpoint"

    def incomplete(directory):
        (Path(directory) / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="no safetensors shards"):
        prep._staged(str(output), incomplete, identity={})

    assert not output.exists()


def test_trainer_only_packages_do_not_change_checkpoint_identity(monkeypatch):
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    recipe = SimpleNamespace(
        SOURCE_MODEL="org/model",
        SOURCE_REVISION="a" * 40,
        TRAINER_EXTRA_PIP_PACKAGES=("agent-runtime==1",),
        miles=SimpleNamespace(),
    )

    first = prep._preparation_identity(recipe, "bf16")
    recipe.TRAINER_EXTRA_PIP_PACKAGES = ("agent-runtime==2",)

    assert prep._preparation_identity(recipe, "bf16") == first
