from pathlib import Path

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
        prep._staged(
            str(output), _checkpoint, identity={"source_revision": "b" * 40}
        )

    assert (output / "model.safetensors").read_bytes() == b"weights"


def test_incomplete_checkpoint_is_not_published(tmp_path):
    output = tmp_path / "checkpoint"

    def incomplete(directory):
        (Path(directory) / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="no safetensors shards"):
        prep._staged(str(output), incomplete, identity={})

    assert not output.exists()
