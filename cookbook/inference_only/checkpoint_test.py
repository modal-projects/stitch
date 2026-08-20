import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.inference_only.checkpoint import (
    claim_boot_pointer,
    ensure_boot_pointer,
    require_checkpoint,
)
from stitch.types import VersionRef


def test_require_checkpoint_accepts_config_with_safetensors(tmp_path: Path) -> None:
    """Single-file model with safetensors."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")

    assert require_checkpoint(model) == model


def test_require_checkpoint_rejects_missing_config(tmp_path: Path) -> None:
    """No config.json → incomplete."""
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(RuntimeError, match="missing config.json"):
        require_checkpoint(model)


def test_require_checkpoint_missing_directory(tmp_path: Path) -> None:
    """Missing directory → FileNotFoundError with prep hint."""
    model = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="prep_app::download_model"):
        require_checkpoint(model)


def test_require_checkpoint_with_index_json_all_shards_present(tmp_path: Path) -> None:
    """Real-shaped index.json (weight_map + total_size only, NO metadata.version)."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")

    # Create index.json with weight_map but no metadata.version
    index = {
        "metadata": {"total_size": 100 * 1024**3},  # 100 GB, no 'version' key
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00002-of-00002.safetensors",
        },
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index))

    # Create shard files
    (model / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1000)
    (model / "model-00002-of-00002.safetensors").write_bytes(b"x" * 1000)

    assert require_checkpoint(model) == model


def test_require_checkpoint_with_index_json_missing_shard(tmp_path: Path) -> None:
    """Missing shard in index.json → RuntimeError with prep hint."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")

    index = {
        "metadata": {"total_size": 100 * 1024**3},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        },
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1000)
    # Missing model-00002-of-00002.safetensors

    with pytest.raises(RuntimeError, match="missing shards.*prep_app::download_model"):
        require_checkpoint(model)


def test_require_checkpoint_with_index_json_empty_shard(tmp_path: Path) -> None:
    """Empty shard file → RuntimeError with prep hint."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")

    index = {
        "metadata": {"total_size": 100 * 1024**3},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        },
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1000)
    (model / "model-00002-of-00002.safetensors").write_bytes(b"")  # Empty!

    with pytest.raises(RuntimeError, match="empty shards.*prep_app::download_model"):
        require_checkpoint(model)


def test_require_checkpoint_single_file_model_present(tmp_path: Path) -> None:
    """Single-file model with no index.json."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"x" * 1000)

    assert require_checkpoint(model) == model


def test_require_checkpoint_single_file_model_missing(tmp_path: Path) -> None:
    """No index.json and no safetensors files → incomplete."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")

    with pytest.raises(RuntimeError, match="no safetensors.*prep_app::download_model"):
        require_checkpoint(model)


def test_require_checkpoint_index_json_not_a_dict(tmp_path: Path) -> None:
    """Valid JSON that isn't an object → clean failure with prep hint, not AttributeError."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps(["not", "a", "dict"]))

    with pytest.raises(RuntimeError, match="not a JSON object.*prep_app::download_model"):
        require_checkpoint(model)


def test_require_checkpoint_index_json_missing_weight_map(tmp_path: Path) -> None:
    """Index dict without weight_map → fail closed with prep hint, not silent pass."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    index = {"metadata": {"total_size": 100 * 1024**3}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(RuntimeError, match="missing weight_map.*prep_app::download_model"):
        require_checkpoint(model)


def test_require_checkpoint_index_json_invalid_utf8(tmp_path: Path) -> None:
    """Non-UTF-8 index.json → clean failure with prep hint, not UnicodeDecodeError."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_bytes(b'\xff\xfe{"weight_map": {}}')

    with pytest.raises(RuntimeError, match="corrupted index.json.*prep_app::download_model"):
        require_checkpoint(model)


def test_claim_boot_pointer_claims_when_pointer_absent() -> None:
    """No pointer yet → the claim writes the boot version."""
    claimed = []
    store = SimpleNamespace(
        refresh=lambda: None,
        read_pointer=lambda: None,
        claim=lambda boot: claimed.append(boot),
    )

    claim_boot_pointer(store, "run-a")

    assert claimed == [VersionRef("run-a", 0)]


def test_claim_boot_pointer_no_write_when_pointer_present() -> None:
    """An existing pointer (boot or any later publish) is left untouched — never rewind."""
    store = SimpleNamespace(
        refresh=lambda: None,
        read_pointer=lambda: VersionRef("run-a", 3),
        claim=lambda boot: pytest.fail("claim must not write when a pointer exists"),
    )

    claim_boot_pointer(store, "run-a")


def test_ensure_boot_pointer_waits_for_launcher_claim() -> None:
    """Replica waits for launcher to set the pointer."""
    boot = VersionRef("run-a", 0)
    refreshes = 0

    def refresh():
        nonlocal refreshes
        refreshes += 1

    store = SimpleNamespace()
    store.refresh = refresh
    store.read_pointer = lambda: boot if refreshes >= 2 else None

    # Should wait and find the pointer after a few refreshes
    ensure_boot_pointer(store, "run-a")
    assert refreshes >= 2


def test_ensure_boot_pointer_fails_if_launcher_never_claims() -> None:
    """Replica fails fast if launcher doesn't claim within timeout."""
    store = SimpleNamespace()
    store.refresh = lambda: None
    store.read_pointer = lambda: None

    with pytest.raises(RuntimeError, match="boot pointer.*launcher"):
        ensure_boot_pointer(store, "run-a", timeout_seconds=1)
