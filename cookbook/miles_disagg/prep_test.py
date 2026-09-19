from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import prep


def _checkpoint(path: str | Path, *, config: dict | None = None) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or {}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}})
    )
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def _experiment(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        SOURCE_MODEL="example/model",
        SOURCE_REVISION="a" * 40,
        BF16_CHECKPOINT_PATH=root / "bf16",
        TORCH_DIST_CHECKPOINT_PATH=root / "torch-dist",
        SERVED_CHECKPOINT_FORMAT="bf16",
        PREP_ENV={},
        miles=SimpleNamespace(
            hf_checkpoint=str(root / "bf16"), megatron_model_type="test"
        ),
        modal=SimpleNamespace(
            torch_dist_prep_nodes=1,
            torch_dist_prep_gpus_per_node=1,
            torch_dist_convert_extra_args="--tensor-model-parallel-size 1",
        ),
    )


def test_staged_rejects_unidentified_existing_bytes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text("{}")

    with pytest.raises(RuntimeError, match="cannot reuse checkpoint.*output"):
        prep._staged(str(output), lambda _: pytest.fail("must not build"), identity={})

    assert (output / "config.json").read_text() == "{}"


def test_staged_reuses_only_a_complete_matching_artifact(tmp_path: Path) -> None:
    output = tmp_path / "output"
    identity = prep._preparation_identity(_experiment(tmp_path), "bf16")
    prep._staged(str(output), _checkpoint, identity=identity)

    prep._staged(
        str(output),
        lambda _: pytest.fail("must reuse the verified artifact"),
        identity=identity,
    )

    manifest = json.loads((output / prep._COMPLETION_FILE).read_text())
    assert manifest["identity"] == identity
    assert manifest["files"]["model.safetensors"] == len(b"weights")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SOURCE_MODEL", "another/model"),
        ("SOURCE_REVISION", "b" * 40),
        ("UNPACK_FUSED_EXPERTS", True),
        ("PREP_ENV", {"NVTE_NVFP4_4OVER6_ERR_MODE": "MSE"}),
        ("MILES_REPO_REF", "c" * 40),
        ("TRAINER_EXTRA_PIP_PACKAGES", ("transformer-engine==2.17",)),
    ],
)
def test_staged_rejects_changed_preparation_inputs(
    tmp_path: Path, field: str, value: object
) -> None:
    exp = _experiment(tmp_path)
    output = tmp_path / "output"
    prep._staged(
        str(output), _checkpoint, identity=prep._preparation_identity(exp, "bf16")
    )
    setattr(exp, field, value)

    with pytest.raises(RuntimeError, match="identity differs"):
        prep._staged(
            str(output), _checkpoint, identity=prep._preparation_identity(exp, "bf16")
        )

    assert (output / "model.safetensors").read_bytes() == b"weights"


def test_identity_tracks_precision_and_carveouts(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    original = prep._preparation_identity(exp, "nvfp4")
    assert original != prep._preparation_identity(exp, "bf16")
    exp.miles.num_layers_at_end_in_bf16 = 2
    exp.miles.extra_high_precision_layers_hf = ("model.layers.3",)

    changed = prep._preparation_identity(exp, "nvfp4")

    assert original != changed
    assert json.loads(json.dumps(changed)) == changed


@pytest.mark.parametrize("revision", [None, "main", "release-v1"])
def test_identity_requires_an_immutable_source_revision(
    tmp_path: Path, revision
) -> None:
    exp = _experiment(tmp_path)
    exp.SOURCE_REVISION = revision

    with pytest.raises(ValueError, match="pinned SOURCE_REVISION"):
        prep._preparation_identity(exp, "bf16")


@pytest.mark.parametrize("revision", [None, "main", "13fa3952"])
def test_identity_requires_an_immutable_converter_revision(tmp_path, revision):
    exp = _experiment(tmp_path)
    exp.MILES_REPO_REF = revision

    with pytest.raises(ValueError, match="full MILES_REPO_REF commit hash"):
        prep._preparation_identity(exp, "nvfp4")


def test_local_overlay_cannot_reuse_a_prepared_artifact(tmp_path, monkeypatch):
    source = _checkpoint(tmp_path / "source")
    exp = _experiment(tmp_path)
    prep.prepare_checkpoints(
        exp,
        SimpleNamespace(reload=lambda: None, commit=lambda: None),
        source_snapshot=str(source),
    )
    manifest = exp.BF16_CHECKPOINT_PATH / prep._COMPLETION_FILE
    original = manifest.read_bytes()
    monkeypatch.setenv("MILES_LOCAL_DIR", str(tmp_path / "mutable-miles"))

    with pytest.raises(ValueError, match="MILES_LOCAL_DIR.*full MILES_REPO_REF"):
        prep.prepare_checkpoints(
            exp,
            SimpleNamespace(
                reload=lambda: None, commit=lambda: pytest.fail("must not reuse")
            ),
            source_snapshot=str(source),
        )

    assert manifest.read_bytes() == original


@pytest.mark.parametrize("damage", ["missing", "empty", "truncated"])
def test_staged_rejects_damaged_completed_shards(tmp_path: Path, damage: str) -> None:
    output = tmp_path / "output"
    prep._staged(str(output), _checkpoint, identity={})
    shard = output / "model.safetensors"
    if damage == "missing":
        shard.unlink()
    else:
        shard.write_bytes(b"" if damage == "empty" else b"w")

    with pytest.raises(RuntimeError):
        prep._staged(
            str(output), lambda _: pytest.fail("must not overwrite"), identity={}
        )


def test_staged_does_not_publish_a_builder_with_missing_shards(tmp_path: Path) -> None:
    output = tmp_path / "output"

    def incomplete(path: str) -> None:
        _checkpoint(path)
        (Path(path) / "model.safetensors").unlink()

    with pytest.raises(RuntimeError, match="missing or empty"):
        prep._staged(str(output), incomplete, identity={})

    assert not output.exists()
    assert not (tmp_path / "output.partial" / prep._COMPLETION_FILE).exists()


def test_staged_preserves_an_existing_partial_directory(tmp_path: Path) -> None:
    partial = _checkpoint(tmp_path / "output.partial")

    with pytest.raises(RuntimeError, match="unfinished preparation"):
        prep._staged(str(tmp_path / "output"), _checkpoint, identity={})

    assert (partial / "model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("quant_method", ["fp8", "compressed-tensors", "unknown"])
def test_quantized_source_is_not_relabelled_as_bf16(
    tmp_path: Path, nested: bool, quant_method: str
) -> None:
    config = {"quantization_config": {"quant_method": quant_method}}
    source = _checkpoint(
        tmp_path / "source", config={"text_config": config} if nested else config
    )
    exp = _experiment(tmp_path)
    volume = SimpleNamespace(
        reload=lambda: None, commit=lambda: pytest.fail("must not commit")
    )

    with pytest.raises(ValueError, match="quantized.*explicit BF16 conversion"):
        prep.prepare_checkpoints(exp, volume, source_snapshot=str(source))

    assert not exp.BF16_CHECKPOINT_PATH.exists()
    assert (source / "model.safetensors").read_bytes() == b"weights"


def test_prepare_commits_only_after_hf_completion(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source")
    exp = _experiment(tmp_path)
    commits = []
    volume = SimpleNamespace(
        reload=lambda: None,
        commit=lambda: commits.append(
            (exp.BF16_CHECKPOINT_PATH / prep._COMPLETION_FILE).exists()
        ),
    )

    prep.prepare_checkpoints(exp, volume, source_snapshot=str(source))

    assert commits == [True]


def _torch_dist_checkpoint(root: Path) -> dict[str, int]:
    checkpoint = root / "iter_0000001"
    checkpoint.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("1")
    for name in (".metadata", "common.pt", "__0_0.distcp"):
        (checkpoint / name).write_bytes(b"checkpoint")
    return prep._file_sizes(
        root,
        (str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()),
    )


def test_torch_dist_requires_receipts_from_every_node(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    exp.modal.torch_dist_prep_nodes = 2
    identity = prep._preparation_identity(exp, "torch_dist")
    files = _torch_dist_checkpoint(exp.TORCH_DIST_CHECKPOINT_PATH)
    prep._write_completion(
        exp.TORCH_DIST_CHECKPOINT_PATH, identity, files, name=".stitch-prep-node-0.json"
    )

    with pytest.raises(RuntimeError, match="stitch-prep-node-1"):
        prep._torch_dist_reusable(exp.TORCH_DIST_CHECKPOINT_PATH, identity)

    remote_shard = exp.TORCH_DIST_CHECKPOINT_PATH / "iter_0000001/__1_0.distcp"
    remote_shard.write_bytes(b"other node weights")
    prep._write_completion(
        exp.TORCH_DIST_CHECKPOINT_PATH,
        identity,
        {"iter_0000001/__1_0.distcp": remote_shard.stat().st_size},
        name=".stitch-prep-node-1.json",
    )
    assert prep._torch_dist_reusable(exp.TORCH_DIST_CHECKPOINT_PATH, identity)
    remote_shard.unlink()
    with pytest.raises(RuntimeError, match="__1_0.distcp"):
        prep._torch_dist_reusable(exp.TORCH_DIST_CHECKPOINT_PATH, identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch_dist_prep_nodes", 2),
        ("torch_dist_prep_gpus_per_node", 8),
        ("torch_dist_convert_extra_args", "--tensor-model-parallel-size 2"),
    ],
)
def test_torch_dist_rejects_changed_topology(
    tmp_path: Path, field: str, value: object
) -> None:
    exp = _experiment(tmp_path)
    identity = prep._preparation_identity(exp, "torch_dist")
    files = _torch_dist_checkpoint(exp.TORCH_DIST_CHECKPOINT_PATH)
    prep._write_completion(
        exp.TORCH_DIST_CHECKPOINT_PATH, identity, files, name=".stitch-prep-node-0.json"
    )
    setattr(exp.modal, field, value)

    with pytest.raises(RuntimeError, match="identity differs"):
        prep._torch_dist_reusable(
            exp.TORCH_DIST_CHECKPOINT_PATH,
            prep._preparation_identity(exp, "torch_dist"),
        )


def test_torch_dist_commits_data_before_its_completion_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    source = _checkpoint(tmp_path / "source")
    exp = _experiment(tmp_path)
    exp.MATERIALIZE_BF16_MASTERS = False
    receipt = exp.TORCH_DIST_CHECKPOINT_PATH / ".stitch-prep-node-0.json"
    commits = []
    volume = SimpleNamespace(
        reload=lambda: None, commit=lambda: commits.append(receipt.exists())
    )
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.external_utils.model_args_utils",
        SimpleNamespace(load_model_args=lambda _: "--num-layers 2"),
    )
    monkeypatch.setattr(
        prep.subprocess,
        "run",
        lambda *args, **kwargs: _torch_dist_checkpoint(exp.TORCH_DIST_CHECKPOINT_PATH),
    )

    prep.prepare_torch_dist(
        exp, volume, rank=0, master_addr="localhost", source_snapshot=str(source)
    )

    assert commits == [False, True]
    assert prep._torch_dist_reusable(
        exp.TORCH_DIST_CHECKPOINT_PATH, prep._preparation_identity(exp, "torch_dist")
    )


def test_torch_dist_does_not_mark_a_failed_data_commit(
    tmp_path: Path, monkeypatch
) -> None:
    source = _checkpoint(tmp_path / "source")
    exp = _experiment(tmp_path)
    exp.MATERIALIZE_BF16_MASTERS = False

    def fail_commit() -> None:
        raise OSError("volume commit failed")

    volume = SimpleNamespace(reload=lambda: None, commit=fail_commit)
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.external_utils.model_args_utils",
        SimpleNamespace(load_model_args=lambda _: "--num-layers 2"),
    )
    monkeypatch.setattr(
        prep.subprocess,
        "run",
        lambda *args, **kwargs: _torch_dist_checkpoint(exp.TORCH_DIST_CHECKPOINT_PATH),
    )

    with pytest.raises(OSError, match="volume commit failed"):
        prep.prepare_torch_dist(
            exp, volume, rank=0, master_addr="localhost", source_snapshot=str(source)
        )

    assert not (exp.TORCH_DIST_CHECKPOINT_PATH / ".stitch-prep-node-0.json").exists()
