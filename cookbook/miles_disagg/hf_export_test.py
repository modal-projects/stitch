"""HF export preservation contracts, exercised inside the Miles trainer image."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("miles")
    from miles.backends.megatron_utils import hf_export

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
    )
    monkeypatch.setattr(hf_export, "get_gloo_group", lambda: None)
    monkeypatch.setattr(
        hf_export,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0)),
    )
    yield hf_export
    torch.distributed.destroy_process_group()


@pytest.fixture
def checkpoint(tmp_path, exporter):
    import safetensors.torch
    import torch

    base = tmp_path / "base"
    base.mkdir()
    weights = {
        "model.language_model.weight": torch.ones(2, dtype=torch.bfloat16),
        "model.language_model.missing.weight": torch.ones(3, dtype=torch.bfloat16),
        "model.visual.weight": torch.tensor([2, 3], dtype=torch.bfloat16),
        "model.language_model.input_scale": torch.tensor(0.25),
        "model.language_model.rotary_emb.inv_freq": torch.tensor([1.0, 0.01]),
    }
    safetensors.torch.save_file(weights, base / "model.safetensors")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "vision_config": {"hidden_size": 2},
    }
    (base / "config.json").write_text(json.dumps(config))
    return base, weights, config


def _export(exporter, monkeypatch, base, output, current, prefixes=None):
    class Iterator:
        def __init__(self, *args, **kwargs):
            pass

        def get_hf_weight_chunks(self, weights):
            yield list(current.items())

    monkeypatch.setattr(exporter, "HfWeightIteratorDirect", Iterator)
    args = SimpleNamespace(hf_checkpoint=str(base))
    if prefixes is not None:
        args.hf_export_static_weight_prefixes = prefixes
    exporter._start_direct_export(
        args,
        [],
        output,
        model_name="qwen3_6",
        quantization_config=None,
        megatron_local_weights={},
        complete=True,
    ).finish()


@pytest.mark.parametrize("prefixes", [None, ["model.visual."]])
def test_complete_export_preserves_declared_static_weights(
    exporter, checkpoint, tmp_path, monkeypatch, prefixes
):
    import safetensors.torch
    import torch

    base, original, config = checkpoint
    output = tmp_path / "export"
    current = {"model.language_model.weight": torch.full((2,), 9, dtype=torch.bfloat16)}
    _export(exporter, monkeypatch, base, output, current, prefixes)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    expected = {
        **current,
        **{
            name: tensor
            for name, tensor in original.items()
            if name.endswith((".input_scale", ".inv_freq"))
        },
    }
    if prefixes:
        expected["model.visual.weight"] = original["model.visual.weight"]
    assert set(index["weight_map"]) == set(expected)
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in expected.values()
    )
    for name, tensor in expected.items():
        restored = safetensors.torch.load_file(output / index["weight_map"][name])[name]
        assert restored.dtype == tensor.dtype
        assert torch.equal(restored, tensor)
    assert json.loads((output / "config.json").read_text()) == config
    assert (output / ".complete").is_file()


def test_existing_exported_static_weights_take_precedence(
    exporter, checkpoint, tmp_path, monkeypatch
):
    import safetensors.torch
    import torch

    base, original, _ = checkpoint
    output = tmp_path / "export"
    current = {name: torch.full_like(tensor, 9) for name, tensor in original.items()}
    _export(exporter, monkeypatch, base, output, current, ["model.visual."])

    index = json.loads((output / "model.safetensors.index.json").read_text())
    for name, tensor in current.items():
        restored = safetensors.torch.load_file(output / index["weight_map"][name])[name]
        assert torch.equal(restored, tensor)
    assert not (output / "model-static.safetensors").exists()


@pytest.mark.parametrize(
    "prefixes", ["model.visual.", "", [], [""], [7], ["model.visual.", None]]
)
def test_invalid_static_prefixes_cannot_mark_export_complete(
    exporter, checkpoint, tmp_path, monkeypatch, prefixes
):
    base, original, _ = checkpoint
    output = tmp_path / "export"
    output.mkdir()
    (output / ".complete").touch()
    with pytest.raises(RuntimeError, match="nonempty list of nonempty strings"):
        _export(exporter, monkeypatch, base, output, original, prefixes)
    assert not (output / ".complete").exists()


@pytest.mark.parametrize(
    "broken_source", ["unmatched", "no-index", "no-shard", "no-directory"]
)
def test_missing_static_source_cannot_mark_export_complete(
    exporter, checkpoint, tmp_path, monkeypatch, broken_source
):
    base, original, _ = checkpoint
    prefixes = ["model.visual."]
    if broken_source == "unmatched":
        prefixes = ["model.audio."]
    elif broken_source == "no-index":
        (base / "model.safetensors.index.json").unlink()
    elif broken_source == "no-shard":
        (base / "model.safetensors").unlink()
    else:
        base = tmp_path / "absent"
    output = tmp_path / "export"
    current = {"model.language_model.weight": original["model.language_model.weight"]}
    with pytest.raises(RuntimeError, match="HF export failed"):
        _export(exporter, monkeypatch, base, output, current, prefixes)
    assert not (output / ".complete").exists()
