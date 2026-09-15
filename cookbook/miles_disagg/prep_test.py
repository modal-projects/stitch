from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cookbook.miles_disagg import prep


@pytest.mark.parametrize("nested", [False, True])
def test_fp8_source_is_dequantized_before_nvfp4(tmp_path, monkeypatch, nested):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
    }
    if nested:
        config = {"text_config": config}
    (source / "config.json").write_text(json.dumps(config))
    exp = SimpleNamespace(
        BF16_CHECKPOINT_PATH=tmp_path / "bf16",
        SERVED_CHECKPOINT_FORMAT="nvfp4",
        MATERIALIZE_BF16_MASTERS=False,
        miles=SimpleNamespace(hf_checkpoint=str(tmp_path / "nvfp4")),
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "--output-bf16-hf-path" in command:
            output = Path(command[command.index("--output-bf16-hf-path") + 1])
            (output / "config.json").write_text(json.dumps(config))

    monkeypatch.setattr(prep.subprocess, "run", run)
    prep.prepare_checkpoints(
        exp, Mock(), source_snapshot=str(source), rollout_snapshot=None
    )

    conversion = next(c for c in commands if c[1].endswith("/fp8_cast_bf16.py"))
    assert conversion[conversion.index("--input-fp8-hf-path") + 1] == str(source)
    assert prep._bf16_masters(exp, str(source)) == str(exp.BF16_CHECKPOINT_PATH)
    nvfp4 = next(c for c in commands if c[1].endswith("/convert_hf_to_nvfp4.py"))
    assert nvfp4[nvfp4.index("--model-dir") + 1] == str(exp.BF16_CHECKPOINT_PATH)
    output_config = json.loads((exp.BF16_CHECKPOINT_PATH / "config.json").read_text())
    assert "quantization_config" not in output_config
    assert "quantization_config" not in output_config.get("text_config", {})


def test_fp8_conversion_dereferences_snapshot_metadata(tmp_path, monkeypatch):
    source = tmp_path / "cache" / "snapshots" / "revision"
    source.mkdir(parents=True)
    blobs = source.parents[1] / "blobs"
    blobs.mkdir()
    config = {
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
    }
    metadata = {
        "config.json": json.dumps(config),
        "chat_template.jinja": "{{ messages }}",
        "tokenizer.json": "{}",
    }
    for name, content in metadata.items():
        (blobs / name).write_text(content)
        (source / name).symlink_to(Path("../../blobs") / name)
    (source / "model.safetensors.index.json").write_text("source index")
    exp = SimpleNamespace(
        BF16_CHECKPOINT_PATH=tmp_path / "bf16",
        SERVED_CHECKPOINT_FORMAT="bf16",
        miles=SimpleNamespace(hf_checkpoint=str(tmp_path / "bf16")),
    )

    def run(command, **kwargs):
        if "--output-bf16-hf-path" in command:
            output = Path(command[command.index("--output-bf16-hf-path") + 1])
            for name in metadata:
                shutil.copyfile(source / name, output / name, follow_symlinks=False)
            (output / "model.safetensors.index.json").write_text("converted index")

    monkeypatch.setattr(prep.subprocess, "run", run)
    prep.prepare_checkpoints(
        exp, Mock(), source_snapshot=str(source), rollout_snapshot=None
    )
    shutil.rmtree(source.parents[1])

    for name in ("chat_template.jinja", "tokenizer.json"):
        assert (exp.BF16_CHECKPOINT_PATH / name).read_text() == metadata[name]
        assert not (exp.BF16_CHECKPOINT_PATH / name).is_symlink()
    assert "quantization_config" not in json.loads(
        (exp.BF16_CHECKPOINT_PATH / "config.json").read_text()
    )
    assert (
        exp.BF16_CHECKPOINT_PATH / "model.safetensors.index.json"
    ).read_text() == "converted index"
