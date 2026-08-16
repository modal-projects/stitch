from __future__ import annotations

import json
from pathlib import Path

from cookbook.common import serving_image


class _Image:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.environment: dict[str, str] = {}
        self.local_files: list[tuple[Path, str, bool]] = []

    def run_commands(self, command: str) -> _Image:
        self.commands.append(command)
        return self

    def pip_install(self, *_packages: str) -> _Image:
        return self

    def env(self, values: dict[str, str]) -> _Image:
        self.environment.update(values)
        return self

    def add_local_file(
        self, local_path: str | Path, remote_path: str, *, copy: bool = False
    ) -> _Image:
        self.local_files.append((Path(local_path), remote_path, copy))
        return self

    def add_local_python_source(self, _module: str) -> _Image:
        return self

    def add_local_dir(self, *_args, **_kwargs) -> _Image:
        return self


def test_build_serving_image_uses_selected_runtime(monkeypatch) -> None:
    image = _Image()
    selected_image: list[str] = []
    monkeypatch.setattr(
        serving_image.modal.Image,
        "from_registry",
        lambda name: selected_image.append(name) or image,
    )
    runtime = serving_image.SGLangRuntime(
        image="example/sglang:image",
        repository="https://example.com/sglang.git",
        branch="model-release",
        commit="0123456789abcdef",
    )

    serving_image.build_serving_image(
        hf_cache_path="/cache",
        experiment="test",
        runtime=runtime,
    )

    assert selected_image == [runtime.image]
    source_overlay = image.commands[0]
    assert runtime.repository in source_overlay
    assert runtime.branch in source_overlay
    assert runtime.commit in source_overlay
    assert "sglang-fastsafetensors-config.patch" in source_overlay


def test_build_serving_image_bakes_fastsafetensors_config(monkeypatch) -> None:
    image = _Image()
    monkeypatch.setattr(
        serving_image.modal.Image,
        "from_registry",
        lambda _name: image,
    )

    serving_image.build_serving_image(
        hf_cache_path="/cache",
        experiment="test",
    )

    config_path = "/etc/fastsafetensors.json"
    patch_path = "/tmp/sglang-fastsafetensors-config.patch"
    local_files = {
        remote_path: (local_path, copy)
        for local_path, remote_path, copy in image.local_files
    }
    assert image.environment["FASTSAFETENSORS_CONFIG"] == config_path
    assert local_files[config_path] == (
        Path(serving_image.__file__).with_name("fastsafetensors.json"),
        True,
    )
    assert local_files[patch_path] == (
        Path(serving_image.__file__).with_name("sglang_fastsafetensors_config.patch"),
        True,
    )
    assert json.loads(local_files[config_path][0].read_text()) == {
        "loader": "base",
        "base": {
            "copier_type": "nogds",
            "bbuf_size_kb": 65536,
            "max_threads": 4,
        },
    }
