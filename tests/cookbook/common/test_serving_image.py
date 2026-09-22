from __future__ import annotations

from cookbook.common import serving_image


class _Image:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run_commands(self, command: str) -> _Image:
        self.commands.append(command)
        return self

    def pip_install(self, *_packages: str) -> _Image:
        return self

    def env(self, _values: dict[str, str]) -> _Image:
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
