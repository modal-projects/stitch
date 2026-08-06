from __future__ import annotations

from cookbook.common import serving_image


class _Image:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.local_files: list[tuple[str, str, bool]] = []

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

    def add_local_file(
        self, local_path: str, remote_path: str, *, copy: bool = False
    ) -> _Image:
        self.local_files.append((local_path, remote_path, copy))
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


def test_default_runtime_applies_source_hotfixes(monkeypatch) -> None:
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

    assert image.local_files == [
        (
            str(serving_image.DFLASH_PREFILL_ATTN_TP_PADDING_PATCH),
            "/tmp/stitch-sglang-patch-0.patch",
            True,
        )
    ]
    assert "git -C /tmp/stitch-sglang-overlay apply --check" in image.commands[0]
    assert "git -C /tmp/stitch-sglang-overlay apply /tmp/stitch-sglang-patch-0.patch" in image.commands[0]
