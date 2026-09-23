from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from cookbook.common import serving_image

GEMMA_PATCH = Path(serving_image.DEFAULT_SGLANG_RUNTIME.patches[0])


class _Image:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.local_files: list[tuple[str, str, bool]] = []

    def add_local_file(self, local: str, remote: str, copy: bool = False) -> _Image:
        self.local_files.append((local, remote, copy))
        return self

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


def _build(monkeypatch, runtime: serving_image.SGLangRuntime) -> _Image:
    image = _Image()
    monkeypatch.setattr(serving_image.modal.Image, "from_registry", lambda _name: image)
    serving_image.build_serving_image(
        hf_cache_path="/cache", experiment="test", runtime=runtime
    )
    return image


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
    assert image.local_files == []
    assert "git apply" not in source_overlay


def test_build_serving_image_applies_patches_before_overlay_copy(
    monkeypatch, tmp_path
) -> None:
    patch = tmp_path / "example.patch"
    patch.write_text("")
    runtime = serving_image.SGLangRuntime(
        image="example/sglang:image",
        repository="https://example.com/sglang.git",
        branch="model-release",
        commit="0123456789abcdef",
        patches=(str(patch),),
    )

    image = _build(monkeypatch, runtime)

    remote = "/tmp/stitch-sglang-patches/example.patch"
    assert image.local_files == [(str(patch), remote, True)]
    steps = image.commands[0].split(" && ")
    checkout = f"git -C /tmp/stitch-sglang-overlay checkout --detach {runtime.commit}"
    check = f"git -C /tmp/stitch-sglang-overlay apply --check {remote}"
    apply = f"git -C /tmp/stitch-sglang-overlay apply {remote}"
    copy = "cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
    assert [step for step in steps if step in {checkout, check, apply, copy}] == [
        checkout,
        check,
        apply,
        copy,
    ]


def test_build_serving_image_rejects_missing_patch(monkeypatch, tmp_path) -> None:
    runtime = serving_image.SGLangRuntime(
        image="example/sglang:image",
        repository="https://example.com/sglang.git",
        branch="model-release",
        commit="0123456789abcdef",
        patches=(str(tmp_path / "missing.patch"),),
    )

    with pytest.raises(FileNotFoundError, match="missing.patch"):
        _build(monkeypatch, runtime)


def test_default_runtime_applies_gemma_rmsnorm_patch(monkeypatch) -> None:
    assert GEMMA_PATCH.is_file()
    image = _build(monkeypatch, serving_image.DEFAULT_SGLANG_RUNTIME)
    assert image.local_files[0][0] == str(GEMMA_PATCH)
    assert (
        f"apply --check /tmp/stitch-sglang-patches/{GEMMA_PATCH.name}"
        in image.commands[0]
    )


def test_gemma_rmsnorm_patch_isolates_derived_buffer_from_loader() -> None:
    """The staged-load fix must (1) stop the loader from writing a buffer that the
    CPU shadow module shares with the live GPU module and (2) rederive it after
    commit. Both are visible in the patch text without importing SGLang."""
    patch = GEMMA_PATCH.read_text()
    files = re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)
    assert files == ["python/sglang/srt/layers/layernorm.py"]
    added = [line[1:] for line in patch.splitlines() if line.startswith("+")]
    assert any(
        line.strip().startswith("def get_derived_weight_tensors(self)")
        for line in added
    )
    assert any('"gemma_weight", self.gemma_weight' in line for line in added)
    assert any(
        line.strip().startswith("def process_weights_after_weight_commit(self)")
        for line in added
    )
    # The loader keeps writing in place (stable storage for CUDA graphs) but only
    # ever through the module the shadow parameter belongs to.
    assert "torch.add(param.data, 1.0, out=self.gemma_weight)" in patch


def test_gemma_rmsnorm_patch_applies_to_pinned_fork_commit(tmp_path) -> None:
    """Network test: a blob-less, sparse, depth-1 fetch of the pinned fork commit
    (a few MB) followed by ``git apply --check``. Skipped when the fork cannot be
    fetched; fails if the patch no longer applies to the pin."""
    runtime = serving_image.DEFAULT_SGLANG_RUNTIME
    touched = re.findall(
        r"^\+\+\+ b/(.+)$", GEMMA_PATCH.read_text(), flags=re.MULTILINE
    )
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "remote", "add", "origin", runtime.repository], check=True)
    subprocess.run([*git, "sparse-checkout", "set", "--no-cone", *touched], check=True)
    fetch = subprocess.run(
        [*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin"]
        + [runtime.commit],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if fetch.returncode != 0:
        pytest.skip(f"cannot fetch {runtime.repository}: {fetch.stderr.strip()}")
    subprocess.run([*git, "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)

    check = subprocess.run(
        [*git, "apply", "--check", str(GEMMA_PATCH)], capture_output=True, text=True
    )
    assert check.returncode == 0, check.stderr
    reverse = subprocess.run(
        [*git, "apply", "--check", "--reverse", str(GEMMA_PATCH)],
        capture_output=True,
        text=True,
    )
    assert reverse.returncode != 0, "patch is already part of the pinned commit"
