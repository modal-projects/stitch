from __future__ import annotations

from pathlib import Path

from cookbook.miles_disagg import trainer_image


class _RecordingImage:
    def __init__(self) -> None:
        self.operations: list[tuple[str, object]] = []

    def _record(self, name: str, *args, **kwargs) -> _RecordingImage:
        self.operations.append((name, (args, kwargs)))
        return self

    def entrypoint(self, *args, **kwargs) -> _RecordingImage:
        return self._record("entrypoint", *args, **kwargs)

    def pip_install(self, *args, **kwargs) -> _RecordingImage:
        return self._record("pip_install", *args, **kwargs)

    def apt_install(self, *args, **kwargs) -> _RecordingImage:
        return self._record("apt_install", *args, **kwargs)

    def run_commands(self, *args, **kwargs) -> _RecordingImage:
        return self._record("run_commands", *args, **kwargs)

    def add_local_file(self, *args, **kwargs) -> _RecordingImage:
        return self._record("add_local_file", *args, **kwargs)


def test_source_patch_is_copied_and_applied_before_install(
    monkeypatch, tmp_path: Path
) -> None:
    image = _RecordingImage()
    patch = tmp_path / "qwen.patch"
    patch.write_text("diff --git a/a b/a\n")
    monkeypatch.setattr(
        trainer_image.modal.Image,
        "from_registry",
        lambda *_args, **_kwargs: image,
    )
    monkeypatch.setattr(
        trainer_image.common_trainer_image,
        "add_common_layers",
        lambda current_image, **_kwargs: current_image,
    )

    trainer_image.build_trainer_image(
        hf_cache_path="/cache",
        experiment="qwen",
        source_patches=(patch,),
    )

    patch_copy_index = next(
        index
        for index, (operation, (args, kwargs)) in enumerate(image.operations)
        if operation == "add_local_file" and args[0] == str(patch) and kwargs["copy"]
    )
    patch_apply_index, patch_apply_command = next(
        (index, args[0])
        for index, (operation, (args, _kwargs)) in enumerate(image.operations)
        if operation == "run_commands" and "git -C /root/miles apply --check" in args[0]
    )

    assert patch_copy_index < patch_apply_index
    assert (
        "git -C /root/miles apply /root/miles-source-patches/00-qwen.patch"
        in patch_apply_command
    )
    assert patch_apply_command.index(
        "git -C /root/miles apply"
    ) < patch_apply_command.index("pip install")
