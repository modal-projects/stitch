from __future__ import annotations

from cookbook.miles_disagg import trainer_image


class _OrderingImage:
    """Modal build layers may not follow runtime source mounts."""

    def __init__(self) -> None:
        self.has_runtime_mount = False

    def _build(self, *_args, **_kwargs) -> _OrderingImage:
        assert not self.has_runtime_mount, (
            "build layer added after a runtime source mount"
        )
        return self

    entrypoint = apt_install = run_commands = pip_install = env = _build

    def add_local_file(self, *_args, copy: bool = False, **_kwargs) -> _OrderingImage:
        if copy:
            return self._build()
        self.has_runtime_mount = True
        return self

    add_local_dir = add_local_python_source = add_local_file


def _build(
    monkeypatch,
    *,
    copy_source: bool = False,
    miles_local: str | None = None,
) -> _OrderingImage:
    image = _OrderingImage()
    monkeypatch.setattr(
        trainer_image.modal.Image, "from_registry", lambda *_args, **_kwargs: image
    )
    return trainer_image.build_trainer_image(
        hf_cache_path="/cache",
        experiment="test",
        copy_source=copy_source,
        miles_local=miles_local,
    )


def test_sources_remain_fast_runtime_mounts_by_default(monkeypatch) -> None:
    image = _build(monkeypatch)

    assert image.has_runtime_mount


def test_copy_source_image_can_be_extended_after_local_fork_overlay(
    monkeypatch,
) -> None:
    image = _build(monkeypatch, copy_source=True, miles_local="/local/miles")

    assert not image.has_runtime_mount
    image.env({"TRAINER_GPU": "B300"})
    image.add_local_file("patch", "/root/patch", copy=True)
    image.run_commands(f"cd {trainer_image.MILES_ROOT} && git apply /root/patch")
