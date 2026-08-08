from __future__ import annotations

from typing import Any

import pytest

from cookbook.miles_disagg import trainer_image as miles_trainer_image
from cookbook.slime_disagg import trainer_image as slime_trainer_image


# Cross-integration coverage lives at the cookbook boundary so ``cookbook.common``
# never imports the trainer-specific packages that consume it.
class _OrderingImage:
    """Small Modal Image ordering model: build layers may not follow runtime mounts."""

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
    trainer_image: Any,
    local_source_arg: str,
    *,
    copy_source: bool,
    local_source: str | None = None,
) -> _OrderingImage:
    image = _OrderingImage()
    monkeypatch.setattr(
        trainer_image.modal.Image, "from_registry", lambda *_args, **_kwargs: image
    )
    return trainer_image.build_trainer_image(
        hf_cache_path="/cache",
        experiment="test",
        copy_source=copy_source,
        **({local_source_arg: local_source} if local_source else {}),
    )


@pytest.mark.parametrize(
    ("trainer_image", "local_source_arg"),
    [
        (miles_trainer_image, "miles_local"),
        (slime_trainer_image, "slime_local"),
    ],
    ids=["miles", "slime"],
)
def test_sources_remain_fast_runtime_mounts_by_default(
    monkeypatch,
    trainer_image: Any,
    local_source_arg: str,
) -> None:
    image = _build(
        monkeypatch,
        trainer_image,
        local_source_arg,
        copy_source=False,
    )

    assert image.has_runtime_mount


@pytest.mark.parametrize(
    ("trainer_image", "local_source_arg", "local_source", "trainer_root"),
    [
        (miles_trainer_image, "miles_local", "/local/miles", "/root/miles"),
        (slime_trainer_image, "slime_local", "/local/slime", "/root/slime"),
    ],
    ids=["miles", "slime"],
)
def test_copy_source_image_can_be_extended_after_local_fork_overlay(
    monkeypatch,
    trainer_image: Any,
    local_source_arg: str,
    local_source: str,
    trainer_root: str,
) -> None:
    image = _build(
        monkeypatch,
        trainer_image,
        local_source_arg,
        copy_source=True,
        local_source=local_source,
    )

    assert not image.has_runtime_mount
    image.env({"TRAINER_GPU": "B300"})
    image.add_local_file("patch", "/root/patch", copy=True)
    image.run_commands(f"cd {trainer_root} && git apply /root/patch")
