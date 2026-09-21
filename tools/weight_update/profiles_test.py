from __future__ import annotations

from pathlib import PurePosixPath

from tools.weight_update.profiles import glm5_3_nvfp4 as profiler


def test_local_draft_checkpoint_is_mounted():
    draft = PurePosixPath(
        profiler.model.SGLANG_SERVER_ARGS["--speculative-draft-model-path"]
    )

    assert draft.is_absolute()
    assert any(
        draft.is_relative_to(mount) for mount in profiler.benchmark.spec.volumes
    ), f"{draft} is absent from the profiler's volume mounts"
