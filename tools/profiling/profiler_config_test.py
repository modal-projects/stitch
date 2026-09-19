from __future__ import annotations

import importlib
from pathlib import PurePosixPath

import pytest


@pytest.mark.parametrize("model_name", ["glm5_2_nvfp4", "glm5_3_nvfp4"])
def test_local_draft_checkpoint_is_mounted(model_name):
    profiler = importlib.import_module(
        f"tools.profiling.{model_name}_delta_weight_update"
    )
    draft = PurePosixPath(
        profiler.model.SGLANG_SERVER_ARGS["--speculative-draft-model-path"]
    )

    assert draft.is_absolute()
    assert any(
        draft.is_relative_to(mount) for mount in profiler.benchmark.spec.volumes
    ), f"{draft} is absent from the profiler's volume mounts"
