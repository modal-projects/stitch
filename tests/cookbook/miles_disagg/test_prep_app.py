import runpy
from pathlib import Path

import pytest

from cookbook.miles_disagg import prep, trainer_image
from cookbook.miles_disagg.configs import qwen3_4b_math


@pytest.mark.parametrize("mutable_converter", ["overlay", "branch"])
def test_preparation_rejects_mutable_converter_before_building_images(
    monkeypatch, tmp_path, mutable_converter
):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_4b_math")
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    if mutable_converter == "overlay":
        monkeypatch.setenv("MILES_LOCAL_DIR", str(tmp_path / "miles"))
    else:
        monkeypatch.setattr(qwen3_4b_math, "MILES_REPO_REF", "main", raising=False)
    monkeypatch.setattr(
        trainer_image,
        "build_trainer_image",
        lambda **kwargs: pytest.fail("must reject before constructing the image"),
    )

    with pytest.raises(ValueError, match="full MILES_REPO_REF commit hash"):
        runpy.run_path(str(Path(prep.__file__).with_name("prep_app.py")))
