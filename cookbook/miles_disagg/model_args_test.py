from pathlib import Path

import pytest

from cookbook.miles_disagg.model_args import load_pinned_model_args


def test_load_pinned_model_args_preserves_bash_array_values(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "models" / "qwen-demo.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        'echo "sourcing pinned script"\n'
        "MODEL_ARGS=(--alpha 'two words' '' '$literal')\n"
    )

    assert load_pinned_model_args(tmp_path, "qwen-demo") == [
        "--alpha",
        "two words",
        "",
        "$literal",
    ]


def test_load_pinned_model_args_rejects_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="basename"):
        load_pinned_model_args(tmp_path, "../other-model")
