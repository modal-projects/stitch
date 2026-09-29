from __future__ import annotations

import json

import pytest

from cookbook.miles_disagg.mimo_v2_6 import (
    _code_setup_script,
    _code_verifier_script,
    _materialize_code_task,
    _patch_paths,
    normalize_row,
    normalize_rows,
)


def _row(*, image: str = "source:image", instance_id: str = "task-1") -> dict:
    return {
        "data_source": "opensource-code",
        "ability": "swe",
        "agent_name": "mimo_swe_agent",
        "prompt": [{"role": "user", "content": "Fix the issue"}],
        "extra_info": {
            "instance_json": json.dumps(
                {
                    "instance_id": instance_id,
                    "dataset_type": "opensource-code",
                    "docker_image": image,
                    "cwd": "/testbed",
                    "test_patch": "diff --git a/a b/a",
                    "test_command": "true",
                    "verifier_timeout_sec": 1,
                }
            )
        },
    }


def test_normalize_code_row_resolves_image_without_exposing_hidden_state() -> None:
    output = normalize_row(
        "code",
        _row(),
        index=0,
        image_mapping={"source:image": "registry/image:tag"},
    )

    assert output["prompt"] == "Fix the issue"
    assert output["metadata"]["domain"] == "code"
    assert output["metadata"]["image"] == "registry/image:tag"
    assert output["metadata"]["instance"]["test_command"] == "true"
    assert "test_patch" not in output["prompt"]


def test_normalize_music_preserves_task_constraints() -> None:
    output = normalize_row(
        "music",
        {
            "data_source": "music",
            "ability": "music_generation",
            "prompt": [{"role": "user", "content": "Write ABC music"}],
            "extra_info": {"src_id": "music-1", "bpm": 120, "nvoice_want": 3},
        },
        index=0,
        image_mapping={},
    )

    assert output["metadata"]["image"] is None
    assert output["metadata"]["task"]["nvoice_want"] == 3


def test_normalize_rejects_unknown_images_and_duplicate_instances() -> None:
    with pytest.raises(ValueError, match="unmapped Docker image"):
        normalize_row("code", _row(), index=0, image_mapping={})

    with pytest.raises(ValueError, match="duplicate instance identities"):
        normalize_rows(
            "code",
            [_row(), _row()],
            image_mapping={"source:image": "registry/image:tag"},
        )


def test_normalize_rejects_prompt_shape_changes() -> None:
    row = _row()
    row["prompt"].append({"role": "assistant", "content": "leak"})
    with pytest.raises(ValueError, match="exactly one prompt message"):
        normalize_row(
            "code",
            row,
            index=0,
            image_mapping={"source:image": "registry/image:tag"},
        )


def test_patch_paths_include_rename_source_and_target() -> None:
    patch = "diff --git a/old name.py b/new name.py\n"
    assert _patch_paths(patch) == ["new name.py", "old name.py"]


def test_materialize_code_task_keeps_tests_out_of_environment(tmp_path) -> None:
    row = normalize_row(
        "code",
        _row(),
        index=0,
        image_mapping={"source:image": "registry/image:tag"},
    )
    task_dir = _materialize_code_task(tmp_path, row)

    assert (task_dir / "environment" / "Dockerfile").read_text() == (
        "FROM registry/image:tag\n"
    )
    assert not (task_dir / "environment" / "test.patch").exists()
    assert (task_dir / "tests" / "test.patch").is_file()
    assert row["metadata"]["task_dir"] == str(task_dir)
    assert "instance" not in row["metadata"]


@pytest.mark.parametrize(
    "script",
    [_code_setup_script("/testbed"), _code_verifier_script("/testbed", "true")],
)
def test_generated_code_scripts_are_valid(script: str) -> None:
    import subprocess

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_code_setup_uses_checked_out_head_when_other_refs_exist(tmp_path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "MiMo"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "mimo@example.invalid"],
        cwd=repo,
        check=True,
    )
    (repo / "baseline.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "baseline.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=repo, check=True)
    baseline = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    subprocess.run(["git", "checkout", "-q", "-b", "other-ref"], cwd=repo, check=True)
    (repo / "other.txt").write_text("other\n")
    subprocess.run(["git", "add", "other.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "other"], cwd=repo, check=True)
    subprocess.run(
        ["git", "checkout", "-q", "--detach", baseline], cwd=repo, check=True
    )

    subprocess.run(["bash"], input=_code_setup_script(str(repo)), text=True, check=True)

    recorded = subprocess.check_output(
        ["git", "rev-parse", "refs/miles/task-baseline"], cwd=repo, text=True
    ).strip()
    assert recorded == baseline
