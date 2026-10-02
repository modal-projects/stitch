from __future__ import annotations

import subprocess

import pytest

from cookbook.miles_disagg.swebench_pro import (
    _parse_string_list,
    _patched_paths,
    _setup_script,
    _verifier_script,
)


def test_parse_string_list_rejects_non_string_items() -> None:
    assert _parse_string_list("['a', 'b']", "tests", "task") == ["a", "b"]
    with pytest.raises(TypeError, match="list of strings"):
        _parse_string_list("['a', 1]", "tests", "task")


def test_patched_paths_are_unique_and_sorted() -> None:
    patch = "\n".join(
        [
            "diff --git a/z.py b/z.py",
            "diff --git a/a.py b/a.py",
            "diff --git a/z.py b/z.py",
        ]
    )
    assert _patched_paths(patch) == ["a.py", "z.py"]


@pytest.mark.parametrize(
    "script",
    [_setup_script("true"), _verifier_script(["test_a.py", "test b.py"])],
)
def test_generated_shell_is_valid(script: str) -> None:
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_generated_verifier_python_is_valid() -> None:
    script = _verifier_script(["test_a.py"])
    source = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(source, "verifier.py", "exec")


def test_setup_gives_login_shells_the_image_path(tmp_path) -> None:
    """The restoring lines run against a stand-in for PID 1's environment and
    profile directory, then a login-style shell reads the result."""
    environ = tmp_path / "environ"
    environ.write_bytes(b"HOME=/root\0PATH=/go/bin:/usr/local/go/bin:/usr/bin:/bin\0")
    profile = tmp_path / "profile.d"
    profile.mkdir()
    script = _setup_script("true")
    restore = script[: script.index("cd /app")]
    restore = restore.replace("/proc/1/environ", str(environ)).replace(
        "/etc/profile.d", str(profile)
    )
    subprocess.run(["bash", "-c", restore], check=True)

    path = subprocess.run(
        ["bash", "-c", f"PATH=/usr/bin:/bin; . {profile}/zz-image-path.sh; echo $PATH"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert path == "/go/bin:/usr/local/go/bin:/usr/bin:/bin"
