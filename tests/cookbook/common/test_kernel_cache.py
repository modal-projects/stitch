import subprocess
from importlib import metadata

import pytest

from cookbook.common import kernel_cache


def _nvidia_smi(stdout: str):
    def run(cmd, **_kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("10.3\n", "sm103"),
        ("10.0\n10.0\n10.0\n", "sm100"),
        ("9.0\n", "sm90"),
    ],
)
def test_compute_capability_from_nvidia_smi(monkeypatch, stdout, expected):
    monkeypatch.setattr(kernel_cache.subprocess, "run", _nvidia_smi(stdout))

    assert kernel_cache.compute_capability() == expected


def test_compute_capability_without_a_gpu_falls_back(monkeypatch, capsys):
    def run(cmd, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(kernel_cache.subprocess, "run", run)

    assert kernel_cache.compute_capability() == kernel_cache.UNKNOWN_GPU
    assert "compute capability" in capsys.readouterr().out


def test_cache_key_separates_arch_and_toolchain():
    key = kernel_cache.cache_key(
        compute_capability="sm103",
        torch_version="2.11.0+cu130",
        triton_version="3.6.0",
    )

    assert key == "sm103/torch-2.11.0+cu130-triton-3.6.0"


def test_environment_creates_keyed_cache_dirs_under_root(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_cache.subprocess, "run", _nvidia_smi("10.3\n"))
    versions = {"torch": "2.11.0+cu130", "triton": "3.6.0"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])

    env = kernel_cache.environment(tmp_path)

    base = tmp_path / "sm103" / "torch-2.11.0+cu130-triton-3.6.0"
    assert env == {
        "TRITON_CACHE_DIR": str(base / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(base / "inductor"),
    }
    assert (base / "triton").is_dir()
    assert (base / "inductor").is_dir()
