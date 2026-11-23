"""The trainer's hook: each node's first rank commits the run's Volume."""

import os
from types import SimpleNamespace

import stitch.trainer
from stitch.trainer import commit_adapters, commit_versions


def test_each_write_commits_the_volume_the_environment_names(monkeypatch):
    commits = []
    monkeypatch.setenv("STITCH_VERSIONS_VOLUME", "runs")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(
        stitch.trainer.modal.Volume,
        "from_name",
        lambda name: SimpleNamespace(commit=lambda: commits.append(name)),
    )

    commit_versions(None, "/data/r1/versions/weight_v000003", [])
    commit_versions(None, "/data/r1/versions", [])

    assert commits == ["runs", "runs"]


def test_only_each_nodes_first_rank_commits(monkeypatch):
    commits = []
    monkeypatch.setenv("STITCH_VERSIONS_VOLUME", "runs")
    monkeypatch.setattr(
        stitch.trainer.modal.Volume,
        "from_name",
        lambda name: SimpleNamespace(commit=lambda: commits.append(name)),
    )

    for local_rank in ("0", "3"):
        monkeypatch.setenv("LOCAL_RANK", local_rank)
        commit_versions(None, "/data/r1/versions/weight_v000003", [])

    assert commits == ["runs"]


def _export(directory, written_ns: int) -> str:
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(b"lora")
    for path in (directory / "adapter_model.safetensors", directory):
        os.utime(path, ns=(written_ns, written_ns))
    return str(directory)


def test_a_sampler_export_needs_no_commit_once_a_later_commit_covers_it(
    tmp_path, monkeypatch
):
    commits = []
    monkeypatch.setenv("STITCH_ADAPTERS_VOLUME", "adapters")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(
        stitch.trainer.modal.Volume,
        "from_name",
        lambda name: SimpleNamespace(commit=lambda: commits.append(name)),
    )
    monkeypatch.setattr(
        stitch.trainer,
        "_ADAPTER_COMMITS",
        stitch.trainer._Commits("STITCH_ADAPTERS_VOLUME"),
    )
    monkeypatch.setattr(stitch.trainer, "time", SimpleNamespace(time_ns=lambda: 2_000))
    first, second = _export(tmp_path / "a", 1_000), _export(tmp_path / "b", 1_500)
    late = _export(tmp_path / "c", 3_000)

    commit_adapters(None, first, [])
    commit_adapters(None, second, [])
    assert commits == ["adapters"], (
        "the first commit started after the second export was written"
    )

    commit_adapters(None, late, [])
    assert commits == ["adapters", "adapters"]
