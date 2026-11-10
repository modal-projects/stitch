"""The trainer's hook: each node's first rank commits the run's Volume."""

from types import SimpleNamespace

import stitch.trainer
from stitch.trainer import commit_versions


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
