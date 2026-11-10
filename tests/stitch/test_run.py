"""A run's config: what the app's containers rebuild it from, and one config per run id."""

import dataclasses
from types import SimpleNamespace

import pytest

import stitch.run
from stitch.control import Clocks
from stitch.run import PoolConfig, Run, RunConfig, create_run
from stitch.sglang import SGLangConfig

CONFIG = RunConfig(
    run_id="r1",
    image="me/sglang:1",
    volumes={"/checkpoints": "checkpoints", "/data": "runs"},
    checkpoint="/checkpoints/base",
    versions="/data/r1/versions",
    engine=SGLangConfig(
        server_args={"--tp-size": "4"}, env={"A": "1"}, pause_mode="retract"
    ),
    pool=PoolConfig(gpu="H200:4", cpu=16, memory=262144),
    image_commands=["pip install my-plugin"],
    clocks=Clocks(patience=60),
)


def test_a_run_config_comes_back_whole_from_its_json():
    assert RunConfig.from_json(CONFIG.to_json()) == CONFIG


def test_a_run_id_already_deployed_with_another_config_is_refused(monkeypatch):
    deployed = dataclasses.replace(CONFIG, clocks=Clocks()).to_json()
    control = SimpleNamespace(run_config=SimpleNamespace(remote=lambda: deployed))
    monkeypatch.setattr(
        stitch.run.modal.Cls, "from_name", lambda app_name, name: lambda: control
    )

    with pytest.raises(ValueError, match="a new config is a new run id"):
        create_run(CONFIG)


def test_a_run_reports_its_control_service_status(monkeypatch):
    status = {"head": None, "serving": {}, "catching_up": 2}
    control = SimpleNamespace(status=SimpleNamespace(remote=lambda: status))
    monkeypatch.setattr(
        stitch.run.modal.Cls,
        "from_name",
        lambda app_name, name: (
            (lambda: control) if (app_name, name) == ("stitch-r1", "Control") else None
        ),
    )

    assert (
        Run(
            url="https://control", generate_url="https://engines", app_name="stitch-r1"
        ).status()
        == status
    )


def test_stopping_a_run_already_stopped_changes_nothing(monkeypatch):
    def lookup(app_name):
        raise stitch.run.modal.exception.NotFoundError(f"App '{app_name}' not found")

    monkeypatch.setattr(stitch.run.modal.App, "lookup", lookup)

    Run(
        url="https://control", generate_url="https://engines", app_name="stitch-r1"
    ).stop()


def test_the_versions_volume_is_the_one_mounted_closest_above_them():
    nested = dataclasses.replace(
        CONFIG, volumes={"/data": "runs", "/data/r1": "r1-versions"}
    )
    assert CONFIG.versions_volume == "runs"
    assert nested.versions_volume == "r1-versions"


def _deployed(monkeypatch, config):
    """create_run finds `config` deployed, so it only looks up the run's URLs."""
    control = SimpleNamespace(
        run_config=SimpleNamespace(remote=config.to_json),
        api=SimpleNamespace(get_web_url=lambda: "https://control"),
    )
    monkeypatch.setattr(
        stitch.run.modal.Cls, "from_name", lambda app_name, name: lambda: control
    )
    monkeypatch.setattr(
        stitch.run.modal.Server,
        "from_name",
        lambda app_name, name: SimpleNamespace(get_url=lambda: "https://engines"),
    )


def test_versions_on_the_checkpoints_volume_warn(monkeypatch, recwarn):
    shared = dataclasses.replace(CONFIG, versions="/checkpoints/r1/versions")
    _deployed(monkeypatch, shared)

    with pytest.warns(UserWarning, match="share the Volume checkpoints"):
        create_run(shared)

    _deployed(monkeypatch, CONFIG)
    recwarn.clear()
    create_run(CONFIG)
    assert not recwarn.list
