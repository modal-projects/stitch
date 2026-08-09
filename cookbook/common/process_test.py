from __future__ import annotations

from cookbook.common import process, storage


def test_start_sidecar_passes_s3_store_settings(monkeypatch) -> None:
    commands: list[list[str]] = []

    def popen(command: list[str], **_kwargs):
        commands.append(command)
        return object()

    monkeypatch.setattr(process.subprocess, "Popen", popen)

    process.start_sidecar(
        sidecar_port=8000,
        sglang_port=8001,
        bulletin_root="/cache/run-a",
        base_checkpoint_dir="/model",
        local_checkpoint_dir=None,
        delta_update_mode="cpu",
        disk_load_format="auto",
        store_backend=storage.S3,
        volume_name="",
        s3_root="s3://bucket/experiment/run-a",
        s3_endpoint_url="https://s3.example.test",
        run_id="run-a",
        boot_version=0,
        commit_mode="in_place",
    )

    command = commands[0]
    assert command[command.index("--store-backend") + 1] == storage.S3
    assert command[command.index("--s3-root") + 1] == ("s3://bucket/experiment/run-a")
    assert command[command.index("--s3-endpoint-url") + 1] == (
        "https://s3.example.test"
    )
    assert command[command.index("--boot-version") + 1] == "0"
