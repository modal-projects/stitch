"""Disagg rollout-pool smoke test on a minimal Qwen model.

Two Modal entrypoints validate the recent pool-control work end to end on a tiny
model (Qwen2.5-0.5B-Instruct), with no Kimi/large config and no Megatron trainer:

  * ``control_plane_test`` (GPU-free) exercises PR #5's explicit pool ownership
    semantics and PR #6's consolidation, against a fake SGLang upstream:
      - ``claim`` resets the pool to base (empty pointer ``<run_id>/weight_v000000``);
      - publishing a real disk-delta chain and reconciling patches the *local*
        checkpoint to the exact trainer-intended bytes (real slime decoder);
      - the monotonic pointer rejects same-run rewinds and a reused run_id
        (fresh-run-id enforcement);
      - a fresh run (a restart) re-claims, resets the engine to base, and replays
        its own chain (1:1 trainer-call <-> pool epoch);
      - every consolidated module + thin adapter imports off the whole-cookbook
        mount (the PR #6 "did consolidation break the runnable flow" regression).

  * ``serving_smoke`` (1x GPU) runs a real SGLang server on tiny Qwen behind the
    *real* consolidated sidecar (``python3 -m cookbook.slime_disagg.sidecar``),
    publishes a delta, and confirms the engine reloads it (real
    ``update_weights_from_disk``) and serves a version-pinned completion.

Run (Modal creds required; the smoke needs one GPU). Invoke by module path
(``-m``), not file path, so the entrypoint resolves to its qualified package
name ``cookbook.disagg_smoke.app`` and the container can import it (a bare file
path names it ``app`` and the remote import fails):

    modal run -m cookbook.disagg_smoke.app::control_plane_test
    modal run -m cookbook.disagg_smoke.app::serving_smoke

``control_plane_test`` is the primary, cheap validation and asserts everything
itself (it raises on failure). ``serving_smoke`` is the live-engine confirmation.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal


APP_NAME = "disagg-smoke"
app = modal.App(APP_NAME)

# Tiny model: single safetensors shard, loads on one GPU in seconds.
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIR = "/cache"
HF_CACHE_DIR = f"{CACHE_DIR}/hf"
BASE_DIR = f"{CACHE_DIR}/qwen-base"

# The decoder's slime checkout, pinned --no-deps exactly like the serving image:
# the sidecar only needs slime.utils.disk_delta (numpy/zstandard/xxhash), never
# Megatron. `disk_delta` is a fork feature (absent from upstream `main`), so the
# default ref is the SAME fork SHA the slime_disagg trainer image pins
# (cookbook/slime_disagg/modal_train.py) — encoder == decoder. Override
# SLIME_SMOKE_REPO/REF to track the trainer if it rolls slime forward.
SLIME_REPO = os.environ.get("SLIME_SMOKE_REPO", "https://github.com/modal-projects/slime.git")
SLIME_REF = os.environ.get("SLIME_SMOKE_REF", "ebfe153949b1a69c39e92f947ed5d475166dd724")
SLIME_ROOT = "/opt/slime"

cache_volume = modal.Volume.from_name("disagg-smoke-cache", create_if_missing=True)

# The whole cookbook package is mounted (not just this subdir) and stitch is
# added as local source, with include_source=False — so these images reproduce
# the exact import surface the consolidated sidecar runs under, and the test
# genuinely fails if the PR #6 whole-cookbook mount regresses.
COOKBOOK_DIR = Path(__file__).parent.parent
_MOUNT_KWARGS = dict(remote_path="/root/cookbook", ignore=["**/__pycache__"])

_DELTA_DEPS = ["numpy", "zstandard", "xxhash", "blake3", "huggingface_hub", "hf_transfer"]
_SIDECAR_DEPS = ["fastapi", "httpx", "uvicorn"]


def _with_slime_and_cookbook(image: modal.Image) -> modal.Image:
    """Clone slime --no-deps (host-side decoder only) + mount the cookbook spine.

    Mirrors ``cookbook/serving.py``: the pool never trains, so Megatron is absent
    and only ``slime.utils.disk_delta`` is importable.
    """
    return (
        image.run_commands(
            f"git clone {SLIME_REPO} {SLIME_ROOT}"
            f" && cd {SLIME_ROOT}"
            f" && git fetch --depth 1 origin {SLIME_REF}"
            f" && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_DIR})
        .add_local_python_source("stitch")
        .add_local_dir(COOKBOOK_DIR, **_MOUNT_KWARGS)
    )


# Control plane needs no GPU and no SGLang — just the decoder + sidecar deps.
control_plane_image = _with_slime_and_cookbook(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*_DELTA_DEPS, *_SIDECAR_DEPS)
)

# Serving smoke runs a real SGLang engine on one GPU. A vanilla single-GPU SGLang
# image (not the cookbook's Blackwell fa4 fork build, which targets B200s) keeps
# the smoke runnable on a common 1x GPU; the whole-cookbook mount + decoder layers
# are identical, so the consolidation/reload path is validated the same way.
serving_image = _with_slime_and_cookbook(
    modal.Image.from_registry("lmsysorg/sglang:v0.5.12")
    .run_commands("rm -rf /root/.cache/huggingface")
    .pip_install(*_DELTA_DEPS, *_SIDECAR_DEPS)
)


def _ensure_base_checkpoint() -> str:
    """Download the tiny Qwen base into the cache volume once (idempotent)."""
    from huggingface_hub import snapshot_download

    if not (Path(BASE_DIR) / "config.json").exists():
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=BASE_DIR,
            ignore_patterns=["*.pt", "*.bin", "*.gguf", "original/*"],
        )
        cache_volume.commit()
    return BASE_DIR


class _FakeSGLangUpstream:
    """In-process stand-in for the SGLang engine control plane.

    Answers exactly the endpoints :class:`SGLangDiskDeltaAdapter` calls during a
    reconcile (flush/pause/continue/update_weights_from_disk) so the GPU-free
    test drives the real manager + real host-side delta apply without a GPU. The
    reload payloads are recorded so the test can assert the engine was reloaded
    at each version.
    """

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.reloads: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # silence access log
                pass

            def _ok(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self._ok({"ok": True})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                if self.path.rstrip("/").endswith("update_weights_from_disk"):
                    outer.reloads.append(json.loads(raw or b"{}"))
                self._ok({"success": True})

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()


def _import_consolidated_modules() -> list[str]:
    """Import every consolidated module + thin adapter off the cookbook mount.

    This is the PR #6 regression check: with include_source=False, these are only
    importable because the whole cookbook package is mounted. A subdir-only mount
    (the bug we fixed) makes the shared spine imports below raise ImportError.
    """
    import importlib

    modules = [
        "cookbook.sidecar",
        "cookbook.serving",
        "cookbook.trainer_helpers",
        "cookbook.rollout_control",
        "cookbook.bulletin_hooks",
        "cookbook.slime_disagg.sidecar",
        "cookbook.slime_disagg.helpers",
        "cookbook.slime_disagg.hooks",
    ]
    for name in modules:
        importlib.import_module(name)
    return modules


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@app.function(
    image=control_plane_image,
    volumes={CACHE_DIR: cache_volume},
    include_source=False,
    timeout=1800,
)
def control_plane_test() -> dict:
    """GPU-free end-to-end test of PR #5 semantics + PR #6 consolidation."""
    import asyncio
    import uuid

    import numpy as np

    from cookbook.disagg_smoke.delta import (
        DeltaPublisher,
        read_local_tensor,
        select_delta_tensors,
    )
    from cookbook.sidecar import build_manager
    from stitch.protocol import PointerRewind

    imported = _import_consolidated_modules()
    base_dir = _ensure_base_checkpoint()
    names = select_delta_tensors(base_dir, count=3)

    bulletin_root = "/tmp/bulletin"
    local_ckpt = "/tmp/local-checkpoint"
    Path(bulletin_root).mkdir(parents=True, exist_ok=True)

    upstream = _FakeSGLangUpstream()

    def tensors_match(directory: str, publisher: DeltaPublisher) -> bool:
        return all(
            np.array_equal(read_local_tensor(directory, n), publisher.expected_tensor(n))
            for n in names
        )

    def tensors_equal_base(base_publisher_view: dict) -> bool:
        return all(
            np.array_equal(read_local_tensor(local_ckpt, n), base_publisher_view[n]) for n in names
        )

    async def run() -> dict:
        manager = build_manager(
            upstream_url=upstream.url,
            bulletin_root=bulletin_root,
            local_checkpoint_dir=local_ckpt,
            base_checkpoint_dir=base_dir,
            disk_delta_module="slime.utils.disk_delta",
            inject_apply_deltas=False,
            volume_name="",
            commit_mode="in_place",
        )
        board = manager.board

        # Base bytes captured before any claim, to verify reset-to-base later.
        base_bytes = {n: read_local_tensor(base_dir, n) for n in names}

        # --- PR #5: claim resets the pool to the empty pointer (base) ---
        run_a = uuid.uuid4().hex
        claim_a = board.claim(run_a)
        _assert(claim_a.reset and claim_a.version == 0, f"claim should reset to base: {claim_a}")
        _assert(board.read_latest() == (run_a, 0), f"pointer not at claim: {board.read_latest()}")

        # Startup converges to the claim: base materialized, no deltas yet.
        await manager.startup_sync()
        _assert(
            manager.current_run_id == run_a and manager.current_version == 0,
            f"startup not at claimed base: {manager.current_run_id}/{manager.current_version}",
        )
        _assert(tensors_equal_base(base_bytes), "local checkpoint != base after claim")

        # --- PR #5 + #6: publish a real delta chain, reconcile, verify bytes ---
        pub_a = DeltaPublisher(base_dir, f"{bulletin_root}/{run_a}", names)
        board.advance(run_a, pub_a.publish_next())
        board.advance(run_a, pub_a.publish_next())
        await manager.sync_to()
        _assert(manager.current_version == 2, f"did not reconcile to v2: {manager.current_version}")
        _assert(tensors_match(local_ckpt, pub_a), "local checkpoint != trainer-intended v2 bytes")
        _assert(len(upstream.reloads) >= 1, "engine was never reloaded")
        _assert(
            upstream.reloads[-1]["weight_version"] == "2"
            and upstream.reloads[-1]["model_path"] == local_ckpt,
            f"unexpected reload payload: {upstream.reloads[-1]}",
        )

        # --- PR #5: monotonic pointer rejects rewind + reused run_id ---
        rewind_rejected = False
        try:
            board.advance(run_a, 1)
        except PointerRewind:
            rewind_rejected = True
        _assert(rewind_rejected, "same-run rewind was not rejected")

        reused_run_rejected = False
        try:
            board.claim(run_a)  # reused run_id (a restart that forgot to refresh the epoch)
        except PointerRewind:
            reused_run_rejected = True
        _assert(reused_run_rejected, "re-claiming a live run_id was not rejected")

        # --- PR #5: a fresh run (restart) re-claims, resets to base, replays ---
        run_b = uuid.uuid4().hex
        claim_b = board.claim(run_b)
        _assert(claim_b.reset and claim_b.version == 0, f"fresh-run claim should reset: {claim_b}")
        await manager.sync_to()
        _assert(
            manager.current_run_id == run_b and manager.current_version == 0,
            f"did not switch to fresh run base: {manager.current_run_id}/{manager.current_version}",
        )
        _assert(tensors_equal_base(base_bytes), "engine not reset to base on run switch")

        pub_b = DeltaPublisher(base_dir, f"{bulletin_root}/{run_b}", names)
        board.advance(run_b, pub_b.publish_next())
        await manager.sync_to()
        _assert(manager.current_version == 1, f"fresh run did not reach v1: {manager.current_version}")
        _assert(tensors_match(local_ckpt, pub_b), "fresh-run local checkpoint != v1 bytes")

        return {
            "run_a": run_a,
            "run_b": run_b,
            "delta_tensors": names,
            "engine_reloads": len(upstream.reloads),
        }

    try:
        result = asyncio.run(run())
    finally:
        upstream.close()

    result["imported_modules"] = imported
    print("control_plane_test PASSED:", json.dumps(result, indent=2))
    return result


def _wait_for_http(url: str, *, timeout: float, what: str) -> None:
    import httpx

    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=5.0).status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {what} at {url}: {last}")


@app.function(
    image=serving_image,
    gpu="H100",
    volumes={CACHE_DIR: cache_volume},
    include_source=False,
    timeout=2400,
)
def serving_smoke() -> dict:
    """1-GPU live confirmation: real SGLang reload + version-pinned completion.

    Runs the real consolidated sidecar (``python3 -m cookbook.slime_disagg.sidecar``)
    in front of a real SGLang server, publishes one delta, triggers a reconcile,
    and asserts the engine reloaded to v1 and serves a request pinned to v1.
    """
    import httpx

    base_dir = _ensure_base_checkpoint()
    from cookbook.disagg_smoke.delta import DeltaPublisher, select_delta_tensors

    _import_consolidated_modules()
    names = select_delta_tensors(base_dir, count=3)

    bulletin_root = "/tmp/bulletin"
    local_ckpt = "/tmp/local-checkpoint"
    Path(bulletin_root).mkdir(parents=True, exist_ok=True)
    run_id = "smoke-" + str(int(time.time()))

    sglang_port = 30000
    sidecar_port = 8000
    sglang_url = f"http://127.0.0.1:{sglang_port}"
    sidecar_url = f"http://127.0.0.1:{sidecar_port}"

    sglang = subprocess.Popen(
        [
            "python3", "-m", "sglang.launch_server",
            "--model-path", base_dir,
            "--host", "127.0.0.1",
            "--port", str(sglang_port),
            "--mem-fraction-static", "0.7",
            "--attention-backend", "torch_native",
        ]
    )
    sidecar = subprocess.Popen(
        [
            "python3", "-m", "cookbook.slime_disagg.sidecar",
            "--upstream-url", sglang_url,
            "--host", "127.0.0.1",
            "--port", str(sidecar_port),
            "--bulletin-root", bulletin_root,
            "--base-checkpoint-dir", base_dir,
            "--local-checkpoint-dir", local_ckpt,
            "--run-id", run_id,
            "--commit-mode", "quiesce",
        ],
        env={**os.environ, "PYTHONPATH": "/root"},
    )
    try:
        _wait_for_http(f"{sglang_url}/health", timeout=900, what="sglang server")
        _wait_for_http(f"{sidecar_url}/health", timeout=120, what="sidecar")

        # Drive the pool the way the trainer does: claim this run (resets the
        # pool to base), publish a real delta, then ask the sidecar to reconcile.
        from stitch.bulletin import FilesystemBulletinBoard

        board = FilesystemBulletinBoard(bulletin_root, layout="slime")
        board.claim(run_id)
        publisher = DeltaPublisher(base_dir, f"{bulletin_root}/{run_id}", names)
        board.advance(run_id, publisher.publish_next())
        httpx.post(f"{sidecar_url}/rpc_sync_from_bulletin_board", json={"target_version": 1}, timeout=60)

        deadline = time.time() + 600
        current = -1
        while time.time() < deadline:
            info = httpx.get(f"{sidecar_url}/server_info", timeout=30).json()
            current = int(info["current_version"])
            if current >= 1 and info["sync_state"] == "idle":
                break
            time.sleep(3.0)
        _assert(current == 1, f"engine did not reload to v1 (current={current})")

        # Version-pinned completion: requires the engine to be at >= v1.
        completion = httpx.post(
            f"{sidecar_url}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                "weight_version": {"min_required_version": 1},
            },
            timeout=120,
        )
        _assert(completion.status_code == 200, f"pinned completion failed: {completion.status_code}")
        meta = completion.json().get("meta_info", {})
        _assert(
            str(meta.get("weight_version")) == "1",
            f"completion not served at v1: {meta.get('weight_version')}",
        )
        result = {"run_id": run_id, "current_version": current, "completion_meta": meta}
        print("serving_smoke PASSED:", json.dumps(result, indent=2))
        return result
    finally:
        for proc in (sidecar, sglang):
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
