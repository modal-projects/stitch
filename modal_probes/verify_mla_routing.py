"""M0 verify gate for the async-MoE disaggregated demo.

The cheap question this answers before any training config is written: does the
SGLang build pinned in this repo's slime image serve a DeepSeek-V3-architecture
(multi-head latent attention) MoE *and* return per-token routed-expert ids?

Why this is the one gate worth running first. Kimi K2.6 is a DeepSeek-V3-arch
model -- MLA (``q_a/q_b`` + ``kv_a_proj_with_mqa``/``kv_b`` projections) plus
DeepSeek-MoE (sigmoid router, shared experts, grouped top-k). The disk-delta
weight sync and routing replay (``--use-rollout-routing-replay``) are already
validated on the GLM-4.x MoEs in bf16 and fp8 -- but GLM-4.x is GQA, not MLA.
MLA is the single axis those runs never exercised. Moonlight-16B-A3B is
Moonshot's small same-family proxy (MLA + DeepSeek-MoE, *real* HF weights), so
if SGLang here serves it and emits ``meta_info['routed_experts']``, the
routing-replay rollout path works for the Kimi proxy too.

Scope: this checks the *serving* axis only. It deliberately needs real HF
weights, which is why it serves Moonlight rather than the deepseek-v3-5layer
arch wrapper (that wrapper is a Megatron arch config with no HF checkpoint to
serve). The *trainer-side* MLA round-trip -- bridge-mode HF load into Megatron
and disk-delta export producing HF keys that match the served safetensors -- is
exercised by the first M1 bring-up run, where deepseek-v3-5layer is the fast
iteration vehicle.

    alias m="uv run --extra modal modal"
    m run -m modal_probes.verify_mla_routing::download_model
    m run -m modal_probes.verify_mla_routing::verify

PASS prints the SGLang version and a routed-experts tensor that reshapes to
[tokens, num_layers, moe_router_topk] with expert ids inside [0, num_experts).
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

# The pinned slime image carries the SGLang build under test; serving from it is
# what makes this check meaningful. We add nothing to the sglang wheel.
SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
HF_CACHE_PATH = "/root/.cache/huggingface"

# Moonlight-16B-A3B: small DeepSeek-V3-arch (MLA) MoE with real weights. Override
# VERIFY_MODEL to point at another DeepSeek/Kimi-family HF checkpoint.
MODEL_NAME = os.environ.get("VERIFY_MODEL", "moonshotai/Moonlight-16B-A3B-Instruct")

MINUTES = 60
SGLANG_PORT = 8001
STARTUP_TIMEOUT = 25 * MINUTES

image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    # The base image bakes in an HF cache; remove it so it cannot shadow the
    # cache volume mounted at the same path.
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("verify-mla-routing")
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_NAME)
    hf_cache_volume.commit()
    print(f"Downloaded {MODEL_NAME} into the huggingface-cache volume.")


@app.function(
    image=image,
    gpu="H200:1",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    timeout=30 * MINUTES,
)
def verify() -> None:
    """Serve the proxy model on the pinned SGLang and assert routed-expert output."""
    import numpy as np
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(MODEL_NAME, local_files_only=True)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    num_layers = int(config["num_hidden_layers"])
    topk = int(config["num_experts_per_tok"])
    num_experts = int(config.get("n_routed_experts") or config.get("num_experts") or 0)
    print(
        f"Serving {MODEL_NAME}: num_hidden_layers={num_layers}, "
        f"num_experts_per_tok={topk}, n_routed_experts={num_experts or '?'}"
    )

    proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_dir,
            "--port",
            str(SGLANG_PORT),
            "--trust-remote-code",
            "--enable-return-routed-experts",  # the capability under test
            "--mem-fraction-static",
            "0.85",
            "--cuda-graph-max-bs",
            "8",
        ],
        start_new_session=True,
    )
    try:
        info = _wait_ready(proc, STARTUP_TIMEOUT)
        print(
            f"SGLang up: version={info.get('version')!r} "
            f"attention_backend={info.get('attention_backend')!r}"
        )

        payload = {
            "text": "Give me a one-sentence fun fact about the Moon.",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 24},
            "return_logprob": True,
            "return_routed_experts": True,  # what the rollout path sends
        }
        out = _post_json(
            f"http://127.0.0.1:{SGLANG_PORT}/generate", payload, timeout=180
        )
        meta = out["meta_info"]

        if "routed_experts" not in meta:
            raise SystemExit(
                "FAIL: meta_info has no 'routed_experts'. The pinned SGLang build does not emit "
                "routed experts for this MLA model -- routing replay for the Kimi proxy is blocked "
                "at the engine. (Confirm --enable-return-routed-experts is supported by this build.)"
            )

        # Mirror the slime rollout decode exactly: base64 -> int32, reshape to
        # [tokens, num_layers, moe_router_topk] (see slime sglang_rollout.generate).
        flat = np.frombuffer(
            base64.b64decode(meta["routed_experts"].encode("ascii")), dtype=np.int32
        )
        per_layer_topk = num_layers * topk
        if per_layer_topk == 0 or flat.size % per_layer_topk != 0:
            raise SystemExit(
                f"FAIL: routed_experts size {flat.size} is not divisible by "
                f"num_layers*topk ({num_layers}*{topk}); the rollout reshape contract is broken."
            )
        tokens = flat.size // per_layer_topk
        matrix = flat.reshape(tokens, num_layers, topk)

        completion = meta.get("completion_tokens")
        prompt = meta.get("prompt_tokens")
        print(
            f"routed_experts: {flat.size} int32 -> [tokens={tokens}, layers={num_layers}, topk={topk}] "
            f"(prompt_tokens={prompt}, completion_tokens={completion})"
        )
        lo, hi = int(matrix.min()), int(matrix.max())
        print(
            f"  expert id range [{lo}, {hi}]; sample (token 0, layer 0): {matrix[0, 0].tolist()}"
        )
        if num_experts and not (0 <= lo and hi < num_experts):
            raise SystemExit(
                f"FAIL: expert ids [{lo}, {hi}] fall outside [0, {num_experts}); "
                "the routed-experts payload is malformed for this arch."
            )
        print(
            "PASS: pinned SGLang serves the MLA MoE and emits well-formed routed_experts."
        )
    finally:
        _terminate(proc)


def _wait_ready(proc: subprocess.Popen, timeout: int) -> dict:
    """Poll until SGLang answers server_info; return it (carries the version)."""
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"FAIL: sglang.launch_server exited early (code={proc.returncode})."
            )
        for endpoint in ("/get_server_info", "/server_info"):
            try:
                return _get_json(
                    f"http://127.0.0.1:{SGLANG_PORT}{endpoint}", timeout=10
                )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = f"{endpoint}: {type(exc).__name__}: {exc}"
        time.sleep(5)
    raise SystemExit(
        f"FAIL: SGLang did not become ready in {timeout}s; last error: {last_error}"
    )


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
