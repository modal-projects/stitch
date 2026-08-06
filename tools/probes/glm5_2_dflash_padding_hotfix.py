"""Validate the GLM-5.2 DFlash attention-TP padding image hotfix on Modal."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import modal

from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DRAFT_PATH,
    HF_CACHE_PATH,
    SGLANG_CACHE_PATH,
)
from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg.configs import glm5_2_nvfp4 as model

APP_NAME = "probe-glm5-2-dflash-padding-hotfix"
PORT = 8000

app = modal.App(APP_NAME)
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints", create_if_missing=False, version=2
)
draft_volume = modal.Volume.from_name(
    model.DFLASH_VOLUME, create_if_missing=False, version=2
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)

serving_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment="glm5_2_nvfp4",
    extra_env=model.SGLANG_SERVER_ENV,
)


def _probe_server_args() -> dict[str, str]:
    args = dict(model.SGLANG_SERVER_ARGS)
    # Weight staging is orthogonal to prefill alignment and would add a
    # model-sized background initialization to this focused regression.
    for flag in (
        "--enable-cpu-weight-cache",
        "--cpu-weight-cache-max-compile-group-gb",
        "--cpu-weight-cache-canonical-checkpoint-dir",
    ):
        args.pop(flag, None)
    return args


@app.function(image=serving_image, cpu=2, memory=4096, timeout=15 * 60)
def preflight() -> str:
    """Confirm that the source patch is present and rejects non-padding gaps."""
    import torch
    from sglang.srt.speculative.dflash_worker_v2 import (
        _trim_dflash_prefill_hidden_states,
    )

    hidden_states = torch.arange(16).reshape(8, 2)
    trimmed = _trim_dflash_prefill_hidden_states(
        hidden_states,
        cache_loc_tokens=5,
        attention_tp_size=4,
    )
    if not torch.equal(trimmed, hidden_states[:5]):
        raise RuntimeError("attention-TP padding rows were not trimmed")

    try:
        _trim_dflash_prefill_hidden_states(
            hidden_states[:6],
            cache_loc_tokens=5,
            attention_tp_size=4,
        )
    except ValueError as exc:
        rejection = str(exc)
    else:
        raise RuntimeError("non-padding shape mismatch was accepted")

    verdict = {
        "status": "PASS",
        "input_rows": 8,
        "output_rows": int(trimmed.shape[0]),
        "invalid_shape_rejection": rejection,
    }
    rendered = json.dumps(verdict, sort_keys=True)
    print(f"VERDICT dflash_padding_preflight PASS {rendered}", flush=True)
    return rendered


def _request(prompt_tokens: int, request_id: int) -> dict[str, Any]:
    # Distinct non-special token sequences prevent radix-prefix reuse from
    # hiding the exact prefill length under test.
    input_ids = [1000 + request_id * 100 + index % 97 for index in range(prompt_tokens)]
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "sampling_seed": 20260807 + request_id,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "return_sampling_mask": True,
        "return_routed_experts": True,
        "stream": False,
    }


async def _run_requests() -> list[dict[str, Any]]:
    import httpx

    lengths = (65, 66, 67, 16101)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}", timeout=10 * 60, trust_env=False
    ) as client:
        responses = await asyncio.gather(
            *(
                client.post("/generate", json=_request(length, index))
                for index, length in enumerate(lengths)
            )
        )

    summaries = []
    for prompt_tokens, response in zip(lengths, responses, strict=True):
        try:
            body = response.json()
        except ValueError:
            body = {"text": response.text}
        if response.status_code != 200:
            raise RuntimeError(
                f"prompt_tokens={prompt_tokens} failed with "
                f"status={response.status_code}: {body}"
            )
        output_ids = body.get("output_ids") or []
        meta = body.get("meta_info") or {}
        if len(output_ids) != 8:
            raise RuntimeError(
                f"prompt_tokens={prompt_tokens} returned {len(output_ids)} tokens"
            )
        if len(meta.get("output_token_sampling_logprobs") or []) != 8:
            raise RuntimeError(
                f"prompt_tokens={prompt_tokens} returned unaligned sampling logprobs"
            )
        if len(meta.get("output_token_sampling_mask") or []) != 8:
            raise RuntimeError(
                f"prompt_tokens={prompt_tokens} returned unaligned sampling masks"
            )
        if not meta.get("routed_experts"):
            raise RuntimeError(
                f"prompt_tokens={prompt_tokens} did not return routed experts"
            )
        summaries.append(
            {
                "prompt_tokens": prompt_tokens,
                "attention_tp_padding": (-prompt_tokens) % 4,
                "output_tokens": len(output_ids),
            }
        )
    return summaries


@app.function(
    image=serving_image,
    gpu=f"{model.modal.rollout_gpu}:{model.ROLLOUT_GPUS_PER_ENGINE}",
    cpu=64,
    memory=model.modal.rollout_memory_mib,
    ephemeral_disk=model.modal.rollout_ephemeral_disk_mib,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        str(DRAFT_PATH): draft_volume.read_only(),
        SGLANG_CACHE_PATH: sglang_cache_volume,
    },
    timeout=3 * 60 * 60,
)
def run_probe() -> str:
    """Run concurrent TP4 prefills covering every nonzero padding suffix."""
    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    endpoint = SGLangEndpoint(
        model_path=str(model.ROLLOUT_CHECKPOINT_PATH),
        worker_port=PORT,
        tp=model.ROLLOUT_GPUS_PER_ENGINE,
        extra_server_args=_probe_server_args(),
        health_timeout=60 * 60,
        health_poll_interval=10,
        log_requests_level=-1,
    )
    result: dict[str, Any] = {"status": "RUNNING"}
    try:
        print("PROGRESS phase=server_start", flush=True)
        endpoint.start()
        print("PROGRESS phase=server_ready", flush=True)
        result["requests"] = asyncio.run(_run_requests())
        health = httpx.get(
            f"http://127.0.0.1:{PORT}/health", timeout=30, trust_env=False
        )
        if health.status_code != 200:
            raise RuntimeError(f"post-request health returned {health.status_code}")
        result["status"] = "PASS"
        rendered = json.dumps(result, sort_keys=True)
        print(f"VERDICT dflash_padding_gpu PASS {rendered}", flush=True)
        return rendered
    except BaseException as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"VERDICT dflash_padding_gpu FAIL {json.dumps(result, sort_keys=True)}",
            flush=True,
        )
        raise
    finally:
        endpoint.stop()
