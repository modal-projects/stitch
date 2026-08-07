"""CPU-only Modal preflight for the Qwen3.6 NVFP4 DFlash experiment.

Run before checkpoint preparation or a GPU server launch:

    uv run --extra modal modal run -e stitch-dev \
      tools/probes/qwen3_6_nvfp4_dflash_preflight.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import modal

from cookbook.common.constants import HF_CACHE_PATH
from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_nvfp4 as model

APP_NAME = "probe-qwen3-6-nvfp4-dflash-preflight"
EXPERIMENT = "qwen3_6_35b_a3b_nvfp4"

app = modal.App(APP_NAME)
trainer_preflight_image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    miles_repo_ref=model.MILES_REPO_REF,
    extra_pip_packages=model.TRAINER_EXTRA_PIP_PACKAGES,
    image_run_commands=model.TRAINER_IMAGE_RUN_COMMANDS,
    source_patches=model.MILES_SOURCE_PATCHES,
)
serving_preflight_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env=model.SGLANG_SERVER_ENV,
)


def _load_miles_module(name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.function(image=trainer_preflight_image, cpu=2, memory=8192, timeout=15 * 60)
def validate_trainer_contract() -> dict:
    import torch

    prep_converter = _load_miles_module(
        "qwen36_prep_converter",
        "/root/miles/tools/convert_hf_to_nvfp4.py",
    )
    live_quantizer = _load_miles_module(
        "qwen36_live_quantizer",
        "/root/miles/miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py",
    )
    gate_up = torch.arange(2 * 8 * 16, dtype=torch.bfloat16).reshape(2, 8, 16)
    down = torch.arange(2 * 16 * 4, dtype=torch.bfloat16).reshape(2, 16, 4)
    params = [
        ("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.language_model.layers.0.mlp.experts.down_proj", down),
    ]
    live = live_quantizer._expand_batched_expert_params(params)
    prep = []
    for name, tensor in params:
        prep.extend(prep_converter._expand_batched_expert_params(name, tensor))

    live_names = [name for name, _tensor in live]
    prep_names = [name for name, _tensor in prep]
    if live_names != prep_names or len(live_names) != 6:
        raise RuntimeError(
            f"prep/live Qwen expert layouts diverged: prep={prep_names}, live={live_names}"
        )
    if live[0][1].shape != (4, 16) or live[-1][1].shape != (16, 4):
        raise RuntimeError("expanded Qwen expert projection shapes are incorrect")

    tail_prefixes = prep_converter._decoder_layer_prefixes(39)
    tail_name = "model.language_model.layers.39.mlp.experts.gate_up_proj"
    if prep_converter.should_quantize(
        tail_name,
        gate_up,
        skip_weight_substrings=tail_prefixes,
    ):
        raise RuntimeError("Qwen's nested final-layer BF16 carve-out was ignored")

    validation_source = Path("/root/miles/miles/utils/arguments.py").read_text()
    if 'speculative_algorithm.upper() != "DFLASH"' not in validation_source:
        raise RuntimeError("Miles does not permit aligned DFlash top-p mask replay")

    result = {
        "status": "PASS",
        "expanded_expert_tensors": len(live_names),
        "tail_prefixes": tail_prefixes,
        "dflash_top_p_validation": "enabled",
    }
    print(
        f"VERDICT qwen36_trainer_preflight PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.function(image=serving_preflight_image, cpu=2, memory=8192, timeout=15 * 60)
def validate_server_args() -> dict:
    from sglang.srt.server_args import ServerArgs

    argv = ["--model-path", "/tmp/qwen3-6-nvfp4"]
    for flag, value in model.SGLANG_SERVER_ARGS.items():
        argv.append(flag)
        if value:
            argv.append(value)

    parser = argparse.ArgumentParser(prog="qwen36-preflight")
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(argv)
    expected = {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": model.DFLASH_MODEL,
        "speculative_dflash_block_size": 8,
        "speculative_draft_attention_backend": "fa4",
        "attention_backend": "trtllm_mha",
        "linear_attn_prefill_backend": "flashinfer",
        "linear_attn_decode_backend": "flashinfer",
        "mamba_radix_cache_strategy": "extra_buffer",
    }
    actual = {name: getattr(parsed, name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"parsed SGLang arguments differ: {actual} != {expected}")

    result = {"status": "PASS", **actual}
    print(
        f"VERDICT qwen36_server_args PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    trainer_result = validate_trainer_contract.remote()
    server_result = validate_server_args.remote()
    result = {
        "status": "PASS",
        "trainer": trainer_result,
        "server": server_result,
    }
    print(
        f"VERDICT qwen36_cpu_preflight PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
