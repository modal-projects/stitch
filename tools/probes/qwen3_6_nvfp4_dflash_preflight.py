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
from types import SimpleNamespace

import modal

from cookbook.common.constants import HF_CACHE_PATH
from cookbook.common.serving_image import DEFAULT_SGLANG_RUNTIME, build_serving_image
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

    tail = live_quantizer.quantize_params_nvfp4(
        SimpleNamespace(
            first_last_layers_bf16=True,
            num_layers=40,
            num_layers_at_start_in_bf16=0,
            num_layers_at_end_in_bf16=6,
            extra_high_precision_layers_megatron=(),
            fp4_param=False,
            fp4_param_gather=False,
        ),
        "decoder.layers.39.mlp.experts.linear_fc1.weight0",
        params,
        {"quant_algo": "NVFP4", "ignore": []},
    )
    tail_names = [name for name, _tensor in tail]
    if tail_names != live_names:
        raise RuntimeError(f"live BF16 tail kept fused expert tensors: {tail_names}")

    tail_prefixes = prep_converter._decoder_layer_prefixes(39)
    tail_name = "model.language_model.layers.39.mlp.experts.gate_up_proj"
    if prep_converter.should_quantize(
        tail_name,
        gate_up,
        skip_weight_substrings=tail_prefixes,
    ):
        raise RuntimeError("Qwen's nested final-layer BF16 carve-out was ignored")
    prep_source = Path("/root/miles/tools/convert_hf_to_nvfp4.py").read_text()
    if (
        "q_weights.update(dict(_expand_batched_expert_params(key, tensor)))"
        not in prep_source
    ):
        raise RuntimeError("prepared BF16 tail does not emit per-expert tensors")
    expert_aliases = prep_converter._batched_expert_module_aliases(tail_name)
    if expert_aliases != (
        "model.language_model.layers.39.mlp.experts",
        "model.layers.39.mlp.experts",
    ):
        raise RuntimeError(
            f"BF16 FusedMoE ignore aliases are incorrect: {expert_aliases}"
        )

    a_log = torch.ones(32, dtype=torch.bfloat16)
    normalized_a_log = prep_converter._normalize_checkpoint_dtype(
        "model.language_model.layers.0.linear_attn.A_log", a_log
    )
    if normalized_a_log.dtype != torch.float32:
        raise RuntimeError(
            "prepared Qwen A_log does not match the trainer's fp32 dtype"
        )
    ordinary = prep_converter._normalize_checkpoint_dtype(
        "model.language_model.layers.0.linear_attn.dt_bias", a_log
    )
    if ordinary.dtype != torch.bfloat16:
        raise RuntimeError("checkpoint dtype normalization changed an unrelated tensor")

    validation_source = Path("/root/miles/miles/utils/arguments.py").read_text()
    if 'speculative_algorithm.upper() != "DFLASH"' not in validation_source:
        raise RuntimeError("Miles does not permit aligned DFlash top-p mask replay")

    converter_source = Path("/root/miles/tools/convert_hf_to_torch_dist.py").read_text()
    if 'not os.environ.get("CONVERT_KEEP_PP1")' not in converter_source:
        raise RuntimeError("Miles converter does not expose the PP1 preservation gate")
    if model.PREP_ENV.get("CONVERT_KEEP_PP1") != "1":
        raise RuntimeError("Qwen torch-dist prep does not preserve PP1 with EP8")

    result = {
        "status": "PASS",
        "expanded_expert_tensors": len(live_names),
        "bf16_expert_layout": "per_expert",
        "bf16_fused_moe_ignore": "exact_aliases",
        "a_log_dtype": "float32",
        "tail_prefixes": tail_prefixes,
        "dflash_top_p_validation": "enabled",
        "torch_dist_topology": "PP1_EP8",
    }
    print(
        f"VERDICT qwen36_trainer_preflight PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.function(image=serving_preflight_image, cpu=2, memory=8192, timeout=15 * 60)
def validate_server_args() -> dict:
    import inspect

    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.dflash_utils import DFlashSamplingMaskOutput

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
        "speculative_draft_model_quantization": "unquant",
        "speculative_dflash_block_size": 8,
        "speculative_draft_attention_backend": "fa4",
        "attention_backend": "trtllm_mha",
        "linear_attn_prefill_backend": "flashinfer",
        "linear_attn_decode_backend": "flashinfer",
        "mamba_ssm_dtype": "bfloat16",
        "mamba_radix_cache_strategy": "extra_buffer",
    }
    actual = {name: getattr(parsed, name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"parsed SGLang arguments differ: {actual} != {expected}")

    expected_sglang_commit = "be31674effe497f6b3d4449ca2687088865d58c3"
    if DEFAULT_SGLANG_RUNTIME.commit != expected_sglang_commit:
        raise RuntimeError(
            "serving image SGLang pin differs: "
            f"{DEFAULT_SGLANG_RUNTIME.commit} != {expected_sglang_commit}"
        )

    dflash_source = inspect.getsource(DFlashSamplingMaskOutput)
    required_dflash_fragments = (
        "map_device_tensors",
        "support_mask",
        "finalize",
    )
    missing_dflash = [
        fragment
        for fragment in required_dflash_fragments
        if fragment not in dflash_source
    ]
    if missing_dflash:
        raise RuntimeError(
            f"SGLang is missing deferred DFlash mask handling: {missing_dflash}"
        )

    chat_source = inspect.getsource(OpenAIServingChat._convert_to_internal_request)
    if "miles_return_sampling_mask" not in chat_source:
        raise RuntimeError(
            "SGLang is missing Miles sampling-mask compatibility metadata"
        )

    result = {
        "status": "PASS",
        "sglang_commit": expected_sglang_commit,
        "deferred_dflash_masks": True,
        "miles_sampling_mask_metadata": True,
        **actual,
    }
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
