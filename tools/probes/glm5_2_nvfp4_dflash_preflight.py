"""CPU-only Modal preflight for the GLM-5.2 NVFP4 DFlash experiment.

Run before any GPU launch:

    modal run -e stitch-dev tools/probes/glm5_2_nvfp4_dflash_preflight.py
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import modal

from cookbook.common.constants import HF_CACHE_PATH
from cookbook.common.serving_image import DEFAULT_SGLANG_RUNTIME, build_serving_image
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import glm5_2_nvfp4 as model

APP_NAME = "probe-glm5-2-nvfp4-dflash-preflight"
EXPERIMENT = "glm5_2_nvfp4"

app = modal.App(APP_NAME)
trainer_preflight_image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_pip_packages=model.TRAINER_EXTRA_PIP_PACKAGES,
    image_run_commands=model.TRAINER_IMAGE_RUN_COMMANDS,
)
serving_preflight_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env=model.SGLANG_SERVER_ENV,
)


@app.function(image=trainer_preflight_image, cpu=2, memory=8192, timeout=15 * 60)
def validate_trainer_contract() -> dict:
    from miles.utils.external_utils.model_args_utils import load_model_args

    model_args = shlex.split(load_model_args(model.miles.megatron_model_type))
    expected_prefix = [
        "--spec",
        "miles_plugins.models.glm5.glm5",
        "get_glm5_spec",
    ]
    if model_args[:3] != expected_prefix:
        raise RuntimeError(
            f"pinned GLM-5.2 model arguments are incorrect: {model_args[:3]}"
        )

    validation_source = Path("/root/miles/miles/utils/arguments.py").read_text()
    if 'speculative_algorithm.upper() != "DFLASH"' not in validation_source:
        raise RuntimeError("Miles does not permit aligned DFlash top-p mask replay")

    sparse_mla_root = Path("/root/miles/miles_plugins/models/glm5/ops")
    forward_source = (sparse_mla_root / "tilelang_sparse_mla_fwd.py").read_text()
    backward_source = (sparse_mla_root / "tilelang_sparse_mla_bwd.py").read_text()
    for direction, source in (
        ("forward", forward_source),
        ("backward", backward_source),
    ):
        if "kv_i" not in source or "T.if_then_else(mask" not in source:
            raise RuntimeError(
                f"Miles GLM sparse-MLA {direction} kernel lacks padded-index handling"
            )

    result = {
        "status": "PASS",
        "miles_commit": trainer_image.MILES_REPO_REF,
        "model_script_argument_count": len(model_args),
        "model_script_source": "pinned_checkout",
        "dflash_top_p_validation": "enabled",
        "sparse_mla_padded_indices": "guarded",
    }
    print(
        f"VERDICT glm52_trainer_preflight PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.function(image=serving_preflight_image, cpu=2, memory=8192, timeout=15 * 60)
def validate_server_contract() -> dict:
    from sglang.srt.arg_groups.speculative_hook import _handle_dflash
    from sglang.srt.server_args import ServerArgs

    argv = ["--model-path", "dummy"]
    for flag, value in model.SGLANG_SERVER_ARGS.items():
        argv.append(flag)
        if value:
            argv.append(value)

    parser = argparse.ArgumentParser(prog="glm52-preflight")
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(argv)
    parsed.device = "cuda"
    _handle_dflash(parsed)

    expected = {
        "speculative_algorithm": "DFLASH",
        "enable_dp_attention": False,
        "ep_size": model.ROLLOUT_GPUS_PER_ENGINE,
        "max_running_requests": model.ROLLOUT_MAX_RUNNING_REQUESTS,
        "max_queued_requests": model.ROLLOUT_MAX_QUEUED_REQUESTS,
    }
    actual = {name: getattr(parsed, name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"parsed SGLang arguments differ: {actual} != {expected}")

    result = {
        "status": "PASS",
        "sglang_commit": DEFAULT_SGLANG_RUNTIME.commit,
        **actual,
    }
    print(
        f"VERDICT glm52_server_preflight PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    trainer_result = validate_trainer_contract.remote()
    server_result = validate_server_contract.remote()
    result = {
        "status": "PASS",
        "trainer": trainer_result,
        "server": server_result,
    }
    print(
        f"VERDICT glm52_cpu_preflight PASS {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
