"""miles checkpoint preparation, ported to plain functions the app registers as Modal
functions: build the bf16 masters + the served base (bf16 / fp8 / nvfp4) and the raw-mode
torch_dist ref_load. All read their experiment constants off the selected config module.
"""

from __future__ import annotations

import json
import os
import subprocess

from cookbook.common.constants import PREP_PATH
from cookbook.miles_disagg.trainer_image import MEGATRON_PATH, MILES_ROOT, TORCH_DIST_CONVERT_WRAPPER


def prepare_checkpoints(exp, prep_volume) -> None:
    """Build the bf16 masters (trainer arch source) + the served base on a GPU.

    masters (bf16): a quantized source (Kimi INT4) is dequantized; a bf16 source IS the
    masters. served base: bf16 = masters; fp8 = the published ROLLOUT_SOURCE_MODEL; nvfp4 =
    miles' TE-direct quantizer over the masters (packing == the trainer's export packing).
    """
    if getattr(exp, "DISABLE_HF_XET", False):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    if getattr(exp, "DISABLE_HF_TRANSFER", False):
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    # Quantizer envs the conversion must share with the trainer (e.g. the 4/6 recipe's
    # NVTE_NVFP4_4OVER6 family): the served base and the trainer export must be produced
    # under identical settings or the delta baseline diverges from the export layout.
    os.environ.update(getattr(exp, "PREP_ENV", {}))
    from huggingface_hub import snapshot_download

    prep_volume.reload()
    tag = exp.MODEL_TAG
    bf16_dir, fp8_dir, nvfp4_dir = f"{PREP_PATH}/{tag}/bf16", f"{PREP_PATH}/{tag}/fp8", f"{PREP_PATH}/{tag}/nvfp4"
    served_format = getattr(exp, "SERVED_CHECKPOINT_FORMAT", "nvfp4")
    if served_format not in {"bf16", "fp8", "nvfp4"}:
        raise SystemExit(f"unsupported SERVED_CHECKPOINT_FORMAT={served_format!r}")
    tools = f"{MILES_ROOT}/tools"

    src = snapshot_download(exp.SOURCE_MODEL)
    is_int4 = _is_int4(src)  # read the source's quant scheme, not its repo name

    def _build_bf16(out: str) -> None:
        if is_int4:
            subprocess.run(["python", f"{tools}/convert_kimi_int4_to_bf16.py", "--model-dir", src, "--output-dir", out], check=True)
        else:
            subprocess.run(f"cp -aL {src}/. {out}/", shell=True, check=True)  # -L: real files, not cache symlinks
        _strip_stale_quant_config(os.path.join(out, "config.json"))

    _staged(bf16_dir, _build_bf16)

    if served_format == "bf16":
        prep_volume.commit()
        print(f"Prepared masters={bf16_dir} served_base={bf16_dir}")
        return

    if served_format == "fp8":
        fp8_source = getattr(exp, "ROLLOUT_SOURCE_MODEL", None)
        if not fp8_source:
            raise SystemExit("SERVED_CHECKPOINT_FORMAT='fp8' requires ROLLOUT_SOURCE_MODEL")
        _staged(fp8_dir, lambda out: subprocess.run(f"cp -aL {snapshot_download(fp8_source)}/. {out}/", shell=True, check=True))
        prep_volume.commit()
        print(f"Prepared masters={bf16_dir} served_base={fp8_dir}")
        return

    # nvfp4: miles' TE-direct quantizer. bf16 carve-outs must match the trainer's
    # --num-layers-at-start/end-in-bf16 so the served base == the export layout.
    carveouts: list[str] = []
    if (n := getattr(exp.miles, "num_layers_at_start_in_bf16", None)) is not None:
        carveouts += ["--num-layers-at-start-in-bf16", str(n)]
    if (n := getattr(exp.miles, "num_layers_at_end_in_bf16", None)) is not None:
        carveouts += ["--num-layers-at-end-in-bf16", str(n)]
    _staged(nvfp4_dir, lambda out: subprocess.run(
        ["python", f"{tools}/convert_hf_to_nvfp4.py", "--model-dir", bf16_dir, "--save-dir", out, *carveouts], check=True))
    prep_volume.commit()
    print(f"Prepared masters={bf16_dir} served_base={nvfp4_dir}")


def prepare_torch_dist(exp, prep_volume, *, rank: int, master_addr: str) -> None:
    """Build {tag}/torch_dist (the raw-mode ref_load) from the {tag}/bf16 masters via a
    clustered torchrun conversion (large MoE won't fit an 8-way split)."""
    prep_volume.reload()
    tag = exp.MODEL_TAG
    bf16_dir, torch_dist_dir = f"{PREP_PATH}/{tag}/bf16", f"{PREP_PATH}/{tag}/torch_dist"
    if os.path.exists(os.path.join(torch_dist_dir, "latest_checkpointed_iteration.txt")):
        print(f"reusing existing torch_dist {torch_dist_dir}")
        return
    if not exp.miles.miles_model_script:
        raise SystemExit("prepare_torch_dist requires miles_model_script (MODEL_ARGS)")
    nodes = exp.modal.torch_dist_prep_nodes
    use_wrapper = nodes > 1 and getattr(exp, "USE_MODAL_TORCH_DIST_WRAPPER", False)
    convert = TORCH_DIST_CONVERT_WRAPPER if use_wrapper else f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
    inner = (
        f"source {MILES_ROOT}/{exp.miles.miles_model_script} && "
        f"PYTHONPATH={MEGATRON_PATH} torchrun"
        f" --nnodes {nodes} --node-rank {rank} --master-addr {master_addr} --master-port 29500"
        f" --nproc-per-node {exp.modal.torch_dist_prep_gpus_per_node}"
        f" {convert} ${{MODEL_ARGS[@]}}"
        f" --hf-checkpoint {bf16_dir} --save {torch_dist_dir} --megatron-to-hf-mode raw"
        f" {exp.modal.torch_dist_convert_extra_args}"
    )
    env = {**os.environ}
    if use_wrapper:
        env["SKIP_RELEASE_RENAME"] = "1"
    subprocess.run(["bash", "-c", inner], check=True, env=env)
    # Every node commits its own distcp shards (disjoint files merge on the Volume);
    # a rank-0-only commit would drop the other nodes' shards.
    prep_volume.commit()
    if rank == 0:
        print(f"Prepared torch_dist={torch_dist_dir}")


def _staged(final_dir: str, build) -> None:
    """Build into a .partial sibling and atomically rename, so an interrupted step never
    leaves a half-built dir the reuse check mistakes for complete."""
    if os.path.isdir(final_dir) and os.listdir(final_dir):
        print(f"reusing existing {final_dir}")
        return
    partial = f"{final_dir}.partial"
    subprocess.run(["rm", "-rf", partial], check=True)
    os.makedirs(partial, exist_ok=True)
    build(partial)
    os.rename(partial, final_dir)


def _strip_stale_quant_config(config_path: str) -> None:
    """Drop any quantization_config (top-level and text_config-nested) from an HF config,
    so the bf16 masters don't claim the source's quant scheme."""
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    removed = bool(cfg.pop("quantization_config", None))
    if isinstance(cfg.get("text_config"), dict):
        removed = bool(cfg["text_config"].pop("quantization_config", None)) or removed
    if removed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"stripped stale quantization_config from {config_path}")


def _is_int4(model_dir: str) -> bool:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f) or {}
    # VLMs (Kimi K2.x) nest the quant config under text_config.
    qc = (cfg.get("text_config") or {}).get("quantization_config") or cfg.get("quantization_config") or {}
    return qc.get("quant_method") == "compressed-tensors"
