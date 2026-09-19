"""Preparation and serving settings for the GLM-5.2 NVFP4 delta profiler."""

from pathlib import Path
from types import SimpleNamespace

SOURCE_MODEL = "zai-org/GLM-5.2"

SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"

BF16_CHECKPOINT_PATH = Path("/checkpoints/glm5-2-bf16")

ROLLOUT_CHECKPOINT_PATH = Path("/checkpoints/glm5-2-nvfp4")

ROLLOUT_GPUS_PER_ENGINE = 4

PREP_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
}

TRAINER_EXTRA_PIP_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.3",
)

TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --system --break-system-packages flashinfer-python==0.6.15.post1",
    "uv pip install --system --break-system-packages --no-deps --index-url "
    "https://flashinfer.ai/whl flashinfer-cubin==0.6.15.post1",
    "uv pip install --system --break-system-packages --no-deps --index-url "
    "https://flashinfer.ai/whl/cu130 flashinfer-jit-cache==0.6.15.post1+cu130",
)

SGLANG_SERVER_ARGS = {
    "--tp": "4",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--quantization": "modelopt_fp4",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": "65544",
    "--attention-backend": "dsa",
    "--kv-cache-dtype": "fp8_e4m3",
    "--dsa-prefill-backend": "flashmla_sparse",
    "--dsa-decode-backend": "flashmla_kv",
    "--dsa-topk-backend": "flashinfer",
    "--page-size": "64",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "16384",
    "--max-running-requests": "24",
    "--max-queued-requests": "4",
    "--schedule-conservativeness": "1.0",
    "--schedule-policy": "lpm",
    "--cuda-graph-config": '{"decode":{"backend":"full","max_bs":24,"bs":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]}}',
    "--speculative-algorithm": "DFLASH",
    "--speculative-attention-mode": "decode",
    "--speculative-dflash-block-size": "8",
    "--speculative-num-draft-tokens": "8",
    "--speculative-num-steps": "1",
    "--speculative-eagle-topk": "1",
    "--speculative-draft-attention-backend": "flashinfer",
    "--speculative-draft-load-format": "fastsafetensors",
    "--speculative-draft-model-path": "/draft/zai-org/GLM-5.2/dflash/draft-step-103000",
    "--speculative-draft-model-quantization": "unquant",
    "--speculative-draft-window-size": "4096",
    "--enable-return-routed-experts": "",
}

SGLANG_SERVER_ENV = {
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
}

SERVED_CHECKPOINT_FORMAT = "nvfp4"
MATERIALIZE_BF16_MASTERS = False
UNPACK_FUSED_EXPERTS = False
DISABLE_HF_XET = False
DISABLE_HF_TRANSFER = False

miles = SimpleNamespace(
    hf_checkpoint="/checkpoints/glm5-2-nvfp4",
    pipeline_model_parallel_size=4,
    decoder_first_pipeline_num_layers=18,
    decoder_last_pipeline_num_layers=20,
    num_layers_at_start_in_bf16=3,
    num_layers_at_end_in_bf16=12,
    extra_high_precision_layers_hf=[".shared_experts."],
)

modal = SimpleNamespace(
    gpu="B300",
    trainer_memory_mib=(1048576, 3145728),
    rollout_memory_mib=(1048576, 3145728),
)
