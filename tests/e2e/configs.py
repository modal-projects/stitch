"""The e2e runs: a model's checkpoints, how Miles trains it and how SGLang serves it."""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Config:
    gpu: str
    # Folders under the checkpoints Volume: the rollout checkpoint the engines load and
    # whose format the trainer exports, and what the trainer starts from: Miles' torch_dist
    # conversion, or a BF16 checkpoint Miles reads through mbridge.
    rollout_checkpoint: str
    trainer_checkpoint: str
    # Miles' scripts/models entry
    megatron_model_type: str
    train_args: str
    engine_args: str
    env: dict[str, str] = field(default_factory=dict)
    # Written to a file for --te-precision-config-file
    te_precision_config: str | None = None
    train_gpus: int = 4
    engine_gpus: int = 4


GLM5_ENV = {
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "NVSHMEM_DISABLE_NCCL": "1",
}
# The CuTe DSL NVFP4 runner in W4A16, with the deterministic two-stage finalize.
_CUTEDSL_W4A16_ENV = {
    "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
    "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
}
NVFP4_ENV = _CUTEDSL_W4A16_ENV | {
    "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG": "1",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}
# A pure BF16 trainer; four-over-six on the weights with E4M3 block scales up to 256 reproduces
# DeepSeek-V4.1-Flash's MXFP4 expert values exactly.
DSV41_NVFP4_ENV = _CUTEDSL_W4A16_ENV | {"NVTE_NVFP4_4OVER6": "weights"}
# The NVFP4 recipe for a pure BF16 trainer: no fake QAT.
BF16_TRAINER_NVFP4_ENV = {
    k: v for k, v in NVFP4_ENV.items() if k != "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"
}
# As Miles' V4.1 rollouts set them: SGLang's online updates refuse the mHC weight split computed at
# load, and an NVFP4 ignore list names wq_a and wkv, not their fused wqkv_a.
DSV41_ENV = {
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
}
# The DSV4 override does not read the checkpoint's quantization before the MoE runner check, so the
# flag names it. Under DP attention it would pick FlashMLA's mega kernel, which runs wo_a from an FP8
# copy the trainer does not model; plain FlashMLA keeps wo_a in BF16, as Miles' TP rollouts run it.
DSV41_NVFP4_ENGINE = (
    "--enable-dp-lm-head --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutedsl "
    "--dsv4-attn-backend flashmla "
)

# Adam on the CPU, as Miles' recipes run it where the training GPUs cannot hold the optimizer (they
# colocate): only our 4-GPU configs. The 8-GPU node configs fit on the GPUs, and the offload's per-tensor
# pinned copies stretch a Qwen3.6 startup past NCCL's timeout.
_CPU_ADAM = "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "

# Miles' GLM-5 5-layer CI recipe (tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py):
# CP and EP over the 4 training GPUs, DSA on Megatron's cuDNN kernels.
_GLM5_TRAIN = (
    "--megatron-to-hf-mode raw --dsa-impl megatron --dsa-kernel-backend cudnn "
    "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
    "--context-parallel-size 4 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--use-dynamic-batch-size --max-tokens-per-gpu 2048 --data-pad-size-multiplier 1024 "
    "--log-probs-chunk-size 16384 "
    "--transformer-impl transformer_engine --bf16 --qkv-format thd "
    "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
    "--use-precision-aware-optimizer "
    "--attention-softmax-in-fp32 --moe-router-use-torch-mm --attention-backend flash "
    "--cp-comm-type allgather --moe-enable-deepep --moe-token-dispatcher-type flex "
    + _CPU_ADAM
)
# Fake QDQ supplies the NVFP4 expert weights; TE runs BF16 forward and backward GEMMs.
_TE_BF16 = (
    "configs:\n  bf16:\n    transformer_engine_config_type: TEQuantizationParams\n"
    "    training_recipe: {}\nmatchers:\n  default_bf16:\n    type: glob\n    enabled: true\n"
    '    pattern: "*"\n    config: bf16\n'
)
_GLM5_NVFP4_TRAIN = (
    "--no-gradient-accumulation-fusion --miles-dsa-topk-backend flashinfer "
    "--extra-high-precision-layers-hf .shared_experts. "
    "--extra-high-precision-layers-megatron .shared_experts.linear_fc1 .shared_experts.linear_fc2 "
)
# Miles' DeepSeek-V4.1 recipe (scripts/run_deepseek_v4_1.py) on 4 GPUs: TP and EP over all four,
# its own V4.1 kernels, the router gate and expert bias frozen. Gradients reduce in BF16, as the
# recipe offers: each GPU holds 24B expert parameters, whose FP32 gradients would not fit beside
# them. Router logits in FP32, as SGLang computes them and as Miles' GLM-5.3-Flash args set: in
# BF16 near-tied experts route differently in the trainer and the engine.
_DSV41_TRAIN = (
    "--megatron-to-hf-mode raw --dsv4-impl miles --model-name deepseekv41 --qkv-format bshd "
    "--moe-router-freeze-gate --freeze-e-score-correction-bias "
    "--tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--micro-batch-size 1 --max-tokens-per-gpu 2048 "
    "--transformer-impl transformer_engine --bf16 "
    "--attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 "
    "--grad-reduce-in-bf16 --moe-router-dtype fp32 "
)
# Full fine-tuning's Adam state (1.2 TB over the four GPUs) outgrows host memory, so it streams
# through local NVMe, the recipe's disk offload, with at most 8 GiB of it per GPU in the page cache.
_DSV41_STREAMED_ADAM = (
    "--stream-optimizer-state-to-disk --stream-optimizer-state-page-cache-mb 8192 "
)
# Miles' GLM-5.3-Flash recipe (scripts/run_glm5_3_flash.py) on 4 GPUs: TP 2, EP 4.
_GLM53_FLASH_TRAIN = (
    "--megatron-to-hf-mode raw --model-name glm5_next --qkv-format thd "
    "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--micro-batch-size 1 --max-tokens-per-gpu 8192 "
    "--attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 "
    "--accumulate-allreduce-grads-in-fp32 --use-precision-aware-optimizer " + _CPU_ADAM
)


def dp_attention(gpus: int, experts: bool = True) -> str:
    """One engine over `gpus` GPUs: DP attention, and EP for the experts."""
    sizes = f"--tp-size {gpus} --attn-dp-size {gpus} "
    return sizes + (f"--ep-size {gpus} --moe-a2a-backend none " if experts else "")


# Muse Spark's language model in BF16 on one node: DP attention 8 (weight delta takes attention TP 1) with EP8;
# the KV cache in BF16 as Miles' Muse Spark recipe keeps it (SGLang would pick FP8).
MUSE_SPARK_BF16_ENGINE = (
    dp_attention(8)
    + "--enable-dp-lm-head --dtype bfloat16 --kv-cache-dtype bfloat16 --language-model-only "
)


# The recipe's engine: DP attention, EP, dense layers and LM head per GPU.
_GLM5_ENGINE = (
    "--moe-dense-tp-size 1 "
    "--enable-dp-lm-head --enable-fp32-lm-head --disable-shared-experts-fusion "
    "--attention-backend nsa --dsa-decode-backend flashmla_kv --dsa-prefill-backend flashmla_sparse "
    "--kv-cache-dtype fp8_e4m3 --page-size 64 "
    "--cuda-graph-max-bs-decode 256 --max-running-requests 512 --chunked-prefill-size 8192 "
    "--watchdog-timeout 3600 "
)
# The MTP layer as EAGLE's draft (not trained: its own weights stay at the checkpoint's), as in
# the rollout configs of the engine smokes.
_GLM5_MTP = "--speculative-algorithm EAGLE --speculative-eagle-topk 1 "
GLM5_NVFP4_ENGINE = (
    _GLM5_ENGINE
    + "--moe-runner-backend flashinfer_cutedsl --dsa-topk-backend flashinfer "
    + _GLM5_MTP
    + "--speculative-num-steps 3 --speculative-num-draft-tokens 4 "
    "--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none "
    "--speculative-draft-model-quantization unquant "
)
GLM5_FP8_ENGINE = (
    _GLM5_ENGINE
    + _GLM5_MTP
    + "--speculative-num-steps 1 --speculative-num-draft-tokens 2 "
    "--speculative-draft-attention-backend dsa "
)
# Miles' GLM-5.3-Flash rollouts (scripts/run_glm5_3_flash.py) on DP attention, the MTP layer as
# EAGLE's draft as for GLM-5.3.
GLM53_FLASH_NVFP4_ENGINE = (
    "--enable-dp-lm-head --moe-runner-backend flashinfer_cutedsl "
    "--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype bfloat16 "
    "--chunked-prefill-size 8192 --disable-radix-cache "
    + _GLM5_MTP
    + "--speculative-num-steps 3 --speculative-num-draft-tokens 4 "
    "--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none "
    "--speculative-draft-model-quantization unquant "
)

# Miles' Qwen3.6 MTP recipe (scripts/run_qwen3_6_35b_a3b_mtp.py) without its MTP training: EP over
# every training GPU.
_QWEN3_6_NO_MTP_TRAIN = (
    "--megatron-to-hf-mode raw "
    "--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 --expert-tensor-parallel-size 1 "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--use-dynamic-batch-size "
    "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
    "--use-precision-aware-optimizer "
    "--attention-softmax-in-fp32 --attention-backend flash --moe-token-dispatcher-type flex "
)
# The recipe as published: its MTP layer trained.
_QWEN3_6_TRAIN = (
    _QWEN3_6_NO_MTP_TRAIN
    + "--enable-mtp-training --mtp-num-layers 1 --mtp-loss-scaling-factor 0.2 "
)
# Miles has no dense Qwen3.5 recipe. TP over the 4 training GPUs: at TP 1 a GPU would hold
# 55 GB of BF16 weights and 110 GB of FP32 gradients.
_QWEN3_8_TRAIN = (
    "--megatron-to-hf-mode raw "
    "--tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--use-dynamic-batch-size "
    "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
    "--use-precision-aware-optimizer "
    "--attention-softmax-in-fp32 --attention-backend flash "
)
# A pure BF16 trainer whose export quantizes the routed experts to NVFP4; the shared expert stays BF16.
_QWEN3_6_NVFP4_TRAIN = (
    "--extra-high-precision-layers-hf .shared_expert. "
    "--extra-high-precision-layers-megatron .shared_experts.linear_fc1 .shared_experts.linear_fc2 "
)
_QWEN3_5_SERVE = (
    "--mem-fraction-static 0.7 --cuda-graph-max-bs-decode 256 --max-running-requests 256 "
    "--chunked-prefill-size 8192 --watchdog-timeout 3600 "
    "--mamba-radix-cache-strategy extra_buffer "
)
# The Qwen3.6 recipe's engine, the MTP layer as EAGLE's draft.
_QWEN3_5_ENGINE = (
    _QWEN3_5_SERVE
    + "--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 "
    "--speculative-num-draft-tokens 3 "
)
_QWEN3_6_NVFP4_ENGINE = (
    "--moe-runner-backend flashinfer_cutedsl "
    "--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none "
)
# Full models from here on. BF16 experts on Hopper run on the Triton MoE runner.
_QWEN3_6 = Config(
    gpu="H200",
    rollout_checkpoint="qwen3-6-35b-a3b-bf16",
    trainer_checkpoint="qwen3-6-35b-a3b_torch_dist",
    megatron_model_type="qwen3.6-35B-A3B",
    train_args=_QWEN3_6_TRAIN
    + "--expert-model-parallel-size 4 --max-tokens-per-gpu 8192 "
    + _CPU_ADAM,
    engine_args=dp_attention(4)
    + "--enable-dp-lm-head "
    + _QWEN3_5_ENGINE
    + "--moe-runner-backend triton ",
)
_QWEN3_8 = Config(
    gpu="H200",
    rollout_checkpoint="qwen3-8-27b-bf16",
    trainer_checkpoint="qwen3-8-27b_torch_dist",
    megatron_model_type="qwen3.8-27B",
    train_args=_QWEN3_8_TRAIN + "--max-tokens-per-gpu 8192 " + _CPU_ADAM,
    engine_args=dp_attention(4, experts=False)
    + "--enable-dp-lm-head "
    + _QWEN3_5_ENGINE,
)

CONFIGS = {
    "glm5.3-cut-nvfp4": Config(
        gpu="B200",
        rollout_checkpoint="glm5-3-bf16-5layer-nvfp4",
        trainer_checkpoint="glm5-3-bf16-5layer_torch_dist",
        megatron_model_type="glm5.2-744B-A40B_5layer",
        train_args=_GLM5_TRAIN + _GLM5_NVFP4_TRAIN,
        engine_args=dp_attention(4) + GLM5_NVFP4_ENGINE + "--mem-fraction-static 0.7 ",
        env=GLM5_ENV | NVFP4_ENV,
        te_precision_config=_TE_BF16,
    ),
    # The trainer runs BF16 and exports 128x128 block FP8 with FP32 scales on Hopper.
    "glm5.3-cut-fp8": Config(
        gpu="H200",
        rollout_checkpoint="glm5-3-fp8-miles-5layer",
        trainer_checkpoint="glm5-3-bf16-5layer_torch_dist",
        megatron_model_type="glm5.2-744B-A40B_5layer",
        train_args=_GLM5_TRAIN,
        engine_args=dp_attention(4) + GLM5_FP8_ENGINE + "--mem-fraction-static 0.7 ",
        env=GLM5_ENV,
    ),
    # Layers 0-3, 20, 21 and 24 (tests/e2e/checkpoints.py::cut); the trainer reads the BF16 cut.
    "dsv4.1-cut-nvfp4": Config(
        gpu="B200",
        rollout_checkpoint="deepseek-v4-1-flash-bf16-7layer-nvfp4",
        trainer_checkpoint="deepseek-v4-1-flash-bf16-7layer",
        megatron_model_type="deepseek-v4.1-7layer",
        train_args=_DSV41_TRAIN + _DSV41_STREAMED_ADAM,
        engine_args=dp_attention(4) + DSV41_NVFP4_ENGINE + "--mem-fraction-static 0.7 ",
        env=DSV41_ENV | DSV41_NVFP4_ENV,
    ),
    # Layers 0-4 and the MTP layer; the trainer reads the BF16 cut.
    "glm5.3-flash-cut-nvfp4": Config(
        gpu="B200",
        rollout_checkpoint="glm5-3-flash-bf16-5layer-nvfp4",
        trainer_checkpoint="glm5-3-flash-bf16-5layer",
        megatron_model_type="glm5.3-flash-5layer",
        train_args=_GLM53_FLASH_TRAIN,
        engine_args=dp_attention(4)
        + GLM53_FLASH_NVFP4_ENGINE
        + "--mem-fraction-static 0.7 ",
        env=BF16_TRAINER_NVFP4_ENV,
    ),
    "qwen3.6-35b-a3b-bf16": _QWEN3_6,
    "qwen3.8-27b-bf16": _QWEN3_8,
    # Miles' converter run on the BF16 checkpoint, 128x128 blocks with FP32 scales, keeping
    # in BF16 what the official FP8 release keeps (tests/e2e/checkpoints.py).
    "qwen3.6-35b-a3b-fp8": replace(
        _QWEN3_6, rollout_checkpoint="qwen3-6-35b-a3b-fp8-miles"
    ),
    "qwen3.8-27b-fp8": replace(_QWEN3_8, rollout_checkpoint="qwen3-8-27b-fp8-miles"),
    # Routed experts NVFP4 (W4A16 on the CuTe DSL runner), the rest BF16. SGLang builds the MTP
    # draft unquantized from a serialized NVFP4 checkpoint on its own.
    "qwen3.6-35b-a3b-nvfp4": replace(
        _QWEN3_6,
        gpu="B200",
        rollout_checkpoint="qwen3-6-35b-a3b-nvfp4-miles",
        train_args=_QWEN3_6_TRAIN
        + "--expert-model-parallel-size 4 --max-tokens-per-gpu 8192 "
        + _QWEN3_6_NVFP4_TRAIN
        + _CPU_ADAM,
        engine_args=dp_attention(4)
        + "--enable-dp-lm-head "
        + _QWEN3_5_ENGINE
        + _QWEN3_6_NVFP4_ENGINE,
        env=_QWEN3_6.env | BF16_TRAINER_NVFP4_ENV,
    ),
}


_ONE_NODE = dict(train_gpus=8, engine_gpus=1)
# An 8k-token response and its prompt in one GPU's micro-batch.
_ONE_NODE_TOKENS = "--max-tokens-per-gpu 16384 "
CONFIGS |= {
    "qwen3.6-35b-a3b-nvfp4-node": replace(
        CONFIGS["qwen3.6-35b-a3b-nvfp4"],
        train_args=_QWEN3_6_TRAIN
        + "--expert-model-parallel-size 8 "
        + _ONE_NODE_TOKENS
        + _QWEN3_6_NVFP4_TRAIN,
        engine_args="--tp-size 1 " + _QWEN3_5_ENGINE + _QWEN3_6_NVFP4_ENGINE,
        **_ONE_NODE,
    ),
    # incoai/Qwen3.6-35B-A3B-DFlash2 drafts instead of the MTP layer. The drafter comes from its own
    # checkpoint and is never trained: it keeps the weights it loaded.
    "qwen3.6-35b-a3b-nvfp4-node-dflash2": replace(
        CONFIGS["qwen3.6-35b-a3b-nvfp4"],
        train_args=_QWEN3_6_NO_MTP_TRAIN
        + "--expert-model-parallel-size 8 "
        + _ONE_NODE_TOKENS
        + _QWEN3_6_NVFP4_TRAIN,
        engine_args="--tp-size 1 "
        + _QWEN3_5_SERVE
        + "--speculative-algorithm DFLASH "
        "--speculative-draft-model-path /checkpoints/qwen3-6-35b-a3b-dflash2 "
        "--speculative-num-draft-tokens 8 --moe-runner-backend flashinfer_cutedsl ",
        **_ONE_NODE,
    ),
    "qwen3.6-35b-a3b-fp8-node": replace(
        CONFIGS["qwen3.6-35b-a3b-fp8"],
        # Adam on the H200s leaves no room to train the MTP layer; engines draft with its base.
        train_args=_QWEN3_6_NO_MTP_TRAIN
        + "--expert-model-parallel-size 8 "
        + _ONE_NODE_TOKENS,
        engine_args="--tp-size 1 " + _QWEN3_5_ENGINE + "--moe-runner-backend triton ",
        **_ONE_NODE,
    ),
    "qwen3.8-27b-bf16-node": replace(
        CONFIGS["qwen3.8-27b-bf16"],
        gpu="B200",
        train_args=_QWEN3_8_TRAIN + _ONE_NODE_TOKENS,
        engine_args="--tp-size 1 " + _QWEN3_5_ENGINE,
        **_ONE_NODE,
    ),
    "qwen3.8-27b-fp8-node": replace(
        CONFIGS["qwen3.8-27b-fp8"],
        train_args=_QWEN3_8_TRAIN + _ONE_NODE_TOKENS,
        engine_args="--tp-size 1 " + _QWEN3_5_ENGINE,
        **_ONE_NODE,
    ),
}
# SGLang refuses sampling masks with EAGLE, so replay serves and trains without the MTP draft;
# score centering reads float32 LM-head logits, as Miles turns on for its own engines.
CONFIGS["qwen3.6-35b-a3b-fp8-node-replay"] = replace(
    CONFIGS["qwen3.6-35b-a3b-fp8-node"],
    train_args=_QWEN3_6_NO_MTP_TRAIN
    + "--expert-model-parallel-size 8 "
    + _ONE_NODE_TOKENS,
    engine_args="--tp-size 1 "
    + _QWEN3_5_SERVE
    + "--moe-runner-backend triton --enable-return-routed-experts --enable-fp32-lm-head ",
)
# The sampler's own log-probs need the same engine as replay: SGLang captures them only
# without EAGLE.
CONFIGS |= {
    f"qwen3.6-35b-a3b-fp8-node-{name}": CONFIGS["qwen3.6-35b-a3b-fp8-node-replay"]
    for name in ("top-p", "top-p-score-centering")
}
# LoRA merged into the weights at every sync: the engines and the sync are full fine-tuning's.
_MERGED_LORA = (
    "--lora-rank 32 --lora-alpha 32 --lr 1e-5 --no-gradient-accumulation-fusion "
    "--target-modules all-linear --experts-shared-outer-loras --lora-merge-for-rollout "
)
CONFIGS["qwen3.6-35b-a3b-fp8-node-lora"] = replace(
    CONFIGS["qwen3.6-35b-a3b-fp8-node"],
    train_args=CONFIGS["qwen3.6-35b-a3b-fp8-node"].train_args + _MERGED_LORA,
)
# Miles streams no LoRA optimizer to disk; the adapters' Adam state fits on the GPUs.
CONFIGS["dsv4.1-cut-nvfp4-lora"] = replace(
    CONFIGS["dsv4.1-cut-nvfp4"],
    train_args=_DSV41_TRAIN + _MERGED_LORA,
)
# The absorbed DSA never runs kv_b_proj as a linear, so it takes no adapter.
CONFIGS["glm5.3-flash-cut-nvfp4-lora"] = replace(
    CONFIGS["glm5.3-flash-cut-nvfp4"],
    train_args=CONFIGS["glm5.3-flash-cut-nvfp4"].train_args
    + _MERGED_LORA
    + "--exclude-modules '*.self_attn.kv_b_proj' ",
)
# Miles' GLM-5.3-Flash cut layout (scripts/run_glm5_3_flash.py) on the 4 training GPUs: TP 2, PP 2, EP 2,
# the first stage holding the three dense layers. Each tensor is exported by one rank of its stage.
# Muse Spark's last four layers (RadixArk/Muse-Spark-Open-last4-partial), full fine-tuning as Miles' Muse Spark
# recipe trains it: BF16 params, FP32 grads, Adam's FP32 state streamed to local disk; the trainer loads the
# stacked release through Miles' Muse Spark bridge, the engines the same weights one tensor per expert.
_MUSE_SPARK_TRAIN = (
    "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
    "--context-parallel-size 4 --cp-comm-type a2a --expert-model-parallel-size 4 "
    "--expert-tensor-parallel-size 1 --qkv-format thd --attention-backend flash "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--use-dynamic-batch-size --max-tokens-per-gpu 4096 --log-probs-chunk-size 4096 "
    "--use-distributed-optimizer "
)
CONFIGS["muse-spark-cut-bf16"] = Config(
    gpu="B300",
    rollout_checkpoint="muse-spark-open-last4-partial-per-expert",
    trainer_checkpoint="muse-spark-open-last4-partial",
    megatron_model_type="muse-spark-last4-partial",
    train_args=_MUSE_SPARK_TRAIN
    + "--stream-optimizer-state-to-disk --stream-optimizer-state-moment-dtype fp32 "
    "--offload-train-disk-chunk-mb 256 --offload-train-disk-dir /tmp/optimizer-state ",
    engine_args=dp_attention(4)
    + "--enable-dp-lm-head --dtype bfloat16 --kv-cache-dtype bfloat16 --language-model-only "
    "--mem-fraction-static 0.8 ",
)
# Muse Spark's adapters as Miles' Muse Spark LoRA trains them: q/gate/k/v/o and the routed experts; the
# latent MoE's projections stay at base. The adapters' Adam state fits on the GPUs.
CONFIGS["muse-spark-cut-bf16-lora"] = replace(
    CONFIGS["muse-spark-cut-bf16"],
    train_args=_MUSE_SPARK_TRAIN + _MERGED_LORA,
)
CONFIGS["glm5.3-flash-cut-nvfp4-pp2"] = replace(
    CONFIGS["glm5.3-flash-cut-nvfp4"],
    train_args=_GLM53_FLASH_TRAIN.replace(
        "--pipeline-model-parallel-size 1 ",
        "--pipeline-model-parallel-size 2 --decoder-first-pipeline-num-layers 3 ",
    ).replace("--expert-model-parallel-size 4 ", "--expert-model-parallel-size 2 "),
)
