"""The engine test's configs: delta sets (a model in one precision) and the engines serving them."""

from dataclasses import dataclass

from configs import (
    BF16_TRAINER_NVFP4_ENV,
    DSV41_ENV,
    DSV41_NVFP4_ENGINE,
    DSV41_NVFP4_ENV,
    GLM5_ENV,
    GLM5_FP8_ENGINE,
    GLM5_NVFP4_ENGINE,
    GLM53_FLASH_NVFP4_ENGINE,
    MUSE_SPARK_BF16_ENGINE,
    NVFP4_ENV,
    dp_attention,
)
from configs import CONFIGS as E2E


@dataclass(frozen=True)
class DeltaSet:
    # The GPU type its rollout checkpoint was quantized on; its versions are quantized there too.
    gpu: str
    # Checkpoints: what the engine loads, and the BF16 tensors the simulated trainer's masters
    # start from where the rollout checkpoint is quantized (hub/... in the Hugging Face cache).
    rollout_checkpoint: str
    bf16_checkpoint: str
    # Name patterns of the tensors the trainer never exports: they keep the rollout bytes.
    frozen: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class EngineConfig:
    delta_set: str
    gpu: str
    gpus: int
    engine_args: str
    env: dict[str, str]


def _delta_set(name: str, **kwargs) -> DeltaSet:
    """The model and precision of the e2e config `name`."""
    config = E2E[name]
    return DeltaSet(
        gpu=config.gpu,
        rollout_checkpoint=config.rollout_checkpoint,
        env=config.env,
        **kwargs,
    )


def _engine(name: str) -> EngineConfig:
    """The engine of the e2e config `name`, on the delta set of the same name."""
    config = E2E[name]
    return EngineConfig(
        delta_set=name,
        gpu=config.gpu,
        gpus=config.engine_gpus,
        engine_args=config.engine_args,
        env=config.env,
    )


def _glm5_3_flash_frozen(mtp_layer: int) -> tuple[str, ...]:
    """Miles' GLM-5.3-Flash recipe trains no MTP layer, expert bias or vision tower."""
    return (
        f"model.language_model.layers.{mtp_layer}.",
        "e_score_correction_bias",
        "model.visual.",
        ".input_scale",
    )


_QWEN3_6 = dict(
    bf16_checkpoint="qwen3-6-35b-a3b-bf16",
    frozen=("model.visual.", ".input_scale"),
)
# Miles' model args for it train no MTP layer.
_QWEN3_8 = dict(
    bf16_checkpoint="qwen3-8-27b-bf16",
    frozen=("model.visual.", "mtp."),
)
# Miles' GLM-5 recipes train no MTP layer (layer 78) and no expert bias.
_GLM5_3 = dict(
    bf16_checkpoint="hub/models--zai-org--GLM-5.3-BF16/snapshots/abf42e6c0e53bc29a875511301e7b7fcd220315b",
    frozen=("model.layers.78.", "e_score_correction_bias", ".input_scale"),
)
# Miles' V4.1 export leaves out the engram table, the vision routing bias and the vision side.
_DSV4_1_FROZEN = (
    ".engram.embed.",
    ".gate.bias_vl",
    "vision.",
    "aligner.",
    "image_",
    ".input_scale",
)
# Miles' Muse Spark recipe trains the language model only, its expert bias frozen.
_MUSE_SPARK_FROZEN = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.audio_tower.",
    "e_score_correction_bias",
)

DELTA_SETS = {
    "qwen3.6-35b-a3b-bf16": _delta_set("qwen3.6-35b-a3b-bf16", **_QWEN3_6),
    "qwen3.6-35b-a3b-fp8": _delta_set("qwen3.6-35b-a3b-fp8", **_QWEN3_6),
    "qwen3.6-35b-a3b-nvfp4": _delta_set("qwen3.6-35b-a3b-nvfp4", **_QWEN3_6),
    "qwen3.8-27b-bf16": _delta_set("qwen3.8-27b-bf16", **_QWEN3_8),
    "qwen3.8-27b-fp8": _delta_set("qwen3.8-27b-fp8", **_QWEN3_8),
    "glm5.3-nvfp4": DeltaSet(
        gpu="B200",
        rollout_checkpoint="glm5-3-nvfp4",
        env=GLM5_ENV | NVFP4_ENV,
        **_GLM5_3,
    ),
    "glm5.3-fp8": DeltaSet(
        gpu="H200",
        rollout_checkpoint="glm5-3-fp8-miles",
        env=GLM5_ENV,
        **_GLM5_3,
    ),
    "glm5.3-flash-nvfp4": DeltaSet(
        gpu="B200",
        rollout_checkpoint="glm5-3-flash-nvfp4",
        bf16_checkpoint="hub/models--zai-org--GLM-5.3-Flash-BF16/snapshots/f9fed3bbc82291073373d67452e9f25317ef7577",
        frozen=_glm5_3_flash_frozen(mtp_layer=45),
        env=BF16_TRAINER_NVFP4_ENV,
    ),
    "glm5.3-flash-cut-nvfp4": _delta_set(
        "glm5.3-flash-cut-nvfp4",
        bf16_checkpoint="glm5-3-flash-bf16-5layer",
        frozen=_glm5_3_flash_frozen(mtp_layer=5),
    ),
    # Miles' V4.1 recipe trains no MTP layer either.
    "dsv4.1-nvfp4": DeltaSet(
        gpu="B200",
        rollout_checkpoint="deepseek-v4-1-flash-1017-nvfp4-miles",
        bf16_checkpoint="deepseek-v4-1-flash-1017-bf16-miles",
        frozen=(*_DSV4_1_FROZEN, "mtp."),
        env=DSV41_ENV | DSV41_NVFP4_ENV,
    ),
    "dsv4.1-cut-nvfp4": _delta_set(
        "dsv4.1-cut-nvfp4",
        bf16_checkpoint="deepseek-v4-1-flash-bf16-7layer",
        frozen=_DSV4_1_FROZEN,
    ),
    # The last four layers (RadixArk/Muse-Spark-Open-last4-partial), one tensor per expert as Miles exports.
    "muse-spark-cut-bf16": DeltaSet(
        gpu="B300",
        rollout_checkpoint="muse-spark-open-last4-partial-per-expert",
        bf16_checkpoint="muse-spark-open-last4-partial-per-expert",
        frozen=_MUSE_SPARK_FROZEN,
        env={},
    ),
}

ENGINES = {
    name: _engine(name)
    for name in (
        "qwen3.6-35b-a3b-bf16",
        "qwen3.6-35b-a3b-fp8",
        "qwen3.6-35b-a3b-nvfp4",
        "qwen3.8-27b-bf16",
        "qwen3.8-27b-fp8",
        "glm5.3-flash-cut-nvfp4",
        "dsv4.1-cut-nvfp4",
    )
} | {
    "glm5.3-nvfp4": EngineConfig(
        delta_set="glm5.3-nvfp4",
        gpu="B200",
        gpus=8,
        engine_args=dp_attention(8) + GLM5_NVFP4_ENGINE + "--mem-fraction-static 0.7 ",
        env=GLM5_ENV | NVFP4_ENV,
    ),
    "glm5.3-fp8": EngineConfig(
        delta_set="glm5.3-fp8",
        gpu="H200",
        gpus=8,
        # Its weights take 93 of an H200's 141 GB; Miles' GLM-5 recipe runs full-model engines at 0.85.
        engine_args=dp_attention(8) + GLM5_FP8_ENGINE + "--mem-fraction-static 0.85 ",
        env=GLM5_ENV,
    ),
    "glm5.3-flash-nvfp4": EngineConfig(
        delta_set="glm5.3-flash-nvfp4",
        gpu="B200",
        gpus=4,
        engine_args=dp_attention(4)
        + GLM53_FLASH_NVFP4_ENGINE
        + "--mem-fraction-static 0.7 ",
        env=BF16_TRAINER_NVFP4_ENV,
    ),
    # 555 GB of weights: more than 0.7 of four B200s.
    "dsv4.1-nvfp4": EngineConfig(
        delta_set="dsv4.1-nvfp4",
        gpu="B200",
        gpus=8,
        engine_args=dp_attention(8) + DSV41_NVFP4_ENGINE + "--mem-fraction-static 0.7 ",
        env=DSV41_ENV | DSV41_NVFP4_ENV,
    ),
    "muse-spark-cut-bf16": EngineConfig(
        delta_set="muse-spark-cut-bf16",
        gpu="B300",
        gpus=8,
        engine_args=MUSE_SPARK_BF16_ENGINE + "--mem-fraction-static 0.8 ",
        env={},
    ),
}
