"""Matching humans& NVFP4 settings for preparation, training, and serving."""

from typing import Any, Literal


def environments(
    *, error_mode: Literal["MAE", "MSE"]
) -> tuple[dict[str, str], dict[str, str]]:
    """Return fresh TE and FlashInfer settings; the error metric changes weight bytes."""
    if error_mode not in {"MAE", "MSE"}:
        raise ValueError(f"Unsupported NVFP4 error metric: {error_mode!r}")
    training = {
        "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "NVTE_NVFP4_DISABLE_RHT": "1",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
        "NVTE_BACKWARD_OVERRIDE": "dequantized",
        "NVTE_USE_FAST_MATH": "0",
        "NVTE_NVFP4_4OVER6": "all",
        "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
        "NVTE_NVFP4_4OVER6_ERR_MODE": error_mode,
        "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    }
    serving = {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": error_mode,
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    }
    return training, serving


def validate_environments(recipe: Any) -> None:
    """Require the recipe's quantization contract at each execution boundary."""
    declared_training = getattr(recipe, "NVFP4_TRAINING_ENV", {})
    error_mode = declared_training.get("NVTE_NVFP4_4OVER6_ERR_MODE")
    training, serving = environments(error_mode=error_mode)
    for name, actual, expected in (
        ("NVFP4_TRAINING_ENV", declared_training, training),
        ("NVFP4_SERVING_ENV", getattr(recipe, "NVFP4_SERVING_ENV", {}), serving),
        ("PREP_ENV", getattr(recipe, "PREP_ENV", {}), training),
        ("miles.environment", recipe.miles.environment, training),
        ("SGLANG_SERVER_ENV", getattr(recipe, "SGLANG_SERVER_ENV", {}), serving),
    ):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"{name}[{key!r}] must be {value!r} for this NVFP4 recipe"
                )


def routed_expert_precision() -> dict:
    """Quantize routed expert GEMMs; attention, shared experts, and other layers stay BF16."""
    return {
        "configs": {
            "nvfp4": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp4_quantization_recipe": "nvfp4"},
            },
            "bf16": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {},
            },
        },
        "matchers": {
            **{
                f"routed_experts_{name}_nvfp4": {
                    "type": "glob",
                    "enabled": True,
                    "pattern": f"*.mlp.experts.linear_{name}",
                    "config": "nvfp4",
                }
                for name in ("fc1", "fc2")
            },
            "default_bf16": {
                "type": "glob",
                "enabled": True,
                "pattern": "*",
                "config": "bf16",
            },
        },
    }
