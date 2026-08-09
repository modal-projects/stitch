"""Contract tests for the Qwen3.6 NVFP4 DFlash rollout recipe."""

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_nvfp4 as config


def test_rollout_combines_qwen_dflash_with_rl_metadata() -> None:
    args = config.SGLANG_SERVER_ARGS

    assert config.SOURCE_MODEL == "Qwen/Qwen3.6-35B-A3B"
    assert config.DFLASH_MODEL == "modal-labs/Qwen3.6-35B-A3B-DFlash"
    assert args["--speculative-algorithm"] == "DFLASH"
    assert args["--speculative-draft-model-path"] == config.DFLASH_MODEL
    assert args["--speculative-draft-model-quantization"] == "unquant"
    assert args["--speculative-dflash-block-size"] == "8"
    assert args["--speculative-draft-attention-backend"] == "fa4"
    assert args["--attention-backend"] == "trtllm_mha"
    assert args["--linear-attn-prefill-backend"] == "flashinfer"
    assert args["--linear-attn-decode-backend"] == "flashinfer"
    assert args["--mamba-ssm-dtype"] == "bfloat16"
    assert args["--mamba-radix-cache-strategy"] == "extra_buffer"
    assert args["--mem-fraction-static"] == "0.65"
    assert "--enable-return-routed-experts" in args
    assert "--enable-dp-attention" not in args


def test_training_keeps_nvfp4_routing_replay_and_top_p_masking() -> None:
    miles = config.miles

    assert miles.miles_model_script == "scripts/models/qwen3.6-35B-A3B.sh"
    assert miles.model_name == "qwen3_6"
    assert miles.fp4_recipe == "nvfp4"
    assert miles.use_rollout_routing_replay is True
    assert miles.rollout_top_p == 0.95
    assert miles.input_key == "prompt"
    assert miles.apply_chat_template is False
    assert miles.colocate is False
    assert miles.custom_config_path["rollout_request_retry_attempts"] == 1200
    assert miles.actor_num_nodes == 1
    assert miles.rollout_num_gpus_per_engine == 1
    assert config.modal.rollout_min_containers == 16
    assert config.modal.rollout_max_containers == 16


def test_quantizer_contract_matches_across_prep_training_and_serving() -> None:
    prep_quantizer_env = {
        key: config.PREP_ENV[key] for key in config.NVFP4_TRAINING_ENV
    }
    assert prep_quantizer_env == config.NVFP4_TRAINING_ENV
    assert config.PREP_ENV["CONVERT_KEEP_PP1"] == "1"
    assert config.PREP_ENV["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert (
        config.NVFP4_TRAINING_ENV["NVTE_NVFP4_4OVER6_ERR_MODE"]
        == config.NVFP4_SERVING_ENV["FLASHINFER_NVFP4_4OVER6_ERR_MODE"]
    )
    assert config.MILES_SOURCE_PATCHES == (config.QWEN36_NVFP4_DFLASH_MILES_PATCH,)
    patch = config.QWEN36_NVFP4_DFLASH_MILES_PATCH.read_text()
    assert "return _expand_batched_expert_params(converted_named_params)" in patch
    assert "q_weights.update(dict(_expand_batched_expert_params(key, tensor)))" in patch
    assert "modules_to_not_convert.extend(_batched_expert_module_aliases(key))" in patch
    assert "tensor = _normalize_checkpoint_dtype(key, tensor)" in patch
    assert 'name.endswith(".linear_attn.A_log")' in patch
