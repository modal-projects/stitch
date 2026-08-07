"""Contract tests for the Qwen3.6 NVFP4 DFlash rollout recipe."""

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_nvfp4 as config


def test_rollout_combines_qwen_dflash_with_rl_metadata() -> None:
    args = config.SGLANG_SERVER_ARGS

    assert config.SOURCE_MODEL == "Qwen/Qwen3.6-35B-A3B"
    assert config.DFLASH_MODEL == "modal-labs/Qwen3.6-35B-A3B-DFlash"
    assert args["--speculative-algorithm"] == "DFLASH"
    assert args["--speculative-draft-model-path"] == config.DFLASH_MODEL
    assert args["--speculative-dflash-block-size"] == "8"
    assert args["--speculative-draft-attention-backend"] == "fa4"
    assert args["--attention-backend"] == "trtllm_mha"
    assert args["--linear-attn-prefill-backend"] == "flashinfer"
    assert args["--linear-attn-decode-backend"] == "flashinfer"
    assert args["--mamba-radix-cache-strategy"] == "extra_buffer"
    assert "--enable-return-routed-experts" in args
    assert "--enable-dp-attention" not in args


def test_training_keeps_nvfp4_routing_replay_and_top_p_masking() -> None:
    miles = config.miles

    assert miles.miles_model_script == "scripts/models/qwen3.6-35B-A3B.sh"
    assert miles.model_name == "qwen3_6"
    assert miles.fp4_recipe == "nvfp4"
    assert miles.use_rollout_routing_replay is True
    assert miles.rollout_top_p == 0.95
    assert miles.actor_num_nodes == 1
    assert miles.rollout_num_gpus_per_engine == 1
    assert config.modal.rollout_min_containers == 16
    assert config.modal.rollout_max_containers == 16


def test_quantizer_contract_matches_across_prep_training_and_serving() -> None:
    assert config.PREP_ENV == config.NVFP4_TRAINING_ENV
    assert (
        config.NVFP4_TRAINING_ENV["NVTE_NVFP4_4OVER6_ERR_MODE"]
        == config.NVFP4_SERVING_ENV["FLASHINFER_NVFP4_4OVER6_ERR_MODE"]
    )
    assert config.MILES_SOURCE_PATCHES == (config.QWEN36_NVFP4_DFLASH_MILES_PATCH,)
