from __future__ import annotations

from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import glm5_2_nvfp4 as config


def test_training_image_matches_the_working_flashinfer_runtime() -> None:
    commands = "\n".join(config.TRAINER_IMAGE_RUN_COMMANDS)

    assert trainer_image.MILES_IMAGE_TAG == "radixark/miles:dev-202607290235"
    assert config.MILES_REPO_REF == "7cdfcf78c8f7ab3f2111dafe9881aa99b716a0c5"
    assert "flashinfer-python==0.6.15.post1" in commands
    assert "flashinfer-cubin==0.6.15.post1" in commands
    assert "flashinfer-jit-cache==0.6.15.post1+cu130" in commands


def test_checkpoint_artifacts_are_separate_from_run_state() -> None:
    assert str(config.BF16_CHECKPOINT_PATH) == "/checkpoints/glm5-2-bf16"
    assert config.miles.hf_checkpoint == "/checkpoints/glm5-2-nvfp4"
    assert config.miles.ref_load == "/checkpoints/glm5-2-torch-dist"
    assert not hasattr(config.miles, "update_weight_disk_dir")


def test_training_topology_uses_all_trainer_gpus() -> None:
    miles = config.miles
    trainer_gpus = config.TRAINER_NODES * config.GPUS_PER_TRAINER_NODE
    model_parallel_size = (
        miles.tensor_model_parallel_size
        * miles.pipeline_model_parallel_size
        * miles.context_parallel_size
    )

    assert trainer_gpus == model_parallel_size
    assert miles.expert_model_parallel_size == (
        miles.tensor_model_parallel_size * miles.context_parallel_size
    )

    middle_layers, remainder = divmod(
        78
        - miles.decoder_first_pipeline_num_layers
        - miles.decoder_last_pipeline_num_layers,
        miles.pipeline_model_parallel_size - 2,
    )
    assert remainder == 0
    stage_starts = [1, 1 + miles.decoder_first_pipeline_num_layers]
    for _ in range(miles.pipeline_model_parallel_size - 2):
        stage_starts.append(stage_starts[-1] + middle_layers)
    assert all(layer <= 3 or (layer - 3) % 4 == 0 for layer in stage_starts)


def test_rollout_capacity_and_resources_match_the_plan() -> None:
    trainer_gpus = config.TRAINER_NODES * config.GPUS_PER_TRAINER_NODE
    assert config.ROLLOUT_TO_TRAINER_GPU_RATIO == 1
    assert config.ROLLOUT_ENGINES * config.ROLLOUT_GPUS_PER_ENGINE == (
        trainer_gpus * config.ROLLOUT_TO_TRAINER_GPU_RATIO
    )
    assert config.modal.rollout_min_containers == config.ROLLOUT_ENGINES
    assert config.modal.rollout_max_containers == config.ROLLOUT_ENGINES


def test_training_and_serving_precision_environments_are_separate() -> None:
    assert all(key.startswith("NVTE_") for key in config.NVFP4_TRAINING_ENV)
    assert not set(config.NVFP4_TRAINING_ENV) & set(config.NVFP4_SERVING_ENV)
    assert set(config.NVFP4_TRAINING_ENV) <= set(config.miles.environment)
    assert set(config.NVFP4_SERVING_ENV) <= set(config.SGLANG_SERVER_ENV)
    assert not set(config.NVFP4_SERVING_ENV) & set(config.miles.environment)
    assert not set(config.NVFP4_TRAINING_ENV) & set(config.SGLANG_SERVER_ENV)


def test_training_limit_has_only_the_required_serving_margin() -> None:
    assert config.miles.max_seq_len == config.MAX_SEQ_LEN
    assert config.SGLANG_CONTEXT_LENGTH == config.MAX_SEQ_LEN + 8
    assert config.SGLANG_SERVER_ARGS["--context-length"] == str(
        config.SGLANG_CONTEXT_LENGTH
    )
