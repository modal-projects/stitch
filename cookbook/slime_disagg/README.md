# Disaggregated SLIME with Sparse-Delta Sync and Elastic Rollouts

This example runs Qwen3-4B GRPO with SLIME training on a Ray actor cluster and
an elastic Modal Flash SGLang rollout pool. The trainer writes retained sparse
delta transitions to a v2 Modal Volume bulletin board; each Flash container
reloads the Volume and applies the requested weight version before serving
version-pinned rollout requests.

Scope for this first implementation:

- Full-parameter fine-tuning through SLIME sparse delta compression.
- Modal Flash gateway for blind load balancing across `Server` containers.
- Direct container wakeups with `modal.experimental.flash_get_containers`.
- `Volume.commit()` on trainer ranks and `Volume.reload()` in the rollout sidecar.
- No LoRA path and no stale KV cache reuse.

## Layout

- `modal_app.py`: Modal app, Flash `Server` class, train/download/reset entrypoints.
- `modal_helpers.py`: example-specific Modal launch, smoke, and process helpers.
- `configs/qwen3_4b_delta_flash.py`: Qwen3-4B GSM8K GRPO config.
- `configs/qwen3_4b_delta_flash_hillclimb.py`: Longer GSM8K config for reward hillclimb validation.
- `vendor/`: launcher config base classes and Modal cluster helpers vendored from [modal-projects/multinode-training-guide](https://github.com/modal-projects/multinode-training-guide).
- `stitch` (this repo): reusable protocol, bulletin-board, SGLang sidecar, provider adapters, and SLIME adapter code. The image installs the enclosing checkout by default; override with `STITCH_REPO_PATH`.

Config should use the extracted package paths:

- `stitch.trainers.slime.commit_delta_volume`
- `stitch.trainers.slime.publish_delta_version`
- `stitch.trainers.slime.generate_rollout`

The Modal image starts from the nightly SLIME image, installs the local
`stitch` package, then replaces `/root/slime` with the fork
branch that contains the generic HTTP rollout endpoint and publish-only
disk-delta hooks. Override these when testing another branch:

- `SLIME_REPO_URL`: defaults to `https://github.com/modal-projects/slime.git`
- `SLIME_REPO_REF`: defaults to `jvmncs/rollout-endpoint`

## Run

The app expects a standard Modal secret named `huggingface-secret` when model
or dataset access requires Hugging Face credentials.

```bash
uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::download_model
uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::prepare_dataset
uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::reset_bulletin_board --confirm
MIN_CONTAINERS=2 TARGET_INPUTS=64 uv run --extra modal modal deploy --strategy recreate cookbook/slime_disagg/modal_app.py
PYTHONPATH=cookbook uv run --extra modal python -u -c "from slime_disagg.modal_app import run_flash_pool_smoke; run_flash_pool_smoke(weight_version=0, expect_min_containers=2)"
uv run --extra modal modal run --detach cookbook/slime_disagg/modal_app.py::launch_train --experiment qwen3_4b_delta_flash
PYTHONPATH=cookbook uv run --extra modal python -u -c "from slime_disagg.modal_app import run_flash_pool_smoke; run_flash_pool_smoke(weight_version=3, expect_min_containers=2)"
```

`launch_train` first looks up the deployed app's `train` function with
`modal.Function.from_name`. If that function is not found, it falls back to the
ephemeral `train` function from the current `modal run`; keep `--detach` so the
fallback can outlive the local client process.

Reset the bulletin board before starting a fresh rollout pool. If a running pool
has already synced old retained versions, reset first and then deploy with
`--strategy recreate` so every container starts again from base version 0:

```bash
uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::reset_bulletin_board --confirm
MIN_CONTAINERS=2 TARGET_INPUTS=64 uv run --extra modal modal deploy --strategy recreate cookbook/slime_disagg/modal_app.py
```

Useful knobs:

- `SLIME_DELTA_APP_NAME`: Modal app name override. Defaults to the selected config's app name.
- `MIN_CONTAINERS`: minimum Flash rollout containers. Defaults to `2`.
- `TARGET_INPUTS`: Modal per-container concurrency and SGLang max running requests. Defaults to `64`.
- `ROLLOUT_GPU`: rollout GPU type. Defaults to the config's Modal GPU.
- `ROLLOUT_GATEWAY_URL`: explicit Flash gateway URL override for training.
- `SGLANG_CONTEXT_LENGTH`, `SGLANG_MEM_FRACTION_STATIC`, `SGLANG_CHUNKED_PREFILL_SIZE`: SGLang server tuning.
- `SIDECAR_DEBUG_REQUESTS=1`: log each versioned sidecar proxy request. This is noisy under rollout load.

For debugging, stream focused deployed-app logs with Modal's log search:

```bash
uv run --extra modal modal app logs slime-qwen3-4b-delta-flash --since 30m --search "Published sparse delta version"
uv run --extra modal modal app logs slime-qwen3-4b-delta-flash --since 30m --search "Training qwen3_4b_delta_flash"
```

## Reward Hillclimb Run

Use `qwen3_4b_delta_flash_hillclimb` when the acceptance signal is training
quality rather than protocol/perf smoke. It keeps the same disaggregated sparse
delta transport as the short config, but runs 120 rollouts, enables
`log_passrate`, and evaluates GSM8K every 20 rollouts.

```bash
EXPERIMENT_CONFIG=qwen3_4b_delta_flash_hillclimb uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::download_model
EXPERIMENT_CONFIG=qwen3_4b_delta_flash_hillclimb uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::prepare_dataset
EXPERIMENT_CONFIG=qwen3_4b_delta_flash_hillclimb uv run --extra modal modal run cookbook/slime_disagg/modal_app.py::reset_bulletin_board --confirm
EXPERIMENT_CONFIG=qwen3_4b_delta_flash_hillclimb MIN_CONTAINERS=2 TARGET_INPUTS=64 uv run --extra modal modal deploy --strategy recreate cookbook/slime_disagg/modal_app.py
EXPERIMENT_CONFIG=qwen3_4b_delta_flash_hillclimb uv run --extra modal modal run --detach cookbook/slime_disagg/modal_app.py::launch_train --experiment qwen3_4b_delta_flash_hillclimb
```

Primary pass signal: `passrate/pass@1` and `passrate/pass@8` should trend up
over the run. Secondary sanity signal: `eval/gsm8k` should improve from the
pre-train baseline or, at minimum, not regress while train pass-rate improves.

Useful log searches:

```bash
uv run --extra modal modal app logs slime-qwen3-4b-delta-flash-hillclimb --since 4h --search "passrate "
uv run --extra modal modal app logs slime-qwen3-4b-delta-flash-hillclimb --since 4h --search "eval "
uv run --extra modal modal app logs slime-qwen3-4b-delta-flash-hillclimb --since 4h --search "Published sparse delta"
```

## Protocol Notes

The trainer publishes one complete manifest per weight version at
`/delta-bulletin/versions/weight_vNNNNNN/manifest.json`, then updates
`/delta-bulletin/latest.json`. The extracted SGLang sidecar applies versions in
order from its current version to the requested target. SLIME's generic HTTP
rollout endpoint mode sends exact version requests, so a container returns
`409 WeightVersionNotReady` until it has caught up and SLIME retries until the
target version is ready.

The retained-chain approach is simple and recovery-friendly for this example.
Production runs may later add periodic checkpoint or accumulated-delta anchors
to bound cold-start replay length.
