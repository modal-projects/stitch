# Disaggregated miles on Modal

Train Kimi K2.6 or Moonlight with GRPO under **native NVFP4 QAT**, or run
GLM-4.5-Air as a BF16-trainer H200 experiment with BF16 or native HF FP8
rollout weights, while rollouts run on a separate, elastic pool of SGLang
servers. The miles twin of
[`../slime_disagg`](../slime_disagg): same two-half architecture and the same
`stitch` bulletin-board / sidecar machinery, but the trainer is **miles**.

The app has two halves.

- **`Trainer`** is a clustered miles/Ray job. For NVFP4 configs it QAT-trains
  with `--fp4-format e2m1` / `NVFP4BlockScaling`; for GLM-4.5-Air it trains and
  exports BF16. After each step it writes a sparse XOR weight delta to a Modal
  Volume (the "bulletin board") and publishes a new weight version.
- **`Server`** is a Modal Flash pool of SGLang servers on the configured GPU
  type: B200 for NVFP4 configs, H200 for GLM-4.5-Air. A sidecar in each
  container watches the bulletin board, applies deltas in order, and serves
  rollouts pinned to an exact weight version. Requests for a version a container
  hasn't reached get `409` and miles retries.

Weights flow through the Volume; rollout traffic flows through the Flash
gateway; either side scales or restarts on its own.

## NVFP4 is all-Blackwell

NVFP4 QAT is native Megatron FP4 training (`NVFP4BlockScaling`, TransformerEngine
≥ 2.7.0.dev0). FP4 GEMM requires Blackwell, so **both** the trainer and the
rollout pool run on B200 — unlike the INT4 recipe, which fake-quantizes on H200.
There is no simulated/non-Blackwell NVFP4 weight-QAT path.

`glm45_air_bf16_disagg` and `glm45_air_fp8_disagg` are the exceptions: they are
BF16 trainer experiments on H200. They do not use NVFP4 QAT or run
`convert_hf_to_nvfp4.py`; the rollout base is either the prepared BF16 Hugging
Face checkpoint or the native `zai-org/GLM-4.5-Air-FP8` checkpoint.

## Checkpoint lifecycle (three roles)

`prepare_checkpoints` builds them on a GPU (see `modal_train.py`). The
`tools/convert_*.py` scripts it invokes live in the pinned miles fork, not in
this repo.

1. **BF16 masters** (`--ref-load`): the trainable parameters. Moonlight ships
   bf16 (masters = the download); Kimi K2.6 ships INT4, so masters are
   dequantized with `tools/convert_kimi_int4_to_bf16.py`.
2. **Served NVFP4 base** (`--hf-checkpoint`): produced from the masters with
   miles' own `tools/convert_hf_to_nvfp4.py`, so the served packing equals the
   trainer's export packing **by construction** (smallest deltas, no byte-exact
   risk).
3. **Megatron torch_dist** (`--load`/`--save`): trainer-internal rollout ckpts.

The trainer reads the NVFP4 base for both the export quant config and the diff
baseline, so applying delta_vN reproduces export_vN byte-for-byte — the served
weights become the trainer's NVFP4 export.

## Run it

You need a Modal account and a `huggingface-secret` Modal secret. Work from the
repo root, with this alias:

```bash
alias m="uv run --extra modal modal"

# Start with the small, runnable de-risk (single Blackwell node).
export EXPERIMENT_CONFIG=moonlight_nvfp4_disagg

# One-time setup: fetch + convert checkpoints (GPU), and the dataset.
m run -m cookbook.miles_disagg.modal_train::prepare_checkpoints
m run -m cookbook.miles_disagg.modal_train::prepare_dataset

# Deploy the rollout pool + trainer.
m deploy -m cookbook.miles_disagg.modal_train

# Wait for the pool to come up and answer at version 0.
m run -m cookbook.miles_disagg.modal_train::smoke_flash_pool

# Train. Returns immediately; the run continues on Modal.
m run -m cookbook.miles_disagg.modal_train::launch_train

# Smoke-check the pool at a given version (the chain advances one per rollout).
m run -m cookbook.miles_disagg.modal_train::smoke_flash_pool --weight-version 3
```

To start a run against the already-deployed app without a client-tied `m run`
(whose ephemeral app context can stop the deployed serving app), use plain
Python instead: `python -m cookbook.miles_disagg._spawn_into_deployed
<experiment>`.

The full `kimi_k2_6_nvfp4_disagg` recipe is a 32×8 B200 trainer footprint — run
the Moonlight de-risk first to validate the QAT → NVFP4-export → XOR-delta →
SGLang-reload loop, then scale.

### GLM-4.5-Air BF16 variant

`glm45_air_bf16_disagg` runs `zai-org/GLM-4.5-Air` as BF16 on H200; the
weight-sync loop is the same disk-delta loop, applied to BF16 HF tensors. It
prepares two checkpoints: `/prep/glm45-air-bf16/bf16` (served base + delta
baseline) and `/prep/glm45-air-bf16/torch_dist` (`--ref-load`, built by
`prepare_torch_dist` via the multi-node conversion wrapper in
`convert_hf_to_torch_dist_modal.py`). Run the same flow as above with
`EXPERIMENT_CONFIG=glm45_air_bf16_disagg`, plus
`m run --detach -m cookbook.miles_disagg.modal_train::prepare_torch_dist`
between the prep and deploy steps. Keep `POOL_MIN_CONTAINERS=0` during prep so
warm containers don't crash-loop on the missing model path, and let each prep
job finish before starting the next.

### GLM-4.5-Air FP8 rollout variant

`glm45_air_fp8_disagg` trains from the same BF16 Megatron `torch_dist` checkpoint
as the BF16 variant, but serves the native Hugging Face FP8 checkpoint
`zai-org/GLM-4.5-Air-FP8` in SGLang. The prepared paths are:

- `/prep/glm45-air-bf16/bf16`: BF16 Hugging Face masters used to build
  `torch_dist`.
- `/prep/glm45-air-bf16/fp8`: native FP8 rollout base; this is miles
  `--hf-checkpoint`, the SGLang served base, and the disk-delta baseline/export
  quant config source.
- `/prep/glm45-air-bf16/torch_dist`: Megatron raw-mode checkpoint loaded by the
  trainer via `--ref-load`.

Run the same GLM flow with `EXPERIMENT_CONFIG=glm45_air_fp8_disagg`. Keep
`POOL_MIN_CONTAINERS=0` for `prepare_checkpoints`, `prepare_torch_dist`, and
`prepare_dataset`; deploy afterward, smoke version 0, then launch training. The
rollout pool is capped at two H200 containers, each using four GPUs. Each
rollout container applies `patches/sglang-fp8-reload-attrs.patch` to the SGLang
checkout at startup so online FP8 reload preserves SGLang's sharded parameter
loaders.

### Fork dependencies

The image pins `modal-projects/miles` branch `nvfp4-disagg-v2` by commit
(`MILES_REPO_REF`). That branch carries the disaggregated-rollout features, the
publish-only / NVFP4 fixes, and the GLM-Air native FP8 disk-delta export support
needed by `glm45_air_fp8_disagg`.

The megatron routing-replay (R3) fix is applied at trainer startup as
`patches/megatron-r3-dispatch.patch` via `MEGATRON_RUNTIME_PATCHES` against the
`/root/Megatron-LM` checkout (idempotent).

Dev iteration: overlay a local miles checkout at deploy time (no rebuild, no
push). This is optional; the committed GLM-Air FP8 path does not require a
local overlay. This only overlays miles; the megatron R3 fix still comes from
the runtime patch.

```bash
MILES_LOCAL_DIR=/path/to/miles EXPERIMENT_CONFIG=moonlight_nvfp4_disagg \
  m deploy --strategy recreate -m cookbook.miles_disagg.modal_train
```

## Configuration

Each experiment is a module in `configs/` holding a `ModalConfig` (GPU type,
pool size, regions), a `MilesConfig` (every miles/Megatron CLI arg), and the
names of the Modal resources it owns. `launch_train` ships the training args as
plain data, so editing or adding a `MilesConfig` needs no redeploy; only
infrastructure (GPU, nodes, pool size, Volume names, serving flags) does.

```bash
EXPERIMENT_CONFIG=<experiment> m deploy --strategy recreate -m cookbook.miles_disagg.modal_train
```

Useful log searches (the app name is the config's `APP_NAME`):

```bash
m app logs <app-name> --since 4h --search "passrate "
m app logs <app-name> --since 4h --search "weight_v"
```

## Bring-up checklist

- The miles image's TransformerEngine is ≥ 2.7.0.dev0 and the trainer runs on
  Blackwell (NVFP4 BlockScaling).
- SGLang serves the prepared NVFP4 base on Blackwell — verify on a warm
  container.
- The `convert_hf_to_nvfp4.py` quantization scope (which tensors get NVFP4 +
  exclude rules) matches the export processor's scope, so the XOR delta aligns.
  A scope mismatch fails loud on the first delta apply (checksum/shape).
- `miles.utils.disk_delta` is import-light in the `--no-deps` serving image.
