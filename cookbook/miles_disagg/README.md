# Disaggregated miles on Modal

Train Kimi K2.6 or Moonlight with GRPO under **native NVFP4 QAT**, or run
GLM-4.5-Air as a BF16 H200 experiment, while rollouts run on a separate,
elastic pool of SGLang servers. The miles twin of
[`../slime_disagg`](../slime_disagg): same two-half architecture and the same
`stitch` bulletin-board / sidecar machinery, but the trainer is **miles**.

The app has two halves.

- **`Trainer`** is a clustered miles/Ray job. For NVFP4 configs it QAT-trains
  with `--fp4-format e2m1` / `NVFP4BlockScaling`; for GLM-4.5-Air it trains and
  exports BF16. After each step it writes a sparse XOR weight delta to a Modal
  Volume (the "bulletin board") and publishes a new weight version.
- **`Server`** is a Modal Flash pool of SGLang servers on the configured GPU
  type: B200 for NVFP4 configs, H200 for GLM-4.5-Air BF16. A sidecar in each
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

`glm45_air_bf16_disagg` is the exception: a BF16 experiment on H200 with no
NVFP4 anywhere (see the variant note below).

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

### Fork dependencies

The image pins a miles fork commit (`MILES_REPO_REF` in `modal_train.py`) that
carries the disaggregated-rollout features plus the NVFP4/publish-only fixes
this cookbook needs; push the ref to `modal-projects/miles` before deploying.
The megatron routing-replay (R3) fix is baked into the trainer image at build
time (idempotent — a no-op once the fork ships it).

Dev iteration: overlay a local miles checkout at deploy time (no rebuild, no
push). This only overlays miles; the megatron R3 fix still comes from the bake.

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
