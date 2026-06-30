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

`glm45_air_bf16_disagg` is the exception: it is a BF16 experiment on H200. It
does not use NVFP4 QAT, does not run `convert_hf_to_nvfp4.py`, and serves the
prepared BF16 Hugging Face checkpoint directly.

## Checkpoint lifecycle (three roles)

`prepare_checkpoints` builds them on a GPU (see `modal_train.py`):

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

## GLM-4.5-Air BF16 on H200

Use `glm45_air_bf16_disagg` for `zai-org/GLM-4.5-Air`.

This config has two prepared checkpoint paths:

- `/prep/glm45-air-bf16/bf16`: the prepared Hugging Face BF16 checkpoint. This
  is both the SGLang served base (`--hf-checkpoint`) and the disk-delta baseline.
- `/prep/glm45-air-bf16/torch_dist`: the Megatron raw-mode checkpoint loaded by
  the trainer (`--ref-load`).

The weight-sync loop is still the same disk-delta loop: miles exports BF16 HF
tensors after each update, XORs the new bytes against the previous bytes, writes
the delta to the Modal Volume bulletin board, and the stitch sidecar applies the
delta onto each rollout container's local BF16 checkpoint copy.

The GLM path has two Modal-specific details:

- `prepare_checkpoints` disables Xet and `hf_transfer`; the standard Hugging
  Face downloader was the path that finished reliably for this large checkpoint.
- `prepare_torch_dist` uses a small wrapper around miles'
  `convert_hf_to_torch_dist.py` so multi-node Modal Volume commits merge all
  `iter_0000001` shard files instead of only rank 0's renamed `release` dir.

Run from the repo root:

```bash
alias m="uv run --extra modal modal"
export EXPERIMENT_CONFIG=glm45_air_bf16_disagg

# Long-running one-time prep. Keep the rollout pool down until the served base
# exists, otherwise warm containers crash-loop on the missing model path. If you
# use --detach, wait for each prep job to finish before starting the next one.
POOL_MIN_CONTAINERS=0 m run --detach -m cookbook.miles_disagg.modal_train::prepare_checkpoints
POOL_MIN_CONTAINERS=0 m run --detach -m cookbook.miles_disagg.modal_train::prepare_torch_dist
POOL_MIN_CONTAINERS=0 m run -m cookbook.miles_disagg.modal_train::prepare_dataset

# Deploy the H200 rollout pool and trainer app.
m deploy --strategy recreate -m cookbook.miles_disagg.modal_train

# Verify SGLang serves the prepared BF16 base, then launch training.
m run -m cookbook.miles_disagg.modal_train::smoke_flash_pool
m run -m cookbook.miles_disagg.modal_train::launch_train

# Optional: check a later synced weight version.
m run -m cookbook.miles_disagg.modal_train::smoke_flash_pool --weight-version 1
```

Expected prepared outputs in the `miles-prep-checkpoints` Volume:

```text
glm45-air-bf16/bf16/
glm45-air-bf16/torch_dist/latest_checkpointed_iteration.txt
glm45-air-bf16/torch_dist/iter_0000001/
```

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

The full `kimi_k2_6_nvfp4_disagg` recipe is a 32×8 B200 trainer footprint that
exceeds the de-risk budget — run the Moonlight de-risk first to validate the
QAT → NVFP4-export → XOR-delta → SGLang-reload loop, then scale.

### Fork dependencies

The image pins the miles fork branch `nvfp4-disagg-fixes` (`MILES_REPO_REF`),
which carries the disaggregated-rollout features plus the publish-only / NVFP4
fixes this cookbook needs: NVFP4 export dispatch
(`megatron_to_hf/processors/__init__.py`), the publish-only rollout semaphore
and HTTP client, the 0-dim NVFP4-scale delta encode, and the `encoding_dsv4`
import guard. Push that branch before deploying.

The megatron routing-replay (R3) fix lives in `radixark/Megatron-LM` and is
**baked into the trainer image at build time** (a `.run_commands` step in
`modal_train.py`; source diff in `megatron_r3_num_out_tokens.patch`). The bake
is idempotent — it becomes a no-op once the fork itself ships the fix.

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

## Bring-up checklist (flagged, validated by the Moonlight run)

- The miles image's TransformerEngine is ≥ 2.7.0.dev0 and the trainer runs on
  Blackwell (NVFP4 BlockScaling).
- SGLang serves the prepared NVFP4 base on Blackwell (the `serving.py` fork is
  proven for NVFP4) — verify on a warm container.
- The `convert_hf_to_nvfp4.py` quantization scope (which tensors get NVFP4 +
  exclude rules) matches the export processor's scope, so the XOR delta aligns.
  A scope mismatch fails loud on the first delta apply (checksum/shape) — which
  is exactly what the Moonlight run catches cheaply.
- `miles.utils.disk_delta` is import-light in the `--no-deps` serving image.
