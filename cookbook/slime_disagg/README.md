# Disaggregated SLIME on Modal

Train Qwen3-4B with GRPO while rollouts run on a separate, elastic pool of
SGLang servers.

The app has two halves.

- **`Trainer`** is a clustered SLIME/Ray job. After each optimizer step it
  writes a sparse weight delta to a Modal Volume (the "bulletin board") and
  publishes a new weight version.
- **`Server`** is a Modal Flash pool of SGLang servers. A sidecar in each
  container watches the bulletin board, applies deltas in order, and serves
  rollout requests pinned to an exact weight version. Requests for a version
  the container hasn't reached yet get `409` and SLIME retries.

The two halves never talk directly. Weights flow through the Volume, rollout
traffic flows through the Flash gateway, and either side can scale or restart
on its own.

## Layout

| File | What it is |
|---|---|
| `modal_app.py` | The Modal app. Image, `Server` pool, `Trainer` cluster, entrypoints |
| `helpers.py` | Ray startup, sidecar process management, smoke checks |
| `configs/base.py` | `ModalConfig` (infra) and `SlimeConfig` (training args) base classes |
| `configs/qwen3_4b_delta_flash.py` | Qwen3-4B GSM8K GRPO, 3 rollouts. Protocol smoke test |
| `configs/qwen3_4b_delta_flash_hillclimb.py` | Same, 120 rollouts with evals. Reward hillclimb |

The `stitch` package (this repo) provides the bulletin-board protocol, the
SGLang sidecar, and the SLIME hooks. Both it and this example are mounted
into containers at startup, so code edits never rebuild the image.

## Run it

You need a Modal account and a `huggingface-secret` Modal secret. Work from
the repo root, with this alias to keep the commands short:

```bash
alias m="uv run --extra modal modal"

# One-time setup. Fetch the model and dataset onto Volumes.
m run -m cookbook.slime_disagg.modal_app::download_model
m run -m cookbook.slime_disagg.modal_app::prepare_dataset

# Start every rollout container from weight version 0.
m run -m cookbook.slime_disagg.modal_app::reset_bulletin_board --confirm
m deploy --strategy recreate -m cookbook.slime_disagg.modal_app

# Wait for the pool to come up and answer at version 0.
m run -m cookbook.slime_disagg.modal_app::smoke_flash_pool

# Train. Returns immediately; the run continues on Modal.
m run -m cookbook.slime_disagg.modal_app::launch_train

# The 3-rollout config should leave the pool at version 3.
m run -m cookbook.slime_disagg.modal_app::smoke_flash_pool --weight-version 3
```

To train again, run `launch_train` again. The `Trainer` cluster starts Ray
once per container in `@modal.enter()`, so a warm cluster goes straight to
training. To rerun from scratch, reset the bulletin board and redeploy with
`--strategy recreate` so every container starts again from version 0.

## Configuration

Each experiment is a module in `configs/` holding a `ModalConfig` (GPU type,
pool size, regions), a `SlimeConfig` (every SLIME CLI arg), and the names of
the Modal resources it owns.

`launch_train` imports the experiment from your local working tree and ships
the training arguments to the deployed `Trainer` as plain data. Editing or
adding a `SlimeConfig` therefore needs no redeploy; just `launch_train` it.
Infrastructure binds at deploy time, so changes to GPU type, node count,
pool size, Volume names, or SGLang server flags still need a deploy.

The only environment variable is `EXPERIMENT_CONFIG`, which picks the config
module at deploy time. Each experiment becomes its own Modal app:

```bash
EXPERIMENT_CONFIG=qwen3_4b_delta_flash_hillclimb m deploy --strategy recreate -m cookbook.slime_disagg.modal_app
```

The hillclimb run is the same transport with a real acceptance signal.
`passrate/pass@1` and `passrate/pass@8` should trend up, and `eval/gsm8k`
should not regress. Useful log searches:

```bash
m app logs slime-qwen3-4b-delta-flash-hillclimb --since 4h --search "passrate "
m app logs slime-qwen3-4b-delta-flash-hillclimb --since 4h --search "Published sparse delta"
```

## Protocol notes

The trainer publishes one manifest per weight version at
`/delta-bulletin/versions/weight_vNNNNNN/manifest.json`, then updates
`/delta-bulletin/latest.json`. Sidecars apply versions strictly in order from
their current version to the requested one. A retained chain of deltas keeps
recovery simple, since a fresh container replays from version 0. Bounding
that replay with periodic checkpoints is left for later.

Sidecars default to `quiesce` commit mode, which drains in-flight requests
before applying a delta. The `in_place` mode (set `SIDECAR_COMMIT_MODE` in
the config module) applies without draining and relies on version-namespaced
KV keys. It needs an SGLang build with the overlap-drain fix, described in
`docs/kv-version-namespace-design.md`.
