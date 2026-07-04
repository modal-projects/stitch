# Disaggregated SLIME on Modal

Train a model with GRPO while rollouts run on a separate, elastic pool of
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

`modal_train.py` defines the Modal app (the `Server` pool, the `Trainer`
cluster, and the setup/launch entrypoints); `configs/` holds one module per
experiment. The `stitch` package (this repo) provides the bulletin-board
protocol, the SGLang sidecar, and the SLIME hooks. Both `stitch` and this
example are mounted into containers at startup, so code edits never rebuild
the image.

## Run it

You need a Modal account and a `huggingface-secret` Modal secret. Work from
the repo root, with this alias to keep the commands short:

```bash
alias m="uv run --extra modal modal"

# One-time setup. Fetch the model and dataset onto Volumes.
m run -m cookbook.slime_disagg.modal_train::download_model
m run -m cookbook.slime_disagg.modal_train::prepare_dataset

# Deploy the rollout pool + trainer.
m deploy -m cookbook.slime_disagg.modal_train

# Wait for the pool to come up and answer at version 0.
m run -m cookbook.slime_disagg.modal_train::smoke_flash_pool

# Train. Returns immediately; the run continues on Modal.
m run -m cookbook.slime_disagg.modal_train::launch_train

# Smoke-check the pool at a given version (the chain advances one per rollout).
m run -m cookbook.slime_disagg.modal_train::smoke_flash_pool --weight-version 3
```

To train again, just run `launch_train` again. Each launch gets a fresh
`run_id`: the trainer writes that run's delta chain under
`/delta-bulletin/<run_id>/weight_v{N}/` and a single `latest` pointer names the
active snapshot (`<run_id>/weight_v{N}`), so sequential runs never collide — no
bulletin-board reset between runs. Sidecars apply versions in order; when the
pointer moves to a new run they re-materialize the base and replay that run's
chain. The warm `Trainer` cluster (Ray started once per container in
`@modal.enter()`) goes straight to training.

Sidecars default to `quiesce` commit mode, which drains in-flight requests
before applying a delta. The `in_place` mode (set `SIDECAR_COMMIT_MODE` in the
config module) applies without draining and relies on version-namespaced KV
keys.

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
EXPERIMENT_CONFIG=<experiment> m deploy --strategy recreate -m cookbook.slime_disagg.modal_train
```

A config with a real acceptance signal (rather than the smoke-test protocol
check) runs the same transport while the reward metrics climb. The
`passrate/pass@1` and `passrate/pass@8` metrics should trend up. Useful log
searches (the app name is the config's `APP_NAME`):

```bash
m app logs <app-name> --since 4h --search "passrate "
m app logs <app-name> --since 4h --search "Published sparse delta"
```
