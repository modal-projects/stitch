# Stitch

Autoscaled SGLang engines on Modal that follow a trainer's weights through lossless weight deltas.

## Use

Install with `pip install "git+https://github.com/modal-projects/stitch@dev"` and select your Modal environment.

```python
from stitch.run import PoolConfig, RunConfig, create_run
from stitch.sglang import SGLangConfig

# The engines need an SGLang that serves weight deltas: our branch, and the Miles whose receiver it imports,
# copied over the image's.
SGLANG = (
    "python3 -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12"
    " && git clone --filter=blob:none https://github.com/modal-projects/sglang /tmp/sglang"
    " && git -C /tmp/sglang checkout e09dd6c2c4"
    " && cp -a /tmp/sglang/python/sglang/. /sgl-workspace/sglang/python/sglang/"
    " && git clone --filter=blob:none https://github.com/modal-projects/miles /tmp/miles"
    " && git -C /tmp/miles checkout ed864dce11"
    " && cp -a /tmp/miles/miles/. /root/miles/miles/"
)

run = create_run(RunConfig(
    run_id="my-run",
    image="radixark/miles:dev-202611220647",
    image_commands=[SGLANG],
    volumes={"/models": "my-models", "/runs": "my-runs"},
    checkpoint="/models/my-model",  # every publication chain starts from it
    # On another Volume than the checkpoint: an engine's first reload drops what it read through that Volume. A v2
    # Volume (`modal volume create --version 2`): the trainer seals each version with a hard link, which v1 refuses.
    versions="/runs/my-run/versions",
    engine=SGLangConfig(server_args={"--tp-size": "8", "--expert-parallel-size": "8"}),
    pool=PoolConfig(gpu="B200:8", min_containers=2, max_containers=8),
))
```

Train with Miles from
[`modal-projects/miles@stitch-dev`](https://github.com/modal-projects/miles/tree/stitch-dev):

```
--update-weight-transfer-mode gpu-delta --update-weight-disk-dir /runs/my-run/versions --fully-async
--rollout-external-url <run.generate_url> --rollout-external-control-url <run.url>
--custom-update-weight-post-write-path stitch.trainer.commit_versions
```

with `STITCH_VERSIONS_VOLUME` set to `config.versions_volume` (`my-runs`) and `MILES_ROLLOUT_EXTERNAL_API_KEY` set to a Modal proxy token
(`modal workspace proxy-tokens create`, then `<id>.<secret>`). Call `run.stop()` when training ends.

An engine's cold start (loading the checkpoint) usually outlasts the trainer's, so start the trainer once
`run.status()["catching_up"]` counts the engines you want loaded.

Another trainer calls `POST /start {"version", "publication"}` once, then for each version writes its delta under
`versions`, commits the Volume and calls `POST /versions {"version", "base_version", "start", "start_id", "publication"}`.
`publication` names the version's GPU-delta publication as Miles announces it: `manifest` (relative to `versions`),
`manifest_sha256`, `stream_id`, `base_stream_id`, `base_version`, `target_version`, `plan_digest` and
`payload_checksum_format`. Each version builds on the one before; numbers only need to grow. An engine catches up
by following each publication's base back to its own weights, so publications stay on the Volume for the run. A
trainer that polls `GET /status` passes `start`, `start_id` and its newest version as `head`.

Modal's load balancer drops a connection that stays silent for 350 s: a client whose generations run longer
streams them or sets TCP keepalive on its sockets, as Miles' branch does.

## Adapters on demand

A trainer that saves every adapter version under a new name and lets each request name the version it samples
(Tinker-style multi-LoRA) publishes the adapters instead of weight versions. Give them a Volume mount of their own,
on a v2 Volume, and start the engines with SGLang's LoRA flags, without attention DP: adapters rotate through
SGLang's LoRA slots, and attention DP serves only pinned ones.

```python
RunConfig(
    ...,
    volumes={"/models": "my-models", "/runs": "my-runs", "/adapters": "my-adapters"},
    adapters="/adapters/my-run",
    engine=SGLangConfig(server_args={
        "--enable-lora": "", "--max-lora-rank": "8", "--lora-target-modules": "all",
        "--max-loras-per-batch": "8", "--max-loaded-loras": "32",
    }),
)
```

Engines keep a local copy of each adapter at the path a request names. Miles' Tinker gateway publishes each sampler
save with:

```
--rollout-external-url <run.generate_url> --rollout-external-control-url <run.url>
--custom-update-weight-post-write-path stitch.trainer.commit_adapters --worker-comm-backend rpc
--tinker-sampler-root /adapters/my-run --tinker-checkpoint-root <training checkpoints, not under adapters>
```

with `STITCH_ADAPTERS_VOLUME` set to `config.adapters_volume` (`my-adapters`) and `MILES_ROLLOUT_EXTERNAL_API_KEY` as
above. A save returns once every serving engine holds the adapter. It holds the trainer only for the export: the
Volume commit runs beside training (`--worker-comm-backend rpc`) and one commit covers every export made before it.
A commit crosses to the Volume's region, so a trainer far from it waits longer for each save.

Another trainer calls `POST /start {"version": -1}` once (the checkpoint's weights). Each save is a new directory
under `adapters` that a symlink points to, as Miles writes `.<name>.<commit>/` and switches `<name>` to it; a linked
directory never changes. After committing the Volume the trainer calls `POST /adapters {"path", "start", "start_id"}`
with the directory, and again once it is deleted. Engines read what the Volume holds: a joining engine, or one that
finds the control service restarted, copies every linked directory.

### In place

With `adapters_in_place=True` (Miles' `--tinker-sampler-backend in_place`; set the two together), each link is one
adapter that requests name by its path under `adapters`, and each save replaces it on every engine: SGLang swaps it
at a step boundary if it is loaded, and otherwise only learns where the new version is. The trainer writes
`version.json` (`{"version": n}`, counting that adapter's saves) into each directory and calls `POST /adapters` with
the link; the call returns once every serving engine serves the save, and responses report each token's adapter
version. Deleting the link removes the adapter from every engine.

## Scaling

Modal's autoscaler sizes the pool: with `target_concurrency` set, it keeps about that many requests in flight per
engine between `min_containers` and `max_containers`. An engine it removes drains: it takes no new requests and has
up to `exit_grace_period` seconds (3600 by default, Modal's limit) to finish those in flight, stopping once they
end. Without a trainer call for `clocks.idle` seconds, the pool's minimum drops to 0 until the trainer is back.

### Auto mode

`PoolConfig(autoscale=True)` runs the engines that keep the trainer fed; the control service sets the count and
Modal runs it. After each training step the rollout posts it, `POST /steps {"seconds", "waited", "held",
"consumed", "produced"}`: the step's length, how long the trainer waited for groups (0 when its batch was ready),
how long the producer was held back by a full buffer or by admission, the groups the trainer took and the usable
groups produced meanwhile. A step's need is the serving engines times the trainer's rate over the rollout's,
`consumed * (seconds - held) / ((seconds - waited) * produced)`.

- The count rises when the trainer waited in each of three steps in a row, and falls when each of them would still
  be fed by one engine fewer; either way it moves halfway to their need, since production does not grow in
  proportion to engines.
- A rise promises production in proportion to the engines it adds, up to the need. One that raised production by
  less than a quarter of that is undone, and the count does not rise again while production stays there: the
  rollout, not the engines, is then the limit. A fall after which the trainer waited is not tried again from that
  count while production stays there.
- A step counts only if the serving engines and the count held through it, so a preempted engine or one still
  starting delays decisions rather than skewing them. A change takes three steps plus an engine's start.
- It assumes a group lives at most about two steps, so the steps after a change see what it did, and one policy
  per pool.

Each engine reports SGLang's `/metrics` every 30 s, and every minute the rollout posts its load, `POST /load
{"finished_sessions", "finished_groups", "session_seconds", "turn_seconds", "prefix_tokens", "uncached_prefix_tokens",
"sessions_in_flight", "waiting"}`: the sessions and groups finished since its last report, their seconds in all and
in requests, the tokens their turns shared with the turns they continued and of those the ones the engine did not
find cached, the sessions in flight, and how long the trainer has waited so far. The reply carries `capacity`, the
requests the serving engines run at once, which Miles' admission follows. From both, in every run, auto mode
or not, the control service solves each minute the count that auto mode's next rule would set and logs it (`shadow:
engines 4 -> 6 (trainer starving, engines stressed): prefix kept 0.62`); `/status` shows it under `shadow`. It does
not set the count yet.

To use it:

- Stitch: `PoolConfig(autoscale=True, min_containers=..., max_containers=...)`. The run starts at `min_containers`
  (at least 1) and never goes below it; `max_containers` caps the cost. Leave `target_concurrency` unset, and keep
  `exit_grace_period` long enough for the longest request, so a removed engine finishes what it runs.
- Miles, fully async in endpoint mode (`--fully-async --rollout-external-url ... --rollout-external-control-url
  ...`): the rollout posts its steps and its load itself.
  - `--async-max-concurrent-samples` is the work in flight that more engines can spread; about the staleness
    window times the batch.
  - `--sglang-server-concurrency` caps the requests in flight in endpoint mode: set it at least as high as
    `--async-max-concurrent-samples`.
  - `--max-weight-staleness N --async-submit-within-staleness` keep what is generated within what the trainer will
    use, if wanted.
- Any other rollout posts `/steps` after each training step, and may post `/load`; without it the shadow judges by
  the engines alone.

The control service logs each change (`engines 2 -> 3, 2 serving: needs [...]`), each undone rise, and a trainer
that waits with the engines at `max_containers`.

## Tests

See [`tests/README.md`](tests/README.md).
