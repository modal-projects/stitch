# Stitch

Autoscaled SGLang engines on Modal that follow a trainer's weights through lossless weight deltas.

## Use

Install with `pip install "git+https://github.com/modal-projects/stitch@dev"` and select your Modal environment.

```python
from stitch.run import PoolConfig, RunConfig, create_run
from stitch.sglang import SGLangConfig

# The engines need an SGLang that serves weight deltas: our branch, copied over the image's.
SGLANG = (
    "python3 -m pip install --no-deps nvidia-libnvcomp-cu13==5.3.0.16"
    " && git clone --filter=blob:none https://github.com/modal-projects/sglang /tmp/sglang"
    " && git -C /tmp/sglang checkout 00d4a6283b"
    " && cp -a /tmp/sglang/python/sglang/. /sgl-workspace/sglang/python/sglang/"
)

run = create_run(RunConfig(
    run_id="my-run",
    image="radixark/miles:dev-202611121556",
    image_commands=[SGLANG],
    volumes={"/models": "my-models", "/runs": "my-runs"},
    checkpoint="/models/my-model",  # version 0
    # On another Volume than the checkpoint: an engine's first reload drops what it read through that Volume.
    versions="/runs/my-run/versions",
    engine=SGLangConfig(server_args={"--tp-size": "8", "--expert-parallel-size": "8"}),
    pool=PoolConfig(gpu="B200:8", min_containers=2, max_containers=8),
))
```

Train with Miles from
[`modal-projects/miles@stitch-miles-weight-delta`](https://github.com/modal-projects/miles/tree/stitch-miles-weight-delta):

```
--update-weight-transfer-mode weight-delta --update-weight-disk-dir /runs/my-run/versions --fully-async
--rollout-external-url <run.generate_url> --rollout-external-control-url <run.url>
--custom-update-weight-post-write-path stitch.trainer.commit_versions
```

with `STITCH_VERSIONS_VOLUME` set to `config.versions_volume` (`my-runs`) and `MILES_ROLLOUT_EXTERNAL_API_KEY` set to a Modal proxy token
(`modal workspace proxy-tokens create`, then `<id>.<secret>`). Call `run.stop()` when training ends.

An engine's cold start (loading the checkpoint) usually outlasts the trainer's, so start the trainer once
`run.status()["catching_up"]` counts the engines you want loaded.

Another trainer calls `POST /start {"version"}` once, then for each version writes its delta under `versions`,
commits the Volume and calls `POST /versions {"version", "base_version", "start", "start_id"}`. Each version
builds on the one before; numbers only need to grow. A trainer that polls `GET /status` passes `start`,
`start_id` and its newest version as `head`.

Modal's load balancer drops a connection that stays silent for 350 s: a client whose generations run longer
streams them or sets TCP keepalive on its sockets, as Miles' branch does.

## Tests

See [`tests/README.md`](tests/README.md).
