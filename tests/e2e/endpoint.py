"""Miles trains against a Stitch run while an engine joins; each engine must match the trainer's last save.

modal run tests/e2e/endpoint.py --config glm5.3-cut-nvfp4
With --no-check-state, no engine joins and only each version's apply checksums are checked.

A training run: a setup other than the config's own, more rollouts, and W&B:
modal run -d tests/e2e/endpoint.py \
    --config qwen3.6-35b-a3b-nvfp4-node --setup dapo-training --no-check-state \
    --num-rollout 30 --wandb-group qwen3.6-35b-a3b-nvfp4-fft-math
"""

import ast
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import modal

# The repo root here, for Stitch; "/" in the container, where this file is /root/endpoint.py.
sys.path.insert(0, str(Path(__file__).parents[1].parent))
import harness  # noqa: E402
from configs import CONFIGS  # noqa: E402

NUM_ROLLOUT = 5
# Half the batch of Miles' DAPO math runs (scripts/run_qwen3_30b_a3b.py, scripts/run_qwen3_dense.py).
DAPO_ROLLOUT = (
    "--rm-type deepscaler "
    "--rollout-batch-size 16 --n-samples-per-prompt 8 --rollout-max-response-len 8192 "
    "--rollout-temperature 1 --global-batch-size 128 --balance-data "
)
# The batch of Miles' DAPO math runs, for training rather than testing.
DAPO_TRAINING_ROLLOUT = (
    "--rm-type deepscaler "
    "--rollout-batch-size 32 --n-samples-per-prompt 8 --rollout-max-response-len 8192 "
    "--rollout-temperature 1 --global-batch-size 256 --balance-data "
)
# Every reply carries its sampling support and routed experts.
REPLAY_ROLLOUT = (
    DAPO_ROLLOUT
    + "--rollout-top-p 0.9 --rollout-top-k 128 --use-rollout-sampling-support-replay "
    + "--use-rollout-routing-replay "
)
# Filtered sampling without replay: the rollout records the sampler's own log-probs, and the
# trainer scores the full vocabulary.
TOP_P_ROLLOUT = DAPO_ROLLOUT + "--rollout-top-p 0.9 "
TOP_P_TOP_K_ROLLOUT = DAPO_ROLLOUT + "--rollout-top-p 0.9 --rollout-top-k 128 "
# Score centering over the replayed support, as its README enables it, with MIS weights.
SCORE_CENTERING_LOSS = (
    "--loss-type score_centering --advantage-estimator grpo --rollout-top-logprobs-num 128 "
    "--score-centering-is mis --use-rollout-logprobs --disable-grpo-std-normalization "
    "--calculate-per-token-loss "
)
# Without replay score centering normalizes over the full vocabulary, so it runs in chunks
# and is recomputed in backward instead of saved.
SCORE_CENTERING_FULL_VOCABULARY_LOSS = (
    SCORE_CENTERING_LOSS + "--log-probs-chunk-size 2048 --recompute-loss-function "
)


@dataclass(frozen=True)
class Setup:
    engines: int
    # One more engine joins once the run publishes this version.
    join_at: int
    rollout: str = harness.SEEDED_ROLLOUT
    loss: str = harness.POLICY_LOSS
    # Saved after this step, the run resumes there and publishes the versions after it again.
    resume_at: int | None = None
    # Saves every this many steps into the run's folder on the Volume, where they stay; otherwise
    # once, at the end, on the trainer's disk.
    save_interval: int | None = None


# The GLM-5.3 cut writes meaningless text, so only the seeded reward fits it.
SETUPS = {
    "glm5.3-cut-nvfp4": Setup(engines=2, join_at=4, resume_at=2),
    "glm5.3-cut-fp8": Setup(engines=2, join_at=4, resume_at=2),
    "muse-spark-cut-bf16": Setup(engines=2, join_at=4, resume_at=2),
    "muse-spark-cut-bf16-lora": Setup(engines=2, join_at=4),
    **{
        name: Setup(engines=4, join_at=3, rollout=DAPO_ROLLOUT)
        for name in (
            "qwen3.6-35b-a3b-nvfp4-node",
            "qwen3.6-35b-a3b-fp8-node",
            "qwen3.6-35b-a3b-fp8-node-lora",
            "qwen3.6-35b-a3b-nvfp4-node-dflash2",
            "qwen3.8-27b-bf16-node",
            "qwen3.8-27b-fp8-node",
        )
    },
    "qwen3.6-35b-a3b-fp8-node-replay": Setup(
        engines=4, join_at=3, rollout=REPLAY_ROLLOUT, loss=SCORE_CENTERING_LOSS
    ),
    "qwen3.6-35b-a3b-fp8-node-top-p": Setup(
        engines=4, join_at=3, rollout=TOP_P_ROLLOUT
    ),
    "qwen3.6-35b-a3b-fp8-node-top-p-score-centering": Setup(
        engines=4,
        join_at=3,
        rollout=TOP_P_TOP_K_ROLLOUT,
        loss=SCORE_CENTERING_FULL_VOCABULARY_LOSS,
    ),
    "dapo-training": Setup(
        engines=8, join_at=3, rollout=DAPO_TRAINING_ROLLOUT, save_interval=10
    ),
}

app = modal.App(
    "stitch-endpoint-e2e",
    image=harness.image.add_local_python_source("harness", "stitch"),
)
TRAINER = dict(
    **harness.NODE,
    secrets=[
        modal.Secret.from_name("stitch-proxy-auth"),
        modal.Secret.from_name("wandb-secret"),
    ],
)


@app.function(gpu="B200:8", **TRAINER)
@modal.clustered(size=1)
def train_b200(
    name: str, setup: str, run_dir: str, stitch_run: dict, options: dict
) -> dict:
    return _train_against_run(name, SETUPS[setup], Path(run_dir), stitch_run, **options)


@app.function(gpu="H200:8", **TRAINER)
@modal.clustered(size=1)
def train_h200(
    name: str, setup: str, run_dir: str, stitch_run: dict, options: dict
) -> dict:
    return _train_against_run(name, SETUPS[setup], Path(run_dir), stitch_run, **options)


@app.function(gpu="B300:8", **TRAINER)
@modal.clustered(size=1)
def train_b300(
    name: str, setup: str, run_dir: str, stitch_run: dict, options: dict
) -> dict:
    return _train_against_run(name, SETUPS[setup], Path(run_dir), stitch_run, **options)


def _train_against_run(
    name: str,
    setup: Setup,
    run_dir: Path,
    stitch_run: dict,
    check_state: bool,
    num_rollout: int,
    wandb_group: str,
) -> dict:
    config = CONFIGS[name]
    logs = run_dir / "logs"
    if logs.exists():
        raise RuntimeError(
            f"{run_dir} has a trainer already: Modal retried this one after losing its first worker"
        )
    logs.mkdir(parents=True)
    result = {
        "revisions": harness.install_overlays(),
        "run_dir": str(run_dir),
        # Where the trainer landed sets its Volume commit's round trips.
        "trainer_place": {
            key: os.environ.get(key) for key in ("MODAL_REGION", "MODAL_CLOUD_PROVIDER")
        },
    }
    # Ray's processes inherit them from `ray start`; arguments would reach W&B and the checkpoints.
    os.environ["MILES_ROLLOUT_EXTERNAL_API_KEY"] = (
        f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}"
    )
    os.environ["STITCH_VERSIONS_VOLUME"] = stitch_run["versions_volume"]
    pool = modal.Server.from_name(stitch_run["app_name"], "EnginePool")
    base_args = (
        f"{harness.recipe_args(config, run_dir, setup.rollout, setup.loss)}--num-rollout {num_rollout} "
        f"--rollout-external-url {stitch_run['generate_url']} "
        f"--rollout-external-control-url {stitch_run['control_url']} "
        "--custom-update-weight-post-write-path stitch.trainer.commit_versions "
        "--no-save-optim "
    )
    if wandb_group:
        # W&B reads its key from the environment the wandb-secret sets.
        base_args += f"--use-wandb --wandb-project stitch --wandb-group {wandb_group} "
    saves = harness.SAVE if setup.save_interval is None else str(run_dir / "save")
    phases = ["train"] if setup.resume_at is None else ["train", "resume"]
    try:
        for phase in phases:
            save = f"{saves}/{phase}"
            args = f"{base_args}--save {save}/torch_dist --save-hf {save}/hf/step_{{rollout_id}} "
            if phase == "resume":
                result["engines"] = _engines(pool)
                args += (
                    f"--load {saves}/train/torch_dist --ckpt-step {setup.resume_at} "
                    "--no-load-optim --no-load-rng "
                )
            last = phase == phases[-1]
            if last and check_state:
                result["joined"] = _join_once_published(setup, pool, stitch_run)
            interval = (
                (setup.save_interval or num_rollout) if last else setup.resume_at + 1
            )
            harness.train(
                config,
                logs / f"{phase}.log",
                f"{args}--save-interval {interval} ",
                fully_async=True,
                cpus=harness.CPUS,
            )
        if check_state:
            result["reference"] = _reference_state(
                config, f"{saves}/{phases[-1]}/hf/step_{num_rollout - 1}", logs
            )
            result["serving_after_s"] = _wait_serving(
                stitch_run, setup.engines + 1, num_rollout
            )
        result["engines_at_end"] = _engines(pool)
    except Exception as error:
        result["error"] = repr(error)
    finally:
        for phase in phases:
            result[f"{phase}_served"] = _served(logs / f"{phase}.log")
            result[f"{phase}_syncs"] = harness.syncs(logs / f"{phase}.log")
        (run_dir / "result.json").write_text(json.dumps(result, indent=1))
        harness.data.commit()
    return result


def _join_once_published(setup: Setup, pool: modal.Server, stitch_run: dict) -> dict:
    """Adds an engine once the run, first below it, publishes `setup.join_at`; returns where."""
    joined: dict = {}

    def join():
        below = False
        while True:
            head = _status(stitch_run)["head"]
            below |= head is None or head < setup.join_at
            if below and head is not None and head >= setup.join_at:
                break
            time.sleep(10)
        count = setup.engines + 1
        pool.autoscaler.update(min_containers=count, max_containers=count)
        joined["at"] = head

    threading.Thread(target=join, daemon=True).start()
    return joined


def _wait_serving(stitch_run: dict, engines: int, version: int) -> float:
    """Seconds until `engines` serve the last version: a joiner or replacement may still be loading."""
    started = time.monotonic()
    while _status(stitch_run)["serving"].get(str(version), 0) < engines:
        if time.monotonic() - started > 3600:
            raise TimeoutError(
                f"{engines} engines did not serve version {version} in 3600 s"
            )
        time.sleep(10)
    return time.monotonic() - started


def _status(stitch_run: dict) -> dict:
    request = urllib.request.Request(
        f"{stitch_run['control_url']}/status",
        headers={
            "Authorization": f"Bearer {os.environ['MILES_ROLLOUT_EXTERNAL_API_KEY']}"
        },
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                return json.loads(reply.read())
        except urllib.error.URLError as error:
            # The control can answer 5xx while its container starts.
            if getattr(error, "code", 500) < 500 or attempt == 5:
                raise
        time.sleep(10)


def _reference_state(config, save: str, logs: Path) -> dict:
    """Every tensor's checksum in an engine loaded from the trainer's last save, on its node."""
    gpus = ",".join(map(str, range(config.engine_gpus)))
    reference = harness.Engine(
        save,
        config,
        gpus,
        harness.ENGINE_PORT,
        logs / "reference.log",
    )
    try:
        reference.wait()
        return reference.state()
    finally:
        reference.stop()


def _engines(pool: modal.Server) -> list[str]:
    return sorted(c.container_id for c in pool._experimental_list_containers())


def _served(log: Path) -> dict[int, dict]:
    if not log.exists():
        return {}
    lines = re.finditer(
        r"the external rollout serves version (\d+): (\{.*\})",
        log.read_text(errors="replace"),
    )
    return {int(match[1]): ast.literal_eval(match[2]) for match in lines}


# Run in an engine container: SGLang's key and port off its command line, then every checksum,
# written to the run's Volume (argv: path, Volume) as exec's output can lose a large reply.
_STATE_SCRIPT = """
import io, json, os, sys, urllib.request
import modal
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\\0")
    except OSError:
        continue
    if "sglang.launch_server" in argv and "--admin-api-key" in argv:
        break
flag = lambda name: argv[argv.index(name) + 1]
request = urllib.request.Request(
    f"http://127.0.0.1:{flag('--port')}/weights_checker",
    data=json.dumps({"action": "state_checksum"}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + flag("--admin-api-key")},
)
reply = urllib.request.urlopen(request, timeout=1800).read()
# Engines mount the Volume read-only: the reply goes up through the Volume's API.
with modal.Volume.from_name(sys.argv[2]).batch_upload(force=True) as batch:
    batch.put_file(io.BytesIO(reply), sys.argv[1])
"""


def _engine_state(container_id: str, run_dir: Path) -> dict:
    path = run_dir / "states" / f"{container_id}.json"
    command = [sys.executable, "-m", "modal", "container", "exec", "--no-pty"]
    command += [container_id, "--", "python3", "-c", _STATE_SCRIPT]
    subprocess.run(
        [*command, str(path.relative_to(harness.DATA)), harness.data.name], check=True
    )
    reply = b"".join(harness.data.read_file(str(path.relative_to(harness.DATA))))
    return harness.tp_states(json.loads(reply)["ranks"])


@app.local_entrypoint()
def main(
    config: str = "glm5.3-cut-nvfp4",
    check_state: bool = True,
    setup: str = "",
    num_rollout: int = NUM_ROLLOUT,
    wandb_group: str = "",
):
    from stitch.run import PoolConfig, RunConfig, create_run
    from stitch.sglang import SGLangConfig

    setup_name = setup or config
    e2e, setup = CONFIGS[config], SETUPS[setup_name]
    # Runs launched together may start in the same second; the suffix keeps their apps apart.
    stamp = f"{datetime.now().strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    run_dir = Path(harness.DATA, "endpoint", config, stamp)
    run_config = RunConfig(
        run_id=f"endpoint-e2e-{stamp}",
        image=harness.MILES_IMAGE,
        image_commands=harness.ENGINE_COMMANDS,
        volumes={
            harness.CHECKPOINTS: harness.checkpoints.name,
            harness.DATA: harness.data.name,
        },
        checkpoint=f"{harness.CHECKPOINTS}/{e2e.rollout_checkpoint}",
        versions=str(run_dir / "deltas"),
        engine=SGLangConfig(
            server_args=harness.server_args(e2e.engine_args), env=e2e.env
        ),
        pool=PoolConfig(
            gpu=f"{e2e.gpu}:{e2e.engine_gpus}",
            cpu=max(16, 8 * e2e.engine_gpus),
            memory=max(131072, 65536 * e2e.engine_gpus),
            min_containers=setup.engines,
            max_containers=setup.engines,
            # Off reserved capacity; Modal takes this only on a whole-node cluster, as Stitch runs
            # an 8-GPU engine.
            experimental_options=(
                {"exclude_reserved_capacity": True} if e2e.engine_gpus == 8 else {}
            ),
        ),
    )
    run = create_run(run_config)
    # A Volume looked up by name knows its name only where it was looked up: here.
    stitch_run = {
        "generate_url": run.generate_url,
        "control_url": run.url,
        "app_name": run_config.app_name,
        "versions_volume": run_config.versions_volume,
    }
    train = {"B200": train_b200, "H200": train_h200, "B300": train_b300}[e2e.gpu]
    try:
        options = dict(
            check_state=check_state, num_rollout=num_rollout, wandb_group=wandb_group
        )
        result = train.remote(config, setup_name, str(run_dir), stitch_run, options)
        if "reference" in result:
            reference = result.pop("reference")
            result["checks"] = {
                container: harness.compare(_engine_state(container, run_dir), reference)
                for container in result["engines_at_end"]
            }
    finally:
        run.stop()

    print(json.dumps(result, indent=1)[:8000])
    failures = _failures(setup, result, check_state, num_rollout)
    if failures:
        raise SystemExit(f"endpoint e2e FAILED: {failures}")
    print("endpoint e2e passed")


def _failures(
    setup: Setup, result: dict, check_state: bool, num_rollout: int
) -> list[str]:
    if result.get("error"):
        return [result["error"]]
    failures = []
    # A launch's version is served without /versions.
    firsts = (
        {"train": 1}
        if setup.resume_at is None
        else {"train": 1, "resume": setup.resume_at + 2}
    )
    for run, first in firsts.items():
        served = result[f"{run}_served"]
        missing = [
            v
            for v in range(first, num_rollout + 1)
            if "perf/update_weights_external_s" not in served.get(v, {})
        ]
        if missing:
            failures.append(f"{run}: versions {missing} not served through /versions")
    if setup.resume_at is not None and set(result["engines"]) & set(
        result["engines_at_end"]
    ):
        failures.append(
            f"an engine past the resume kept serving: {result['engines']} -> "
            f"{result['engines_at_end']}"
        )
    if not check_state:
        return failures
    if len(result["engines_at_end"]) != setup.engines + 1:
        failures.append(
            f"{len(result['engines_at_end'])} engines at the end, not {setup.engines + 1}: "
            f"joined {result['joined']}"
        )
    for container, check in result["checks"].items():
        if check["differ"] or check["only_one"] or not check["tensors"]:
            failures.append(
                f"engine {container} is not the trainer's save: {json.dumps(check)[:2000]}"
            )
    return failures
