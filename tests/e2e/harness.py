"""Miles trains a model for a few steps while one SGLang engine serves its rollouts and takes
every update as a weight delta. At the end, that engine must hold the same bytes, tensor for
tensor on every GPU, as a fresh engine started from the trainer's own save of the last version, in the
rollout checkpoint's dtypes as the trainer publishes it.

One node: GPUs 0-3 train, GPUs 4-7 serve. The harness starts the engine and Miles reaches it
through --rollout-external-engine-addrs, so the engine outlives the training run. Rollouts are
short and their reward is seeded random: the run exercises the infrastructure, not learning.

With --resume-at S, a first run publishes two versions past S and stops; the run then resumes from
its save at S into a fresh engine, which Miles brings to S with a publication from the checkpoint.

    modal run tests/e2e/harness.py --config glm5.3-cut-nvfp4
"""

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from glob import glob
from pathlib import Path

import modal
from configs import CONFIGS, Config

# Our SGLang and Miles branches, at their heads when the image is built; SGLANG_SOURCE and
# MILES_SOURCE put local trees in their place while developing.
SGLANG_REPO, SGLANG_BRANCH = (
    "https://github.com/modal-projects/sglang.git",
    "stitch-dev",
)
MILES_REPO, MILES_BRANCH = (
    "https://github.com/modal-projects/miles.git",
    "stitch-dev",
)
SGLANG_SOURCE, MILES_SOURCE = (
    os.environ.get("SGLANG_SOURCE"),
    os.environ.get("MILES_SOURCE"),
)
# A Miles build for its Megatron, Megatron-Bridge and TransformerEngine;
# its Miles and SGLang trees are replaced by ours.
MILES_IMAGE = "radixark/miles:dev-202611191721"
MILES, MILES_OVERLAY = "/root/miles", "/root/miles-overlay"
MILES_DIRS = ("miles", "miles_plugins", "scripts", "tools")
MEGATRON_PATH = "/root/Megatron-LM"
SGLANG, SGLANG_OVERLAY = "/sgl-workspace/sglang/python/sglang", "/root/sglang-overlay"
SGLANG_BUILD_PRODUCTS = (
    "_version.py",
    "srt/rust_extensions/*.so",
    "srt/mem_cache/rust_tree_core/*.so",
)
SGLANG_CACHE = "/root/.cache/sglang"
REWARD_DIR = "/root/stitch-e2e"
# lora_slot_check, which a sitecustomize there installs into a served-LoRA engine's processes.
LORA_CHECK_DIR = "/root/lora-check"
CHECKPOINTS, DATA = "/checkpoints", "/data"
SAVE = "/tmp/save"
# The engine takes the GPUs after the trainer's, as many as its config asks for.
ENGINE_GPU_START = 4
CPUS = 64
ENGINE_PORT = 30000
NUM_ROLLOUT = 10
# Miles' name for its single adapter (miles.utils.lora.utils.LORA_ADAPTER_NAME).
LORA_ADAPTER = "miles_lora"
LORA_PROMPTS = [
    "The river",
    "Compute 17 * 23 and explain each step.",
    "Write a haiku about pipelines.",
    "List three prime numbers greater than 100.",
]
# A full MoE's CPU-offload Adam takes most of an hour to build on these nodes.
TRAIN_TIMEOUT = 3 * 3600


def _revision(source: str) -> str:
    git = ["git", "-C", source]
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    changes = subprocess.run(
        [*git, "diff", "HEAD"], capture_output=True, text=True
    ).stdout
    changes += subprocess.run(
        [*git, "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    ).stdout
    if not changes:
        return head
    return f"{head}+dirty-{hashlib.sha256(changes.encode()).hexdigest()[:12]}"


def _branch_head(repo: str, branch: str) -> str:
    listed = subprocess.run(
        ["git", "ls-remote", repo, f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return listed.split()[0]


def _revisions() -> str:
    if not modal.is_local():
        return os.environ.get("E2E_REVISIONS", "{}")
    return json.dumps(
        {
            "sglang": _revision(SGLANG_SOURCE)
            if SGLANG_SOURCE
            else _branch_head(SGLANG_REPO, SGLANG_BRANCH),
            "miles": _revision(MILES_SOURCE)
            if MILES_SOURCE
            else _branch_head(MILES_REPO, MILES_BRANCH),
        }
    )


REVISIONS = json.loads(_revisions())


def _with_tree(image, source, repo, revision, subdir, target, ignore):
    """`image` with `subdir` of a local tree, or of `repo` at `revision`, at `target`."""
    if source:
        return image.add_local_dir(Path(source, subdir), target, ignore=ignore)
    return image.run_commands(
        f"git clone --filter=blob:none --no-checkout {repo} /tmp/tree "
        f"&& git -C /tmp/tree checkout --quiet {revision} "
        f"&& mv {Path('/tmp/tree', subdir)} {target} && rm -rf /tmp/tree"
    )


IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.so", "_version.py"]
TREES = [
    (
        SGLANG_SOURCE,
        SGLANG_REPO,
        REVISIONS["sglang"],
        "python/sglang",
        SGLANG_OVERLAY,
        IGNORE,
    ),
    (
        MILES_SOURCE,
        MILES_REPO,
        REVISIONS["miles"],
        "",
        MILES_OVERLAY,
        [".git", *IGNORE, "docs/**", "charts/**", "tests/**"],
    ),
]
image = (
    modal.Image.from_registry(MILES_IMAGE)
    .run_commands(
        "python3 -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12",
        f"rm -rf {SGLANG_CACHE}",
    )
    .env({"E2E_REVISIONS": json.dumps(REVISIONS)})
)
# A clone is a build step, which Modal allows only before the local trees it mounts at start-up.
for tree in sorted(TREES, key=lambda tree: tree[0] is not None):
    image = _with_tree(image, *tree)
image = (
    image.add_local_file(
        Path(__file__).with_name("seeded_reward.py"), f"{REWARD_DIR}/seeded_reward.py"
    )
    .add_local_file(
        Path(__file__).with_name("lora_slot_check.py"),
        f"{LORA_CHECK_DIR}/lora_slot_check.py",
    )
    .add_local_python_source("configs")
)
# What a run's engines build onto MILES_IMAGE: our SGLang copied over the image's, so the files the image
# built for its SGLang stay, and our Miles, whose GPU-delta receiver SGLang imports.
ENGINE_COMMANDS = [
    "python3 -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12",
    f"git clone --filter=blob:none --no-checkout {SGLANG_REPO} /tmp/sglang "
    f"&& git -C /tmp/sglang checkout --quiet {REVISIONS['sglang']} "
    f"&& cp -a /tmp/sglang/python/sglang/. {SGLANG}/ && rm -rf /tmp/sglang",
    f"git clone --filter=blob:none --no-checkout {MILES_REPO} /tmp/miles "
    f"&& git -C /tmp/miles checkout --quiet {REVISIONS['miles']} "
    f"&& cp -a /tmp/miles/miles/. {MILES}/miles/ && rm -rf /tmp/miles",
]
app = modal.App("weight-delta-e2e", image=image)
checkpoints = modal.Volume.from_name("miles-checkpoints")
data = modal.Volume.from_name("gpu-delta-tests")
sglang_cache = modal.Volume.from_name("sglang-cache", create_if_missing=True, version=2)
EXPERIMENTAL_OPTIONS = {"exclude_reserved_capacity": True}
NODE = dict(
    cpu=CPUS,
    memory=1048576,
    ephemeral_disk=2097152,
    timeout=4 * 3600,
    volumes={CHECKPOINTS: checkpoints, DATA: data, SGLANG_CACHE: sglang_cache},
    experimental_options=EXPERIMENTAL_OPTIONS,
)


@app.function(gpu="B200:8", **NODE)
@modal.clustered(size=1)
def run_b200(
    name: str,
    profile_step: int,
    fully_async: bool,
    num_rollout: int,
    dump_details: bool,
    resume_at: int,
) -> dict:
    return _run(
        name,
        CONFIGS[name],
        profile_step,
        fully_async,
        num_rollout,
        dump_details,
        resume_at,
    )


@app.function(gpu="H200:8", **NODE)
@modal.clustered(size=1)
def run_h200(
    name: str,
    profile_step: int,
    fully_async: bool,
    num_rollout: int,
    dump_details: bool,
    resume_at: int,
) -> dict:
    return _run(
        name,
        CONFIGS[name],
        profile_step,
        fully_async,
        num_rollout,
        dump_details,
        resume_at,
    )


@app.function(gpu="B300:8", **NODE)
@modal.clustered(size=1)
def run_b300(
    name: str,
    profile_step: int,
    fully_async: bool,
    num_rollout: int,
    dump_details: bool,
    resume_at: int,
) -> dict:
    return _run(
        name,
        CONFIGS[name],
        profile_step,
        fully_async,
        num_rollout,
        dump_details,
        resume_at,
    )


@app.local_entrypoint()
def main(
    config: str,
    profile_step: int = 0,
    fully_async: bool = False,
    num_rollout: int = NUM_ROLLOUT,
    dump_details: bool = False,
    resume_at: int = 0,
):
    run = {"B200": run_b200, "H200": run_h200, "B300": run_b300}[CONFIGS[config].gpu]
    result = run.remote(
        config, profile_step, fully_async, num_rollout, dump_details, resume_at
    )
    print(json.dumps({key: result[key] for key in ("revisions", "run_dir")}, indent=1))
    for version, metrics in sorted(
        result["syncs"].items(), key=lambda item: int(item[0])
    ):
        print(f"v{version}: {json.dumps(metrics)}")
    check = result["check"]
    print(
        f"bit-exact check at v{_last_version(fully_async, num_rollout)}: {json.dumps(check)[:2000]}"
    )
    if result.get("error") or check["differ"] or check["only_one"]:
        raise SystemExit(f"e2e FAILED: {result.get('error') or 'engines differ'}")
    print("e2e passed")


def _engine_gpus(config: Config) -> str:
    return ",".join(str(ENGINE_GPU_START + gpu) for gpu in range(config.engine_gpus))


def _last_version(fully_async: bool, num_rollout: int) -> int:
    """train.py syncs after every training step but the last, train_async.py after every step."""
    return num_rollout if fully_async else num_rollout - 1


def _run(
    name: str,
    config: Config,
    profile_step: int,
    fully_async: bool,
    num_rollout: int,
    dump_details: bool,
    resume_at: int,
) -> dict:
    # Runs launched in the same second get apart.
    run_dir = Path(
        DATA, "e2e", name, f"{datetime.now():%m%d-%H%M%S}-{os.urandom(2).hex()}"
    )
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    result = {
        "revisions": install_overlays(),
        "run_dir": str(run_dir),
        "fully_async": fully_async,
        "host": _host(),
    }
    engines = []

    def start_engine(log: str) -> Engine:
        engine = Engine(
            f"{CHECKPOINTS}/{config.rollout_checkpoint}",
            config,
            _engine_gpus(config),
            ENGINE_PORT,
            logs / log,
        )
        engines.append(engine)
        engine.wait()
        engine.warm_up()
        return engine

    try:
        engine = start_engine("engine.log")
        args = _train_args(config, run_dir, profile_step, num_rollout, dump_details)
        if resume_at:
            first = f"{SAVE}/first"
            train(
                config,
                logs / "first.log",
                _train_args(
                    config,
                    run_dir,
                    0,
                    resume_at + 3,
                    False,
                    save=first,
                    save_interval=resume_at + 1,
                ),
                fully_async,
            )
            result["first_syncs"] = syncs(logs / "first.log")
            engine.stop()
            engine = start_engine("engine-resumed.log")
            # LoRA resumes from the adapter checkpoint, whose schedule ran fewer rollouts.
            if config.served_lora:
                args += (
                    f"--lora-adapter-path {first}/torch_dist/iter_{resume_at:07d}/adapter "
                    "--no-load-optim --override-opt-param-scheduler "
                )
            else:
                args += f"--load {first}/torch_dist --ckpt-step {resume_at} --no-load-optim --no-load-rng "
        train(config, logs / "train.log", args, fully_async)
        result["syncs"] = syncs(logs / "train.log")
        delta_state = engine.state()
        last = _last_version(fully_async, num_rollout)
        if config.served_lora:
            (version_dir,) = (run_dir / "adapters").glob(f"{LORA_ADAPTER}_*_v{last}")
            # Outputs are not compared: nondeterministic kernels move log-probs. Serving
            # them makes the adapter resident for the slot check.
            result["lora"] = {
                "served_version": engine.weight_version(),
                "served": engine.greedy(LORA_ADAPTER),
            }
            result["lora"]["slot"] = engine.check_lora_slot(
                LORA_ADAPTER, str(version_dir)
            )
            engine.load_lora_adapter("reference", str(version_dir))
            result["lora"]["reference"] = engine.greedy("reference")
        engine.stop()
        # Served LoRA keeps the base, so the reference is the checkpoint; full weights:
        # the trainer's save of the last version.
        reference = Engine(
            f"{CHECKPOINTS}/{config.rollout_checkpoint}"
            if config.served_lora
            else in_checkpoint_dtypes(
                Path(f"{SAVE}/hf/step_{last - 1}"),
                Path(CHECKPOINTS, config.rollout_checkpoint),
            ),
            config,
            _engine_gpus(config),
            ENGINE_PORT,
            logs / "reference.log",
        )
        engines.append(reference)
        reference.wait()
        reference_state = reference.state()
        (run_dir / "state.json").write_text(
            json.dumps({"delta": delta_state, "reference": reference_state}, indent=1)
        )
        result["check"] = compare(delta_state, reference_state)
        if config.served_lora:
            lora = result["lora"]
            if lora["served_version"] != str(last):
                result["check"]["differ"].append(
                    f"served weight version {lora['served_version']}, not {last}"
                )
            result["check"]["differ"] += [
                f"{LORA_ADAPTER} slot on rank {rank}: {counts}"
                for rank, counts in enumerate(lora["slot"])
                if not counts["written"]
                or counts["mismatch"]
                or counts["unwritten_nonzero"]
                or counts["written_outside_regions"]
            ]
    except Exception as error:
        result["error"] = repr(error)
        result.setdefault("syncs", syncs(logs / "train.log"))
        result.setdefault("check", {"differ": [], "only_one": []})
    finally:
        for engine in engines:
            engine.stop()
        (run_dir / "result.json").write_text(json.dumps(result, indent=1))
        data.commit()
        sglang_cache.commit()
    return result


def _host() -> dict:
    """What bounds the sync's stages on this node, so runs on different hosts compare."""
    cpu = re.search(r"model name\s*: (.+)", Path("/proc/cpuinfo").read_text())
    pcie = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.split("\n")[0]
    return {"cpu": cpu[1] if cpu else None, "cpus": os.cpu_count(), "pcie": pcie}


def install_overlays() -> dict:
    """Replace the image's SGLang and Miles sources with our trees, keeping SGLang's build
    products."""
    with tempfile.TemporaryDirectory() as stash:
        products = [
            Path(path).relative_to(SGLANG)
            for pattern in SGLANG_BUILD_PRODUCTS
            for path in glob(f"{SGLANG}/{pattern}")
        ]
        for product in products:
            (Path(stash) / product).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f"{SGLANG}/{product}", Path(stash) / product)
        shutil.rmtree(SGLANG)
        shutil.copytree(SGLANG_OVERLAY, SGLANG)
        for product in products:
            shutil.copy2(Path(stash) / product, f"{SGLANG}/{product}")
    for directory in MILES_DIRS:
        shutil.rmtree(f"{MILES}/{directory}", ignore_errors=True)
        shutil.copytree(f"{MILES_OVERLAY}/{directory}", f"{MILES}/{directory}")
    for path in Path(MILES_OVERLAY).glob("*.py"):
        shutil.copy2(path, f"{MILES}/{path.name}")
    return json.loads(os.environ["E2E_REVISIONS"])


class Engine:
    """One SGLang server on `gpus`, logging to `log`."""

    def __init__(
        self, model_path: str, config: Config, gpus: str, port: int, log: Path
    ):
        self.url, self.log = f"http://127.0.0.1:{port}", log
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *shlex.split(config.engine_args),
        ]
        env = {**os.environ, **config.env, "CUDA_VISIBLE_DEVICES": gpus}
        if config.served_lora:
            Path(LORA_CHECK_DIR, "sitecustomize.py").write_text(
                "import lora_slot_check\nlora_slot_check.install()\n"
            )
            env["PYTHONPATH"] = ":".join(
                filter(None, [LORA_CHECK_DIR, env.get("PYTHONPATH")])
            )
        with log.open("w") as out:
            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def wait(self) -> None:
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"engine exited {self.process.returncode}, see {self.log}"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as reply:
                    if reply.status == 200:
                        return
            except OSError:
                pass
            time.sleep(5)
        raise TimeoutError(f"engine not ready after 3600 s, see {self.log}")

    def warm_up(self) -> None:
        """A batch sampled as the rollouts sample, so the kernels SGLang builds on first use
        (EAGLE's sampling kernel, Triton kernels) build before Miles connects. Built inside
        the first rollout, they stall the engine past Miles' router health checks."""
        body = {
            "text": ["Hello"] * 16,
            "sampling_params": {"temperature": 1.0, "max_new_tokens": 16},
        }
        request = urllib.request.Request(
            f"{self.url}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as reply:
            reply.read()

    def load_lora_adapter(self, name: str, path: str) -> None:
        """SGLang's own load of an adapter directory."""
        self._post("/load_lora_adapter", {"lora_name": name, "lora_path": path})

    def check_lora_slot(self, name: str, path: str) -> list[dict]:
        """Each rank's served slot of `name` against SGLang's install of `path` (lora_slot_check)."""
        for stale in glob("/tmp/lora_slot_check_rank*.json"):
            os.unlink(stale)
        self.load_lora_adapter(f"__check__{name}", path)
        files = sorted(
            glob("/tmp/lora_slot_check_rank*.json"),
            key=lambda file: int(re.search(r"rank(\d+)", file)[1]),
        )
        assert files, "lora_slot_check is not installed in the engine"
        return [json.loads(Path(file).read_text()) for file in files]

    def weight_version(self) -> str:
        with urllib.request.urlopen(f"{self.url}/model_info", timeout=60) as reply:
            return json.loads(reply.read())["weight_version"]

    def greedy(self, lora_name: str) -> list[dict]:
        """Greedy outputs of LORA_PROMPTS through `lora_name`, one request at a time."""
        replies = [
            self._post(
                "/generate",
                {
                    "text": prompt,
                    "lora_path": lora_name,
                    "return_logprob": True,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            for prompt in LORA_PROMPTS
        ]
        return [
            {
                "text": reply["text"],
                "logprobs": [
                    token[0] for token in reply["meta_info"]["output_token_logprobs"]
                ],
            }
            for reply in replies
        ]

    def _post(self, path: str, body: dict):
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as reply:
            return json.loads(reply.read())

    def state(self) -> dict[str, dict[str, str]]:
        """Every tensor's checksum on each GPU (draft tensors prefixed by their role), by
        TP rank. Each DP rank's reply carries every GPU's checksums."""
        request = urllib.request.Request(
            f"{self.url}/weights_checker",
            data=json.dumps({"action": "state_checksum"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as reply:
            return tp_states(json.loads(reply.read())["ranks"])

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)


def train(
    config: Config, log: Path, train_args: str, fully_async: bool, cpus: int = CPUS // 4
) -> None:
    """Miles' training script on a Ray cluster of the node's first GPUs; its ranks share `cpus`."""
    gpus = config.train_gpus
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(gpus)))}
    subprocess.run(
        ["ray", "start", "--head", "--num-gpus", str(gpus), "--disable-usage-stats"],
        env=env,
        check=True,
    )
    from miles.utils.external_utils.model_args_utils import shell_safe_model_args

    runtime_env = {
        "env_vars": {
            **config.env,
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            # Modal sets it to the container's CPUs for every process. Past the ranks' share every
            # OpenMP region stalls, and a full MoE's CPU optimizer copies thousands of tensors at init.
            "OMP_NUM_THREADS": str(cpus // gpus),
            # Ahead of the container's own path: Modal's client and the local sources.
            "PYTHONPATH": f"{MILES}:{MEGATRON_PATH}:{REWARD_DIR}:{os.environ['PYTHONPATH']}",
            "MASTER_ADDR": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        }
    }
    command = (
        f"ray job submit --runtime-env-json {shlex.quote(json.dumps(runtime_env))} -- "
        f"python3 {MILES}/{'train_async.py' if fully_async else 'train.py'} "
        f"{shell_safe_model_args(config.megatron_model_type)} "
        f"{train_args}{'--fully-async ' if fully_async else ''}"
    )
    try:
        with log.open("w") as out:
            subprocess.run(
                command,
                shell=True,
                cwd=MILES,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
                # Miles keeps retrying once its router has marked the engine dead.
                timeout=TRAIN_TIMEOUT,
            )
    finally:
        subprocess.run(["ray", "stop", "--force"], env=env)


def _train_args(
    config: Config,
    run_dir: Path,
    profile_step: int,
    num_rollout: int,
    dump_details: bool,
    save: str = SAVE,
    save_interval: int | None = None,
) -> str:
    return (
        f"{recipe_args(config, run_dir)}--num-rollout {num_rollout} "
        f"--rollout-external-engine-addrs 127.0.0.1:{ENGINE_PORT} --router miles "
        f"--rollout-num-gpus {config.engine_gpus} --rollout-num-gpus-per-engine {config.engine_gpus} "
        # By default, saves after the last two steps: one of them holds the last version published.
        f"--save {save}/torch_dist --save-hf {save}/hf/step_{{rollout_id}} "
        f"--save-interval {save_interval or num_rollout - 1} --no-save-optim "
        f"{f'--dump-details {run_dir}/dump ' if dump_details else ''}"
        f"{_profiler_args(profile_step, run_dir / 'profile') if profile_step else ''}"
    )


# A reward that needs no model quality, on short rollouts.
SEEDED_ROLLOUT = (
    "--custom-rm-path seeded_reward.seeded_random_reward "
    "--rollout-batch-size 4 --n-samples-per-prompt 4 "
    "--rollout-max-response-len 512 --rollout-temperature 1 --global-batch-size 16 "
)


POLICY_LOSS = (
    "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    "--use-tis --tis-clip-low 0.5 --tis-clip 2.0 "
)


def recipe_args(
    config: Config,
    run_dir: Path,
    rollout: str = SEEDED_ROLLOUT,
    loss: str = POLICY_LOSS,
) -> str:
    """Everything but where rollouts go and when Miles saves."""
    te_precision = ""
    if config.te_precision_config is not None:
        path = run_dir / "te_precision.yaml"
        path.write_text(config.te_precision_config)
        te_precision = f"--te-precision-config-file {path} "
    return (
        f"--hf-checkpoint {CHECKPOINTS}/{config.rollout_checkpoint}/ "
        f"--ref-load {CHECKPOINTS}/{config.trainer_checkpoint} "
        f"--prompt-data {DATA}/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        f"{rollout}{loss}"
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        f"--update-weight-transfer-mode {'disk-lora' if config.served_lora else 'gpu-delta'} "
        f"--update-weight-disk-dir {run_dir}/{'adapters' if config.served_lora else 'deltas'} "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {config.train_gpus} "
        f"--num-gpus-per-node {config.train_gpus} "
        f"{te_precision}{config.train_args}"
    )


def _profiler_args(step: int, directory: Path) -> str:
    """PyTorch's profiler, with Python call stacks, on every trainer rank from the end of
    training step `step` to the end of the next: the weight sync after `step` and the step
    after it."""
    return (
        f"--use-pytorch-profiler --profile-target train_overall "
        f"--profile-step-start {step} --profile-step-end {step + 1} "
        f"--pytorch-profiler-collect-callstack --tensorboard-dir {directory} "
    )


def syncs(log: Path) -> dict[int, dict]:
    """The weight-update metrics Miles logs with each training step, by the version the sync
    before that step published."""
    syncs = {}
    if not log.exists():
        return syncs
    for match in re.finditer(r"perf (\d+): (\{.*\})", log.read_text(errors="replace")):
        try:
            metrics = ast.literal_eval(match[2])
        except (SyntaxError, ValueError):
            continue
        weight_update = {
            key.removeprefix("perf/").removeprefix("update_weights_"): value
            for key, value in metrics.items()
            if "update_weights" in key or "finalize_and_resume_engines" in key
        }
        if weight_update:
            syncs[int(match[1])] = weight_update
    return syncs


def server_args(engine_args: str) -> dict[str, str]:
    """`--flag value...` as SGLangEndpoint takes it: each flag with its values, or ""."""
    args: dict[str, str] = {}
    for word in shlex.split(engine_args):
        if word.startswith("--"):
            flag, args[word] = word, ""
        else:
            args[flag] = f"{args[flag]} {word}".strip()
    return args


def in_checkpoint_dtypes(save: Path, checkpoint: Path) -> Path:
    """The trainer's save as it publishes it: each float tensor cast to the rollout checkpoint's dtype, as the
    trainer casts every export before the XOR. Files that need no cast are linked."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    floats = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    canonical = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(path, "pt") as f:
            canonical.update((name, f.get_slice(name).get_dtype()) for name in f.keys())
    target = save.with_name(f"{save.name}-checkpoint-dtypes")
    target.mkdir()
    for path in save.iterdir():
        casts = {}
        if path.suffix == ".safetensors":
            with safe_open(path, "pt") as f:
                for name in f.keys():
                    dtype, wanted = f.get_slice(name).get_dtype(), canonical.get(name)
                    if dtype != wanted and dtype in floats and wanted in floats:
                        casts[name] = floats[wanted]
                if casts:
                    tensors = {name: f.get_tensor(name) for name in f.keys()}
                    metadata = f.metadata()
        if not casts:
            (target / path.name).symlink_to(path)
            continue
        save_file(
            {name: t.to(casts.get(name, t.dtype)) for name, t in tensors.items()},
            target / path.name,
            metadata,
        )
    return target


def tp_states(ranks: list[dict]) -> dict[str, dict[str, str]]:
    """A weights_checker state_checksum reply's checksums by TP rank."""
    return {
        f"tp{rank['parallelism_info'][0]['tp_rank']}": rank["checksums"]
        for rank in ranks
    }


def compare(delta: dict, reference: dict) -> dict:
    differ, only_one = [], []
    for gpu in sorted(delta.keys() | reference.keys()):
        a, b = delta.get(gpu, {}), reference.get(gpu, {})
        only_one += [f"{gpu} {name}" for name in sorted(a.keys() ^ b.keys())]
        differ += [
            f"{gpu} {name}"
            for name in sorted(a.keys() & b.keys())
            if a[name] != b[name]
        ]
    return {
        "gpus": len(delta),
        "tensors": sum(len(checksums) for checksums in delta.values()),
        "differ": differ,
        "only_one": only_one,
    }
