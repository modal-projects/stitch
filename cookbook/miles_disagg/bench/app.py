"""The sampler benchmark as a Modal app: the engines of one benchmark configuration, and a
CPU driver that ramps them with replayed trajectories.

Selected by environment, which ``cookbook.miles_disagg.bench.launch`` passes through:

- ``BENCH_CONFIG``: a row of the matrix (``configs.CONFIGS``), e.g. ``b200-bf16``.
- ``BENCH_VARIANTS``: comma-separated variants of that row; all of them by default.
  Each variant is its own Server class with its own engines, so up to four variants
  run side by side, each swept by its own driver call.
- ``BENCH_ENGINES``: engines per variant (default 1).
- ``BENCH_EXPERIMENT``, ``BENCH_RUN``, ``BENCH_VERSION``: the checkpoint, found as an
  eval point finds it (``evaluation.checkpoint_dir``). Version 0 (the default) is the
  base model's static view on the ``miles-checkpoints`` volume; version N of run R is
  ``/source-run/R/hf_checkpoints/weight_v{N-1:06d}/<view>`` on the recipe's volume.
- ``BENCH_TAG``: an optional app-name suffix, to run one configuration twice at once.

An engine is SGLang alone on the container's public port, with its pool's server
arguments and the startup flags ``server.serve_startup`` adds. The Stitch sidecar is
left out: nothing publishes weights here, and the sidecar buffers each response, which
would hide time to first token. Engines must run on runc (``MODAL_FUNCTION_RUNTIME=runc``
when deploying), which the launcher enforces.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from cookbook.common import server, serving_image
from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    MINUTES,
    SERVER_STARTUP_TIMEOUT,
    SGLANG_CACHE_PATH,
    SIDECAR_PORT,
)
from cookbook.miles_disagg import evaluation
from cookbook.miles_disagg.bench import configs
from cookbook.miles_disagg.config import validate_recipe

BENCH_VOLUME_NAME = "stitch-sampler-bench"
BENCH_PATH = Path("/bench")
# The eval volume, where an eval launch with MODAL_SWE_TRAJECTORY_DUMP_DIR writes traces.
TRACES_PATH = Path("/traces")
ENGINE_READY_TIMEOUT_S = 90 * MINUTES

CONFIG = configs.config(os.environ["BENCH_CONFIG"])
VARIANTS = configs.select_variants(CONFIG, os.environ.get("BENCH_VARIANTS", ""))
ENGINES = int(os.environ.get("BENCH_ENGINES") or 1)
EXPERIMENT = os.environ.get("BENCH_EXPERIMENT") or configs.DEFAULT_EXPERIMENT
TAG = os.environ.get("BENCH_TAG", "")
exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
validate_recipe(exp)
POINT = evaluation.EvalPoint(
    experiment=EXPERIMENT,
    run_id=os.environ.get("BENCH_RUN") or None,
    version=int(os.environ.get("BENCH_VERSION") or 0),
    view=CONFIG.weight_view,
)
APP_NAME = configs.app_name(CONFIG, TAG)
POOLS = {
    variant.name: CONFIG.bench_pool(variant, engines=ENGINES) for variant in VARIANTS
}
# Baked into both images so a container's re-import selects the same configuration.
BENCH_ENVIRONMENT = {
    "BENCH_CONFIG": CONFIG.key,
    "BENCH_VARIANTS": ",".join(variant.name for variant in VARIANTS),
    "BENCH_ENGINES": str(ENGINES),
    "BENCH_EXPERIMENT": EXPERIMENT,
    "BENCH_RUN": POINT.run_id or "",
    "BENCH_VERSION": str(POINT.version),
    "BENCH_TAG": TAG,
}


def engine_args(pool: Any) -> dict[str, str]:
    """The SGLang arguments of an engine: what ``server.serve_startup`` passes a
    rollout replica, which boots the checkpoint as version ``POINT.version``."""
    args = {
        "--served-model-name": exp.miles.hf_checkpoint,
        "--trust-remote-code": "",
        **pool.sglang_args,
        "--weight-update-staging": exp.SGLANG_DELTA_UPDATE_MODE,
        "--weight-version": str(POINT.version),
    }
    if exp.LOCAL_CHECKPOINT_PATH is not None:
        args["--weight-update-local-checkpoint-dir"] = str(exp.LOCAL_CHECKPOINT_PATH)
    return args


server_image = serving_image.build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env={**(getattr(exp, "SGLANG_SERVER_ENV", None) or {}), **BENCH_ENVIRONMENT},
    runtime=getattr(exp, "SGLANG_RUNTIME", serving_image.DEFAULT_SGLANG_RUNTIME),
)
driver_image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers renders and tokenizes token prompts with the served checkpoint's
    # own tokenizer and chat template.
    .pip_install("httpx", "fastapi", "transformers>=4.57", "jinja2", "orjson")
    .env({**BENCH_ENVIRONMENT, "EXPERIMENT_CONFIG": EXPERIMENT})
    .add_local_python_source("stitch")
    .add_local_dir(
        str(Path(__file__).resolve().parents[2]),
        remote_path="/root/cookbook",
        ignore=["**/__pycache__"],
    )
)

hf_cache_volume = modal.Volume.from_name("huggingface-cache", version=2).read_only()
checkpoint_volume = modal.Volume.from_name("miles-checkpoints", version=2).read_only()
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
bench_volume = modal.Volume.from_name(
    BENCH_VOLUME_NAME, create_if_missing=True, version=2
)
traces_volume = modal.Volume.from_name(
    evaluation.EVAL_VOLUME_NAME, version=2
).read_only()
engine_volumes: dict[str, Any] = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(CHECKPOINTS_PATH): checkpoint_volume,
    SGLANG_CACHE_PATH: sglang_cache_volume,
}
if not POINT.is_base:
    engine_volumes[str(evaluation.SOURCE_RUN_PATH)] = modal.Volume.from_name(
        exp.EXPERIMENT_VOLUME_NAME, version=2
    ).read_only()

app = modal.App(APP_NAME)


class _BenchEngine:
    pool_config: Any

    @modal.enter()
    def startup(self) -> None:
        from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions

        pool = self.pool_config
        os.environ.update(pool.environment)
        checkpoint = evaluation.checkpoint_dir(
            exp, POINT, source_run_root=evaluation.SOURCE_RUN_PATH
        )
        self.endpoint = SGLangEndpoint(
            model_path=str(checkpoint),
            worker_port=SIDECAR_PORT,
            extra_server_args=engine_args(pool),
            health_timeout=SERVER_STARTUP_TIMEOUT,
            health_poll_interval=10.0,
            log_requests_level=-1,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=SIDECAR_PORT,
            payload={
                "model": exp.miles.hf_checkpoint,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            successful_requests=2,
            request_timeout=120.0,
            max_attempts_per_request=3,
        )
        print(f"Bench engine ready: {pool.name} serving {checkpoint}", flush=True)

    @modal.exit()
    def stop(self) -> None:
        endpoint = getattr(self, "endpoint", None)
        if endpoint is not None:
            endpoint.stop()


def _define_engine(pool: Any) -> Any:
    """One Modal Server per variant, defined at module scope as Modal requires."""
    cls = type(
        pool.name,
        (_BenchEngine,),
        {"__module__": __name__, "__qualname__": pool.name, "pool_config": pool},
    )
    globals()[pool.name] = cls
    decorated = app.server(
        **server.modal_server_options(exp.modal, pool),
        image=server_image,
        volumes=engine_volumes,
    )(cls)
    globals()[pool.name] = decorated
    return decorated


ENGINE_SERVERS = {name: _define_engine(pool) for name, pool in POOLS.items()}


async def ready_targets(pool_name: str, engines: int, *, direct: bool) -> list[Any]:
    """Wait until ``engines`` replicas of a variant answer ``/health``, then return
    one replay target per replica, each addressed through the pool URL so a session
    stays on its engine; with ``direct`` off, one target for Flash to route."""
    import httpx

    from cookbook.miles_disagg.bench.replay import Target
    from stitch.pools.modal_flash import ModalFlashPool

    pool = ModalFlashPool(APP_NAME, pool_name)
    deadline = time.monotonic() + ENGINE_READY_TIMEOUT_S
    async with httpx.AsyncClient(trust_env=False, timeout=15.0) as client:
        while True:
            ready = []
            for replica in await pool.discover_replicas_async():
                url, headers = await asyncio.to_thread(
                    pool.replica_request, replica, "/health"
                )
                try:
                    if (await client.get(url, headers=headers)).status_code == 200:
                        ready.append((url.removesuffix("/health"), headers, replica))
                except httpx.HTTPError:
                    pass
            if len(ready) >= engines:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"{pool_name}: {len(ready)}/{engines} engines ready")
            print(f"[{pool_name}] {len(ready)}/{engines} engines ready", flush=True)
            await asyncio.sleep(30)
    if not direct:
        return [Target(await pool.gateway_url_async(), name=pool_name)]
    return [
        Target(url, dict(headers), replica) for url, headers, replica in ready[:engines]
    ]


@app.function(
    image=driver_image,
    cpu=4.0,
    memory=8192,
    # The engines' volumes too, for the served checkpoint's tokenizer.
    volumes={
        str(BENCH_PATH): bench_volume,
        str(TRACES_PATH): traces_volume,
        **engine_volumes,
    },
    timeout=12 * 60 * MINUTES,
    include_source=False,
)
def sweep_variant(variant: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    """Ramp one variant's engines with the traces at ``options["traces"]`` (relative
    to the eval volume) and write its points to the bench volume under
    ``<config>/<run_id>/<variant>.csv``, committed after every point."""
    import dataclasses

    from cookbook.miles_disagg.bench import replay, sweep

    pool = POOLS[variant]
    bench_variant = CONFIG.variant(variant)
    plan = sweep.SweepConfig(**options["sweep"])
    plan = dataclasses.replace(
        plan, sessions_per_engine=plan.sessions_up_to(bench_variant.max_batch)
    )
    if not plan.sessions_per_engine:
        raise ValueError(
            f"{variant}: no session count fits max_batch {bench_variant.max_batch}"
        )
    replay_config = replay.ReplayConfig(**options.get("replay", {}))
    prompts = None
    if options.get("token_prompts", True):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(
                evaluation.checkpoint_dir(
                    exp, POINT, source_run_root=evaluation.SOURCE_RUN_PATH
                )
            )
        )
        prompts = replay.TokenPrompts(
            tokenizer, options.get("chat_template_kwargs") or {}
        )
    trajectories = replay.load_trajectories(TRACES_PATH / options["traces"])
    if not trajectories:
        raise RuntimeError(f"no trajectories under {options['traces']}")
    targets = asyncio.run(
        ready_targets(pool.name, ENGINES, direct=options.get("direct", True))
    )
    out_dir = BENCH_PATH / CONFIG.key / options["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **options,
        "app": APP_NAME,
        "config": CONFIG.key,
        "variant": variant,
        "server": pool.name,
        "gpu": CONFIG.gpu,
        "gpus_per_engine": CONFIG.gpus_per_engine,
        "engines": ENGINES,
        "targets": [target.name for target in targets],
        "sglang_args": engine_args(pool),
        "environment": pool.environment,
        "checkpoint": str(
            evaluation.checkpoint_path(
                exp, POINT, source_run_root=evaluation.SOURCE_RUN_PATH
            )
        ),
        "trajectories": len(trajectories),
        "model_calls": sum(len(t.calls) for t in trajectories),
        "sweep_resolved": sweep.describe(plan),
    }
    (out_dir / f"{variant}.manifest.json").write_text(json.dumps(manifest, indent=2))
    bench_volume.commit()

    def on_point(row: dict[str, Any]) -> None:
        print(json.dumps(_brief(row)), flush=True)
        bench_volume.commit()

    rows = asyncio.run(
        sweep.run_sweep(
            trajectories,
            targets,
            labels={
                **sweep.variant_labels(CONFIG, bench_variant),
                "run_id": options["run_id"],
            },
            sweep=plan,
            replay_config=replay_config,
            prompts=prompts,
            csv_path=out_dir / f"{variant}.csv",
            on_point=on_point,
            refresh_targets=lambda: ready_targets(
                pool.name, ENGINES, direct=options.get("direct", True)
            ),
        )
    )
    return rows


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "variant",
        "sessions",
        "output_tok_s_per_gpu",
        "requests_s_per_gpu",
        "latency_s_p90",
        "decode_tok_s_p50",
        "loop_lag_s_p90",
        "errors",
        "stop_reason",
    )
    return {key: row.get(key) for key in keys}


def results_path(run_id: str) -> PurePosixPath:
    """Where a launch's CSVs land on the bench volume."""
    return PurePosixPath(CONFIG.key) / run_id
