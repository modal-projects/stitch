"""Kimi-K2.6 NVFP4 benchmark for a host-prepared weight-update loop.

This is deliberately narrower than an RL end-to-end run. It starts one real SGLang
engine, then measures the same two operations the Stitch sidecar drives in
production:

1. ``POST /pull_weights`` reconstructs and checksum-verifies the canonical local
   checkpoint, then advances one persistent pinned CPU runtime image from XOR deltas.
2. ``POST /update_weights_from_prepared`` performs only a complete CPU-to-GPU copy
   into existing runtime storages and switches the serving weight version.

Base seeding is timed separately and excluded from the steady-state loop. HiCache
is not enabled. The same prompt is generated before and after the update.

Run::

    uv run --extra modal modal run -d \
      tools/profiling/kimi_k2_6_reload_baseline.py::benchmark

The defaults use the complete v1 chain currently selected by the delta Volume's
top-level ``latest`` pointer. The mounts intentionally match the current Miles Kimi
recipe: its model, Hugging Face, and SGLang cache Volumes plus the recipe's
model-specific delta Volume. Override ``run_id``/``target_version`` to replay another
complete chain without modifying the source Volume. Set ``validate_against_disk`` to
load the verified local checkpoint through SGLang's ordinary disk loader at the end
and require deterministic token/logprob parity with the host-prepared result.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "kimi-k2-6-reload-baseline"

SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.15.post1"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "stitch-sglang-v0.5.15-post1-host-runtime"
SGLANG_FORK_COMMIT = "1a25ea54d0"

HF_CACHE_VOLUME_NAME = "huggingface-cache"
BASE_VOLUME_NAME = "cognition-kimi-k2-6-nvfp4-base"
DELTA_VOLUME_NAME = "stitch-delta-kimi-k2-6-nvfp4"
SGLANG_CACHE_VOLUME_NAME = "cognition-sglang-cache"

HF_CACHE_PATH = "/root/.cache/huggingface"
BASE_PATH = "/model"
DELTA_PATH = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SGLANG_CACHE_PATH = "/root/.cache/sglang"
SGLANG_PORT = 8001

DEFAULT_RUN_ID = "520c51f61535"
DEFAULT_TARGET_VERSION = 1

MINUTES = 60
STARTUP_TIMEOUT = 45 * MINUTES

_THIS_FILE = Path(__file__).resolve()
_LOCAL_STITCH_SOURCE = (
    _THIS_FILE.parents[2] / "src" / "stitch"
    if len(_THIS_FILE.parents) > 2
    else Path("/root/stitch")
)


app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, version=2)
base_volume = modal.Volume.from_name(BASE_VOLUME_NAME, version=1)
delta_volume = modal.Volume.from_name(DELTA_VOLUME_NAME, version=2)
sglang_cache_volume = modal.Volume.from_name(SGLANG_CACHE_VOLUME_NAME, version=2)

image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .run_commands(
        f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
        f" && git fetch modal-fork {SGLANG_FORK_BRANCH}"
        f" && git checkout {SGLANG_FORK_COMMIT} -- python/"
    )
    .pip_install(
        "autoinference-utils==0.2.3",
        "httpx",
        "zstandard",
        "xxhash",
        "blake3",
        "fastsafetensors",
        "psutil",
        "nvidia-ml-py",
    )
    .env(
        {
            "DELTA_VOLUME_NAME": DELTA_VOLUME_NAME,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_FASTSAFETENSORS_NOGDS": "1",
            "SGLANG_PROFILE_RUNTIME_STATE": "1",
            "SGLANG_PROFILE_WEIGHT_RELOAD": "1",
            "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
        }
    )
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .run_commands(f"rm -rf {SGLANG_CACHE_PATH}")
    # Use the source directory directly so this script also launches from a shell where
    # the editable Stitch package is not installed (e.g. the system Modal CLI venv).
    .add_local_dir(str(_LOCAL_STITCH_SOURCE), remote_path="/root/stitch")
)


SGLANG_SERVER_ARGS = {
    "--served-model-name": BASE_PATH,
    "--load-format": "fastsafetensors",
    "--cuda-graph-max-bs-decode": "32",
    "--max-running-requests": "32",
    "--trust-remote-code": "",
    "--custom-pull-weights-pre-read-hook": (
        "stitch.stores.modal_volume.pull_weights_pre_read_hook"
    ),
    "--tool-call-parser": "kimi_k2",
    "--reasoning-parser": "kimi_k2",
    "--dist-timeout": "3600",
    "--kv-cache-dtype": "fp8_e4m3",
    "--attention-backend": "tokenspeed_mla",
    "--context-length": "32768",
    "--mem-fraction-static": "0.85",
    "--chunked-prefill-size": "16384",
    "--schedule-conservativeness": "0.5",
    "--schedule-policy": "lpm",
    "--skip-server-warmup": "",
    "--enable-return-routed-experts": "",
    "--enable-host-runtime-weight-update": "",
}


def _tree_bytes(root: Path, pattern: str = "*") -> int:
    return sum(path.stat().st_size for path in root.glob(pattern) if path.is_file())


class _ResourceSampler:
    """Sample process, host-I/O, and GPU utilization around one benchmark stage."""

    def __init__(self, label: str, interval_s: float = 0.5):
        self.label = label
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._processes: dict[int, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._nvml: Any = None
        self._gpu_handles: list[Any] = []

    def __enter__(self):
        import psutil

        self._psutil = psutil
        self._root = psutil.Process()
        psutil.cpu_percent(interval=None)
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:
            self._nvml = None
            self._gpu_handles = []
        self._started = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        psutil = self._psutil
        try:
            current = [self._root, *self._root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            current = []
        live_processes = []
        for process in current:
            try:
                cached = self._processes.setdefault(process.pid, process)
                live_processes.append(cached)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        cpu_percent = 0.0
        rss_bytes = 0
        read_bytes = 0
        write_bytes = 0
        for process in live_processes:
            try:
                cpu_percent += process.cpu_percent(interval=None)
                rss_bytes += process.memory_info().rss
                io = process.io_counters()
                read_bytes += io.read_bytes
                write_bytes += io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        disk = psutil.disk_io_counters()
        gpu_utils = []
        gpu_memory_utils = []
        pcie_rx_kib_s = []
        if self._nvml is not None:
            for handle in self._gpu_handles:
                try:
                    utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utils.append(float(utilization.gpu))
                    gpu_memory_utils.append(float(utilization.memory))
                except Exception:
                    pass
                try:
                    pcie_rx_kib_s.append(
                        float(
                            self._nvml.nvmlDeviceGetPcieThroughput(
                                handle, self._nvml.NVML_PCIE_UTIL_RX_BYTES
                            )
                        )
                    )
                except Exception:
                    pass

        self.samples.append(
            {
                "elapsed_s": time.perf_counter() - self._started,
                "process_cpu_cores": cpu_percent / 100.0,
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "rss_bytes": float(rss_bytes),
                "process_read_bytes": float(read_bytes),
                "process_write_bytes": float(write_bytes),
                "disk_read_bytes": float(disk.read_bytes if disk else 0),
                "disk_write_bytes": float(disk.write_bytes if disk else 0),
                "gpu_util_percent": sum(gpu_utils) / len(gpu_utils)
                if gpu_utils
                else 0.0,
                "gpu_memory_util_percent": (
                    sum(gpu_memory_utils) / len(gpu_memory_utils)
                    if gpu_memory_utils
                    else 0.0
                ),
                "pcie_rx_kib_s": (
                    sum(pcie_rx_kib_s) / len(pcie_rx_kib_s) if pcie_rx_kib_s else 0.0
                ),
            }
        )

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"label": self.label, "samples": 0}

        def average(name: str) -> float:
            return sum(sample[name] for sample in self.samples) / len(self.samples)

        first, last = self.samples[0], self.samples[-1]
        duration = max(last["elapsed_s"] - first["elapsed_s"], 1e-9)
        process_read_bytes = max(
            0.0, last["process_read_bytes"] - first["process_read_bytes"]
        )
        disk_read_bytes = max(0.0, last["disk_read_bytes"] - first["disk_read_bytes"])
        return {
            "label": self.label,
            "duration_s": round(duration, 3),
            "samples": len(self.samples),
            "process_cpu_cores_avg": round(average("process_cpu_cores"), 2),
            "process_cpu_cores_max": round(
                max(sample["process_cpu_cores"] for sample in self.samples), 2
            ),
            "system_cpu_percent_avg": round(average("system_cpu_percent"), 2),
            "rss_gb_avg": round(average("rss_bytes") / 1e9, 3),
            "rss_gb_max": round(
                max(sample["rss_bytes"] for sample in self.samples) / 1e9, 3
            ),
            "process_read_gb": round(process_read_bytes / 1e9, 3),
            "process_read_gbps": round(process_read_bytes / duration / 1e9, 3),
            "disk_read_gb": round(disk_read_bytes / 1e9, 3),
            "disk_read_gbps": round(disk_read_bytes / duration / 1e9, 3),
            "gpu_util_percent_avg": round(average("gpu_util_percent"), 2),
            "gpu_util_percent_max": round(
                max(sample["gpu_util_percent"] for sample in self.samples), 2
            ),
            "gpu_memory_util_percent_avg": round(average("gpu_memory_util_percent"), 2),
            "pcie_rx_gbps_avg": round(average("pcie_rx_kib_s") * 1024 / 1e9, 3),
            "pcie_rx_gbps_max": round(
                max(sample["pcie_rx_kib_s"] for sample in self.samples) * 1024 / 1e9,
                3,
            ),
        }


def _post(
    client: Any, url: str, path: str, payload: dict[str, Any], timeout: float | None
) -> dict[str, Any]:
    response = client.post(f"{url}{path}", json=payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{path} returned HTTP {response.status_code}: {response.text[:500]}"
        ) from exc
    if response.status_code != 200 or body.get("success") is False:
        raise RuntimeError(f"{path} failed with HTTP {response.status_code}: {body}")
    return body


def _assert_recorded_delta(
    run_id: str, target_version: int
) -> tuple[Path, dict[str, Any]]:
    run_dir = Path(DELTA_PATH) / run_id
    version_dir = run_dir / f"weight_v{target_version:06d}"
    index_path = version_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"recorded delta index not found: {index_path}")
    index = json.loads(index_path.read_text())
    metadata = index.get("metadata") or {}
    expected = {
        "version": f"{target_version:06d}",
        "base_version": f"{target_version - 1:06d}",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "xxh3-128",
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != value:
            raise ValueError(
                f"unexpected {key} in {index_path}: {metadata.get(key)!r}, expected {value!r}"
            )
    missing = sorted(
        filename
        for filename in set((index.get("weight_map") or {}).values())
        if not (version_dir / filename).is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"recorded delta is incomplete; missing: {missing[:10]}"
        )
    return run_dir, index


def _fluent_completion(model: str) -> dict[str, Any]:
    import httpx

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "In two concise sentences, explain how photosynthesis helps "
                    "life on Earth. Use plain English and no bullet points."
                ),
            }
        ],
        "max_tokens": 160,
        "temperature": 0,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.perf_counter()
    response = httpx.post(
        f"http://127.0.0.1:{SGLANG_PORT}/v1/chat/completions",
        json=payload,
        timeout=300.0,
        trust_env=False,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("choices"):
        raise RuntimeError(f"completion returned no choices: {body}")
    choice = body["choices"][0]
    message = choice.get("message") or {}
    visible_text = message.get("content") or ""
    reasoning_text = message.get("reasoning_content") or ""
    text = visible_text or reasoning_text
    output_channel = "content" if visible_text else "reasoning_content"
    words = text.split()
    if len(words) < 12 or sum(character.isalpha() for character in text) < 40:
        raise RuntimeError(f"completion was not plausibly fluent: {text!r}")
    if "\ufffd" in text:
        raise RuntimeError(f"completion contains replacement characters: {text!r}")
    usage = body.get("usage") or {}
    return {
        "latency_s": round(time.perf_counter() - started, 3),
        "text": text,
        "output_channel": output_channel,
        "word_count": len(words),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def _generation_fingerprint(url: str) -> dict[str, Any]:
    """Deterministic model fingerprint used for prepared-vs-disk parity."""

    import httpx

    response = httpx.post(
        f"{url}/generate",
        json={
            "text": (
                "Correctness probe 7f31c9: Explain in exactly three short clauses "
                "why the Moon has phases."
            ),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 48,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "top_logprobs_num": 1,
            "logprob_start_len": -1,
            "stream": False,
        },
        timeout=300.0,
        trust_env=False,
    )
    response.raise_for_status()
    body = response.json()
    meta = body.get("meta_info") or {}
    raw_logprobs = meta.get("output_token_logprobs") or []
    logprobs = [
        float(item[0] if isinstance(item, (list, tuple)) else item)
        for item in raw_logprobs
    ]
    output_ids = body.get("output_ids") or []
    if not output_ids or len(logprobs) != len(output_ids):
        raise RuntimeError(
            "fingerprint generation did not return aligned token ids/logprobs: "
            f"tokens={len(output_ids)} logprobs={len(logprobs)}"
        )
    return {
        "text": body.get("text") or "",
        "output_ids": output_ids,
        "output_logprobs": logprobs,
    }


def _assert_generation_parity(
    prepared: dict[str, Any],
    disk: dict[str, Any],
    *,
    logprob_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if prepared["output_ids"] != disk["output_ids"]:
        raise RuntimeError(
            "host-prepared and canonical-disk token ids differ: "
            f"prepared={prepared['output_ids']} disk={disk['output_ids']}"
        )
    if prepared["text"] != disk["text"]:
        raise RuntimeError(
            "host-prepared and canonical-disk text differs despite equal token ids"
        )
    differences = [
        abs(left - right)
        for left, right in zip(
            prepared["output_logprobs"],
            disk["output_logprobs"],
            strict=True,
        )
    ]
    max_difference = max(differences, default=0.0)
    if max_difference > logprob_tolerance:
        raise RuntimeError(
            "host-prepared and canonical-disk logprobs differ: "
            f"max_abs_diff={max_difference} tolerance={logprob_tolerance}"
        )
    return {
        "tokens": len(prepared["output_ids"]),
        "max_logprob_abs_diff": max_difference,
        "tolerance": logprob_tolerance,
        "exact_token_ids": True,
        "exact_text": True,
    }


@app.function(
    image=image,
    gpu="B300:4",
    cpu=64,
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        BASE_PATH: base_volume,
        DELTA_PATH: delta_volume,
        SGLANG_CACHE_PATH: sglang_cache_volume,
    },
    ephemeral_disk=819_200,
    memory=(1024, 2 * 1024 * 1024),
    timeout=4 * 60 * MINUTES,
)
def benchmark(
    run_id: str = DEFAULT_RUN_ID,
    target_version: int = DEFAULT_TARGET_VERSION,
    inventory_only: bool = False,
    validate_against_disk: bool = False,
) -> dict[str, Any]:
    """Run one clean v0 -> verified XOR -> host-prepared commit benchmark."""
    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    # The benchmark is intentionally incompatible with the removed partial
    # reload and old prepared-reconstruction experiments.
    forbidden_env = (
        "SGLANG_ENABLE_RELOAD_LOAD_PLAN",
        "SGLANG_ENABLE_PREPARED_RUNTIME_RELOAD",
        "STITCH_PARTIAL_RELOAD",
    )
    for name in forbidden_env:
        os.environ.pop(name, None)

    delta_volume.reload()
    run_dir, index = _assert_recorded_delta(run_id, target_version)
    base_index = Path(BASE_PATH) / "model.safetensors.index.json"
    if not base_index.is_file():
        raise FileNotFoundError(f"base checkpoint index not found: {base_index}")

    version_dir = run_dir / f"weight_v{target_version:06d}"
    results: dict[str, Any] = {
        "sglang_branch": SGLANG_FORK_BRANCH,
        "sglang_commit": SGLANG_FORK_COMMIT,
        "gpu": "B300:4",
        "hf_cache_volume": HF_CACHE_VOLUME_NAME,
        "base_volume": BASE_VOLUME_NAME,
        "sglang_cache_volume": SGLANG_CACHE_VOLUME_NAME,
        "base_path": BASE_PATH,
        "base_gb": round(_tree_bytes(Path(BASE_PATH)) / 1e9, 3),
        "delta_volume": DELTA_VOLUME_NAME,
        "delta_run_id": run_id,
        "target_version": target_version,
        "delta_tensors": len(index.get("weight_map") or {}),
        "delta_payload_gb": round(_tree_bytes(version_dir, "*.safetensors") / 1e9, 3),
        "load_format": "fastsafetensors (env-gated no-GDS host-bounce path)",
        "hicache": "disabled",
    }
    print("=== BASELINE INPUT ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    endpoint = SGLangEndpoint(
        model_path=BASE_PATH,
        worker_port=SGLANG_PORT,
        tp=4,
        extra_server_args=SGLANG_SERVER_ARGS,
        health_timeout=STARTUP_TIMEOUT,
        health_poll_interval=10.0,
    )
    url = f"http://127.0.0.1:{SGLANG_PORT}"
    try:
        started = time.perf_counter()
        with _ResourceSampler("initial_load") as initial_load_resources:
            endpoint.start()
        results["initial_load_s"] = round(time.perf_counter() - started, 3)
        results["initial_load_resources"] = initial_load_resources.summary()
        results["pre_update_generation"] = _fluent_completion(BASE_PATH)
        if inventory_only:
            results["status"] = "passed_inventory_only"
            print("=== RUNTIME INVENTORY RESULT ===", flush=True)
            print(json.dumps(results, indent=2), flush=True)
            return results

        with httpx.Client(timeout=None, trust_env=False) as client:
            # Seed the host-local checkpoint exactly as the production prefetch does. This
            # includes the large base copy and is not part of the steady-state loop.
            started = time.perf_counter()
            with _ResourceSampler("base_seed") as base_seed_resources:
                seed_body = _post(
                    client,
                    url,
                    "/pull_weights",
                    {
                        "local_checkpoint_dir": LOCAL_CHECKPOINT_PATH,
                        "source_dir": str(run_dir),
                        "target_version": 0,
                    },
                    None,
                )
            results["base_seed_s"] = round(time.perf_counter() - started, 3)
            results["base_seed_resources"] = base_seed_resources.summary()
            results["base_seed_message"] = seed_body.get("message")

            # Production-equivalent stage: the pre-read hook reloads the delta Volume,
            # then /pull_weights applies and checksum-verifies the real XOR delta.
            loop_started = time.perf_counter()
            started = time.perf_counter()
            with _ResourceSampler("xor_apply") as xor_resources:
                pull_body = _post(
                    client,
                    url,
                    "/pull_weights",
                    {
                        "local_checkpoint_dir": LOCAL_CHECKPOINT_PATH,
                        "source_dir": str(run_dir),
                        "target_version": target_version,
                        "prepare": "runtime",
                    },
                    None,
                )
            results["xor_apply_s"] = round(time.perf_counter() - started, 3)
            results["xor_apply_resources"] = xor_resources.summary()
            results["xor_apply_message"] = pull_body.get("message")

            started = time.perf_counter()
            _post(client, url, "/pause_generation", {"mode": "in_place"}, 120.0)
            results["pause_s"] = round(time.perf_counter() - started, 3)

            try:
                started = time.perf_counter()
                with _ResourceSampler(
                    "update_weights_from_prepared"
                ) as update_resources:
                    update_body = _post(
                        client,
                        url,
                        "/update_weights_from_prepared",
                        {
                            "weight_version": str(target_version),
                            "flush_cache": False,
                        },
                        None,
                    )
                results["update_weights_from_prepared_s"] = round(
                    time.perf_counter() - started, 3
                )
                results["update_weights_from_prepared_resources"] = (
                    update_resources.summary()
                )
                results["update_message"] = update_body.get("message")
            finally:
                started = time.perf_counter()
                _post(client, url, "/continue_generation", {}, 120.0)
                results["resume_s"] = round(time.perf_counter() - started, 3)

            results["xor_plus_update_s"] = round(
                results["xor_apply_s"]
                + results["update_weights_from_prepared_s"],
                3,
            )
            results["full_loop_through_resume_s"] = round(
                time.perf_counter() - loop_started, 3
            )
            results["post_update_generation"] = _fluent_completion(BASE_PATH)
            if validate_against_disk:
                results["prepared_fingerprint"] = _generation_fingerprint(url)
                started = time.perf_counter()
                _post(
                    client,
                    url,
                    "/pause_generation",
                    {"mode": "in_place"},
                    120.0,
                )
                try:
                    disk_started = time.perf_counter()
                    disk_body = _post(
                        client,
                        url,
                        "/update_weights_from_disk",
                        {
                            "model_path": LOCAL_CHECKPOINT_PATH,
                            "load_format": "fastsafetensors",
                            "weight_version": f"{target_version}-disk-reference",
                            "flush_cache": True,
                        },
                        None,
                    )
                    results["disk_reference_reload_s"] = round(
                        time.perf_counter() - disk_started,
                        3,
                    )
                    results["disk_reference_message"] = disk_body.get("message")
                finally:
                    _post(
                        client,
                        url,
                        "/continue_generation",
                        {},
                        120.0,
                    )
                results["disk_reference_validation_pause_s"] = round(
                    time.perf_counter() - started,
                    3,
                )
                results["disk_fingerprint"] = _generation_fingerprint(url)
                results["prepared_disk_parity"] = _assert_generation_parity(
                    results["prepared_fingerprint"],
                    results["disk_fingerprint"],
                )

        results["local_checkpoint_gb"] = round(
            _tree_bytes(Path(LOCAL_CHECKPOINT_PATH)) / 1e9, 3
        )
        results["status"] = "passed"
        print("=== BASELINE RESULT ===", flush=True)
        print(json.dumps(results, indent=2), flush=True)
        return results
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        print("=== BASELINE FAILURE ===", flush=True)
        print(json.dumps(results, indent=2), flush=True)
        raise
    finally:
        try:
            endpoint.stop()
        except Exception as exc:
            print(f"WARNING: failed to stop SGLang cleanly: {exc}", flush=True)
