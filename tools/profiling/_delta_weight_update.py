"""Shared runner for SGLang delta weight-update profilers.

Model-specific Modal apps supply the checkpoint paths, recorded delta lineage,
GPU shape, and direct SGLang arguments. This module owns the benchmark
sequence: startup through routing readiness, live generation during target
preparation, the commit RPC, and post-update generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

UpdateMode = Literal["disk", "cpu"]
CanonicalStorage = Literal["memory", "disk"]


@dataclass(frozen=True)
class WeightUpdateSpec:
    model_name: str
    base_checkpoint_dir: str
    local_target_checkpoint_dir: str
    local_canonical_checkpoint_dir: str
    server_args: dict[str, str]
    tp_size: int = 4
    port: int = 8001
    max_compile_group_gb: int = 8


def server_args_for_mode(
    server_args: dict[str, str],
    update_mode: UpdateMode,
    canonical_storage: CanonicalStorage | None,
    target_checkpoint_dir: str,
    canonical_checkpoint_dir: str,
    max_compile_group_gb: int,
) -> dict[str, str]:
    """Return direct SGLang arguments for one update mode."""

    legacy_options = {
        "--enable-cpu-weight-cache",
        "--cpu-weight-cache-max-compile-group-gb",
        "--cpu-weight-cache-canonical-checkpoint-dir",
    }
    result = {
        key: value for key, value in server_args.items() if key not in legacy_options
    }
    result["--weight-update-staging"] = update_mode
    result["--weight-version"] = "0"
    result.pop("--weight-update-local-checkpoint-dir", None)
    result.pop("--weight-update-max-compile-group-gb", None)
    if update_mode == "cpu":
        if canonical_storage not in {"memory", "disk"}:
            raise ValueError(
                "canonical_storage must be 'memory' or 'disk' for CPU updates"
            )
        result["--weight-update-max-compile-group-gb"] = str(max_compile_group_gb)
        if canonical_storage == "disk":
            result["--weight-update-local-checkpoint-dir"] = canonical_checkpoint_dir
    elif update_mode == "disk":
        if canonical_storage is not None:
            raise ValueError("canonical_storage applies only to CPU updates")
        result["--weight-update-local-checkpoint-dir"] = target_checkpoint_dir
    else:
        raise ValueError(f"unsupported update mode: {update_mode!r}")
    return result


def server_args_for_native_load(
    server_args: dict[str, str],
    weight_version: int,
) -> dict[str, str]:
    """Return arguments for an independent native load of one full target."""

    staging_options = {
        "--enable-cpu-weight-cache",
        "--cpu-weight-cache-max-compile-group-gb",
        "--cpu-weight-cache-canonical-checkpoint-dir",
        "--weight-update-staging",
        "--weight-update-local-checkpoint-dir",
        "--weight-update-max-compile-group-gb",
    }
    result = {
        key: value for key, value in server_args.items() if key not in staging_options
    }
    result["--weight-version"] = str(weight_version)
    return result


def parse_update_mode(value: str) -> UpdateMode:
    if value not in {"disk", "cpu"}:
        raise ValueError("update_mode must be 'disk' or 'cpu'")
    return value


def parse_canonical_storage(value: str | None) -> CanonicalStorage | None:
    if value not in {None, "memory", "disk"}:
        raise ValueError("canonical_storage must be 'memory' or 'disk'")
    return value


def parse_update_destination(
    update_mode: str,
    canonical_storage: str | None,
) -> tuple[UpdateMode, CanonicalStorage | None]:
    mode = parse_update_mode(update_mode)
    storage = parse_canonical_storage(canonical_storage)
    if mode == "cpu" and storage is None:
        raise ValueError("CPU updates require --canonical-storage memory or disk")
    if mode == "disk" and storage is not None:
        raise ValueError("--canonical-storage applies only to CPU updates")
    return mode, storage


def modal_runtime_label() -> str:
    value = os.environ.get("MODAL_FUNCTION_RUNTIME")
    if not value:
        return "gvisor"
    if value != "runc":
        raise ValueError(
            "MODAL_FUNCTION_RUNTIME must be unset for gvisor or set to 'runc'"
        )
    return value


def _post(
    client: Any,
    url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
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


def _generate(
    url: str,
    *,
    fingerprint: bool = False,
    fingerprint_logprobs: bool = True,
) -> dict[str, Any]:
    import httpx

    started = time.perf_counter()
    models_response = httpx.get(
        f"{url}/v1/models",
        timeout=30,
        trust_env=False,
    )
    models_response.raise_for_status()
    model = models_response.json()["data"][0]["id"]
    messages = [
        {
            "role": "user",
            "content": "Explain in exactly three short clauses why the Moon has phases.",
        }
    ]
    if not fingerprint or not fingerprint_logprobs:
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 96 if fingerprint else 80,
        }
        if fingerprint:
            # Compare one deterministic DP worker rather than treating normal
            # cross-rank numerical drift as a weight-update failure.
            request["routed_dp_rank"] = 0
        response = httpx.post(
            f"{url}/v1/chat/completions",
            json=request,
            timeout=300,
            trust_env=False,
        )
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
        if len(text.split()) < 5 or sum(character.isalpha() for character in text) < 20:
            raise RuntimeError(f"completion was not plausibly fluent: {text!r}")
        result = {
            "wall_s": round(time.perf_counter() - started, 6),
            "text": text,
            "weight_version": (body.get("metadata") or {}).get("weight_version"),
        }
        if fingerprint:
            result["output_ids"] = []
            result["output_logprobs"] = []
        return result

    response = httpx.post(
        f"{url}/generate",
        json={
            "text": "Explain why the Moon has phases in exactly three short clauses.",
            "routed_dp_rank": 0,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 48 if fingerprint else 80,
                "ignore_eos": fingerprint,
            },
            "return_logprob": fingerprint and fingerprint_logprobs,
            "return_text_in_logprobs": False,
            "top_logprobs_num": 1 if fingerprint and fingerprint_logprobs else 0,
            "logprob_start_len": -1,
            "stream": False,
        },
        timeout=300,
        trust_env=False,
    )
    response.raise_for_status()
    body = response.json()
    text = body.get("text") or ""
    result = {
        "wall_s": round(time.perf_counter() - started, 6),
        "text": text,
        "weight_version": (body.get("meta_info") or {}).get("weight_version"),
    }
    if fingerprint:
        raw_logprobs = (body.get("meta_info") or {}).get("output_token_logprobs") or []
        result["output_ids"] = body.get("output_ids") or []
        result["output_logprobs"] = [
            float(item[0] if isinstance(item, (list, tuple)) else item)
            for item in raw_logprobs
        ]
        if fingerprint_logprobs and not result["output_ids"]:
            raise RuntimeError("fingerprint token IDs are incomplete")
        if fingerprint_logprobs and len(result["output_ids"]) != len(
            result["output_logprobs"]
        ):
            raise RuntimeError("fingerprint logprobs are incomplete")
    return result


def _weight_checksums(client: Any, url: str) -> dict[str, Any]:
    result = _post(client, url, "/weights_checker", {"action": "checksum"})
    ranks = result.get("ranks") or []
    if not ranks or not result.get("per_engine_checksum"):
        raise RuntimeError(f"weight checksum response is incomplete: {result}")
    return result


def _validate_weight_checksums(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    expected_ranks: int,
) -> dict[str, Any]:
    before_ranks = before["ranks"]
    after_ranks = after["ranks"]
    if len(before_ranks) != expected_ranks or len(after_ranks) != expected_ranks:
        raise RuntimeError(
            "weight checksum rank count mismatch: "
            f"expected={expected_ranks} before={len(before_ranks)} "
            f"after={len(after_ranks)}"
        )
    if before["per_engine_checksum"] == after["per_engine_checksum"]:
        raise RuntimeError("live engine weight checksum did not change")

    before_draft = {
        (rank_index, name): checksum
        for rank_index, rank in enumerate(before_ranks)
        for name, checksum in rank["checksums"].items()
        if name.startswith("draft.")
    }
    after_draft = {
        (rank_index, name): checksum
        for rank_index, rank in enumerate(after_ranks)
        for name, checksum in rank["checksums"].items()
        if name.startswith("draft.")
    }
    if before_draft != after_draft:
        raise RuntimeError("speculative draft weights changed with target weights")
    return {
        "ranks": len(after_ranks),
        "per_engine_checksum": after["per_engine_checksum"],
        "draft_tensors": len(after_draft),
        "draft_unchanged": True,
    }


def _assert_repeat_consistency(
    fingerprints: list[dict[str, Any]],
) -> dict[str, Any]:
    first, second = fingerprints
    if first["output_ids"] != second["output_ids"]:
        raise RuntimeError("repeated fingerprint token IDs differ")
    if first["text"] != second["text"]:
        raise RuntimeError("repeated fingerprint text differs")
    max_logprob_difference = (
        max(
            (
                abs(left - right)
                for left, right in zip(
                    first["output_logprobs"],
                    second["output_logprobs"],
                    strict=True,
                )
            ),
            default=0.0,
        )
        if first["output_logprobs"]
        else None
    )
    result = {
        "exact_text": True,
        **_fingerprint_hashes(first),
    }
    if first["output_ids"]:
        result["exact_token_ids"] = True
        result["tokens"] = len(first["output_ids"])
    if max_logprob_difference is not None:
        result["repeat_max_logprob_abs_diff"] = max_logprob_difference
    return result


def _fingerprint_hashes(fingerprint: dict[str, Any]) -> dict[str, str]:
    token_ids = fingerprint["output_ids"]
    output_logprobs = fingerprint["output_logprobs"]
    hashes = {"text_sha256": hashlib.sha256(fingerprint["text"].encode()).hexdigest()}
    if token_ids:
        hashes["token_ids_sha256"] = hashlib.sha256(
            struct.pack(f"<{len(token_ids)}q", *token_ids)
        ).hexdigest()
    if output_logprobs:
        hashes["logprobs_sha256"] = hashlib.sha256(
            struct.pack(f"<{len(output_logprobs)}d", *output_logprobs)
        ).hexdigest()
    return hashes


def _assert_target_changed(
    baseline: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    token_ids_changed = baseline["output_ids"] != target["output_ids"]
    text_changed = baseline["text"] != target["text"]
    if baseline["output_logprobs"] and len(baseline["output_logprobs"]) == len(
        target["output_logprobs"]
    ):
        max_logprob_difference = max(
            (
                abs(left - right)
                for left, right in zip(
                    baseline["output_logprobs"],
                    target["output_logprobs"],
                    strict=True,
                )
            ),
            default=0.0,
        )
    else:
        max_logprob_difference = None
    logprobs_changed = (
        max_logprob_difference is not None and max_logprob_difference > 0.0
    )
    if not (token_ids_changed or text_changed or logprobs_changed):
        raise RuntimeError("post-update fingerprint is identical to the base model")
    return {
        "changed_from_base": True,
        "token_ids_changed_from_base": token_ids_changed,
        "text_changed_from_base": text_changed,
        "max_logprob_abs_diff_from_base": max_logprob_difference,
    }


class _GenerationProbe:
    def __init__(self, url: str) -> None:
        self.url = url
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.thread = threading.Thread(
            target=self._run,
            name="generation-during-weight-stage",
            daemon=True,
        )

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self.samples.append(_generate(self.url))
            except Exception as exc:  # noqa: BLE001 - report remote benchmark errors
                self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    def __enter__(self) -> _GenerationProbe:
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        self.thread.join(timeout=310)
        if self.thread.is_alive():
            self.errors.append("generation probe did not stop")

    def summary(self) -> dict[str, Any]:
        latencies = [sample["wall_s"] for sample in self.samples]
        return {
            "samples": len(self.samples),
            "errors": self.errors,
            "weight_versions": sorted(
                {sample["weight_version"] for sample in self.samples},
                key=lambda value: "" if value is None else str(value),
            ),
            "latency_min_s": min(latencies) if latencies else None,
            "latency_median_s": (
                round(statistics.median(latencies), 6) if latencies else None
            ),
            "latency_max_s": max(latencies) if latencies else None,
        }


class _MemoryProbe:
    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.samples: list[dict[str, int | str]] = []
        self.thread = threading.Thread(
            target=self._run,
            name="memory-during-weight-stage",
            daemon=True,
        )

    def _run(self) -> None:
        while not self.stop.is_set():
            self.samples.append(_memory_snapshot())
            self.stop.wait(self.interval_s)

    def __enter__(self) -> _MemoryProbe:
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        self.thread.join(timeout=max(5.0, 2 * self.interval_s))

    def summary(self) -> dict[str, int | str | None]:
        numeric_keys = {
            key
            for sample in self.samples
            for key, value in sample.items()
            if isinstance(value, int)
        }
        result: dict[str, int | str | None] = {"samples": len(self.samples)}
        for key in sorted(numeric_keys):
            values = [
                value
                for sample in self.samples
                if isinstance(value := sample.get(key), int)
            ]
            if key == "MemAvailable_bytes":
                result[f"{key}_min"] = min(values)
            else:
                result[f"{key}_max"] = max(values)
        return result


def _memory_snapshot() -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    cgroup_files = {
        "memory_current": (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
        "memory_peak": (
            Path("/sys/fs/cgroup/memory.peak"),
            Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
        ),
        "memory_max": (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    }
    for key, candidates in cgroup_files.items():
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is not None:
            value = path.read_text().strip()
            result[key] = value if value == "max" else int(value)
    wanted = {"MemTotal", "MemAvailable", "Cached", "AnonPages", "Shmem"}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            result[f"{key}_bytes"] = int(value.split()[0]) * 1024
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_used_mib = [
            int(line.strip()) for line in completed.stdout.splitlines() if line.strip()
        ]
        if gpu_used_mib:
            result["gpu_memory_used_total_bytes"] = sum(gpu_used_mib) << 20
            result["gpu_memory_used_max_rank_bytes"] = max(gpu_used_mib) << 20
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return result


def _local_storage_snapshot(paths: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, root in paths.items():
        total = 0
        if os.path.isdir(root):
            for directory, _, filenames in os.walk(root):
                for filename in filenames:
                    try:
                        total += os.path.getsize(os.path.join(directory, filename))
                    except FileNotFoundError:
                        pass
        result[f"{label}_bytes"] = total
    usage = shutil.disk_usage("/")
    result["filesystem_used_bytes"] = usage.used
    result["filesystem_free_bytes"] = usage.free
    return result


def _cgroup_cpu_usage_s() -> float | None:
    cgroup_v2 = Path("/sys/fs/cgroup/cpu.stat")
    if cgroup_v2.is_file():
        for line in cgroup_v2.read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec":
                return int(value) / 1_000_000
    cgroup_v1 = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if cgroup_v1.is_file():
        return int(cgroup_v1.read_text()) / 1_000_000_000
    return None


def _cgroup_io_snapshot() -> dict[str, int] | None:
    path = Path("/sys/fs/cgroup/io.stat")
    if not path.is_file():
        return None
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        for field in line.split()[1:]:
            key, separator, value = field.partition("=")
            if separator:
                result[key] = result.get(key, 0) + int(value)
    return result


def _io_usage_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {
        key: max(0, after.get(key, 0) - before.get(key, 0))
        for key in sorted(before.keys() | after.keys())
    }


def _host_snapshot() -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model_name = next(
        (
            line.partition(":")[2].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    return {
        "cpu_model": model_name,
        "allowed_cpus": len(os.sched_getaffinity(0)),
        "memory": _memory_snapshot(),
    }


def _cpu_usage_delta(
    before: float | None,
    after: float | None,
    wall_s: float,
) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    cpu_s = max(0.0, after - before)
    return {
        "cpu_s": round(cpu_s, 6),
        "average_cores": round(cpu_s / max(wall_s, 1e-9), 3),
    }


def _print_profile_summary(results: dict[str, Any]) -> None:
    summary = {
        key: results.get(key)
        for key in (
            "model",
            "runtime",
            "update_mode",
            "canonical_storage",
            "sample_id",
            "status",
            "startup_ready_s",
            "startup_cpu",
            "memory_during_startup",
        )
        if key in results
    }
    summary["updates"] = [
        {
            key: update.get(key)
            for key in (
                "from_version",
                "target_version",
                "delta_count",
                "prepare_s",
                "prepare_cpu",
                "commit_rpc_s",
                "commit_cpu",
                "prepare_through_commit_s",
                "generation_during_prepare",
                "critical_rank_prepare_s",
                "critical_rank_commit_s",
                "correctness",
            )
        }
        for update in results.get("updates", [])
    ]
    print(f"PROFILE_SUMMARY={json.dumps(summary, sort_keys=True)}", flush=True)


def _validate_delta_lineage(source_dir: str, target_versions: tuple[int, ...]) -> None:
    if not target_versions or target_versions != tuple(sorted(set(target_versions))):
        raise ValueError("target_versions must be a non-empty increasing sequence")
    if target_versions[0] <= 0:
        raise ValueError("target_versions must follow base version 0")
    for version in range(1, target_versions[-1] + 1):
        index_path = (
            Path(source_dir) / f"weight_v{version:06d}" / "model.safetensors.index.json"
        )
        if not index_path.is_file():
            raise FileNotFoundError(f"delta target is missing: {index_path}")
        metadata = json.loads(index_path.read_text()).get("metadata") or {}
        if metadata.get("delta_encoding") != "xor":
            raise ValueError(f"profiling requires an XOR delta: {index_path}")
        if int(metadata.get("version", -1)) != version:
            raise ValueError(f"delta version metadata is invalid: {index_path}")
        if int(metadata.get("base_version", -1)) != version - 1:
            raise ValueError(f"delta lineage is not contiguous: {index_path}")


def _critical_rank_time(rank_stats: list[dict[str, Any]] | None) -> float | None:
    return max(
        (
            rank.get("wall_s", rank.get("total_wall_s", 0.0))
            for rank in (rank_stats or [])
        ),
        default=None,
    )


def _validate_generation_probe(
    generation: _GenerationProbe,
    expected_version: int,
) -> dict[str, Any]:
    summary = generation.summary()
    if generation.errors or not generation.samples:
        raise RuntimeError(f"generation was not healthy during preparation: {summary}")
    observed = {sample["weight_version"] for sample in generation.samples}
    if observed != {str(expected_version)}:
        raise RuntimeError(
            "generation switched weight versions during preparation: "
            f"expected={expected_version} observed={summary}"
        )
    return summary


def _fingerprint_after_commit(
    url: str,
    *,
    target_version: int,
    previous_fingerprint: dict[str, Any],
    fingerprint_logprobs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = _generate(url)
    fingerprints = [
        _generate(
            url,
            fingerprint=True,
            fingerprint_logprobs=fingerprint_logprobs,
        )
        for _ in range(2)
    ]
    observed_versions = {
        generation["weight_version"],
        *(fingerprint["weight_version"] for fingerprint in fingerprints),
    }
    if observed_versions != {str(target_version)}:
        raise RuntimeError(
            "post-update generation reported unexpected weight versions: "
            f"{sorted(str(value) for value in observed_versions)}"
        )
    correctness = {
        **_assert_repeat_consistency(fingerprints),
        **_assert_target_changed(previous_fingerprint, fingerprints[0]),
    }
    return fingerprints[0], {
        "generation_after": generation,
        "fingerprint_probes": [
            {
                "wall_s": fingerprint["wall_s"],
                "weight_version": fingerprint["weight_version"],
                **_fingerprint_hashes(fingerprint),
            }
            for fingerprint in fingerprints
        ],
        "correctness": correctness,
    }


def run_delta_weight_update(
    spec: WeightUpdateSpec,
    *,
    source_dir: str,
    target_versions: tuple[int, ...],
    update_mode: UpdateMode,
    canonical_storage: CanonicalStorage | None,
    runtime: str,
    sample_id: str,
) -> dict[str, Any]:
    """Run and print one complete disk- or CPU-destination delta weight update."""

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    base_index = Path(spec.base_checkpoint_dir) / "model.safetensors.index.json"
    if not base_index.is_file():
        raise FileNotFoundError(f"base checkpoint is missing: {base_index}")
    _validate_delta_lineage(source_dir, target_versions)

    results: dict[str, Any] = {
        "model": spec.model_name,
        "base_checkpoint_dir": spec.base_checkpoint_dir,
        "source_dir": source_dir,
        "target_versions": list(target_versions),
        "update_mode": update_mode,
        "canonical_storage": canonical_storage,
        "runtime": runtime,
        "sample_id": sample_id,
        "tp_size": spec.tp_size,
        "hicache": "disabled",
        "host": _host_snapshot(),
    }

    shutil.rmtree(spec.local_target_checkpoint_dir, ignore_errors=True)
    if spec.local_canonical_checkpoint_dir != spec.local_target_checkpoint_dir:
        shutil.rmtree(spec.local_canonical_checkpoint_dir, ignore_errors=True)

    endpoint = SGLangEndpoint(
        model_path=spec.base_checkpoint_dir,
        worker_port=spec.port,
        tp=spec.tp_size,
        extra_server_args=server_args_for_mode(
            spec.server_args,
            update_mode,
            canonical_storage,
            spec.local_target_checkpoint_dir,
            spec.local_canonical_checkpoint_dir,
            spec.max_compile_group_gb,
        ),
        # Startup includes the model load and construction of the configured
        # inactive update destination; readiness is the externally useful bound.
        health_timeout=2 * 60 * 60,
        health_poll_interval=10,
        log_requests_level=-1,
    )
    endpoint_running = False
    reference_endpoint = None
    url = f"http://127.0.0.1:{spec.port}"
    try:
        startup_started = time.perf_counter()
        startup_cpu_started = _cgroup_cpu_usage_s()
        startup_io_started = _cgroup_io_snapshot()
        with _MemoryProbe() as memory:
            endpoint.start()
        endpoint_running = True
        results["startup_ready_s"] = round(time.perf_counter() - startup_started, 6)
        results["startup_cpu"] = _cpu_usage_delta(
            startup_cpu_started,
            _cgroup_cpu_usage_s(),
            results["startup_ready_s"],
        )
        results["startup_io"] = _io_usage_delta(
            startup_io_started,
            _cgroup_io_snapshot(),
        )
        results["memory_during_startup"] = memory.summary()
        results["memory_after_startup"] = _memory_snapshot()
        results["storage_after_startup"] = _local_storage_snapshot(
            {
                "target": spec.local_target_checkpoint_dir,
                "canonical": spec.local_canonical_checkpoint_dir,
            }
        )
        results["generation_before"] = _generate(url)
        speculative_algorithm = spec.server_args.get(
            "--speculative-algorithm", ""
        ).upper()
        # DSpark rejects return_logprob requests. The pinned SGLang runtime
        # supports aligned verifier logprobs for DFlash.
        fingerprint_logprobs = speculative_algorithm != "DSPARK"
        baseline_fingerprint = _generate(
            url,
            fingerprint=True,
            fingerprint_logprobs=fingerprint_logprobs,
        )
        if not fingerprint_logprobs:
            _assert_repeat_consistency(
                [
                    baseline_fingerprint,
                    _generate(
                        url,
                        fingerprint=True,
                        fingerprint_logprobs=False,
                    ),
                ]
            )
        results["fingerprint_before"] = {
            "wall_s": baseline_fingerprint["wall_s"],
            "weight_version": baseline_fingerprint["weight_version"],
            **_fingerprint_hashes(baseline_fingerprint),
        }
        if baseline_fingerprint["weight_version"] != "0":
            raise RuntimeError(
                "base generation reported unexpected weight version: "
                f"{baseline_fingerprint['weight_version']!r}"
            )

        results["updates"] = []
        previous_version = 0
        previous_fingerprint = baseline_fingerprint
        with httpx.Client(timeout=None, trust_env=False) as client:
            previous_checksums = _weight_checksums(client, url)
            results["base_weight_checksums"] = {
                "ranks": len(previous_checksums["ranks"]),
                "per_engine_checksum": previous_checksums["per_engine_checksum"],
            }
            for target_version in target_versions:
                update: dict[str, Any] = {
                    "from_version": previous_version,
                    "target_version": target_version,
                    "delta_count": target_version - previous_version,
                }
                prepare_started = time.perf_counter()
                prepare_cpu_started = _cgroup_cpu_usage_s()
                prepare_io_started = _cgroup_io_snapshot()
                with _GenerationProbe(url) as generation, _MemoryProbe() as memory:
                    prepared = _post(
                        client,
                        url,
                        "/prepare_weight_update",
                        {
                            "checkpoint_source_dir": source_dir,
                            "target_version": target_version,
                        },
                    )
                update["prepare_s"] = round(
                    time.perf_counter() - prepare_started,
                    6,
                )
                update["prepare_cpu"] = _cpu_usage_delta(
                    prepare_cpu_started,
                    _cgroup_cpu_usage_s(),
                    update["prepare_s"],
                )
                update["prepare_io"] = _io_usage_delta(
                    prepare_io_started,
                    _cgroup_io_snapshot(),
                )
                update["prepare_rank_stats"] = prepared.get("rank_stats")
                update["critical_rank_prepare_s"] = _critical_rank_time(
                    update["prepare_rank_stats"]
                )
                update["generation_during_prepare"] = _validate_generation_probe(
                    generation,
                    previous_version,
                )
                update["memory_during_prepare"] = memory.summary()
                update["memory_after_prepare"] = _memory_snapshot()
                update["storage_after_prepare"] = _local_storage_snapshot(
                    {
                        "target": spec.local_target_checkpoint_dir,
                        "canonical": spec.local_canonical_checkpoint_dir,
                    }
                )

                commit_started = time.perf_counter()
                commit_cpu_started = _cgroup_cpu_usage_s()
                commit_io_started = _cgroup_io_snapshot()
                committed = _post(
                    client,
                    url,
                    "/commit_weight_update",
                    {
                        "target_version": target_version,
                        "abort_all_requests": False,
                        "torch_empty_cache": False,
                    },
                )
                update["commit_rpc_s"] = round(
                    time.perf_counter() - commit_started,
                    6,
                )
                update["commit_cpu"] = _cpu_usage_delta(
                    commit_cpu_started,
                    _cgroup_cpu_usage_s(),
                    update["commit_rpc_s"],
                )
                update["commit_io"] = _io_usage_delta(
                    commit_io_started,
                    _cgroup_io_snapshot(),
                )
                update["commit_rank_stats"] = committed.get("rank_stats")
                update["critical_rank_commit_s"] = _critical_rank_time(
                    update["commit_rank_stats"]
                )
                update["memory_after_commit"] = _memory_snapshot()
                update["prepare_through_commit_s"] = round(
                    time.perf_counter() - prepare_started,
                    6,
                )
                previous_fingerprint, post_commit = _fingerprint_after_commit(
                    url,
                    target_version=target_version,
                    previous_fingerprint=previous_fingerprint,
                    fingerprint_logprobs=fingerprint_logprobs,
                )
                update.update(post_commit)
                current_checksums = _weight_checksums(client, url)
                update["live_weight_checksums"] = _validate_weight_checksums(
                    previous_checksums,
                    current_checksums,
                    expected_ranks=spec.tp_size,
                )
                results["updates"].append(update)
                previous_version = target_version
                previous_checksums = current_checksums

            failure_version = target_versions[-1] + 1
            try:
                _post(
                    client,
                    url,
                    "/prepare_weight_update",
                    {
                        "checkpoint_source_dir": source_dir,
                        "target_version": failure_version,
                    },
                )
            except RuntimeError as exc:
                failure_message = str(exc)
            else:
                raise RuntimeError(
                    f"missing weight version {failure_version} prepared successfully"
                )
            failure_fingerprint = _generate(
                url,
                fingerprint=True,
                fingerprint_logprobs=fingerprint_logprobs,
            )
            if failure_fingerprint["weight_version"] != str(previous_version):
                raise RuntimeError(
                    "failed preparation changed the served version: "
                    f"{failure_fingerprint['weight_version']!r}"
                )
            failure_checksums = _weight_checksums(client, url)
            if (
                failure_checksums["per_engine_checksum"]
                != previous_checksums["per_engine_checksum"]
            ):
                raise RuntimeError("failed preparation changed live weights")
            results["preparation_failure"] = {
                "target_version": failure_version,
                "error": failure_message,
                "served_version": failure_fingerprint["weight_version"],
                "live_weights_unchanged": True,
            }

        final_version = target_versions[-1]
        final_live_checksums = previous_checksums
        final_live_fingerprint = previous_fingerprint
        endpoint.stop()
        endpoint_running = False

        if update_mode == "disk":
            reference_checkpoint_dir = spec.local_target_checkpoint_dir
            reference_materialization = {"reused_staged_checkpoint": True}
        elif canonical_storage == "disk":
            reference_checkpoint_dir = spec.local_canonical_checkpoint_dir
            reference_materialization = {"reused_canonical_checkpoint": True}
        else:
            from sglang.srt.weight_sync.disk_checkpoint import materialize

            reference_checkpoint_dir = spec.local_target_checkpoint_dir
            shutil.rmtree(reference_checkpoint_dir, ignore_errors=True)
            reference_materialization = materialize(
                local_checkpoint_dir=reference_checkpoint_dir,
                base_checkpoint_dir=spec.base_checkpoint_dir,
                checkpoint_source_dir=source_dir,
                target_version=final_version,
                base_version=0,
            )
        results["reference_materialization"] = reference_materialization

        reference_endpoint = SGLangEndpoint(
            model_path=reference_checkpoint_dir,
            worker_port=spec.port,
            tp=spec.tp_size,
            extra_server_args=server_args_for_native_load(
                spec.server_args,
                final_version,
            ),
            health_timeout=2 * 60 * 60,
            health_poll_interval=10,
            log_requests_level=-1,
        )
        clean_load_started = time.perf_counter()
        reference_endpoint.start()
        results["reference_load_s"] = round(
            time.perf_counter() - clean_load_started,
            6,
        )
        with httpx.Client(timeout=None, trust_env=False) as client:
            reference_checksums = _weight_checksums(client, url)
        if reference_checksums["ranks"] != final_live_checksums["ranks"]:
            raise RuntimeError(
                "clean native load rank checksums differ from live weights"
            )
        if (
            reference_checksums["per_engine_checksum"]
            != final_live_checksums["per_engine_checksum"]
        ):
            raise RuntimeError(
                "clean native load engine checksum differs from live weights"
            )
        reference_fingerprint = _generate(
            url,
            fingerprint=True,
            fingerprint_logprobs=fingerprint_logprobs,
        )
        fingerprint_comparison = _assert_repeat_consistency(
            [final_live_fingerprint, reference_fingerprint]
        )
        results["clean_native_load"] = {
            "checkpoint_dir": reference_checkpoint_dir,
            "weight_version": reference_fingerprint["weight_version"],
            "per_engine_checksum": reference_checksums["per_engine_checksum"],
            "matches_live_runtime": True,
            "fingerprint_comparison": fingerprint_comparison,
            **_fingerprint_hashes(reference_fingerprint),
        }

        results["status"] = "passed"
        _print_profile_summary(results)
        print(json.dumps(results, indent=2), flush=True)
        return results
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        _print_profile_summary(results)
        print(json.dumps(results, indent=2), flush=True)
        raise
    finally:
        for running_endpoint in (
            reference_endpoint,
            endpoint if endpoint_running else None,
        ):
            if running_endpoint is None:
                continue
            try:
                running_endpoint.stop()
            except Exception as exc:  # noqa: BLE001 - preserve benchmark result
                print(f"warning: failed to stop SGLang cleanly: {exc}", flush=True)
