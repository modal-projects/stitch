"""Shared runner for SGLang delta weight-update profilers.

Model-specific Modal apps supply the checkpoint paths, recorded delta, GPU
shape, and direct SGLang arguments. This module owns the benchmark sequence:
initial load, live generation during destination initialization and target
staging, the commit RPC, and post-update generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

UpdateMode = Literal["disk", "cpu"]


@dataclass(frozen=True)
class WeightUpdateSpec:
    model_name: str
    base_checkpoint_dir: str
    local_target_checkpoint_dir: str
    server_args: dict[str, str]
    tp_size: int = 4
    port: int = 8001


def server_args_for_mode(
    server_args: dict[str, str],
    update_mode: UpdateMode,
) -> dict[str, str]:
    """Return direct SGLang arguments for one update mode."""

    result = dict(server_args)
    if update_mode == "cpu":
        result["--enable-cpu-weight-cache"] = ""
        result.setdefault("--cpu-weight-cache-max-compile-group-gb", "8")
    elif update_mode == "disk":
        result.pop("--enable-cpu-weight-cache", None)
        result.pop("--cpu-weight-cache-max-compile-group-gb", None)
        result.pop("--cpu-weight-cache-canonical-checkpoint-dir", None)
    else:
        raise ValueError(f"unsupported update mode: {update_mode!r}")
    return result


def parse_update_mode(value: str) -> UpdateMode:
    if value not in {"disk", "cpu"}:
        raise ValueError("update_mode must be 'disk' or 'cpu'")
    return value


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
        response = httpx.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 96 if fingerprint else 80,
            },
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


def _drop_checkpoint_page_cache(path: str) -> dict[str, Any]:
    """Release setup-only file cache before allocating persistent CPU images."""

    started = time.perf_counter()
    files = [entry.path for entry in os.scandir(path) if entry.is_file()]

    def drop(filename: str) -> int:
        fd = os.open(filename, os.O_RDONLY)
        try:
            if hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            return os.fstat(fd).st_size
        finally:
            os.close(fd)

    with ThreadPoolExecutor(max_workers=min(8, len(files) or 1)) as executor:
        released_bytes = sum(executor.map(drop, files))
    return {
        "files": len(files),
        "bytes": released_bytes,
        "wall_s": round(time.perf_counter() - started, 6),
    }


def _print_profile_summary(results: dict[str, Any]) -> None:
    stage_ranks = results.get("stage_rank_stats") or []
    commit_ranks = results.get("commit_rank_stats") or []
    summary = {
        key: results.get(key)
        for key in (
            "model",
            "runtime",
            "update_mode",
            "sample_id",
            "status",
            "initial_load_s",
            "destination_init_s",
            "destination_init_cpu",
            "generation_during_destination_init",
            "stage_s",
            "stage_cpu",
            "commit_rpc_s",
            "commit_cpu",
            "stage_through_commit_s",
            "generation_during_stage",
            "correctness",
        )
        if key in results
    }
    summary["critical_rank_stage_s"] = max(
        (rank.get("wall_s", rank.get("total_wall_s", 0.0)) for rank in stage_ranks),
        default=None,
    )
    destination_init_ranks = results.get("destination_init_rank_stats") or []
    summary["critical_rank_destination_init_s"] = max(
        (rank.get("wall_s", 0.0) for rank in destination_init_ranks),
        default=None,
    )
    summary["critical_rank_commit_s"] = max(
        (rank.get("wall_s", 0.0) for rank in commit_ranks),
        default=None,
    )
    print(f"PROFILE_SUMMARY={json.dumps(summary, sort_keys=True)}", flush=True)


def run_delta_weight_update(
    spec: WeightUpdateSpec,
    *,
    source_dir: str,
    target_version: int,
    update_mode: UpdateMode,
    runtime: str,
    sample_id: str,
) -> dict[str, Any]:
    """Run and print one complete disk- or CPU-destination delta weight update."""

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    target_dir = Path(source_dir) / f"weight_v{target_version:06d}"
    target_index = target_dir / "model.safetensors.index.json"
    base_index = Path(spec.base_checkpoint_dir) / "model.safetensors.index.json"
    if not base_index.is_file():
        raise FileNotFoundError(f"base checkpoint is missing: {base_index}")
    if not target_index.is_file():
        raise FileNotFoundError(f"delta target is missing: {target_index}")
    target_metadata = json.loads(target_index.read_text()).get("metadata") or {}
    if not target_metadata.get("delta_encoding"):
        raise ValueError(f"profiling requires a delta target: {target_index}")

    results: dict[str, Any] = {
        "model": spec.model_name,
        "base_checkpoint_dir": spec.base_checkpoint_dir,
        "source_dir": source_dir,
        "target_version": target_version,
        "update_mode": update_mode,
        "runtime": runtime,
        "sample_id": sample_id,
        "tp_size": spec.tp_size,
        "hicache": "disabled",
    }

    shutil.rmtree(spec.local_target_checkpoint_dir, ignore_errors=True)

    endpoint = SGLangEndpoint(
        model_path=spec.base_checkpoint_dir,
        worker_port=spec.port,
        tp=spec.tp_size,
        extra_server_args=server_args_for_mode(spec.server_args, update_mode),
        # Allow for a cold, model-sized initial load. Destination initialization
        # is measured separately after the endpoint begins serving.
        health_timeout=2 * 60 * 60,
        health_poll_interval=10,
        log_requests_level=-1,
    )
    url = f"http://127.0.0.1:{spec.port}"
    try:
        initial_load_started = time.perf_counter()
        endpoint.start()
        results["initial_load_s"] = round(
            time.perf_counter() - initial_load_started,
            6,
        )
        results["memory_after_initial_load"] = _memory_snapshot()
        results["generation_before"] = _generate(url)
        speculative_algorithm = spec.server_args.get(
            "--speculative-algorithm", ""
        ).upper()
        # These SGLang speculative workers reject return_logprob requests.
        fingerprint_logprobs = speculative_algorithm not in {"DFLASH", "DSPARK"}
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

        with httpx.Client(timeout=None, trust_env=False) as client:
            destination_init_started = time.perf_counter()
            destination_init_cpu_started = _cgroup_cpu_usage_s()
            with _GenerationProbe(url) as generation:
                init_payload = {
                    "base_checkpoint_dir": spec.base_checkpoint_dir,
                    "target_version": 0,
                    "destination": update_mode,
                }
                if update_mode == "disk":
                    init_payload["local_checkpoint_dir"] = (
                        spec.local_target_checkpoint_dir
                    )
                initialized = _post(
                    client,
                    url,
                    "/stage_weight_update",
                    init_payload,
                )
            results["destination_init_s"] = round(
                time.perf_counter() - destination_init_started,
                6,
            )
            results["destination_init_cpu"] = _cpu_usage_delta(
                destination_init_cpu_started,
                _cgroup_cpu_usage_s(),
                results["destination_init_s"],
            )
            results["destination_init_rank_stats"] = initialized.get("rank_stats")
            results["generation_during_destination_init"] = generation.summary()
            results["memory_after_destination_init"] = _memory_snapshot()
            if generation.errors or not generation.samples:
                raise RuntimeError(
                    "generation was not healthy during destination initialization: "
                    f"{generation.summary()}"
                )
            if {sample["weight_version"] for sample in generation.samples} != {
                baseline_fingerprint["weight_version"]
            }:
                raise RuntimeError(
                    "generation switched weight versions during destination "
                    f"initialization: {generation.summary()}"
                )
            if update_mode == "disk":
                results["destination_init_cache_drop"] = _drop_checkpoint_page_cache(
                    spec.local_target_checkpoint_dir
                )

            stage_started = time.perf_counter()
            stage_cpu_started = _cgroup_cpu_usage_s()
            with _GenerationProbe(url) as generation:
                stage_payload = {
                    "base_checkpoint_dir": spec.base_checkpoint_dir,
                    "checkpoint_source_dir": source_dir,
                    "target_version": target_version,
                    "destination": update_mode,
                }
                if update_mode == "disk":
                    stage_payload["local_checkpoint_dir"] = (
                        spec.local_target_checkpoint_dir
                    )
                staged = _post(
                    client,
                    url,
                    "/stage_weight_update",
                    stage_payload,
                )
            results["stage_s"] = round(time.perf_counter() - stage_started, 6)
            results["stage_cpu"] = _cpu_usage_delta(
                stage_cpu_started,
                _cgroup_cpu_usage_s(),
                results["stage_s"],
            )
            results["stage_rank_stats"] = staged.get("rank_stats")
            results["generation_during_stage"] = generation.summary()
            results["memory_after_stage"] = _memory_snapshot()
            if generation.errors or not generation.samples:
                raise RuntimeError(
                    f"generation was not healthy during staging: {generation.summary()}"
                )
            if {sample["weight_version"] for sample in generation.samples} != {
                baseline_fingerprint["weight_version"]
            }:
                raise RuntimeError(
                    "generation switched weight versions during staging: "
                    f"{generation.summary()}"
                )

            commit_started = time.perf_counter()
            commit_cpu_started = _cgroup_cpu_usage_s()
            if update_mode == "cpu":
                commit_path = "/update_weights_from_cpu"
                commit_payload = {
                    "target_version": target_version,
                    "flush_cache": False,
                }
            else:
                commit_path = "/update_weights_from_disk"
                commit_payload = {
                    "model_path": spec.local_target_checkpoint_dir,
                    "load_format": spec.server_args.get("--load-format", "auto"),
                    "weight_version": str(target_version),
                    "flush_cache": False,
                }
            committed = _post(
                client,
                url,
                commit_path,
                commit_payload,
            )
            results["commit_rpc_s"] = round(
                time.perf_counter() - commit_started,
                6,
            )
            results["commit_cpu"] = _cpu_usage_delta(
                commit_cpu_started,
                _cgroup_cpu_usage_s(),
                results["commit_rpc_s"],
            )
            results["commit_rank_stats"] = committed.get("rank_stats")
            results["stage_through_commit_s"] = round(
                time.perf_counter() - stage_started,
                6,
            )

        results["generation_after"] = _generate(url)
        fingerprints = [
            _generate(
                url,
                fingerprint=True,
                fingerprint_logprobs=fingerprint_logprobs,
            ),
            _generate(
                url,
                fingerprint=True,
                fingerprint_logprobs=fingerprint_logprobs,
            ),
        ]
        observed_versions = {
            results["generation_after"]["weight_version"],
            *(fingerprint["weight_version"] for fingerprint in fingerprints),
        }
        if observed_versions != {str(target_version)}:
            raise RuntimeError(
                "post-update generation reported unexpected weight versions: "
                f"{sorted(str(value) for value in observed_versions)}"
            )
        results["fingerprint_probes"] = [
            {
                "wall_s": fingerprint["wall_s"],
                "weight_version": fingerprint["weight_version"],
            }
            for fingerprint in fingerprints
        ]
        results["correctness"] = {
            **_assert_repeat_consistency(fingerprints),
            **_assert_target_changed(baseline_fingerprint, fingerprints[0]),
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
        try:
            endpoint.stop()
        except Exception as exc:  # noqa: BLE001 - preserve the benchmark result
            print(f"warning: failed to stop SGLang cleanly: {exc}", flush=True)
