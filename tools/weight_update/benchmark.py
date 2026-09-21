"""Shared runner for SGLang delta weight-update profilers.

Model-specific Modal apps supply the checkpoint paths, recorded delta lineage,
GPU shape, and direct SGLang arguments. This module owns the benchmark
sequence: startup through routing readiness, live generation during target
preparation, the commit RPC, and post-update generation.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tools.weight_update.metrics import (
    _cgroup_cpu_usage_s,
    _cgroup_io_snapshot,
    _cpu_usage_delta,
    _io_usage_delta,
    _local_storage_snapshot,
    _memory_snapshot,
    _MemoryProbe,
    _print_profile_summary,
    _resource_snapshot,
)
from tools.weight_update.sglang import (
    _assert_post_mutation_failure_is_fatal,
    _compare_fingerprints,
    _fingerprint,
    _fingerprint_after_commit,
    _fingerprint_hashes,
    _generate,
    _post,
    _validate_weight_checksums,
    _weight_checksum_differences,
    _weight_checksums,
)

UpdateMode = Literal["disk", "cpu"]
CanonicalStorage = Literal["memory", "disk"]
_VALIDATION_RANDOM_SEED = "42"


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
    weight_version: int = 0,
) -> dict[str, str]:
    """Return direct SGLang arguments for one update mode."""

    result = dict(server_args)
    result["--weight-update-staging"] = update_mode
    result["--weight-version"] = str(weight_version)
    result.setdefault("--random-seed", _VALIDATION_RANDOM_SEED)
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

    result = dict(server_args)
    for key in (
        "--weight-update-staging",
        "--weight-update-local-checkpoint-dir",
        "--weight-update-max-compile-group-gb",
    ):
        result.pop(key, None)
    result["--weight-version"] = str(weight_version)
    result.setdefault("--random-seed", _VALIDATION_RANDOM_SEED)
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
        "resources": _resource_snapshot(),
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
        baseline_fingerprint = _fingerprint(
            url,
            fingerprint_logprobs=fingerprint_logprobs,
        )
        if not fingerprint_logprobs:
            _compare_fingerprints(
                [
                    baseline_fingerprint,
                    _fingerprint(url, fingerprint_logprobs=False),
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
                        "flush_cache": False,
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
                    require_draft_weights=bool(
                        spec.server_args.get("--speculative-algorithm")
                    ),
                )
                results["updates"].append(update)
                previous_version = target_version
                previous_checksums = current_checksums

            # The destructive fault probe owns the adjacent v5 artifact. Keep
            # this pre-mutation probe on the next unpublished version so the
            # independent validations cannot affect each other.
            failure_version = target_versions[-1] + 2
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
            failure_fingerprint = _fingerprint(
                url,
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
            results["clean_native_load_checksum_differences"] = (
                _weight_checksum_differences(
                    final_live_checksums,
                    reference_checksums,
                )
            )
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
        reference_fingerprints = [
            _fingerprint(url, fingerprint_logprobs=fingerprint_logprobs)
            for _ in range(2)
        ]
        reference_versions = {
            fingerprint["weight_version"] for fingerprint in reference_fingerprints
        }
        if reference_versions != {str(final_version)}:
            raise RuntimeError(
                "clean native load reported unexpected weight versions: "
                f"{sorted(str(value) for value in reference_versions)}"
            )
        reference_fingerprint = reference_fingerprints[0]
        reference_repeat = _compare_fingerprints(reference_fingerprints)
        results["clean_native_load"] = {
            "checkpoint_dir": reference_checkpoint_dir,
            "weight_version": reference_fingerprint["weight_version"],
            "per_engine_checksum": reference_checksums["per_engine_checksum"],
            "matches_live_weight_checksums": True,
            "repeat_fingerprint": reference_repeat,
            **_fingerprint_hashes(reference_fingerprint),
        }
        results["clean_native_load"]["fingerprint_comparison"] = _compare_fingerprints(
            [final_live_fingerprint, reference_fingerprint]
        )

        results["status"] = "passed"
        _print_profile_summary(results)
        print(json.dumps(results, indent=2), flush=True)
        return results
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["memory_after_failure"] = _memory_snapshot()
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


def run_post_mutation_failure(
    spec: WeightUpdateSpec,
    *,
    source_dir: str,
    served_version: int,
    failure_version: int,
    update_mode: UpdateMode,
    canonical_storage: CanonicalStorage | None,
    runtime: str,
    sample_id: str,
) -> dict[str, Any]:
    """Prove that a distributed failure during live mutation is terminal."""

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint
    from sglang.srt.weight_sync.disk_checkpoint import materialize

    if failure_version != served_version + 1:
        raise ValueError("failure_version must immediately follow served_version")
    _validate_delta_lineage(source_dir, (failure_version,))

    results: dict[str, Any] = {
        "model": spec.model_name,
        "runtime": runtime,
        "update_mode": update_mode,
        "canonical_storage": canonical_storage,
        "sample_id": sample_id,
        "served_version": served_version,
        "failure_version": failure_version,
        "resources": _resource_snapshot(),
    }
    shutil.rmtree(spec.local_target_checkpoint_dir, ignore_errors=True)
    shutil.rmtree(spec.local_canonical_checkpoint_dir, ignore_errors=True)
    if served_version == 0:
        # The immutable base already is the complete v0 checkpoint. Boot it
        # directly instead of copying model-sized bytes merely to inject a
        # failure into the first live update.
        served_checkpoint_dir = spec.base_checkpoint_dir
        results["served_checkpoint_materialization"] = {
            "operation": "immutable_base",
            "target_version": 0,
        }
    else:
        # A disk reload needs an inactive destination distinct from the
        # checkpoint the engine boots. CPU staging does not use the target
        # path, so it can boot there and reserve the canonical path for NVMe.
        served_checkpoint_dir = (
            spec.local_canonical_checkpoint_dir
            if update_mode == "disk"
            else spec.local_target_checkpoint_dir
        )
        results["served_checkpoint_materialization"] = materialize(
            local_checkpoint_dir=served_checkpoint_dir,
            base_checkpoint_dir=spec.base_checkpoint_dir,
            checkpoint_source_dir=source_dir,
            target_version=served_version,
            base_version=0,
        )

    endpoint = SGLangEndpoint(
        model_path=served_checkpoint_dir,
        worker_port=spec.port,
        tp=spec.tp_size,
        extra_server_args=server_args_for_mode(
            spec.server_args,
            update_mode,
            canonical_storage,
            spec.local_target_checkpoint_dir,
            spec.local_canonical_checkpoint_dir,
            spec.max_compile_group_gb,
            weight_version=served_version,
        ),
        health_timeout=2 * 60 * 60,
        health_poll_interval=10,
        log_requests_level=-1,
    )
    endpoint_running = False
    url = f"http://127.0.0.1:{spec.port}"
    try:
        startup_started = time.perf_counter()
        endpoint.start()
        endpoint_running = True
        results["startup_ready_s"] = round(time.perf_counter() - startup_started, 6)
        baseline = _fingerprint(url)
        if baseline["weight_version"] != str(served_version):
            raise RuntimeError(
                "fault probe booted an unexpected weight version: "
                f"{baseline['weight_version']!r}"
            )
        with httpx.Client(timeout=None, trust_env=False) as client:
            checksums_before = _weight_checksums(client, url)
            prepare_started = time.perf_counter()
            with _GenerationProbe(url) as generation:
                prepared = _post(
                    client,
                    url,
                    "/prepare_weight_update",
                    {
                        "checkpoint_source_dir": source_dir,
                        "target_version": failure_version,
                    },
                )
            results["preparation"] = {
                "wall_s": round(time.perf_counter() - prepare_started, 6),
                "rank_stats": prepared.get("rank_stats"),
                "generation": _validate_generation_probe(
                    generation,
                    served_version,
                ),
            }
            checksums_after_prepare = _weight_checksums(client, url)
        if checksums_after_prepare != checksums_before:
            raise RuntimeError("preparation changed live target or draft weights")

        results["post_mutation_failure"] = _assert_post_mutation_failure_is_fatal(
            url,
            target_version=failure_version,
            expected_ranks=spec.tp_size,
            injection_delay_s=5.0 if update_mode == "disk" else 0.1,
            settle_s=180.0 if update_mode == "disk" else 30.0,
        )
        results["status"] = "passed"
        _print_profile_summary(results)
        print(json.dumps(results, indent=2), flush=True)
        return results
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["memory_after_failure"] = _memory_snapshot()
        _print_profile_summary(results)
        print(json.dumps(results, indent=2), flush=True)
        raise
    finally:
        if endpoint_running:
            try:
                endpoint.stop()
            except Exception as exc:  # noqa: BLE001 - preserve benchmark result
                print(f"warning: failed to stop SGLang cleanly: {exc}", flush=True)
