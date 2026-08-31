"""Probe: clustered + single_use_containers + retries lifecycle on Modal.

The reentrant trainer (``cookbook.miles_disagg.app``) rests on gang semantics
the SDK does not document. This probe measures them on a live 2×T4:8 gang.

Verified 2026-08-31 in ``stitch-dev`` (modal 1.5.4.dev11):

- ``clustered`` requires GPUs, in full nodes only (T4 nodes are 8-wide).
- Inputs broadcast to every rank; the caller's ``.get()`` resolves when rank 0
  returns, with rank 0's value. A worker returning early does NOT abort
  rank 0's in-flight execution.
- A still-executing worker is NOT torn down when rank 0 returns successfully
  (it kept running ≥57s after rank 0 finished). A worker that outlives its
  usefulness therefore burns its node until the attempt timeout: worker holds
  must self-terminate (``ray_cluster.hold_worker_node``).
- When rank 0's attempt FAILS, Modal reaps still-running workers (~58s later
  here) and only then schedules the retry as a complete fresh gang: new
  containers on every rank, ``@modal.enter()`` reruns,
  ``modal.current_function_call_id()`` is stable across attempts.
- With ``single_use_containers=True`` each rank's container exits after its
  single input, so every retry attempt starts from clean containers.

Run:
  uv run --extra modal modal run -e stitch-dev -m tools.probes.gang_lifecycle_spike --scenario all
"""

from __future__ import annotations

import os
import time

import modal
import modal.experimental

APP_NAME = "gang-lifecycle-spike"
DICT_NAME = "gang-lifecycle-spike-events"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12")


def _events() -> modal.Dict:
    return modal.Dict.from_name(DICT_NAME, create_if_missing=True)


def _record(scenario: str, event: str, **fields) -> None:
    key = f"{scenario}/{time.time():.3f}/{event}"
    _events()[key] = {"event": event, **fields}
    print(f"EVENT {key} {fields}", flush=True)


@app.cls(
    image=image,
    gpu="T4:8",  # cheapest full node modal schedules a gang on
    timeout=15 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
    single_use_containers=True,
    include_source=True,
)
@modal.experimental.clustered(2)
class Gang:
    @modal.enter()
    def enter(self) -> None:
        info = modal.experimental.get_cluster_info()
        self.rank = info.rank
        self.task_id = os.environ.get("MODAL_TASK_ID", "unknown")
        print(f"ENTER rank={self.rank} task={self.task_id}", flush=True)

    @modal.method()
    def run(self, scenario: str, fail_first_attempt: bool = False) -> dict:
        call_id = modal.current_function_call_id()
        events = _events()
        attempt_key = f"{scenario}/attempts/rank{self.rank}"
        attempt = events.get(attempt_key, 0)
        events[attempt_key] = attempt + 1
        _record(
            scenario,
            "start",
            rank=self.rank,
            task=self.task_id,
            attempt=attempt,
            call_id=call_id,
        )

        if self.rank != 0:
            if scenario == "early-return":
                _record(scenario, "worker-early-return", rank=self.rank, task=self.task_id)
                return {"rank": self.rank, "task": self.task_id}
            # hold "forever": heartbeat until torn down or the attempt times out
            start = time.monotonic()
            while True:
                time.sleep(5)
                _record(
                    scenario,
                    "worker-heartbeat",
                    rank=self.rank,
                    task=self.task_id,
                    held_s=round(time.monotonic() - start, 1),
                )

        # rank 0
        if fail_first_attempt and attempt == 0:
            time.sleep(15)
            _record(scenario, "rank0-raising", task=self.task_id, attempt=attempt)
            raise RuntimeError(f"synthetic failure on attempt {attempt}")
        work_s = 60 if scenario == "early-return" else 30
        time.sleep(work_s)
        _record(scenario, "rank0-done", task=self.task_id, attempt=attempt, call_id=call_id)
        return {"rank": 0, "task": self.task_id, "attempt": attempt, "call_id": call_id}


def _drain_events(scenario: str) -> list[tuple[str, dict]]:
    return sorted(
        (k, v)
        for k, v in _events().items()
        if k.startswith(f"{scenario}/") and "/attempts/" not in k
    )


def _run_scenario(scenario: str, fail_first_attempt: bool, watch_after_s: int) -> None:
    print(f"\n===== scenario: {scenario} (fail_first_attempt={fail_first_attempt}) =====")
    events = _events()
    for key in list(events.keys()):
        if key.startswith(f"{scenario}/"):
            events.pop(key)
    call = Gang().run.spawn(scenario, fail_first_attempt)
    print(f"spawned {call.object_id}")
    t0 = time.monotonic()
    if scenario == "cancel":
        time.sleep(45)  # let the gang form and hold
        print("cancelling…")
        call.cancel(terminate_containers=True)
        result: object = "CANCELLED"
    else:
        result = call.get()
    print(f"result after {time.monotonic() - t0:.1f}s: {result}")
    # watch for straggler heartbeats after the result: does the held worker
    # keep running, or was it torn down with rank 0?
    time.sleep(watch_after_s)
    print(f"--- events ({scenario}) ---")
    for key, val in _drain_events(scenario):
        print(f"  {key}: {val}")


@app.local_entrypoint()
def main(scenario: str = "all") -> None:
    scenarios = (
        ["early-return", "hold", "retry", "cancel"] if scenario == "all" else [scenario]
    )
    for name in scenarios:
        _run_scenario(name, fail_first_attempt=(name == "retry"), watch_after_s=60)
