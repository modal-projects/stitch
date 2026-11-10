"""The engine test: one SGLang engine serves steady traffic while it takes every version of a delta
set as Stitch syncs it (prepare while serving, then pause in place, update, continue). Per version:
the engine's prepare and apply stats, the pause as the client sees it, and decode throughput while
preparing against serving undisturbed. On a correctness set, SGLang checks each version's checksums
and at the end the engine must hash byte for byte like a fresh engine on the expected checkpoint.

    modal run tests/engine/engine_test.py \
        --engine qwen3.6-35b-a3b-fp8 --run 1029-180000
"""

import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))
import harness  # noqa: E402
from engine_configs import DELTA_SETS, ENGINES  # noqa: E402

PORT, CONCURRENCY = harness.ENGINE_PORT, 64
URL = f"http://127.0.0.1:{PORT}"
WINDOW_S = 10
FAILED_WAIT_S = 5
DECODE = re.compile(
    r"^\[(\S+ \S+)( DP(\d+))?[^\]]*\] Decode batch.*gen throughput \(token/s\): ([\d.]+)"
)

app = modal.App(
    "weight-delta-engine-test",
    image=harness.image.add_local_python_source("harness", "engine_configs"),
)


def post(
    path: str,
    body: dict | None = None,
    timeout: float = 1800,
    url: str = URL,
    headers: dict[str, str] | None = None,
) -> dict:
    """Status 0 when the request never got an HTTP answer."""
    request = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            status, text = reply.status, reply.read().decode()
    except urllib.error.HTTPError as error:
        status, text = error.code, error.read().decode()
    except OSError as error:
        status, text = 0, repr(error)
    return {
        "status": status,
        "seconds": time.monotonic() - started,
        "reply": json.loads(text) if text.startswith(("{", "[")) else text,
    }


class Traffic:
    """Requests in flight all the time, sampled as rollouts sample; each must run to its length."""

    def __init__(
        self, concurrency: int, url: str = URL, headers: dict[str, str] | None = None
    ) -> None:
        self.url, self.headers = url, headers
        self.stop, self.done, self.failed = threading.Event(), [], []
        self.threads = [
            threading.Thread(target=self._loop, args=(index,), daemon=True)
            for index in range(concurrency)
        ]

    def _loop(self, index: int) -> None:
        while not self.stop.is_set():
            body = {
                "text": f"Request {index}: write a long story about weight synchronization.",
                "sampling_params": {
                    "max_new_tokens": 256,
                    "ignore_eos": True,
                    "temperature": 1.0,
                },
            }
            result = post("/generate", body, url=self.url, headers=self.headers)
            reply = result["reply"] if isinstance(result["reply"], dict) else {}
            finish = (reply.get("meta_info") or {}).get("finish_reason") or {}
            ok = result["status"] == 200 and finish.get("type") == "length"
            (self.done if ok else self.failed).append((time.time(), result["status"]))
            if not ok:
                # With no engine up, Modal answers at once: never resend in a tight loop.
                self.stop.wait(FAILED_WAIT_S)

    def __enter__(self):
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=1800)


def _throughput(log: str, windows: list[tuple[str, float, float]]) -> dict[str, float]:
    """Per window: decode tokens/s summed over DP ranks of each rank's mean."""
    samples = []
    for line in log.splitlines():
        match = DECODE.match(line)
        if match:
            when = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").timestamp()
            samples.append((when, int(match[3] or 0), float(match[4])))
    out = {}
    for name, start, end in windows:
        ranks = {}
        for when, rank, value in samples:
            if start - 1 < when <= end:
                ranks.setdefault(rank, []).append(value)
        out[name] = sum(sum(v) / len(v) for v in ranks.values()) if ranks else None
    return out


def syncs(manifests: list[Path], windows: list) -> list[dict]:
    syncs = []
    for version, manifest in enumerate(manifests, 1):
        prepare_started = time.time()
        prepared = post("/prepare_weights_from_delta", {"manifest": str(manifest)})
        windows.append((f"v{version} prepare", prepare_started, time.time()))
        if prepared["status"] != 200:
            raise RuntimeError(f"v{version} prepare: {prepared}")
        time.sleep(WINDOW_S)
        paused = time.monotonic()
        post("/pause_generation", {"mode": "in_place"})
        applied = post(
            "/update_weights_from_delta",
            {
                "manifest": str(manifest),
                "weight_version": str(version),
                "flush_cache": False,
            },
        )
        post("/continue_generation")
        pause_s = time.monotonic() - paused
        if applied["status"] != 200:
            raise RuntimeError(f"v{version} apply: {applied}")
        syncs.append(
            {
                "version": version,
                "prepare_s": prepared["seconds"],
                "pause_s": pause_s,
                "apply_s": applied["seconds"],
                "prepare": prepared["reply"],
                "apply": applied["reply"],
            }
        )
        time.sleep(WINDOW_S)
    return syncs


NODE = dict(
    cpu=32,
    memory=524288,
    timeout=6 * 3600,
    volumes={
        harness.CHECKPOINTS: harness.checkpoints,
        harness.DATA: harness.data,
        harness.SGLANG_CACHE: harness.sglang_cache,
    },
    experimental_options=harness.EXPERIMENTAL_OPTIONS,
)


@app.function(gpu="B200:8", **NODE)
@modal.clustered(size=1)
def engine_test_b200(engine: str, run: str) -> dict:
    return _engine_test(engine, run)


@app.function(gpu="H200:8", **NODE)
@modal.clustered(size=1)
def engine_test_h200(engine: str, run: str) -> dict:
    return _engine_test(engine, run)


@app.function(gpu="B300:8", **NODE)
@modal.clustered(size=1)
def engine_test_b300(engine: str, run: str) -> dict:
    return _engine_test(engine, run)


def _engine_test(name: str, run: str) -> dict:
    config = ENGINES[name]
    gpus = ",".join(str(gpu) for gpu in range(config.gpus))
    run_dir = Path(harness.DATA, "engine", config.delta_set, run)
    out_dir = run_dir / f"{name}-{datetime.now().strftime('%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True)
    manifests = sorted((run_dir / "deltas").glob("weight_v*/manifest.json"))
    report = json.loads((run_dir / "report.json").read_text())
    result = {
        "revisions": harness.install_overlays(),
        "delta_set": report,
        "versions": len(manifests),
    }
    log = out_dir / "engine.log"
    engines = []
    try:
        engine = harness.Engine(
            f"{harness.CHECKPOINTS}/{DELTA_SETS[config.delta_set].rollout_checkpoint}",
            config,
            gpus,
            PORT,
            log,
        )
        engines.append(engine)
        engine.wait()
        engine.warm_up()
        # The engine serves version 0, the base, as Miles labels it before its first sync.
        post("/update_weight_version", {"new_version": "0"})
        windows = []
        with Traffic(CONCURRENCY) as traffic:
            started = time.time()
            time.sleep(2 * WINDOW_S)
            windows.append(("baseline", started + WINDOW_S, time.time()))
            result["syncs"] = syncs(manifests, windows)
        last_resume = windows[-1][2]
        result["traffic"] = {
            "done": len(traffic.done),
            "failed": traffic.failed[:20],
            "done_after_last_update": sum(
                when > last_resume for when, _ in traffic.done
            ),
        }
        result["throughput"] = _throughput(log.read_text(errors="replace"), windows)
        if report["mode"] != "correctness":
            return result
        delta_state = engine.state()
        engine.stop()
        reference = harness.Engine(
            str(run_dir / "expected"),
            config,
            gpus,
            PORT,
            out_dir / "reference.log",
        )
        engines.append(reference)
        reference.wait()
        result["check"] = harness.compare(delta_state, reference.state())
    except Exception as error:
        result["error"] = repr(error)
    finally:
        for engine in engines:
            engine.stop()
        (out_dir / "result.json").write_text(json.dumps(result, indent=1))
        harness.data.commit()
        harness.sglang_cache.commit()
    return result


@app.local_entrypoint()
def main(engine: str, run: str):
    test = {
        "B200": engine_test_b200,
        "H200": engine_test_h200,
        "B300": engine_test_b300,
    }[ENGINES[engine].gpu]
    result = test.remote(engine, run)
    for sync in result.get("syncs", []):
        stats = {k: round(v, 4) for k, v in sync.items() if isinstance(v, float)}
        print(f"v{sync['version']}: {json.dumps(stats)}")
    print("throughput:", json.dumps(result.get("throughput")))
    print("traffic:", json.dumps(result.get("traffic")))
    # Version 1 is an outlier: Adam's first step moves every weight, and the engine builds its
    # per-run state.
    steady = result.get("syncs", [])[1:]
    if steady:
        delta_set = result["delta_set"]
        print(
            "steady state, mean of v2 on:",
            json.dumps(
                {
                    "delta_mb": statistics.mean(delta_set["wire_nbytes"][1:]) / 1e6,
                    "changed": statistics.mean(delta_set["density"][1:]),
                    "prepare_s": statistics.mean(sync["prepare_s"] for sync in steady),
                    "pause_s": statistics.mean(sync["pause_s"] for sync in steady),
                }
            ),
        )
    check = result.get("check")
    if check is not None:
        print("bit-exact check:", json.dumps(check)[:2000])
    if (
        result.get("error")
        or (check and (check["differ"] or check["only_one"]))
        or result.get("traffic", {}).get("failed")
    ):
        raise SystemExit(f"engine test FAILED: {result.get('error') or 'see result'}")
    print("engine test passed")
