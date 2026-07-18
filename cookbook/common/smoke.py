"""Flash-pool smoke check — shared by every recipe. Confirms the pool serves
completions at an expected weight version, through the gateway and each replica."""

from __future__ import annotations

import json
import time
import urllib.request

from stitch.pools.modal_flash import ModalFlashPool
from stitch.types import VersionRef


class VersionAheadError(RuntimeError):
    """A monotonic pool has already advanced past the smoke's expected version."""


def smoke_flash_pool(*, app_name: str, cls_name: str, model_name: str, weight_version: int, timeout_seconds: int) -> None:
    """Poll until the pool serves completions at ``weight_version`` — through the gateway
    (which also wakes a scaled-down pool; Flash holds the request through the cold start)
    and then each live replica's ``/server_info``."""
    pool = ModalFlashPool(app_name, cls_name)
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            gateway = pool.gateway_url()
            print(f"Gateway URL: {gateway}")
            # A fresh pool serves the base but has no claimed run, so version 0 (run-scoped)
            # is unpinnable — an exact-version request would 409. Pre-claim, gate on plain
            # serving; the version check only makes sense once a run has claimed the pool.
            if _get_json(f"{gateway}/server_info", timeout=60).get("run_id") is None:
                data = _post_json(f"{gateway}/v1/chat/completions", _completion(model_name), timeout=900)
                _check_serves(data)
                print(f"Pool serves base (unclaimed): {data.get('choices')}")
                return
            data = _post_json(f"{gateway}/v1/chat/completions", _completion(model_name, weight_version), timeout=900)
            print(f"Gateway completion: {data}")
            _check_completion(data, weight_version)
            for target in pool.discover_replicas():
                info = _get_json(f"{target}/server_info", timeout=30)
                print(f"{target} server_info={info}")
                _check_version(_applied_version(info), weight_version, target)
            return
        except VersionAheadError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(f"Flash pool smoke did not pass before timeout: {last_error}")
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def _applied_version(info: dict) -> int:
    applied = info.get("applied")
    return VersionRef.parse(applied).version if applied else -1


def _check_version(current: int, expected: int, target: str) -> None:
    if current > expected:
        raise VersionAheadError(f"{target} applied={current} already past expected {expected}")
    if current != expected:
        raise RuntimeError(f"{target} applied={current}, expected {expected}")


def _check_completion(data: dict, expected: int) -> None:
    start, end = int(data.get("weight_version_start", -1)), int(data.get("weight_version_end", -1))
    if start > expected or end > expected:
        raise VersionAheadError(f"gateway served {start}->{end}, already past expected {expected}")
    if start != expected or end != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _check_serves(data: dict) -> None:
    choices = data.get("choices") or []
    if not choices or not ((choices[0].get("message") or {}).get("content")):
        raise RuntimeError(f"pool did not return a completion: {data}")


def _completion(model_name: str, expected: int | None = None) -> dict:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if expected is not None:  # pin the version only against a claimed pool
        payload["weight_version"] = {"exact_version": expected}
    return payload


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)
