"""Modal Sandbox execution environment for repository-repair rollouts."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import shlex
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

_OUTPUT_HEAD_BYTES = 5000
_OUTPUT_TAIL_BYTES = 5000
_OUTPUT_LIMIT_BYTES = _OUTPUT_HEAD_BYTES + _OUTPUT_TAIL_BYTES
_OUTPUT_HARD_LIMIT_BYTES = 16 * 1024 * 1024
_RESOURCE_CACHE_LOCK = threading.Lock()
_APP_LOOKUP_LOCK = asyncio.Lock()
_APP_CACHE: dict[str, Any] = {}
_IMAGE_CACHE: dict[str, Any] = {}
_MODAL_OBJECT_NAME_RE = re.compile(r"^[a-zA-Z0-9-_.]+$")
_MODAL_APP_ID_RE = re.compile(r"^ap-[a-zA-Z0-9]{22}$")
_SANDBOX_READY_PATH = "/tmp/miles-sandbox-ready"

# Run the policy command inside the Sandbox and bound its output before it
# crosses Modal's command-router connection. mini-swe-agent only exposes the
# first/last 5,000 characters to the model, so transporting hundreds of MiB
# first is pure waste. stdout and stderr are drained concurrently to avoid pipe
# backpressure; their presentation remains stdout followed by stderr, matching
# the previous adapter.
_BOUNDED_COMMAND_RUNNER = r"""
import base64
import json
import os
import selectors
import signal
import subprocess
import sys
import time

command = sys.stdin.buffer.read().decode("utf-8", errors="surrogateescape")
head_limit = int(sys.argv[1])
tail_limit = int(sys.argv[2])
timeout_seconds = float(sys.argv[3])
output_hard_limit = int(sys.argv[4])
output_limit = head_limit + tail_limit


class Capture:
    def __init__(self):
        self.total = 0
        self.head = bytearray()
        self.tail = bytearray()
        self.full = bytearray()

    def add(self, chunk):
        self.total += len(chunk)
        if self.full is not None:
            if len(self.full) + len(chunk) <= output_limit:
                self.full.extend(chunk)
            else:
                self.full = None
        if len(self.head) < head_limit:
            self.head.extend(chunk[: head_limit - len(self.head)])
        self.tail.extend(chunk)
        if len(self.tail) > tail_limit:
            del self.tail[: len(self.tail) - tail_limit]


def combined_head(stdout, stderr):
    if stdout.total >= head_limit:
        return bytes(stdout.head[:head_limit])
    return bytes(stdout.head) + bytes(stderr.head[: head_limit - stdout.total])


def combined_tail(stdout, stderr):
    if stderr.total >= tail_limit:
        return bytes(stderr.tail[-tail_limit:])
    need_from_stdout = tail_limit - stderr.total
    return bytes(stdout.tail[-need_from_stdout:]) + bytes(stderr.tail)


started = time.monotonic()
deadline = started + timeout_seconds
process = subprocess.Popen(
    ["bash", "-l"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
)
try:
    process.stdin.write(command.encode("utf-8", errors="surrogateescape"))
except BrokenPipeError:
    # The script may deliberately exit before consuming trailing input.
    pass
finally:
    process.stdin.close()
captures = {
    process.stdout.fileno(): Capture(),
    process.stderr.fileno(): Capture(),
}
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)
selector.register(process.stderr, selectors.EVENT_READ)
timed_out = False
output_limited = False


def stop_process_group():
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        pass
    # The shell may exit while a descendant keeps the captured pipes open.
    # Always kill the remaining process group after the grace period.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


while selector.get_map():
    remaining = deadline - time.monotonic()
    if not timed_out and remaining <= 0:
        timed_out = True
        stop_process_group()
    wait_seconds = 0.25 if (timed_out or output_limited) else min(0.25, max(0.0, remaining))
    for key, _ in selector.select(timeout=wait_seconds):
        chunk = os.read(key.fd, 1 << 16)
        if chunk:
            captures[key.fd].add(chunk)
            total_captured = sum(capture.total for capture in captures.values())
            if not output_limited and total_captured >= output_hard_limit:
                # The model only receives the bounded head/tail. Continuing to
                # produce and drain gigabytes can monopolize a sandbox for the
                # complete command timeout without adding any observation.
                output_limited = True
                stop_process_group()
        else:
            selector.unregister(key.fileobj)
return_code = process.wait()
remote_seconds = time.monotonic() - started
stdout_capture = captures[process.stdout.fileno()]
stderr_capture = captures[process.stderr.fileno()]
total_bytes = stdout_capture.total + stderr_capture.total
truncated = total_bytes > output_limit
if truncated:
    output = b""
    output_head = combined_head(stdout_capture, stderr_capture)
    output_tail = combined_tail(stdout_capture, stderr_capture)
else:
    output = bytes(stdout_capture.full) + bytes(stderr_capture.full)
    output_head = b""
    output_tail = b""
payload = {
    "return_code": 124 if timed_out else (125 if output_limited else return_code),
    "timed_out": timed_out,
    "output_limited": output_limited,
    "remote_seconds": remote_seconds,
    "stdout_bytes": stdout_capture.total,
    "stderr_bytes": stderr_capture.total,
    "total_bytes": total_bytes,
    "truncated": truncated,
    "output_b64": base64.b64encode(output).decode("ascii"),
    "output_head_b64": base64.b64encode(output_head).decode("ascii"),
    "output_tail_b64": base64.b64encode(output_tail).decode("ascii"),
}
print(json.dumps(payload, separators=(",", ":")), flush=True)
"""


class SandboxCommandTimeoutError(TimeoutError):
    """A command inside the Modal Sandbox exceeded its per-command timeout."""

    def __init__(
        self,
        message: str,
        *,
        result: SandboxExecResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


class SandboxTransportError(RuntimeError):
    """The Sandbox command transport violated the adapter protocol."""


@dataclass(frozen=True)
class SandboxExecResult:
    return_code: int
    output: str
    output_head: str
    output_tail: str
    output_total_bytes: int
    output_truncated: bool
    remote_seconds: float
    client_seconds: float
    transferred_bytes: int
    timed_out: bool = False
    output_limited: bool = False

    @property
    def transport_seconds(self) -> float:
        return max(0.0, self.client_seconds - self.remote_seconds)


def _cached_app(name: str, *, create_if_missing: bool = False):
    with _RESOURCE_CACHE_LOCK:
        if name not in _APP_CACHE:
            _APP_CACHE[name] = modal.App.lookup(
                name, create_if_missing=create_if_missing
            )
        return _APP_CACHE[name]


async def ensure_sandbox_app(name: str) -> None:
    """Create the shared Sandbox app once without blocking the rollout loop."""
    async with _APP_LOOKUP_LOCK:
        with _RESOURCE_CACHE_LOCK:
            if name in _APP_CACHE:
                return
        app = await modal.App.lookup.aio(name, create_if_missing=True)
        with _RESOURCE_CACHE_LOCK:
            _APP_CACHE.setdefault(name, app)


def _cached_image(name: str):
    with _RESOURCE_CACHE_LOCK:
        if name not in _IMAGE_CACHE:
            _IMAGE_CACHE[name] = modal.Image.from_registry(name).entrypoint([])
        return _IMAGE_CACHE[name]


def _decode_output(value: str) -> str:
    return base64.b64decode(value).decode("utf-8", errors="replace")


def _parse_bounded_command_response(
    *,
    stdout: bytes,
    stderr: bytes,
    fallback_return_code: int,
    client_seconds: float,
) -> SandboxExecResult:
    """Parse the bounded runner protocol.

    An invalid payload is an infrastructure failure, not policy output.
    """
    transferred_bytes = len(stdout) + len(stderr)
    try:
        payload = json.loads(stdout)
        return SandboxExecResult(
            return_code=int(payload["return_code"]),
            output=_decode_output(payload["output_b64"]),
            output_head=_decode_output(payload["output_head_b64"]),
            output_tail=_decode_output(payload["output_tail_b64"]),
            output_total_bytes=int(payload["total_bytes"]),
            output_truncated=bool(payload["truncated"]),
            remote_seconds=float(payload["remote_seconds"]),
            client_seconds=client_seconds,
            transferred_bytes=transferred_bytes,
            timed_out=bool(payload.get("timed_out", False)),
            output_limited=bool(payload.get("output_limited", False)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        diagnostic = (stdout + stderr).decode("utf-8", errors="replace")[-4000:]
        raise SandboxTransportError(
            "Modal command wrapper returned an invalid response "
            f"(return_code={fallback_return_code}, output_tail={diagnostic!r})"
        ) from error


class ModalSWEEnvironment:
    """mini-swe-agent environment backed by one fresh Modal Sandbox."""

    def __init__(
        self,
        task_dir: str | Path,
        *,
        cwd: str = "/testbed",
        lifetime: int = 3900,
        exec_timeout: int = 120,
        app_name: str = "miles-modal-swe-sandboxes",
        cpu: float | None = None,
        memory_mib: int | None = None,
    ) -> None:
        self.task_dir = Path(task_dir).resolve()
        dockerfile = self.task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"Modal SWE task has no Dockerfile: {self.task_dir}"
            )

        image_name = _dockerfile_base_image(dockerfile)
        image = _cached_image(image_name)
        app = _cached_app(app_name)
        cpu_env = os.getenv("MODAL_SWE_CPUS")
        memory_env = os.getenv("MODAL_SWE_MEMORY_MIB")
        cpu_request = cpu if cpu is not None else (float(cpu_env) if cpu_env else None)
        memory_request = (
            memory_mib
            if memory_mib is not None
            else (int(memory_env) if memory_env else None)
        )

        started = time.perf_counter()
        create_kwargs = {
            "image": image,
            "app": app,
            "timeout": lifetime,
            "workdir": cwd,
            "cpu": cpu_request,
            "memory": memory_request,
            "tags": {"task_id": self.task_dir.name, "runner": "miles-modal-swe"},
            "readiness_probe": modal.Probe.with_exec(
                "test",
                "-f",
                _SANDBOX_READY_PATH,
                interval_ms=250,
            ),
            # The task image is self-contained. Policy commands must not fetch
            # issue solutions, repository history, or benchmark artifacts.
            "block_network": True,
        }
        # Sandbox v2 does not return until the container has been scheduled.
        # Besides supporting higher create throughput, that makes the interval
        # below a real scheduling measurement instead of only an RPC duration.
        self.sandbox = modal.Sandbox._experimental_create(
            "sleep",
            "infinity",
            **create_kwargs,
        )
        scheduled = time.perf_counter()
        self._startup_started = started
        self._scheduled_at = scheduled
        self._stopped = False
        self.schedule_time = scheduled - started
        self.readiness_time: float | None = None
        self.boot_time: float | None = None
        self.cwd = cwd
        self.exec_timeout = exec_timeout
        self.exec_time = 0.0
        self.exec_remote_time = 0.0
        self.exec_durations: list[float] = []
        self.exec_remote_durations: list[float] = []
        self.exec_transport_durations: list[float] = []
        self.command_count = 0
        self.command_timeout_count = 0
        self.command_output_bytes = 0
        self.command_transferred_bytes = 0
        self.command_output_truncated_count = 0
        self.command_output_hard_limit_count = 0
        self.command_input_sizes: list[int] = []
        self.upload_time = 0.0
        self.upload_bytes = 0

    def lifecycle_diagnostics(self) -> dict[str, Any]:
        """Return bounded best-effort state for a failed Sandbox attempt."""
        diagnostics: dict[str, Any] = {
            "sandbox_id": getattr(self.sandbox, "object_id", None),
            "sandbox_schedule_time": self.schedule_time,
            "sandbox_readiness_time": self.readiness_time,
            "sandbox_boot_time": self.boot_time,
        }
        try:
            diagnostics["sandbox_return_code"] = self.sandbox.poll()
        except Exception as error:
            diagnostics["sandbox_poll_error"] = f"{type(error).__name__}: {error}"[
                :1000
            ]
        return diagnostics

    def mark_ready(self) -> None:
        """Publish task setup completion and wait for Modal readiness."""
        if self.boot_time is not None:
            return
        try:
            process = self.sandbox.exec("touch", _SANDBOX_READY_PATH)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"Failed to publish Sandbox readiness marker: {return_code}"
                )
            self.sandbox.wait_until_ready()
        except BaseException:
            # A failed probe does not terminate the Sandbox automatically.
            # Reclaim it before propagating the infrastructure failure.
            self.stop()
            raise
        ready = time.perf_counter()
        self.readiness_time = ready - self._scheduled_at
        self.boot_time = ready - self._startup_started

    def execute(
        self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        """Execute the shape expected by mini-swe-agent's Environment protocol."""
        command = action.get("command", "") if isinstance(action, dict) else str(action)
        try:
            result = self.exec_detailed(command, cwd=cwd or self.cwd, timeout=timeout)
        except SandboxCommandTimeoutError as error:
            # A command selected by the policy timing out is an environment
            # observation, not an infrastructure failure. Let the agent react;
            # if it never solves the task, the verifier supplies reward 0.
            return {
                "output": f"Command timed out: {error}",
                "returncode": 124,
                "exception_info": type(error).__name__,
            }

        if result.return_code == 0 and result.output.lstrip().startswith(
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ):
            from minisweagent.exceptions import Submitted

            lines = result.output.lstrip().splitlines(keepends=True)
            raise Submitted(
                {
                    "role": "exit",
                    "content": "".join(lines[1:]),
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": "".join(lines[1:]),
                    },
                }
            )

        return {
            "output": result.output,
            "output_head": result.output_head,
            "output_tail": result.output_tail,
            "output_total_bytes": result.output_total_bytes,
            "output_elided_bytes": max(
                0,
                result.output_total_bytes - _OUTPUT_LIMIT_BYTES,
            ),
            "output_truncated": result.output_truncated,
            "output_limited": result.output_limited,
            "returncode": result.return_code,
            "exception_info": "",
        }

    def exec(
        self, command: str, *, cwd: str | None = None, timeout: int | None = None
    ) -> tuple[int, str]:
        result = self.exec_detailed(command, cwd=cwd, timeout=timeout)
        output = (
            result.output
            if not result.output_truncated
            else result.output_head + result.output_tail
        )
        return result.return_code, output

    def exec_detailed(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> SandboxExecResult:
        command = command.replace("\x00", "")
        started = time.perf_counter()
        result: SandboxExecResult | None = None
        try:
            shell_command = f"cd {shlex.quote(cwd or self.cwd)} && {command}"
            command_timeout = float(
                timeout if timeout is not None else self.exec_timeout
            )
            if command_timeout <= 0:
                raise ValueError(
                    f"Sandbox command timeout must be positive, got {command_timeout}"
                )
            output_hard_limit = int(
                os.getenv(
                    "MODAL_SWE_OUTPUT_HARD_LIMIT_BYTES",
                    str(_OUTPUT_HARD_LIMIT_BYTES),
                )
            )
            if output_hard_limit < _OUTPUT_LIMIT_BYTES:
                raise ValueError(
                    "Sandbox output hard limit must retain the complete "
                    f"{_OUTPUT_LIMIT_BYTES}-byte head/tail observation, "
                    f"got {output_hard_limit}"
                )
            command_bytes = shell_command.encode("utf-8", errors="surrogateescape")
            process = self.sandbox.exec(
                "python",
                "-c",
                _BOUNDED_COMMAND_RUNNER,
                str(_OUTPUT_HEAD_BYTES),
                str(_OUTPUT_TAIL_BYTES),
                str(command_timeout),
                str(output_hard_limit),
                # The in-sandbox runner owns the semantic deadline and returns
                # bounded partial output. Modal's outer deadline is only a
                # safety net in case that runner itself becomes unresponsive.
                timeout=math.ceil(command_timeout + 15),
                text=False,
                # Modal's STDOUT stream type forwards the child stream to this
                # Ray process; it does not merge child stderr into the captured
                # stdout pipe. Capture both streams so task tracebacks remain
                # observations for the agent instead of flooding cluster logs.
                stdout=modal.stream_type.StreamType.PIPE,
                stderr=modal.stream_type.StreamType.PIPE,
            )
            # Stream the policy command rather than putting it in Modal's CMD
            # argv. Long-context agents can emit commands well above Modal's
            # 65,536-byte CMD limit; stdin preserves them exactly and also lets
            # the in-sandbox runner feed bash without Linux ARG_MAX.
            process.stdin.write(command_bytes)
            process.stdin.write_eof()
            process.stdin.drain()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            return_code = process.wait()
            result = _parse_bounded_command_response(
                stdout=stdout,
                stderr=stderr,
                fallback_return_code=return_code,
                client_seconds=time.perf_counter() - started,
            )
            if result.timed_out:
                self.command_timeout_count += 1
                diagnostic = (
                    result.output_tail if result.output_truncated else result.output
                )
                raise SandboxCommandTimeoutError(
                    f"command exceeded {command_timeout:.0f}s"
                    + (f"; output tail:\n{diagnostic[-4000:]}" if diagnostic else ""),
                    result=result,
                )
            return result
        finally:
            elapsed = time.perf_counter() - started
            self.exec_time += elapsed
            self.exec_durations.append(elapsed)
            self.command_count += 1
            self.command_input_sizes.append(
                len(
                    f"cd {shlex.quote(cwd or self.cwd)} && {command}".encode(
                        "utf-8",
                        errors="surrogateescape",
                    )
                )
            )
            if result is not None:
                self.exec_remote_time += result.remote_seconds
                self.exec_remote_durations.append(result.remote_seconds)
                self.exec_transport_durations.append(result.transport_seconds)
                self.command_output_bytes += result.output_total_bytes
                self.command_transferred_bytes += result.transferred_bytes
                self.command_output_truncated_count += int(result.output_truncated)
                self.command_output_hard_limit_count += int(result.output_limited)

    def upload_tree(self, source: str | Path, destination: str) -> None:
        started = time.perf_counter()
        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(f"Directory to upload does not exist: {source}")

        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            for path in source.rglob("*"):
                archive.add(path, arcname=path.relative_to(source), recursive=False)
        payload.seek(0)
        payload_size = payload.getbuffer().nbytes

        process = self.sandbox.exec(
            "bash",
            "-lc",
            f"mkdir -p {shlex.quote(destination)} && tar -xf - -C {shlex.quote(destination)}",
            text=False,
        )
        while chunk := payload.read(1 << 20):
            process.stdin.write(chunk)
            process.stdin.drain()
        process.stdin.write_eof()
        process.stdin.drain()
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            raise SandboxTransportError(
                f"Failed to upload {source} to sandbox: {stderr}"
            )
        self.upload_time += time.perf_counter() - started
        self.upload_bytes += payload_size

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"cwd": self.cwd, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {"cwd": self.cwd, "task_dir": str(self.task_dir)},
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self.sandbox.terminate()
        except Exception:
            logger.warning("Failed to terminate Modal Sandbox", exc_info=True)
        finally:
            # terminate is asynchronous by default. Detaching immediately
            # releases the command-router connection while Modal reclaims the
            # container in the background. The Sandbox lifetime remains the
            # fallback if this process disappears before cleanup runs.
            try:
                self.sandbox.detach()
            except Exception:
                logger.warning("Failed to detach Modal Sandbox client", exc_info=True)


def _dockerfile_base_image(dockerfile: Path) -> str:
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            parts = stripped.split()
            if len(parts) >= 2 and not parts[1].startswith("--"):
                return parts[1]
            if len(parts) >= 3 and parts[1].startswith("--"):
                return parts[2]
    raise ValueError(f"Could not find a base image in {dockerfile}")


def _validate_modal_app_name(app_name: str) -> str:
    """Fail locally with Modal's object-name constraints before an RPC."""
    if (
        not app_name
        or len(app_name) > 64
        or _MODAL_OBJECT_NAME_RE.fullmatch(app_name) is None
        or _MODAL_APP_ID_RE.fullmatch(app_name) is not None
    ):
        raise ValueError(
            "MODAL_SWE_SANDBOX_APP must be 1-64 characters, contain only "
            "letters, digits, dashes, periods, or underscores, and not look "
            f"like a Modal App ID; got {app_name!r} ({len(app_name)} chars)"
        )
    return app_name


def sandbox_settings() -> dict[str, int | str]:
    return {
        "app_name": _validate_modal_app_name(
            os.getenv("MODAL_SWE_SANDBOX_APP", "miles-modal-swe-sandboxes")
        ),
        "episode_timeout": int(os.getenv("MODAL_SWE_EPISODE_TIMEOUT", "3600")),
        "exec_timeout": int(os.getenv("MODAL_SWE_EXEC_TIMEOUT", "120")),
        "verify_timeout": int(os.getenv("MODAL_SWE_VERIFY_TIMEOUT", "1200")),
    }
