"""What a trainer on Modal runs with Stitch."""

import os
import threading
import time
from pathlib import Path

import modal


def commit_versions(args, version_dir: str, rollout_engines) -> None:
    """Miles' post-write hook, on every rank: a commit covers its container's writes, so each node's first rank commits."""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        modal.Volume.from_name(os.environ["STITCH_VERSIONS_VOLUME"]).commit()


def commit_adapters(args, adapter_dir: str, rollout_engines) -> None:
    """The same hook for a Tinker gateway's sampler exports, which many saves can make at once."""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        _ADAPTER_COMMITS.cover(Path(adapter_dir))


class _Commits:
    """One Volume's commits from this container: a commit covers every write made before it started."""

    def __init__(self, variable: str):
        self.variable = variable
        self.covered_ns = 0
        self.lock = threading.Lock()

    def cover(self, written: Path) -> None:
        with self.lock:
            paths = (written, *written.rglob("*"))
            if max(path.stat().st_mtime_ns for path in paths) < self.covered_ns:
                return
            started = time.time_ns()
            modal.Volume.from_name(os.environ[self.variable]).commit()
            self.covered_ns = started


_ADAPTER_COMMITS = _Commits("STITCH_ADAPTERS_VOLUME")
