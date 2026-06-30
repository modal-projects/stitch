from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from cookbook import sidecar


def _install_fake_decoder(name: str, *, with_internals: bool) -> types.ModuleType:
    """Register a fake ``disk_delta`` module in sys.modules.

    ``with_internals=True`` exposes the apply-lock + applied-version bookkeeping
    the concurrent copier reuses; False omits them so the copier falls back to the
    decoder's own ``init_local_checkpoint``.
    """
    mod = types.ModuleType(name)

    def init_local_checkpoint(local: str, base: str) -> None:
        os.makedirs(local, exist_ok=True)
        Path(local, ".fallback_used").write_text("1")

    mod.init_local_checkpoint = init_local_checkpoint  # type: ignore[attr-defined]

    if with_internals:
        @contextlib.contextmanager
        def _apply_lock(local: str):
            yield

        def _read_applied_version(local: str) -> str | None:
            p = Path(local, ".applied_version")
            return p.read_text().strip() if p.exists() else None

        def _write_applied_version(local: str, version: str) -> None:
            Path(local).mkdir(parents=True, exist_ok=True)
            Path(local, ".applied_version").write_text(version)

        mod._apply_lock = _apply_lock  # type: ignore[attr-defined]
        mod._read_applied_version = _read_applied_version  # type: ignore[attr-defined]
        mod._write_applied_version = _write_applied_version  # type: ignore[attr-defined]
        mod.drop_page_cache = lambda path: None  # type: ignore[attr-defined]

    sys.modules[name] = mod
    return mod


class ParallelInitLocalCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = self.root / "base"
        self.local = self.root / "local"
        self.base.mkdir()
        (self.base / "shard-0.safetensors").write_bytes(b"AAAA")
        (self.base / "shard-1.safetensors").write_bytes(b"BBBB")
        self._names: list[str] = []

    def tearDown(self) -> None:
        for n in self._names:
            sys.modules.pop(n, None)
        self._tmp.cleanup()

    def _decoder(self, name: str, *, with_internals: bool = True) -> str:
        self._names.append(name)
        _install_fake_decoder(name, with_internals=with_internals)
        return name

    def test_copies_all_base_shards_and_records_fingerprint(self) -> None:
        init = sidecar.parallel_init_local_checkpoint(self._decoder("fake_dd_copy"))
        init(str(self.local), str(self.base))
        self.assertEqual((self.local / "shard-0.safetensors").read_bytes(), b"AAAA")
        self.assertEqual((self.local / "shard-1.safetensors").read_bytes(), b"BBBB")
        self.assertTrue((self.local / ".base_fingerprint").exists())
        self.assertEqual((self.local / ".applied_version").read_text(), "000000")

    def test_skips_recopy_when_base_unchanged(self) -> None:
        init = sidecar.parallel_init_local_checkpoint(self._decoder("fake_dd_skip"))
        init(str(self.local), str(self.base))
        # A delta would have patched the local copy in place; an unnecessary
        # re-materialize would clobber it, so prove the second call is a no-op.
        (self.local / "shard-0.safetensors").write_bytes(b"PATCHED")
        init(str(self.local), str(self.base))
        self.assertEqual((self.local / "shard-0.safetensors").read_bytes(), b"PATCHED")

    def test_wipes_and_rematerializes_when_base_changed(self) -> None:
        init = sidecar.parallel_init_local_checkpoint(self._decoder("fake_dd_stale"))
        init(str(self.local), str(self.base))
        (self.local / "stale-marker").write_text("x")
        # Re-prep rewrites a base shard (new size/mtime) — fingerprint changes,
        # so the stale local copy must be wiped and rebuilt against the new base.
        (self.base / "shard-0.safetensors").write_bytes(b"CCCCCCCC")
        init(str(self.local), str(self.base))
        self.assertEqual((self.local / "shard-0.safetensors").read_bytes(), b"CCCCCCCC")
        self.assertFalse((self.local / "stale-marker").exists())

    def test_falls_back_to_decoder_copier_when_internals_missing(self) -> None:
        init = sidecar.parallel_init_local_checkpoint(
            self._decoder("fake_dd_fallback", with_internals=False)
        )
        init(str(self.local), str(self.base))
        self.assertTrue((self.local / ".fallback_used").exists())


if __name__ == "__main__":
    unittest.main()
