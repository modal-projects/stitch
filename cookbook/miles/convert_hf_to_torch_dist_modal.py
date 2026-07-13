"""Modal Volume-safe wrapper for Miles' HF -> torch_dist converter."""

from __future__ import annotations

import os

_UPSTREAM = "/root/miles/tools/convert_hf_to_torch_dist.py"
_RELEASE_RENAME = "shutil.move(source_dir, target_dir)"
_TRACKER_RELEASE = 'f.write("release")'


def _source() -> str:
    with open(_UPSTREAM) as f:
        return f.read()


if os.environ.get("SKIP_RELEASE_RENAME"):
    src = _source()
    rename_count = src.count(_RELEASE_RENAME)
    tracker_count = src.count(_TRACKER_RELEASE)
    if rename_count != 1 or tracker_count != 1:
        raise RuntimeError(
            "unexpected Miles converter release-finalization shape: "
            f"rename_count={rename_count}, tracker_count={tracker_count}"
        )
    src = src.replace(_RELEASE_RENAME, "pass  # SKIP_RELEASE_RENAME")
    src = src.replace(
        _TRACKER_RELEASE,
        'f.write("1")  # SKIP_RELEASE_RENAME: keep iter_0000001',
    )
    exec(compile(src, _UPSTREAM, "exec"))
else:
    exec(compile(_source(), _UPSTREAM, "exec"))
