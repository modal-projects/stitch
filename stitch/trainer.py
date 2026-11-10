"""What a trainer on Modal runs with Stitch."""

import os

import modal


def commit_versions(args, version_dir: str, rollout_engines) -> None:
    """Miles' post-write hook, on every rank: a commit covers its container's writes, so each node's first rank commits."""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        modal.Volume.from_name(os.environ["STITCH_VERSIONS_VOLUME"]).commit()
