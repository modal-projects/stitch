"""Prepare the pinned MiMo-V2.6 OSS environments for Miles rollouts."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATASET_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"

_SOURCE_FILES = {
    "code": "code.parquet",
    "cyber": "cyber.parquet",
    "general": "general/train.parquet",
    "webdev": "webdev.parquet",
    "music": "music.parquet",
}
_EXPECTED_ROWS = {
    "code": 2698,
    "cyber": 1000,
    "general": 989,
    "webdev": 2093,
    "music": 1000,
}
_REWARD_KINDS = {
    "code": "hidden_test_patch",
    "cyber": "sanitizer_match",
    "general": "general_agent",
    "webdev": "group_visual",
    "music": "abc_music",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    targets: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            source = value.get("dataset_image")
            target = value.get("dockerhub_image")
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(f"{path}:{line_number}: invalid image mapping")
            if source in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate image {source!r}")
            if target in targets:
                raise ValueError(f"{path}:{line_number}: duplicate target {target!r}")
            mapping[source] = target
            targets.add(target)
    return mapping


def _prompt_text(row: dict[str, Any], *, domain: str, index: int) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError(f"{domain}[{index}]: expected exactly one prompt message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError(f"{domain}[{index}]: expected one user prompt")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{domain}[{index}]: prompt content is empty")
    return content


def _instance(row: dict[str, Any], *, domain: str, index: int) -> dict[str, Any]:
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        raise ValueError(f"{domain}[{index}]: extra_info must be an object")
    raw = extra.get("instance_json")
    if not isinstance(raw, str):
        raise ValueError(f"{domain}[{index}]: instance_json must remain a JSON string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{domain}[{index}]: invalid instance_json") from error
    if not isinstance(value, dict):
        raise ValueError(f"{domain}[{index}]: instance_json must decode to an object")
    return value


def normalize_row(
    domain: str,
    row: dict[str, Any],
    *,
    index: int,
    image_mapping: dict[str, str],
) -> dict[str, Any]:
    """Convert one released row without changing its task or reward semantics."""
    if domain not in _SOURCE_FILES:
        raise ValueError(f"unknown MiMo domain: {domain!r}")
    prompt = _prompt_text(row, domain=domain, index=index)
    extra = row.get("extra_info")
    assert isinstance(extra, dict)

    if domain == "music":
        instance_id = extra.get("src_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"music[{index}]: missing src_id")
        instance = None
        image = None
        cwd = None
    else:
        instance = _instance(row, domain=domain, index=index)
        instance_id = instance.get("instance_id") or instance.get("task_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{domain}[{index}]: missing instance identity")
        image_alias = instance.get("docker_image")
        if not isinstance(image_alias, str) or image_alias not in image_mapping:
            raise ValueError(
                f"{domain}[{index}]: unmapped Docker image {image_alias!r}"
            )
        image = image_mapping[image_alias]
        cwd = instance.get("cwd")
        if not isinstance(cwd, str) or not cwd.startswith("/"):
            raise ValueError(f"{domain}[{index}]: invalid cwd {cwd!r}")

        if domain == "general" and instance.get("dataset_type") == "general_agent":
            env_task_dir = instance.get("env_task_dir")
            if not isinstance(env_task_dir, str) or not env_task_dir.startswith(
                "envs/"
            ):
                raise ValueError(
                    f"general[{index}]: invalid env_task_dir {env_task_dir!r}"
                )

    metadata = {
        "domain": domain,
        "instance_id": instance_id,
        "data_source": row.get("data_source"),
        "ability": row.get("ability"),
        "source_agent_name": row.get("agent_name"),
        "source_index": index,
        "source_revision": DATASET_REVISION,
        "reward_kind": _REWARD_KINDS[domain],
        "image": image,
        "sandbox_cwd": cwd,
        "instance": instance,
        "task": extra if domain == "music" else None,
    }
    return {"prompt": prompt, "metadata": metadata}


def normalize_rows(
    domain: str,
    rows: Iterable[dict[str, Any]],
    *,
    image_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    output = [
        normalize_row(
            domain,
            row,
            index=index,
            image_mapping=image_mapping,
        )
        for index, row in enumerate(rows)
    ]
    identities = [row["metadata"]["instance_id"] for row in output]
    duplicates = [name for name, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise ValueError(f"{domain}: duplicate instance identities: {duplicates[:5]}")
    return output


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _patch_paths(patch: str) -> list[str]:
    prefix = "diff --git a/"
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith(prefix):
            continue
        old_path, separator, new_path = line[len(prefix) :].partition(" b/")
        if separator and old_path and new_path:
            paths.update((old_path, new_path))
    return sorted(paths)


def _code_setup_script(cwd: str) -> str:
    quoted_cwd = shlex.quote(cwd)
    return f"""#!/bin/bash
set -euo pipefail
cd {quoted_cwd}
git config --global --add safe.directory {quoted_cwd}
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -q
  git add -A
  GIT_AUTHOR_NAME=MiMo GIT_AUTHOR_EMAIL=mimo@example.invalid \\
  GIT_COMMITTER_NAME=MiMo GIT_COMMITTER_EMAIL=mimo@example.invalid \\
    git commit -q -m baseline --allow-empty
fi
base=$(git rev-parse HEAD)
git update-ref refs/miles/task-baseline "$base"
"""


def _code_verifier_script(cwd: str, test_command: str) -> str:
    quoted_cwd = shlex.quote(cwd)
    quoted_command = shlex.quote(test_command)
    return f"""#!/bin/bash
set -uo pipefail
cd {quoted_cwd} || exit 1
base=$(git rev-parse refs/miles/task-baseline) || exit 1

reset_test_paths() {{
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    if git cat-file -e "$base:$path" 2>/dev/null; then
      git checkout "$base" -- "$path" 2>/dev/null || true
    else
      git rm -f --cached -- "$path" >/dev/null 2>&1 || true
      rm -f -- "$path"
    fi
  done < /tests/patch_paths.txt
}}

reset_test_paths || exit 1
if ! git apply --verbose /tests/test.patch; then
  reset_test_paths
  exit 1
fi
bash -lc {quoted_command}
status=$?
mkdir -p /logs/verifier
if [ "$status" -eq 0 ]; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
reset_test_paths
exit "$status"
"""


def _materialize_code_task(task_root: Path, row: dict[str, Any]) -> Path:
    metadata = row["metadata"]
    instance = metadata["instance"]
    assert isinstance(instance, dict)
    task_dir = task_root / metadata["instance_id"]
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    environment_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "Dockerfile").write_text(f"FROM {metadata['image']}\n")
    (environment_dir / "setup.sh").write_text(
        _code_setup_script(metadata["sandbox_cwd"])
    )
    patch = instance["test_patch"]
    (tests_dir / "test.patch").write_text(patch)
    (tests_dir / "patch_paths.txt").write_text(
        "".join(f"{path}\n" for path in _patch_paths(patch))
    )
    (tests_dir / "test.sh").write_text(
        _code_verifier_script(metadata["sandbox_cwd"], instance["test_command"])
    )
    (task_dir / "task.toml").write_text(
        f"[verifier]\ntimeout_sec = {int(instance['verifier_timeout_sec'])}\n"
    )
    metadata["task_dir"] = str(task_dir)
    metadata.pop("instance")
    return task_dir


def prepare_mimo_v2_6(data_root: Path) -> Path:
    """Download, validate, and write a unified Miles JSONL index."""
    from datasets import Dataset
    from huggingface_hub import snapshot_download

    data_root.mkdir(parents=True, exist_ok=True)
    source_root = data_root / "source"
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=source_root,
        allow_patterns=["image-mapping.jsonl", *_SOURCE_FILES.values()],
    )

    mapping_path = source_root / "image-mapping.jsonl"
    image_mapping = _load_image_mapping(mapping_path)
    if len(image_mapping) != 3764:
        raise RuntimeError(f"expected 3764 image mappings; got {len(image_mapping)}")

    all_rows: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    used_images: set[str] = set()
    code_task_root = data_root / "tasks" / "code"
    source_hashes: dict[str, str] = {"image-mapping.jsonl": _sha256(mapping_path)}
    for domain, relative_path in _SOURCE_FILES.items():
        parquet_path = source_root / relative_path
        source_hashes[relative_path] = _sha256(parquet_path)
        rows = Dataset.from_parquet(str(parquet_path))
        normalized = normalize_rows(
            domain,
            rows,
            image_mapping=image_mapping,
        )
        expected = _EXPECTED_ROWS[domain]
        if len(normalized) != expected:
            raise RuntimeError(
                f"{domain}: expected {expected} rows; got {len(normalized)}"
            )
        domain_counts[domain] = len(normalized)
        if domain == "code":
            for row in normalized:
                _materialize_code_task(code_task_root, row)
        used_images.update(
            row["metadata"]["image"]
            for row in normalized
            if row["metadata"]["image"] is not None
        )
        _write_jsonl(data_root / f"{domain}.jsonl", normalized)
        all_rows.extend(normalized)

    if len(used_images) != len(image_mapping):
        raise RuntimeError(
            f"dataset references {len(used_images)} images but mapping has "
            f"{len(image_mapping)}"
        )
    output_path = data_root / "train.jsonl"
    _write_jsonl(output_path, all_rows)
    manifest = {
        "schema": 1,
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "rows": len(all_rows),
        "rows_by_domain": domain_counts,
        "images": len(used_images),
        "source_sha256": source_hashes,
        "normalizations": {
            "image_aliases": "resolved through image-mapping.jsonl",
            "routing": "domain replaces the overloaded source agent_name",
            "webdev": "routed as webdev despite the released mimo_swe_agent label",
            "source_scope": "tables and image mapping only; general task bundles are not materialized",
        },
    }
    temporary = data_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(data_root / "manifest.json")
    print(f"Prepared {len(all_rows)} MiMo-V2.6 rows at {output_path}")
    return output_path
