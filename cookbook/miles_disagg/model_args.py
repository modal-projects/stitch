"""Read Megatron model arguments from the pinned Miles checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path


def load_pinned_model_args(
    miles_root: str | Path, megatron_model_type: str
) -> list[str]:
    """Source a Miles model script and return its ``MODEL_ARGS`` Bash array.

    Miles is layered over a dated trainer image. Importing a model-argument helper
    can therefore resolve to the image's preloaded Miles package instead of the
    pinned checkout. Sourcing the checkout's script in a clean Bash process keeps
    model arguments coupled to the revision that supplies the training code.
    """
    if Path(megatron_model_type).name != megatron_model_type:
        raise ValueError("megatron_model_type must be a model-script basename")

    script = Path(miles_root) / "scripts" / "models" / f"{megatron_model_type}.sh"
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -euo pipefail; source "$1" >/dev/null; '
            'declare -p MODEL_ARGS >/dev/null; '
            'for arg in "${MODEL_ARGS[@]}"; do printf "%s\\0" "$arg"; done',
            "load-pinned-model-args",
            str(script),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    if not result.stdout:
        return []
    return [arg.decode() for arg in result.stdout.removesuffix(b"\0").split(b"\0")]
