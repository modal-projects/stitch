"""Expose fused Hugging Face experts individually without changing tensor bytes."""

from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path

_FUSED_EXPERT = re.compile(r"(.+\.experts)\.(gate_up_proj|down_proj)$")


def unpack_fused_experts(checkpoint: str) -> None:
    """Rewrite a staged checkpoint to the individual-expert layout Miles exports."""
    root = Path(checkpoint)
    weight_map = {}
    converted = 0
    for shard in sorted(root.glob("*.safetensors")):
        with shard.open("rb") as source:
            (header_size,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(header_size))
            output = {}
            for name, info in header.items():
                match = _FUSED_EXPERT.fullmatch(name)
                if not match:
                    output[name] = info
                    continue
                prefix, projection = match.groups()
                experts, rows, columns = info["shape"]
                pieces = 2 if projection == "gate_up_proj" else 1
                start, end = info["data_offsets"]
                if rows % pieces or (end - start) % (experts * pieces):
                    raise ValueError(f"Invalid fused expert layout: {name}: {info}")
                size = (end - start) // (experts * pieces)
                projections = (
                    ("gate_proj", "up_proj") if pieces == 2 else ("down_proj",)
                )
                for expert in range(experts):
                    for part, proj in enumerate(projections):
                        offset = start + (expert * pieces + part) * size
                        target = f"{prefix}.{expert}.{proj}.weight"
                        if target in header or target in output:
                            raise ValueError(f"Duplicate expert tensor: {target}")
                        output[target] = {
                            "dtype": info["dtype"],
                            "shape": [rows // pieces, columns],
                            "data_offsets": [offset, offset + size],
                        }
                converted += 1
            encoded = json.dumps(output, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            temporary = shard.with_suffix(".safetensors.partial")
            with temporary.open("wb") as destination:
                destination.write(struct.pack("<Q", len(encoded)))
                destination.write(encoded)
                shutil.copyfileobj(source, destination, length=16 << 20)
        temporary.replace(shard)
        weight_map.update(
            {name: shard.name for name in output if name != "__metadata__"}
        )
        print(f"Unpacked expert headers: {shard.name}", flush=True)
    if not weight_map:
        raise ValueError(f"No checkpoint tensors in {checkpoint}")
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Unpacked {converted} fused expert tensors", flush=True)
