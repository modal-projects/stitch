import json
import struct

import pytest
from safetensors import deserialize

from cookbook.miles_disagg.unpack_experts import unpack_fused_experts


def _serialize(tensors):
    header = {"__metadata__": {"format": "pt"}}
    data = b""
    for name, tensor in tensors.items():
        end = len(data) + len(tensor["data"])
        header[name] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [len(data), end],
        }
        data += tensor["data"]
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + data


def test_unpack_preserves_expert_values_and_other_tensors(tmp_path):
    prefix = "model.language_model.layers.0.mlp.experts"
    gate_up = bytes(range(32))
    down = bytes(range(100, 116))
    tensors = {
        f"{prefix}.gate_up_proj": {
            "dtype": "BF16",
            "shape": [2, 4, 2],
            "data": gate_up,
        },
        f"{prefix}.down_proj": {"dtype": "BF16", "shape": [2, 2, 2], "data": down},
        "norm.weight": {"dtype": "F32", "shape": [1], "data": struct.pack("<f", 1.5)},
    }
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(_serialize(tensors))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 52}, "weight_map": {}}))

    unpack_fused_experts(str(tmp_path))

    result = dict(deserialize(shard.read_bytes()))
    expected = {"norm.weight": tensors["norm.weight"]}
    for expert in range(2):
        for part, projection in enumerate(("gate_proj", "up_proj")):
            offset = expert * 16 + part * 8
            expected[f"{prefix}.{expert}.{projection}.weight"] = {
                "dtype": "BF16",
                "shape": [2, 2],
                "data": gate_up[offset : offset + 8],
            }
        expected[f"{prefix}.{expert}.down_proj.weight"] = {
            "dtype": "BF16",
            "shape": [2, 2],
            "data": down[expert * 8 : (expert + 1) * 8],
        }
    assert result == expected
    assert json.loads(index.read_text()) == {
        "metadata": {"total_size": 52},
        "weight_map": dict.fromkeys(expected, shard.name),
    }
    before = shard.read_bytes()
    unpack_fused_experts(str(tmp_path))
    assert shard.read_bytes() == before


def test_unpack_rejects_conflicting_individual_expert(tmp_path):
    prefix = "model.layers.0.mlp.experts"
    (tmp_path / "model.safetensors").write_bytes(
        _serialize(
            {
                f"{prefix}.gate_up_proj": {
                    "dtype": "BF16",
                    "shape": [1, 2, 1],
                    "data": b"\0" * 4,
                },
                f"{prefix}.0.gate_proj.weight": {
                    "dtype": "BF16",
                    "shape": [1, 1],
                    "data": b"\0" * 2,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Duplicate expert tensor"):
        unpack_fused_experts(str(tmp_path))
