"""Probe: dump Moonlight's base HF tensor dtypes/shapes to debug the M1
disk-delta export shape mismatch.

The M1 trainer crashed in slime's disk-delta export:

    update_weight_from_disk_delta._encode_delta -> diff_and_compress:
        diff = new ^ old
        ValueError: operands could not be broadcast together with shapes (256,) (128,)

The disk-delta XORs the Megatron->HF *exported* tensor (``new``) against the base
HF checkpoint tensor (``old``) AS FLAT uint8 BYTES (it does
``tensor.view(torch.uint8).reshape(-1)`` first). So this is 256 bytes vs 128
bytes -- the export produced exactly DOUBLE the bytes of the base for one key.
Two candidate causes:

  (a) dtype mismatch -- the base checkpoint stores that tensor in a 1-byte dtype
      (fp8) while the export is 2-byte bf16 (same element count, 2x bytes); or
  (b) element-count doubling in slime's convert_deepseekv3_to_hf for Moonlight's
      MLA variant (no q-LoRA, --qk-layernorm) -- that converter is written for
      DeepSeek-V3.2 (it has the DSA "indexer" wq_b/wk/k_norm branches) and one of
      its hard-coded 128/64 reshapes does not match Moonlight's layout.

This reads the safetensors headers only (dtype + shape + byte span per tensor),
so it is CPU-only and instant -- no weights are materialized. It surfaces the
base side: the dtype histogram tells (a) apart from (b), and the per-tensor byte
sizes pinpoint which base tensor is 128 bytes (the ``old`` operand).

    alias m="uv run --extra modal modal"
    m run -m modal_probes.inspect_moonlight_weights::inspect
"""

from __future__ import annotations

import json
import os
import struct
from collections import Counter
from pathlib import Path

import modal

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_NAME = os.environ.get("VERIFY_MODEL", "moonshotai/Moonlight-16B-A3B-Instruct")

# safetensors dtype -> bytes/element, for spotting fp8 (1B) vs bf16 (2B) bases.
_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
    "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}

image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    # The base image bakes an HF cache here; remove it so the volume can mount.
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("inspect-moonlight-weights")
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


def _read_safetensors_header(path: Path) -> dict:
    """Parse just the JSON header of a .safetensors file (no tensor data)."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


@app.function(
    image=image,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    timeout=20 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def inspect() -> None:
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(MODEL_NAME))  # cached from the M1 download
    config = json.loads((model_dir / "config.json").read_text())
    mla_keys = [
        "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
        "v_head_dim", "qk_head_dim", "num_attention_heads", "num_key_value_heads",
        "hidden_size", "head_dim", "torch_dtype", "quantization_config",
    ]
    print(f"=== {MODEL_NAME} config (MLA-relevant) ===")
    for k in mla_keys:
        if k in config:
            print(f"  {k} = {config[k]}")

    tensors: dict[str, dict] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        tensors.update(_read_safetensors_header(shard))
    print(f"\n=== {len(tensors)} tensors across {len(list(model_dir.glob('*.safetensors')))} shard(s) ===")

    dtypes = Counter(meta["dtype"] for meta in tensors.values())
    print(f"dtype histogram: {dict(dtypes)}")
    if len(dtypes) > 1:
        print("  -> MIXED dtypes: a base tensor in a 1-byte dtype vs a bf16 export")
        print("     would XOR-mismatch by exactly 2x. This is hypothesis (a).")

    def nbytes(meta: dict) -> int:
        off = meta.get("data_offsets")
        if off:
            return off[1] - off[0]
        n = 1
        for d in meta["shape"]:
            n *= d
        return n * _DTYPE_BYTES.get(meta["dtype"], 0)

    # The `old` operand was 128 bytes. List every base tensor at that byte size.
    print("\n=== base tensors that are exactly 128 bytes (the `old` operand) ===")
    for name, meta in sorted(tensors.items()):
        if nbytes(meta) == 128:
            print(f"  {name}: shape={meta['shape']} dtype={meta['dtype']} nbytes=128")

    # Full attention/norm layout for layer 0 (the converter branches act here).
    print("\n=== layer-0 attention + norm tensors ===")
    for name, meta in sorted(tensors.items()):
        if (".layers.0." in name or "layers.0." in name) and (
            "self_attn" in name or "norm" in name.lower()
        ):
            print(f"  {name}: shape={meta['shape']} dtype={meta['dtype']} nbytes={nbytes(meta)}")
