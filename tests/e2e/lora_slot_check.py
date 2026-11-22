"""Test hook in every SGLang scheduler: a served LoRA slot against SGLang's own install of a version."""

import json

import torch

# /load_lora_adapter of "__check__<name>" with a version's path runs the check instead of a load.
CHECK_PREFIX = "__check__"
RESULT = "/tmp/lora_slot_check_rank{rank}.json"
# A bf16 NaN no adapter holds: the positions the install leaves unwritten keep it.
SENTINEL = 0x7FC1


def install() -> None:
    from sglang.srt.lora.lora_manager import LoRAManager

    load = LoRAManager.load_lora_adapter

    def load_lora_adapter(self, lora_ref, upsert=False):
        if not lora_ref.lora_name.startswith(CHECK_PREFIX):
            return load(self, lora_ref, upsert)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        result = _check(
            self, lora_ref.lora_name.removeprefix(CHECK_PREFIX), lora_ref.lora_path
        )
        with open(RESULT.format(rank=rank), "w") as out:
            json.dump(result, out)
        return self.create_lora_update_result(success=True)

    LoRAManager.load_lora_adapter = load_lora_adapter


def _check(manager, name: str, path: str) -> dict:
    from sglang.srt.lora.lora_config import LoRAConfig
    from sglang.srt.lora.lora_registry import LoRARef
    from sglang.srt.lora.lora_slot_weights import LoRASlotWeights

    pool = manager.memory_pool
    (uid,) = [uid for uid, ref in manager.lora_refs.items() if ref.lora_name == name]
    served = manager.loras[uid]
    assert isinstance(served, LoRASlotWeights), type(served).__name__
    slot = pool.uid_to_buffer_id[uid]
    config = LoRAConfig(path, base_vocab_size=manager.base_hf_config.vocab_size)
    reference = manager._create_lora_adapter_from_tensors(
        LoRARef(lora_name="__reference__", lora_path=path),
        config,
        manager._read_adapter_tensors(path),
    )
    installed = _install_into_stand_ins(manager, reference)
    regions = {region.key for region in served.placement.regions}
    counts = dict(written=0, mismatch=0, unwritten_nonzero=0, written_outside_regions=0)
    for key, tensor in pool._iter_slot_buffers():
        assert tensor.element_size() == 2, tensor.dtype
        expected = installed[key].reshape(-1).view(torch.int16)
        actual = tensor[slot].reshape(-1).view(torch.int16)
        written = expected != SENTINEL
        counts["written"] += int(written.sum())
        if key in regions:
            counts["mismatch"] += int((expected[written] != actual[written]).sum())
            counts["unwritten_nonzero"] += int((actual[~written] != 0).sum())
        else:
            counts["written_outside_regions"] += int(written.sum())
    return counts


def _install_into_stand_ins(manager, adapter) -> dict:
    """SGLang's install of `adapter` into one stand-in slot per buffer, the served slots untouched."""
    pool = manager.memory_pool
    attrs = pool._LAYER_SLOT_BUFFERS + pool._MODULE_SLOT_BUFFERS
    saved = {attr: getattr(pool, attr) for attr in attrs}
    stand_ins = {attr: {} for attr in attrs}
    installed = {}
    for key, tensor in pool._iter_slot_buffers():
        attr, name, layer = key
        stand_in = torch.empty_like(tensor[:1])
        stand_in.view(torch.int16).fill_(SENTINEL)
        installed[key] = stand_in[0]
        if layer is None:
            stand_ins[attr][name] = stand_in
        else:
            stand_ins[attr].setdefault(name, []).append(stand_in)
    for attr, value in stand_ins.items():
        setattr(pool, attr, value)
    try:
        pool.load_lora_weight_to_buffer(
            "__reference__",
            0,
            adapter,
            manager.lora_modules,
            manager.embed_tokens_module,
            manager.lm_head_module,
        )
        torch.cuda.synchronize()
    finally:
        for attr, value in saved.items():
            setattr(pool, attr, value)
    return installed
