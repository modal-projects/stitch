"""Probe: inspect DAPO-Math-17k columns so the Moonlight config's prepare_data
writes a jsonl with the fields slime's --input-key prompt / --label-key label +
rm_type=math expect. The slime recipes consume a *preprocessed* jsonl; the raw HF
dataset may put the answer under e.g. reward_model.ground_truth, not `label`.

    m run -m modal_probes.inspect_dapo_dataset::inspect
"""

from __future__ import annotations

import modal

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"

image = modal.Image.from_registry(SLIME_IMAGE_TAG).entrypoint([]).pip_install("datasets")
app = modal.App("inspect-dapo-dataset")


@app.function(image=image, timeout=15 * 60, secrets=[modal.Secret.from_name("huggingface-secret")])
def inspect() -> None:
    import json

    from datasets import load_dataset

    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    print(f"num_rows={ds.num_rows}")
    print(f"column_names={ds.column_names}")
    print(f"features={ds.features}")
    ex = ds[0]
    print("\n=== example[0] (values truncated to 600 chars) ===")
    for k, v in ex.items():
        s = json.dumps(v, default=str)
        print(f"  {k}: {s[:600]}{' …' if len(s) > 600 else ''}")
