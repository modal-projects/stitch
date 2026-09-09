# Checkpoint persistence

Miles saves model, optimizer, scheduler, RNG, and rollout sampler state to local
ephemeral disk. Snapshot creation and its collectives pause training. Once the
files are closed, one uploader per host copies them to
`<EXPERIMENT_VOLUME_NAME>-checkpoints`, mounted at `/training-checkpoints`.

The `/stitch` Volume holds live weight updates and logs. Its commits do not flush
checkpoint uploads. Both volumes still share the host's network and memory
bandwidth; the upload rate limit controls background copying.

## Capacity

Only one snapshot per host can be awaiting persistence. If any host is busy, all
ranks skip the next periodic save. Explicit and final saves wait for the previous
upload, create their snapshot, and wait for durability. Normal loop exit drains
outstanding uploads. Failures reach the next checkpoint operation or final
drain; local payloads remain available after upload failure.

| Modal setting | Default | Purpose |
| --- | --- | --- |
| `checkpoint_upload_mib_per_second` | `256` | Copy rate per host; `0` removes the limit. |
| `checkpoint_min_free_disk_mib` | `65536` | Free-space reserve checked before a snapshot. |
| `checkpoint_upload_timeout_seconds` | `21600` | Deadline checked while copying and waiting for host receipts. |

Size `trainer_ephemeral_disk_mib` for a local snapshot, the Volume write cache,
and runtime scratch. Rank zero also holds the complete HF export. The reserve
check does not estimate the next snapshot's size. CPU optimizer state and export
buffers also require host memory. If persistence is slower than the save cadence,
skipped saves increase the recovery interval.

## Durability and recovery

Each attempt writes immutable snapshots and completion manifests:

```text
<run>/attempts/<attempt>/snapshots/<iteration>/
    checkpoints/iter_<iteration>/...
    checkpoints/rollout/global_dataset_state_dict_<iteration>.pt
    checkpoints/latest_checkpointed_iteration.txt
    hf_checkpoints/weight_v<iteration>/...
    receipts/<iteration>/<host-rank>.json
<run>/completed/<iteration>-<attempt>.json
```

Each host commits its files and a receipt containing sizes and SHA-256 checksums.
The coordinator checks every host's receipt and coverage of the Megatron and HF
indices before publishing the completion manifest. Hosts remove local payloads
only after that manifest is durable.

Automatic Volume commits can expose partial files. Discovery therefore trusts
the completion manifest, not directory presence, the HF `.complete` marker, or
the Megatron tracker. With Volume-backed updates, trainer recovery also requires
a matching published weight version; replica boot chooses an export at or below
the served pointer. Legacy checkpoints on the run Volume remain a fallback.

This supports actor-only Megatron `torch_dist` checkpoints with synchronous local
serialization (`async_save=False`). A snapshot on ephemeral disk alone is not
durable; host loss can discard it and require recovery from an older completed
checkpoint.
