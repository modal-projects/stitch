# Checkpoint persistence

Miles saves model, optimizer, scheduler, RNG, and rollout sampler state to local
ephemeral disk. Snapshot creation and its collectives pause training. Once the
files are closed, one uploader per host copies them to
`<EXPERIMENT_VOLUME_NAME>-checkpoints`, mounted at `/training-checkpoints`.

The `/stitch` Volume holds live weight updates and logs. Its commits do not flush
checkpoint uploads. Both volumes still share the host's network and memory
bandwidth; the upload rate limit controls background copying.

Before a weight update gathers or encodes tensors, all host uploaders stop
admitting checkpoint I/O. Each copy window writes and fsyncs at most 64 MiB.
An active filesystem call or Volume commit cannot be preempted: the trainer
waits up to `checkpoint_delta_quiesce_seconds`, logs any busy hosts, and proceeds
with new checkpoint operations still paused.

After publication, background monitors retain that priority until every discovered
replica is ready on the published version or a newer version of the same run,
with at least `rollout_min_ready` ready (default: 75% of the container floor).
This serving wait does not block training. A failed update releases its lease; serving waits expire after
`checkpoint_delta_serving_seconds`, even if discovery is stuck. A newer update
replaces the previous serving wait without admitting checkpoint I/O between them.

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
| `checkpoint_delta_quiesce_seconds` | `5` | Maximum wait for an active checkpoint I/O call before delta encoding. |
| `checkpoint_delta_serving_seconds` | `300` | Maximum background hold after delta publication. |

Size `trainer_ephemeral_disk_mib` for a local snapshot, the Volume write cache,
and runtime scratch. For staged raw HF exports, host leaders own shards assigned
by cumulative tensor bytes; rank zero writes the global index, model metadata,
and static calibration/rotary buffers. Every rank still participates in gathering
and conversion, so local snapshot creation remains synchronous. Explicit exports
outside the staging root keep a single writer. The reserve check does not estimate the next snapshot's size. CPU optimizer state and export
buffers also require host memory. If persistence is slower than the save cadence,
skipped saves increase the recovery interval.

`CHECKPOINT` events report quiescence, serving completion or timeout, and normal
copy progress. `DELTA_PHASE` records rank-zero encode, file-write, and
publish/notify durations; write time includes collective waits. `HF_EXPORT`
reports total tensor bytes and generated shard bytes assigned to each writer. The upload deadline
includes time paused for deltas; sustained update traffic can delay durability.

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
