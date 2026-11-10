# Tests

- **Unit** (`tests/stitch`, CPU): the control service, the engine container and the SGLang calls against fakes.
  `uv run pytest tests/stitch`
- **Engine** (`tests/engine`, GPUs on Modal): a model's weight deltas synced into SGLang as production syncs them,
  for the sync cost and for byte-exact results. `run_app.py` runs a whole run on Modal with engines failing and
  rejoining along the way; `--autoscale` runs Modal's own load-based scaling on one Volume instead.
  `generate.py --move-global-scale` writes a set whose NVFP4 global scales change every version.
  See `tests/engine/README.md`.
- **End to end** (`tests/e2e`, GPUs on Modal): Miles trains against a Stitch run while an engine joins, and every
  engine ends on the trainer's last save. `modal run tests/e2e/endpoint.py --config <config>`
  `harness.py` runs Miles with an engine of its own, no Stitch, to the same byte-exact end; `--resume-at S`
  resumes it from its save at S into a fresh engine. `modal run tests/e2e/harness.py --config <config>`

Select the Modal environment first (`MODAL_ENVIRONMENT` or your profile).
