"""Identity constants for the glm45_air_fp8 example — GLM-4.5-Air trained in bf16,
served in native HF FP8 through a disaggregated sglang rollout pool.

Both sides of the deployment key off these names, so they live in one place: the
trainer's hooks (which store/pool to publish to and wake) and the serving side (the
sidecar's store + the Flash app the trainer resolves its rollout endpoint from). The
training hyperparameters and Modal-infra config live with the Modal app (modal_app.py).
"""

from __future__ import annotations

APP_NAME = "stitch-glm45-air-fp8"       # the Modal app / Flash pool
SERVER_CLS_NAME = "Server"              # the Flash-served rollout replica class
DELTA_VOLUME_NAME = "stitch-delta-glm45-air-fp8"  # the Store's Modal Volume
DELTA_BULLETIN_ROOT = "/delta-bulletin"  # Store root (holds `latest` + <run_id>/ chains)
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"  # engine's per-host materialized checkpoint
SIDECAR_COMMIT_MODE = "quiesce"         # drain + reload (fp8 reload is exact; no in_place needed)
