"""Exact SGLang source pin used by the v0.5.20 validation matrix."""

from cookbook.common.serving_image import SGLangRuntime

VALIDATION_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.20",
    repository="https://github.com/modal-projects/sglang.git",
    branch="rebuild/stitch-sglang-v0.5.20-20260919",
    commit="8a5804024552d948193ca3ac26dc86d37e6c23d9",
)
