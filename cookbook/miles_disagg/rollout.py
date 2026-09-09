"""Token-based math rollouts through Stitch's external inference endpoint."""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.rollout_request import (
    RolloutRequestContext,
    prepare_rollout_request,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample

from cookbook.common.hooks import sample_affinity_key


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Generate through the external endpoint with session affinity and a version gate."""
    args, sample = input.args, input.sample
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}
    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
    sampling_params = dict(input.sampling_params)
    if sample.response:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        input_ids = sample.tokens
    else:
        input_ids = prompt_ids
    payload, halt_status = compute_request_payload(
        args, input_ids, sampling_params, sample.multimodal_inputs
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    request = await prepare_rollout_request(
        args,
        RolloutRequestContext(session_id=sample_affinity_key(sample)),
        url=f"{args.rollout_endpoint_url.rstrip('/')}/generate",
        payload=payload,
        headers=compute_routing_headers(args, sample),
    )
    if request["retry_sleep"] != 1.0:
        raise ValueError("Miles token rollouts require rollout_request_retry_sleep=1")
    output = await post(
        request["url"],
        request["payload"],
        headers=request["headers"],
        max_retries=request["max_retries"],
    )
    await update_sample_from_response(args, sample, payload, output)
    return GenerateFnOutput(samples=sample)
