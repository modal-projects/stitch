"""SFT rollout that spells special-token strings inside assistant messages as plain tokens.

The Qwen3.8 tokenizer turns the literal text "<|im_end|>" into token 248046, the id that ends a turn. A model trained
on such text emits the end token when it means to write the string, and the server then stops its reply. This
rollout builds ids and loss mask as Miles' SFT rollout does (sft.loss_mask), then replaces each chat-control token
inside an assistant message (not the template's own turn markers) with the plain tokens of its text. User and tool
messages keep the control tokens, as the server tokenizes them.
"""

from cookbook.miles_disagg import sft

try:
    from miles.utils.mask_utils import get_response_lengths
    from miles.utils.processing_utils import load_tokenizer
except ImportError:  # outside the trainer image (tests)
    get_response_lengths = load_tokenizer = None

CONTROL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
TOKENIZER = None
SAMPLE_PRINTED = False


def plain_loss_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    ids, mask = sft.loss_mask(tokenizer, messages)
    control = dict(zip(tokenizer.convert_tokens_to_ids(list(CONTROL)), CONTROL))
    plain = {i: tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
             for i, text in control.items()}
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out_ids, out_mask, start = [], [], 0
    for end in _assistant_ends(mask) + [len(ids)]:
        segment, seg_mask = ids[start:end], mask[start:end]
        trained = [j for j, m in enumerate(seg_mask) if m]
        first, last = (trained[0], trained[-1]) if trained else (len(segment), -1)
        # the message's own closing marker: the last <|im_end|> of its trained span
        closing = max((j for j in range(first, last + 1) if segment[j] == end_id), default=-1)
        for j, (token, m) in enumerate(zip(segment, seg_mask)):
            if token in plain and first <= j <= last and j != closing:
                out_ids += plain[token]
                out_mask += [m] * len(plain[token])
            else:
                out_ids.append(token)
                out_mask.append(m)
        start = end
    return out_ids, out_mask


def _assistant_ends(mask: list[int]) -> list[int]:
    """Positions just after each trained (assistant) span, so spans are handled one at a time."""
    return [j + 1 for j in range(len(mask)) if mask[j] and (j + 1 == len(mask) or not mask[j + 1])]


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Miles' miles.rollout.sft_rollout.generate_rollout with plain_loss_mask in place of the mask generator."""
    global TOKENIZER, SAMPLE_PRINTED
    assert not evaluation
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path,
                                   trust_remote_code=True)
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for i, sample in enumerate(samples):
        (sample,) = sample
        token_ids, loss_mask = plain_loss_mask(TOKENIZER, sample.prompt)
        response_length = get_response_lengths([loss_mask])[0]
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
        if i == 0 and not SAMPLE_PRINTED:
            print(f"sft_plain_tokens: example {len(token_ids)} tokens, response {response_length}", flush=True)
            SAMPLE_PRINTED = True
    return samples
