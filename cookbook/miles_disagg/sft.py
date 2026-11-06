"""SFT pieces shared by the SFT recipe, its data job and tests.

CHAT_TEMPLATE is Qwen3.8-27B's chat template with one change: the default reasoning effort is medium, the
effort RL passes (harbor_config.arguments). Miles' SFT path renders messages without template arguments, and
the stock default (xhigh) adds a system block that Miles' qwen3 loss mask then cuts from every message.
"""

from pathlib import Path

CHAT_TEMPLATE = Path(__file__).parent / "templates" / "qwen3_8_reasoning_medium.jinja"
# common/trainer_image.py mounts cookbook/ at /root/cookbook.
CHAT_TEMPLATE_IN_IMAGE = f"/root/cookbook/miles_disagg/templates/{CHAT_TEMPLATE.name}"
_PREFIX = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}


def loss_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids and loss mask of a conversation as Miles builds them for SFT with --loss-mask-type qwen3
    (MultiTurnLossMaskGenerator in miles/utils/mask_utils.py at harbor_config.MILES_REPO_REF): each message
    is rendered next to a fixed user message and cut out of that render, and an assistant message trains on
    its tokens after the generation prompt."""
    system_length, gen_length = _template_lengths(tokenizer)
    prefix = _render(tokenizer, [_PREFIX])
    ids, mask = [], []
    for i, message in enumerate(messages):
        if i == 0:
            part = _render(tokenizer, [message, _PREFIX])[: -len(prefix)]
        else:
            part = _render(tokenizer, [_PREFIX, message])[len(prefix) :]
            if message["role"] != "system":
                part = part[system_length:]
        trained = len(part) - gen_length if message["role"] == "assistant" else 0
        ids += part
        mask += [0] * (len(part) - trained) + [1] * trained
    return ids, mask


def _template_lengths(tokenizer) -> tuple[int, int]:
    """Miles' get_system_message_length: the tokens the template puts before a message (a system block),
    and the length of the generation prompt."""
    text = "FOR TESTING ONLY"
    pair = [{"role": "user", "content": text}] * 2
    raw = tokenizer(text, add_special_tokens=False)["input_ids"]
    rendered = tokenizer(tokenizer.apply_chat_template(pair, tokenize=False), add_special_tokens=False)["input_ids"]
    first, second = [i for i in range(len(rendered) - len(raw) + 1) if rendered[i : i + len(raw)] == raw]
    end = len(rendered) - len(raw) - second
    gen_length = len(_render(tokenizer, pair, add_generation_prompt=True)) - len(rendered)
    return first - ((second - first) - end - len(raw)), gen_length


def _render(tokenizer, messages: list[dict], **kwargs) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
