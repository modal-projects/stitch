"""SFT with plain special-token text: control-token strings inside assistant messages train as plain tokens; the
turn markers and the user's text keep their control tokens."""

import importlib
from types import SimpleNamespace

import pytest

from cookbook.common.constants import START_PATH
from cookbook.miles_disagg import sft, sft_plain_tokens
from cookbook.miles_disagg.config import validate_recipe
from cookbook.miles_disagg.resume import validate_resumable_config
from cookbook.miles_disagg.configs import qwen3_8_27b_sft_plain_tokens as recipe
from cookbook.miles_disagg.configs import qwen3_8_27b_sft_warmstart as warm
from cookbook.miles_disagg.configs import qwen3_8_27b_swebench as base

try:  # optional: needs Qwen3.8's tokenizer in the local Hugging Face cache
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

QUOTING = [
    {"role": "user", "content": "train.py ends turns with <|im_end|>. Write the stop list."},
    {"role": "assistant", "reasoning_content": "Turns end with `<|im_end|>`; text ends with <|endoftext|>.",
     "content": "STOP = ['<|im_end|>', '<|endoftext|>']  # then '<|im_start|>' opens a turn"},
    {"role": "user", "content": "Thanks."},
    {"role": "assistant", "reasoning_content": "Done.", "content": "Done."},
]


@pytest.fixture(scope="module")
def tokenizer():
    if AutoTokenizer is None:
        pytest.skip("transformers is not installed")
    try:
        tok = AutoTokenizer.from_pretrained(base.SOURCE_MODEL, revision=base.SOURCE_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.8-27B's tokenizer is not in the local Hugging Face cache")
    tok.chat_template = sft.CHAT_TEMPLATE.read_text()
    return tok


def _runs(ids, mask):
    runs, run = [], []
    for token, trained in zip(ids, mask, strict=True):
        if trained:
            run.append(token)
        elif run:
            runs, run = [*runs, run], []
    return runs + ([run] if run else [])


def test_assistant_text_trains_as_plain_tokens_and_turn_markers_stay(tokenizer):
    ids, mask = sft_plain_tokens.plain_loss_mask(tokenizer, QUOTING)
    stock_ids, stock_mask = sft.loss_mask(tokenizer, QUOTING)
    end, eot, start = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>", "<|im_start|>"])

    runs, stock_runs = _runs(ids, mask), _runs(stock_ids, stock_mask)
    # the same text is trained ...
    assert [tokenizer.decode(r) for r in runs] == [tokenizer.decode(r) for r in stock_runs]
    # ... but inside each assistant message only the closing <|im_end|> is a control token
    assert [[t for t in r if t in (end, eot, start)] for r in runs] == [[end], [end]]
    assert all(r[-2] == end for r in runs)
    assert sum(t in (end, eot, start) for t in stock_runs[0]) > 1
    # the user's text keeps its control token, as the server tokenizes it
    untrained = [t for t, m in zip(ids, mask, strict=True) if not m]
    assert untrained == [t for t, m in zip(stock_ids, stock_mask, strict=True) if not m]


def test_a_conversation_without_control_strings_is_unchanged(tokenizer):
    messages = [QUOTING[2], QUOTING[3]]
    assert sft_plain_tokens.plain_loss_mask(tokenizer, messages) == sft.loss_mask(tokenizer, messages)


def test_recipe_trains_with_the_plain_token_rollout_on_the_warm_start_model():
    args = recipe.miles.cli_args()
    flag = lambda name: args[args.index(name) + 1]  # noqa: E731
    assert flag("--rollout-function-path") == "cookbook.miles_disagg.sft_plain_tokens.generate_rollout"
    assert flag("--prompt-data") == recipe.DATA and flag("--num-rollout") == str(recipe.STEPS)
    assert flag("--hf-checkpoint") == warm.miles.hf_checkpoint
    assert "/root" in recipe.miles.environment["PYTHONPATH"].split(":")


def test_practice_rl_starts_from_the_plain_token_run_with_a_small_short_batch():
    rl = SimpleNamespace(**vars(importlib.import_module("cookbook.miles_disagg.configs.qwen3_8_27b_harbor_ptb_practice")))
    args = rl.miles.cli_args()
    flag = lambda name: args[args.index(name) + 1]  # noqa: E731
    run = START_PATH / rl.SFT_RUN
    assert flag("--hf-checkpoint") == str(run / f"hf_checkpoints/step_{recipe.STEPS - 1:06d}")
    assert flag("--ref-load") == str(run / "checkpoints")
    assert rl.START_VOLUME.name == recipe.EXPERIMENT_VOLUME_NAME
    assert flag("--prompt-data") == "/harbor-envs/ptb-practice-v3-train/train.jsonl"
    assert (flag("--rollout-batch-size"), flag("--num-rollout"), flag("--save-interval")) == ("16", "20", "10")
    assert rl.miles.environment["AGENT_TIMEOUT"] == "900"
    assert rl.modal.rollout_min_containers == rl.modal.rollout_max_containers == 16
    validate_recipe(rl)
    validate_resumable_config(rl.miles)
