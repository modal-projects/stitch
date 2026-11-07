"""SFT warm start: the chat template, the trainer-only SFT recipe and app, RL from its checkpoint, and the
data job's harness formats."""

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.common.constants import START_PATH
from cookbook.miles_disagg import sft, sft_data
from cookbook.miles_disagg.config import MilesConfig, validate_recipe
from cookbook.miles_disagg.configs import qwen3_8_27b_swebench as base
from cookbook.miles_disagg.resume import validate_resumable_config

try:  # optional: the template test also needs Qwen3.8's tokenizer in the local Hugging Face cache
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

PROMPT = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment. "
    "Format your response as JSON.\n\nTask Description:\nCount the files in /app.\n\n"
    "Current terminal state:\nCurrent Terminal Screen:\nroot@box:/app# "
)
OUTPUT = "New Terminal Output:\nroot@box:/app# ls | wc -l\n3\nroot@box:/app# "


def _recipe(name):
    recipe = SimpleNamespace(**vars(importlib.import_module(f"cookbook.miles_disagg.configs.{name}")))
    recipe.miles = deepcopy(recipe.miles)
    return recipe


def _flags(cfg):
    args = cfg.cli_args()
    return {
        arg: args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True
        for i, arg in enumerate(args)
        if arg.startswith("--")
    }


def _conversation():
    def answer(analysis, commands, complete):
        return sft_data.render_xml(analysis, "Next step.", commands, complete)

    confirm, _ = sft_data.observation(OUTPUT, "", True, False)
    return [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "reasoning_content": "List it.", "content": answer("Nothing run.", [("ls | wc -l\n", 0.1)], False)},
        {"role": "user", "content": OUTPUT},
        {"role": "assistant", "reasoning_content": "3 files.", "content": answer("It has 3 files.", [], True)},
        {"role": "user", "content": confirm},
        {"role": "assistant", "reasoning_content": "Done.", "content": answer("Done.", [], True)},
    ]


def _runs(ids, mask):
    runs, run = [], []
    for token, trained in zip(ids, mask, strict=True):
        if trained:
            run.append(token)
        elif run:
            runs, run = [*runs, run], []
    return runs + ([run] if run else [])


def test_sft_chat_template_keeps_every_assistant_token_as_rl_renders_them():
    if AutoTokenizer is None:
        pytest.skip("transformers is not installed")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base.SOURCE_MODEL, revision=base.SOURCE_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3.8-27B's tokenizer is not in the local Hugging Face cache")
    stock, ours, messages = tokenizer.chat_template, sft.CHAT_TEMPLATE.read_text(), _conversation()
    assert ours == stock.replace("reasoning_effort|default('xhigh')", "reasoning_effort|default('medium')") != stock

    tokenizer.chat_template = ours
    ids, mask = sft.loss_mask(tokenizer, messages)
    tokenizer.chat_template = stock
    stock_ids, stock_mask = sft.loss_mask(tokenizer, messages)

    # Miles' SFT render is the conversation as RL renders it (stock template, reasoning effort medium) ...
    rl = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, reasoning_effort="medium")
    assert ids == rl
    # ... and it trains every token the model generates there: reasoning, reply and end of turn.
    assert [tokenizer.decode(run) for run in _runs(ids, mask)] == [
        f"{m['reasoning_content']}\n</think>\n\n{m['content']}<|im_end|>\n" for m in messages if m["role"] == "assistant"
    ]
    # The stock default (xhigh) renders a system block and cuts tokens from the later messages.
    assert stock_ids != rl and sum(stock_mask) < sum(mask)


def test_sft_recipe_runs_miles_sft_with_the_rl_model_and_no_rollouts():
    recipe, rl = _recipe("qwen3_8_27b_sft_warmstart"), _recipe("qwen3_8_27b_harbor_mix_v3b_b300_s2")
    flags = _flags(recipe.miles)

    assert {k: flags.get(k) for k in (
        "--debug-train-only", "--rollout-function-path", "--loss-type", "--loss-mask-type", "--chat-template-path",
        "--input-key", "--disable-compute-advantages-and-returns", "--lr", "--lr-decay-style", "--num-rollout",
    )} == {
        "--debug-train-only": True,  # Miles starts no SGLang engine
        "--rollout-function-path": "miles.rollout.sft_rollout.generate_rollout",
        "--loss-type": "sft_loss",
        "--loss-mask-type": "qwen3",
        "--chat-template-path": sft.CHAT_TEMPLATE_IN_IMAGE,
        "--input-key": "messages",
        "--disable-compute-advantages-and-returns": True,
        "--lr": "1e-05",
        "--lr-decay-style": "cosine",
        "--num-rollout": str(recipe.DATA_ROWS // 128),
    }
    rl_only = ("--rollout-endpoint", "--sglang", "--use-session-server", "--custom-", "--update-weight", "--tito",
               "--advantage-estimator", "--score-centering", "--use-fault-tolerance")
    assert not [flag for flag in flags if flag.startswith(rl_only)]
    assert all(getattr(recipe.miles, name) == getattr(rl.miles, name) for name in recipe.SHARED)
    assert [flags[f"--{k}"] for k in ("tensor-model-parallel-size", "context-parallel-size", "actor-num-gpus-per-node")] == ["2", "2", "8"]
    assert recipe.miles.n_train_nodes == 1 and recipe.modal.gpu == "B300"
    # HF exports at 50% and 100% of the steps (Miles also saves at the last step).
    steps, interval = recipe.STEPS, recipe.miles.save_interval
    saves = sorted(i for i in range(steps) if (i + 1) % interval == 0 or i == steps - 1)
    assert len(saves) == 2 and saves[1] == steps - 1 and abs(saves[0] + 1 - steps / 2) <= 1
    # The trainer reads this repository's template (the image mounts cookbook/ at /root/cookbook).
    cookbook = Path(sft.__file__).parents[1]
    assert Path(sft.CHAT_TEMPLATE_IN_IMAGE.replace("/root/cookbook", str(cookbook))) == sft.CHAT_TEMPLATE


def test_sft_smoke_changes_only_the_steps_and_names():
    smoke, full = _recipe("qwen3_8_27b_sft_warmstart_smoke"), _recipe("qwen3_8_27b_sft_warmstart")
    fields, full_fields = smoke.miles._fields(), full.miles._fields()
    assert {k for k in fields | full_fields if fields.get(k) != full_fields.get(k)} == {
        "num_rollout", "save_interval", "wandb_group"
    }
    assert smoke.miles.num_rollout == smoke.miles.save_interval == 2 and full.miles.num_rollout == full.STEPS


def test_sft_app_runs_only_the_trainer(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_8_27b_sft_warmstart")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.sft_app", raising=False)
    sft_app = importlib.import_module("cookbook.miles_disagg.sft_app")

    # No rollout Server and no pool: one trainer function on its own machine.
    assert set(sft_app.app.registered_functions) == {"train"} and not sft_app.app.registered_classes
    assert sft_app.volumes["/sft-data"] is sft_app.exp.TRAINER_EXTRA_VOLUMES["/sft-data"]
    cfg = sft_app.run_paths(MilesConfig.from_payload(sft_app.miles_cfg.to_payload()), tmp_path)
    assert (cfg.save, cfg.load, cfg.save_hf, cfg.use_checkpoint_opt_param_scheduler) == (
        str(tmp_path / "checkpoints"),
        str(tmp_path / "checkpoints"),
        str(tmp_path / "hf_checkpoints/step_{rollout_id:06d}"),
        True,
    )


def test_v3b_from_sft_changes_only_the_starting_checkpoint_and_names():
    a, b = _recipe("qwen3_8_27b_harbor_mix_v3b_b300_s2"), _recipe("qwen3_8_27b_harbor_mix_v3b_b300_from_sft")
    sft_run = _recipe("qwen3_8_27b_sft_warmstart")
    fa, fb = _flags(a.miles), _flags(b.miles)

    assert {k for k in fa | fb if fa.get(k) != fb.get(k)} == {
        "--hf-checkpoint", "--ref-load", "--wandb-group", "--prometheus-run-name"
    }
    run = START_PATH / b.SFT_RUN
    assert fb["--hf-checkpoint"] == str(run / f"hf_checkpoints/step_{sft_run.STEPS - 1:06d}")
    assert fb["--ref-load"] == str(run / "checkpoints")
    assert b.START_VOLUME.name == sft_run.EXPERIMENT_VOLUME_NAME
    env = a.miles.environment | b.miles.environment
    assert {k for k in env if a.miles.environment.get(k) != b.miles.environment.get(k)} == {"HARBOR_ENV_KWARGS"}
    assert b.modal == a.modal
    assert b.TRAINER_EXTRA_VOLUMES["/harbor-trials"].name == "harbor-mix-v3b-from-sft-trials"
    validate_recipe(b)
    validate_resumable_config(b.miles)


def test_app_mounts_the_start_volume_on_the_trainer_and_the_rollout_servers(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_8_27b_harbor_mix_v3b_b300_from_sft")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)
    app = importlib.import_module("cookbook.miles_disagg.app")

    start = app.exp.START_VOLUME
    assert app.train_volumes[str(START_PATH)] is start and app.server_volumes[str(START_PATH)] is start


def test_checkpoints_under_the_start_path_need_a_start_volume():
    recipe = _recipe("qwen3_8_27b_harbor_mix_v3b_b300_from_sft")
    del recipe.START_VOLUME
    with pytest.raises(ValueError, match="START_VOLUME"):
        validate_recipe(recipe)


def test_sft_data_shows_turns_as_the_xml_harness_does():
    json_confirm = (
        f"Current terminal state:\n{OUTPUT}\n\nAre you sure you want to mark the task as complete? This will trigger "
        "your solution to be graded and you won't be able to make any further corrections. If so, include "
        '"task_complete": true in your JSON response again.'
    )
    json_warning = f"Previous response had warnings:\nWARNINGS: - Extra text detected before JSON object\n\n{OUTPUT}"
    assert sft_data.terminal_output(json_confirm) == sft_data.terminal_output(json_warning) == OUTPUT
    assert sft_data.terminal_output("Previous response had parsing errors:\nERROR: No valid JSON") is None

    xml_confirm = json_confirm.replace('"task_complete": true in your JSON response', "<task_complete>true</task_complete>")
    assert sft_data.observation(OUTPUT, "", True, False) == (xml_confirm, True)
    assert sft_data.observation(OUTPUT, "- Command 1 should end with newline", False, True) == (
        f"Previous response had warnings:\nWARNINGS: - Command 1 should end with newline\n\n{OUTPUT}",
        False,
    )
    assert sft_data.render_xml("A.", "P.", [("ls\n", 0.1)], None) == (
        '<response>\n<analysis>\nA.\n</analysis>\n<plan>\nP.\n</plan>\n<commands>\n<keystrokes duration="0.1">ls\n'
        "</keystrokes>\n</commands>\n</response>"
    )

    screen = "Current Terminal Screen:\nroot@box:/app# "
    assert sft_data.first_prompt([{"role": "user", "content": PROMPT}]) == ("Count the files in /app.", screen, 1)
    swe_v3 = [
        {"role": "system", "content": PROMPT.partition("\n\n")[0]},
        {"role": "user", "content": f"Task:\n\nFix the bug.\n\n{screen}"},
    ]
    assert sft_data.first_prompt(swe_v3) == ("Fix the bug.", screen, 2)
