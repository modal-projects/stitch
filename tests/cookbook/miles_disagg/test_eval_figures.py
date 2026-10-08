import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import evaluation
from cookbook.miles_disagg.figures import eval_figures, eval_source
from cookbook.miles_disagg.figures.experiments import BY_KEY, EXPERIMENTS, MAIN_EVAL

REGISTRY = (
    Path(__file__).resolve().parents[3] / "cookbook" / "miles_disagg" / "QWEN36_RUNS.md"
)
TASKS = [f"instance_t{i}" for i in range(6)]
HARD = frozenset(TASKS[:2])
SPEC = SimpleNamespace(
    TASKS=len(TASKS),
    EXCLUDED_TASKS={"instance_excluded": "unreliable"},
    DATASET={"path": "/data/swebench-pro-scale-v2/test.jsonl"},
)


def _manifest(recipe, run_id, version, *, n=4, **overrides):
    manifest = {
        "experiment": recipe,
        "run_id": run_id,
        "version": version,
        "view": "bf16",
        "checkpoint": (
            f"/source-run/{run_id}/hf_checkpoints/weight_v{version - 1:06d}/bf16"
            if recipe
            else "/checkpoints/base/bf16"
        ),
        "dataset": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "n_samples_per_eval_prompt": n,
        },
        "stitch_commit": "55c2193abc",
        "stitch_dirty": False,
    }
    manifest.update(overrides)
    return manifest


def _rows(solved, *, n=4, tasks=TASKS):
    """``solved[task]`` of each task's n samples pass."""
    return [
        {
            "instance_id": task,
            "sample_index": index,
            "reward": float(index < solved.get(task, 0)),
            "exit_status": "Submitted",
        }
        for task in tasks
        for index in range(n)
    ]


class FakeVolume:
    """The eval Volume as a dict of path -> bytes."""

    def __init__(self):
        self.files = {}
        self.reads = 0

    def add(
        self,
        recipe,
        run_id,
        version,
        *,
        solved,
        complete=True,
        manifest=None,
        rows=None,
    ):
        root = (
            "swebench-pro-scale-v2/base/bf16"
            if recipe is None
            else f"swebench-pro-scale-v2/{recipe}/{run_id}/v{version:06d}/bf16"
        )
        self.files[f"{root}/manifest.json"] = json.dumps(
            manifest or _manifest(recipe, run_id, version)
        ).encode()
        self.files[f"{root}/metrics.json"] = json.dumps({"complete": complete}).encode()
        self.files[f"{root}/samples.jsonl"] = "".join(
            json.dumps(row) + "\n" for row in (rows or _rows(solved))
        ).encode()

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return sorted(
            {
                key[len(prefix) :].split("/")[0]
                for key in self.files
                if key.startswith(prefix)
            }
        )

    def read(self, path):
        self.reads += 1
        return self.files[path]


def _source(volume, tmp_path):
    return eval_source.EvalSource(SPEC, tmp_path, reader=volume)


def test_a_clean_point_passes_its_checks_and_scores_like_the_eval():
    rows = _rows({"instance_t0": 4, "instance_t1": 1, "instance_t2": 2})
    manifest = _manifest("recipe_a", "r01", 40)

    assert (
        eval_source.check_point(
            recipe="recipe_a",
            run_id="r01",
            version=40,
            manifest=manifest,
            metrics={"complete": True},
            rows=rows,
            n_tasks=6,
            excluded=["instance_excluded"],
        )
        == []
    )
    point = eval_source.EvalPoint(
        "recipe_a",
        "r01",
        40,
        4,
        "c",
        False,
        "p",
        tuple((r["instance_id"], r["sample_index"], r["reward"]) for r in rows),
    )
    full = evaluation.summarize(rows, n_samples=4, n_tasks=6)
    assert (
        point.scores("full", HARD)["pass@1"] == full["pass@1"] == pytest.approx(7 / 24)
    )
    hard = [r for r in rows if r["instance_id"] in HARD]
    assert (
        point.scores("hard51", HARD)["pass@2"]
        == evaluation.summarize(hard, n_samples=4, n_tasks=2)["pass@2"]
    )


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"experiment": "recipe_b"}, "manifest experiment"),
        ({"run_id": "r02"}, "manifest run_id"),
        ({"version": 80}, "manifest version"),
        (
            {"checkpoint": "/source-run/r01/hf_checkpoints/weight_v000079/bf16"},
            "checkpoint",
        ),
        (
            {
                "dataset": {
                    "temperature": 1.0,
                    "top_p": 0.97,
                    "top_k": 64,
                    "n_samples_per_eval_prompt": 4,
                }
            },
            "decoding",
        ),
    ],
)
def test_a_point_whose_manifest_disagrees_with_its_directory_is_refused(
    change, problem
):
    manifest = {**_manifest("recipe_a", "r01", 40), **change}
    problems = eval_source.check_point(
        recipe="recipe_a",
        run_id="r01",
        version=40,
        manifest=manifest,
        metrics={"complete": True},
        rows=_rows({}),
        n_tasks=6,
        excluded=[],
    )
    assert any(problem in p for p in problems), problems


def test_missing_duplicate_or_excluded_samples_are_refused():
    def problems(rows):
        return eval_source.check_point(
            recipe="recipe_a",
            run_id="r01",
            version=40,
            manifest=_manifest("recipe_a", "r01", 40),
            metrics={"complete": True},
            rows=rows,
            n_tasks=6,
            excluded=["instance_excluded"],
        )

    assert any("tasks" in p for p in problems(_rows({}, tasks=TASKS[:5])))
    assert any(
        "duplicate" in p for p in problems(_rows({}) + _rows({}, tasks=TASKS[:1]))
    )
    assert any(
        "excluded" in p
        for p in problems(_rows({}, tasks=TASKS[:5] + ["instance_excluded"]))
    )
    assert any(
        "samples 0..3" in p
        for p in problems([r for r in _rows({}) if r["sample_index"] != 3])
    )


def test_the_source_finds_only_finished_points_and_caches_them(tmp_path):
    volume = FakeVolume()
    volume.add("recipe_a", "r01", 80, solved={"instance_t0": 4})
    volume.add("recipe_a", "r01", 40, solved={"instance_t0": 2})
    volume.add("recipe_a", "r01", 120, solved={}, complete=False)
    source = _source(volume, tmp_path)

    points = source.run_points("recipe_a", "r01")

    assert [p.version for p in points] == [40, 80]
    reads = volume.reads
    again = _source(volume, tmp_path).point("recipe_a", "r01", 40)
    assert again == points[0]
    # Served from the cache: only metrics.json is read, to check the cache is current.
    assert volume.reads == reads + 1


def test_a_point_rerun_at_the_same_path_is_not_served_from_the_old_cache(tmp_path):
    volume = FakeVolume()
    volume.add("recipe_a", "r01", 40, solved={"instance_t0": 1})
    root = "swebench-pro-scale-v2/recipe_a/r01/v000040/bf16"
    volume.files[f"{root}/metrics.json"] = json.dumps(
        {"complete": True, "pass@1": 0.1}
    ).encode()
    first = _source(volume, tmp_path).point("recipe_a", "r01", 40)

    # The first result is archived and the point rerun: new samples, new metrics.
    volume.add("recipe_a", "r01", 40, solved={"instance_t0": 4, "instance_t1": 4})
    volume.files[f"{root}/metrics.json"] = json.dumps(
        {"complete": True, "pass@1": 0.3}
    ).encode()
    rerun = _source(volume, tmp_path).point("recipe_a", "r01", 40)

    assert rerun != first
    assert sum(reward for _, _, reward in rerun.rewards) == 8


def test_a_sample_over_the_turn_time_limit_scores_as_a_failure(tmp_path):
    rows = _rows({"instance_t0": 4, "instance_t1": 4, "instance_t2": 4})
    # A resent turn that the session server gave up on, a slow turn that still
    # returned, an episode that ended on the limit, and an earlier attempt that
    # aborted on a request deadline: all four fail under the rule.
    rows[0]["agent_metrics"] = {
        "client_model_request_durations_seconds": [3.0, 1800.0, 40.0]
    }
    rows[1]["agent_metrics"] = {"client_model_request_durations_seconds": [310.0]}
    rows[2]["exit_status"] = "TurnTimeLimit"
    rows[3]["aborts"] = [{"agent_error": "BadGatewayError: request deadline exceeded"}]
    # Under the limit, or another abort reason: unchanged.
    rows[4]["agent_metrics"] = {"client_model_request_durations_seconds": [299.0]}
    rows[5]["aborts"] = [{"exit_status": "sandbox_not_found"}]
    volume = FakeVolume()
    volume.add("recipe_a", "r01", 40, solved={}, rows=rows)

    point = _source(volume, tmp_path).point("recipe_a", "r01", 40)

    rewards = {(task, index): reward for task, index, reward in point.rewards}
    assert [rewards["instance_t0", i] for i in range(4)] == [0.0, 0.0, 0.0, 0.0]
    assert [rewards["instance_t1", i] for i in range(4)] == [1.0, 1.0, 1.0, 1.0]
    assert sum(rewards.values()) == 8


def test_a_cache_from_before_the_scoring_rule_is_rebuilt(tmp_path):
    volume = FakeVolume()
    volume.add("recipe_a", "r01", 40, solved={"instance_t0": 4})
    root = "swebench-pro-scale-v2/recipe_a/r01/v000040/bf16"
    rows = _rows({"instance_t0": 4})
    rows[0]["agent_metrics"] = {"client_model_request_durations_seconds": [400.0]}
    volume.files[f"{root}/samples.jsonl"] = "".join(
        json.dumps(row) + "\n" for row in rows
    ).encode()
    cached = tmp_path / "recipe_a/r01/v000040/bf16/point.json"
    cached.parent.mkdir(parents=True)
    # A cache written before the rule: same metrics, recorded rewards.
    cached.write_text(
        json.dumps(
            {
                "recipe": "recipe_a",
                "run_id": "r01",
                "version": 40,
                "n_samples": 4,
                "commit": "",
                "dirty": False,
                "path": root,
                "metrics": {},
                "rewards": [
                    [r["instance_id"], r["sample_index"], r["reward"]] for r in rows
                ],
            }
        )
    )

    point = _source(volume, tmp_path).point("recipe_a", "r01", 40)

    assert sum(reward for _, _, reward in point.rewards) == 3


def test_only_a_missing_directory_lists_as_empty():
    exception = pytest.importorskip("modal.exception")

    def listdir(error):
        def fail(path):
            raise error

        reader = object.__new__(eval_source.VolumeReader)
        reader._volume = SimpleNamespace(listdir=fail)
        return reader.listdir("swebench-pro-scale-v2/recipe_a/r01")

    assert listdir(exception.NotFoundError("path does not exist")) == []
    # A transient API error read as empty would drop a finished point from the figures.
    with pytest.raises(ConnectionError):
        listdir(ConnectionError("deadline exceeded"))


def test_a_transient_internal_error_is_retried_but_a_lasting_one_fails(monkeypatch):
    exception = pytest.importorskip("modal.exception")
    monkeypatch.setattr(eval_source.VolumeReader, "BACKOFF_S", 0.0)
    calls = []

    def reader_failing(times):
        def flaky(path):
            calls.append(path)
            if len(calls) <= times:
                raise exception.InternalError("please contact support@modal.com")
            return [SimpleNamespace(path=f"{path}/metrics.json")]

        reader = object.__new__(eval_source.VolumeReader)
        reader._volume = SimpleNamespace(listdir=flaky)
        return reader

    assert reader_failing(2).listdir("p/v000040") == ["metrics.json"]
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(exception.InternalError):
        reader_failing(eval_source.VolumeReader.ATTEMPTS).listdir("p/v000040")
    assert len(calls) == eval_source.VolumeReader.ATTEMPTS


def test_the_source_refuses_a_bad_point_and_a_different_task_set(tmp_path):
    volume = FakeVolume()
    volume.add(
        "recipe_a", "r01", 40, solved={}, manifest=_manifest("recipe_a", "r09", 40)
    )
    with pytest.raises(ValueError, match="failed its checks"):
        _source(volume, tmp_path).run_points("recipe_a", "r01")

    volume = FakeVolume()
    volume.add("recipe_a", "r01", 40, solved={})
    volume.add(
        "recipe_b",
        "r01",
        40,
        solved={},
        rows=_rows({}, tasks=TASKS[:5] + ["instance_t9"]),
    )
    source = _source(volume, tmp_path / "other")
    source.run_points("recipe_a", "r01")
    with pytest.raises(ValueError, match="different task set"):
        source.run_points("recipe_b", "r01")


def _study_volume(with_real_sc_mis_top_p=False):
    volume = FakeVolume()
    volume.add(None, None, 0, solved={"instance_t0": 2})
    for key, versions in (
        ("l2_grpo", (20, 40)),
        ("l2_icepop", (40, 60)),
        ("l2_sc_top_p_r03", (120,)),
    ):
        experiment = BY_KEY[key]
        for version in versions:
            volume.add(
                experiment.recipe, experiment.run_id, version, solved={"instance_t1": 3}
            )
    if with_real_sc_mis_top_p:
        experiment = BY_KEY["l2_sc_mis_top_p"]
        volume.add(experiment.recipe, experiment.run_id, 20, solved={"instance_t2": 4})
    return volume


def test_every_main_experiment_is_listed_once_with_no_stand_in(tmp_path):
    series, base = eval_figures.collect(_source(_study_volume(), tmp_path))

    assert [entry.experiment.key for entry in series] == list(MAIN_EVAL)
    assert all(entry.proxy is None for entry in series)
    status = {entry.key: entry.status for entry in series}
    assert status["l2_grpo"] == status["l2_icepop"] == "evaluated"
    assert status["l2_sc_mis_top_p"] == "pending"
    assert status["l0_sc_mis_top_p"] == status["l2_grpo_top_p"] == "pending"
    assert base.version == 0


def test_points_csv_has_a_row_per_point(tmp_path):
    pytest.importorskip("matplotlib")
    import csv

    series, base = eval_figures.collect(
        _source(_study_volume(with_real_sc_mis_top_p=True), tmp_path)
    )
    eval_figures.write(tmp_path / "out", series, base, HARD, formats=())

    rows = list(csv.DictReader((tmp_path / "out" / "points.csv").open()))
    assert ("l2_sc_mis_top_p", "20") in {
        (row["experiment"], row["step"]) for row in rows
    }


def test_each_subset_is_one_figure_with_a_panel_per_pass_at_k_and_one_legend(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.legend import Legend

    series, base = eval_figures.collect(_source(_study_volume(), tmp_path))

    fig = eval_figures.draw("full", series, base, HARD)

    assert len(fig.axes) == len(eval_figures.KS)
    (legend,) = [artist for artist in fig.legends if isinstance(artist, Legend)]
    labels = {
        text.get_text(): handle
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        if text.get_text().strip()
    }
    # The base model is a dotted reference line in each panel, keyed in the legend.
    assert {entry.label for entry in series} | {"Base model"} == set(labels)
    assert labels["Base model"].get_linestyle() == ":"
    for entry in series:
        expected = "--" if entry.experiment.fleet == "L0" else "-"
        assert labels[entry.label].get_linestyle() == (
            ":" if entry.proxy is not None else expected
        )
    base_scores = base.scores("full", HARD)
    for ax, k in zip(fig.axes, eval_figures.KS, strict=True):
        assert ax.get_ylabel() == f"pass@{k}"
        (base_line,) = [
            line for line in ax.get_lines() if line.get_gid() == "base-model"
        ]
        assert base_line.get_linestyle() == ":"
        assert base_line.get_ydata()[0] == base_scores[f"pass@{k}"]
        curves = [
            line
            for line in ax.get_lines()
            if len(line.get_xdata()) > 1 and line.get_gid() != "base-model"
        ]
        # Two evaluated runs, each from the base model's step-0 point.
        assert len(curves) == 2
        assert all(
            line.get_xdata()[0] == 0 and line.get_ydata()[0] == base_scores[f"pass@{k}"]
            for line in curves
        )
        assert not ax.collections  # points only, no error bars


def test_figure_four_is_pass_at_4_with_the_hard_subset_beside_the_full_set(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.legend import Legend

    series, base = eval_figures.collect(_source(_study_volume(), tmp_path))

    fig = eval_figures.draw_figure(series, base, HARD)

    assert [ax.get_title() for ax in fig.axes] == ["HARD-51 subset", "Full set"]
    assert all(ax.get_ylabel() == "pass@4" for ax in fig.axes)
    assert len([a for a in fig.legends if isinstance(a, Legend)]) == 1
    for ax, subset in zip(fig.axes, ("hard51", "full"), strict=True):
        (base_line,) = [
            line for line in ax.get_lines() if line.get_gid() == "base-model"
        ]
        assert base_line.get_ydata()[0] == base.scores(subset, HARD)["pass@4"]


def test_a_build_writes_figure_four_beside_the_per_subset_figures(tmp_path):
    pytest.importorskip("matplotlib")

    series, base = eval_figures.collect(_source(_study_volume(), tmp_path))
    written = eval_figures.write(tmp_path / "out", series, base, HARD, formats=("png",))

    assert {path.name for path in written} == {
        "pass4.png",
        "full.png",
        "hard51.png",
        "points.csv",
    }


def test_every_experiment_matches_its_registry_row():
    """Recipe, run id and W&B attempts agree with QWEN36_RUNS.md, so a figure cannot
    show one run's data under another's name."""
    rows = {}
    for line in REGISTRY.read_text().splitlines():
        match = re.match(
            r"\| L[012] \|[^|]*\| `([a-z0-9_]+)`[^|]*\| (r\d+) \| ([^|]*)\|", line
        )
        if match:
            recipe, run_id, wandb = match.groups()
            rows.setdefault((recipe, run_id), set()).update(
                re.findall(r"runs/([a-z0-9]{8})", wandb)
            )
    for experiment in EXPERIMENTS:
        if experiment.run_id is None:
            continue
        assert (experiment.recipe, experiment.run_id) in rows, experiment.key
        assert (
            set(experiment.attempts) == rows[(experiment.recipe, experiment.run_id)]
        ), experiment.key


def test_proxies_name_evaluated_experiments():
    for experiment in EXPERIMENTS:
        if experiment.eval_proxy is not None:
            assert BY_KEY[experiment.eval_proxy].run_id is not None


def test_color_is_the_method_and_line_style_the_pool():
    by_fleet = {
        (BY_KEY[key].arm, BY_KEY[key].sampling, BY_KEY[key].fleet): key
        for key in MAIN_EVAL
    }
    mixed = [key for key in MAIN_EVAL if BY_KEY[key].fleet == "L2"]
    # Each mixed-pool method has its own color.
    assert len({eval_figures.COLORS[key] for key in mixed}) == len(mixed)
    # A uniform-hardware run keeps its method's color; only the line style differs.
    for (arm, sampling, fleet), key in by_fleet.items():
        if fleet == "L0":
            twin = by_fleet[(arm, sampling, "L2")]
            assert eval_figures.COLORS[key] == eval_figures.COLORS[twin]


def _building_blocks(key: str) -> int:
    """What a method adds to plain GRPO: SC, MIS and top-p mask replay count one each."""
    experiment = BY_KEY[key]
    corrections = {"GRPO": 0, "IcePop": 1, "SC": 1, "SC+MIS": 2}[experiment.arm]
    return corrections + (experiment.sampling == "top-p")


def test_the_legend_columns_count_building_blocks():
    columns = eval_figures.LEGEND_COLUMNS
    # The base model alone, then nothing added, one building block, and SC + MIS.
    assert columns[0] == ("base-model",)
    methods = columns[1:]
    assert sorted(key for column in methods for key in column) == sorted(MAIN_EVAL)
    assert {_building_blocks(key) for key in methods[0]} == {0}
    assert {_building_blocks(key) for key in methods[1]} == {1}
    assert all(_building_blocks(key) >= 2 for key in methods[2])
    pytest.importorskip("matplotlib")
    # The eval figures add the base model's handle; the training figures have none.
    handles = {key: object() for key in [*MAIN_EVAL, "base-model"]}
    ordered, ncol = eval_figures.legend_order(handles)
    assert ncol == len(columns)
    depth = max(len(column) for column in columns)
    assert len(ordered) == depth * ncol
    # Filled column by column: the base model heads the first column.
    assert ordered[0] is handles["base-model"]


def test_a_one_column_legend_puts_one_blank_between_groups():
    pytest.importorskip("matplotlib")
    handles = {key: object() for key in MAIN_EVAL}
    stacked, ncol = eval_figures.legend_order(handles, one_column=True)
    # The training figures have no base model, so its column drops out.
    groups = [
        present
        for column in eval_figures.LEGEND_COLUMNS
        if (present := [key for key in column if key in handles])
    ]
    assert ncol == 1
    assert len(stacked) == len(MAIN_EVAL) + len(groups) - 1
    assert [stacked.index(handles[column[0]]) for column in groups] == [
        0,
        len(groups[0]) + 1,
        len(groups[0]) + len(groups[1]) + 2,
    ]


def test_an_episode_that_never_submitted_scores_as_a_failure(tmp_path):
    rows = _rows({"instance_t0": 4, "instance_t1": 4})
    # Its diff passed, but the episode stopped on repeated format errors, on the step
    # or context limit, or on the wall-clock budget: none of them submitted.
    for row, status in zip(
        rows[:3], ("RepeatedFormatError", "LimitsExceeded", "TimeExceeded"), strict=True
    ):
        row["exit_status"] = status
    volume = FakeVolume()
    volume.add("recipe_a", "r01", 40, solved={}, rows=rows)

    point = _source(volume, tmp_path).point("recipe_a", "r01", 40)

    rewards = {(task, index): reward for task, index, reward in point.rewards}
    assert [rewards["instance_t0", i] for i in range(4)] == [0.0, 0.0, 0.0, 1.0]
    assert [rewards["instance_t1", i] for i in range(4)] == [1.0, 1.0, 1.0, 1.0]


def test_a_hidden_point_stays_on_the_volume_but_leaves_the_figures(tmp_path):
    volume = _study_volume()
    grpo = BY_KEY["l2_grpo"]
    volume.add(grpo.recipe, grpo.run_id, 100, solved={"instance_t0": 4})
    assert ("l2_grpo", 100) in eval_figures.HIDDEN_POINTS

    series, _ = eval_figures.collect(_source(volume, tmp_path))

    (entry,) = [s for s in series if s.experiment.key == "l2_grpo" and s.proxy is None]
    assert [point.version for point in entry.points] == [20, 40]


def test_a_hidden_step_leaves_every_methods_curve(tmp_path):
    volume = _study_volume()
    grpo = BY_KEY["l2_grpo"]
    volume.add(grpo.recipe, grpo.run_id, 70, solved={"instance_t0": 4})
    volume.add(grpo.recipe, grpo.run_id, 110, solved={"instance_t0": 4})
    assert {70, 90, 110} <= set(eval_figures.HIDDEN_STEPS)

    series, _ = eval_figures.collect(_source(volume, tmp_path))

    (entry,) = [s for s in series if s.experiment.key == "l2_grpo" and s.proxy is None]
    assert [point.version for point in entry.points] == [20, 40]
