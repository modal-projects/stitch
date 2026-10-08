"""Every experiment the study reports, including the ones that have not run yet.

An experiment is one record run: its recipe and run id name its checkpoints and eval
points, and its W&B attempts name its training history. A planned experiment has neither
yet, but figures list it anyway, so a figure shows from the start what the study compares
and fills in as results land. ``QWEN36_RUNS.md`` is the registry these entries follow.
"""

from __future__ import annotations

from dataclasses import dataclass

FLEETS = ("L0", "L1", "L2")
ARMS = ("GRPO", "IcePop", "SC", "SC+MIS")
SAMPLING = ("full", "top-p")


@dataclass(frozen=True)
class Experiment:
    """``attempts`` are the record run's W&B run ids in the order they ran; a later
    attempt resumed from a save of the one before it. ``eval_proxy`` names an earlier
    experiment whose eval points stood in before this one had its own; the eval figures keep
    them beside this one's, labelled as the stand-in."""

    key: str
    label: str
    fleet: str
    arm: str
    sampling: str
    recipe: str
    run_id: str | None = None
    attempts: tuple[str, ...] = ()
    eval_proxy: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.fleet not in FLEETS:
            raise ValueError(f"{self.key}: fleet must be one of {FLEETS}")
        if self.arm not in ARMS:
            raise ValueError(f"{self.key}: arm must be one of {ARMS}")
        if self.sampling not in SAMPLING:
            raise ValueError(f"{self.key}: sampling must be one of {SAMPLING}")
        if len(set(self.attempts)) != len(self.attempts):
            raise ValueError(f"{self.key}: attempts must be distinct W&B run ids")
        if self.attempts and self.run_id is None:
            raise ValueError(f"{self.key}: W&B attempts need a run id")


_HETERO = "qwen3_6_35b_a3b_hetero_"
_B200_BF16 = "qwen3_6_35b_a3b_b200_bf16_"
_B200_NVFP4 = "qwen3_6_35b_a3b_b200_nvfp4_"

EXPERIMENTS = (
    # The six heterogeneous-pool arms: the study's main question.
    Experiment(
        "l2_grpo", "GRPO", "L2", "GRPO", "full", _HETERO + "grpo", "r07", ("totxahfg",)
    ),
    Experiment(
        "l2_sc",
        "SC",
        "L2",
        "SC",
        "full",
        _HETERO + "score_centering",
        "r11",
        ("va49qscy",),
        note="replicate of r10's collapse",
    ),
    Experiment(
        "l2_icepop",
        "GRPO + MIS",
        "L2",
        "IcePop",
        "full",
        _HETERO + "icepop",
        "r06",
        ("j8nc4xng", "ashnuuol"),
    ),
    Experiment(
        "l2_sc_mis",
        "SC + MIS",
        "L2",
        "SC+MIS",
        "full",
        _HETERO + "score_centering_mis",
        "r02",
        ("tzkhwr7m", "ykc9v9ty", "12ruqi2d"),
    ),
    Experiment(
        "l2_grpo_top_p",
        "GRPO + top-p mask replay",
        "L2",
        "GRPO",
        "top-p",
        _HETERO + "grpo_top_p",
        "r02",
        ("nttqm51w",),
        note="r01 (wy4nvagw) was preempted at step 2 and rerun from scratch as r02",
    ),
    Experiment(
        "l2_sc_mis_top_p",
        "SC + MIS + top-p mask replay",
        "L2",
        "SC+MIS",
        "top-p",
        _HETERO + "score_centering_mis_top_p",
        "r01",
        ("g0c583mu", "fl29kozh", "pxajqlef", "srxgyliz", "qv8y7ue2"),
    ),
    # The final recipe on homogeneous hardware: the study's second question.
    Experiment(
        "l0_sc_mis_top_p",
        "SC + MIS + top-p mask replay, all-B200 BF16",
        "L0",
        "SC+MIS",
        "top-p",
        _B200_BF16 + "score_centering_mis_top_p",
        "r01",
        ("mw9z6ksn", "pnsekp4q", "elweshcy", "8mjqxo6r"),
    ),
    # Context.
    Experiment(
        "l2_sc_top_p_r03",
        "SC + top-p mask replay",
        "L2",
        "SC",
        "top-p",
        _HETERO + "score_centering_advanced",
        "r03",
        ("1vtd813n", "o58xf5g0"),
        note="frozen score_centering_advanced config on older code",
    ),
    Experiment(
        "l2_sc_r10",
        "SC, first run",
        "L2",
        "SC",
        "full",
        _HETERO + "score_centering",
        "r10",
        ("ccytlwt0",),
        note="collapsed from about step 33; stopped at v45",
    ),
    Experiment(
        "l0_grpo",
        "GRPO, all-B200 BF16",
        "L0",
        "GRPO",
        "full",
        _B200_BF16 + "grpo",
        "r02",
        ("7mpv4qdb",),
    ),
    Experiment(
        "l1_grpo",
        "GRPO, all-B200 NVFP4",
        "L1",
        "GRPO",
        "full",
        _B200_NVFP4 + "grpo",
        "r02",
        ("gvo1ldkg",),
    ),
    Experiment(
        "l0_icepop",
        "GRPO + MIS, all-B200 BF16",
        "L0",
        "IcePop",
        "full",
        _B200_BF16 + "icepop",
        "r04",
        ("ghnfty2y", "ylsplqu9"),
    ),
    Experiment(
        "l1_icepop",
        "GRPO + MIS, all-B200 NVFP4",
        "L1",
        "IcePop",
        "full",
        _B200_NVFP4 + "icepop",
        "r02",
        ("quwkb52d", "ke8q6lje", "uiz8nbyf"),
    ),
)
BY_KEY = {experiment.key: experiment for experiment in EXPERIMENTS}

# The main eval figures, in legend order.
MAIN_EVAL = (
    "l2_grpo",
    "l2_sc",
    "l2_icepop",
    "l2_sc_mis",
    "l2_grpo_top_p",
    "l2_sc_mis_top_p",
    "l0_sc_mis_top_p",
    # Uncorrected reference on homogeneous hardware: how much sooner the mixed pool fails.
    "l0_grpo",
)
