"""The cost figure: pass@4 against cumulative rollout cost, for SC + MIS + top-p mask
replay on the mixed pool and on all-B200 BF16.

A run's rollout cost through step ``v`` is the cost of the rollout work its first ``v``
updates trained on: for each step and each sampler pool, the agent turns that pool
served (W&B ``rollout/by_source/<pool>/sample_count`` times ``turns_mean``) at that
pool's cost per turn. A pool's cost per turn is its container's list price per hour over
the turns one container serves per hour, both measured by the throughput benchmark
(``bench/``) at a p90 turn-latency bound. So each fleet is priced as if sized to its
rollout work; the trainer is left out, and so is any idle sampler time.

Turns get longer as training goes on, so the benchmark replays trace sets recorded at
several checkpoints (anchors). A step's cost per turn interpolates linearly between the
anchors around it, and holds the nearest anchor's outside them.

- **As run:** every pool at the configuration it trained with (the benchmark's ``base``
  variant).
- **Projected:** every pool at its best measured variant. All-B200 BF16 keeps BF16
  weights and a BF16 KV cache, which matching the trainer requires; the benchmark only
  varies its batch ceiling. Only ``cost.csv`` carries it: it moved costs by under 3%, so
  the figure draws the runs as run (Nan, 2026-10-08).

A container's price is Modal's list price for its GPUs alone (Nan, 2026-10-07): CPU and
memory requests follow our weight-sync design, not the accelerator.

    uv run --extra modal --with matplotlib python -m cookbook.miles_disagg.figures.cost_figures \\
        --bench 0:~/bench/base --bench 140:~/bench/v140 --out ~/blog-figures
Each ``--bench STEP:DIR`` holds the CSVs of the benchmark run on the traces recorded at
checkpoint STEP, as the bench volume lays them out (``<config>/<run_id>/<variant>.csv``). Writes OUT/cost/cost.{png,svg}, cost.csv and
SOURCES.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.bench import configs, sweep
from cookbook.miles_disagg.figures import eval_figures, eval_source

TARGET_LATENCY_S = 10.0
RUNS = ("l2_sc_mis_top_p", "l0_sc_mis_top_p")
K = 4
PANELS = (("hard51", "HARD-51 subset"), ("full", "Full set"))
# Both curves are SC + MIS + top-p mask replay; the legend names the samplers.
FLEET_LABELS = {"L2": "Mixed pool", "L0": "All-B200 BF16"}


def pool_configs() -> dict[str, str]:
    """Each training pool's server class, mapped to the benchmark row that serves it as
    the fleet did (its ``source``: ``<experiment> <server class>``)."""
    out = {}
    for key, bench in configs.CONFIGS.items():
        parts = bench.source.split()
        if len(parts) == 2 and parts[0].startswith("qwen3_6_35b_a3b_"):
            out[parts[1]] = key
    return out


def container_price(bench: configs.BenchConfig) -> float:
    """One engine per hour at Modal's GPU list price."""
    return bench.price_per_gpu_hour * bench.gpus_per_engine


@dataclass(frozen=True)
class TurnPrice:
    config: str
    variant: str
    turns_per_container_hour: float
    container_price: float

    @property
    def usd(self) -> float:
        return self.container_price / self.turns_per_container_hour


def turn_prices(
    rows: Iterable[dict[str, Any]],
    *,
    projected: bool,
    target: float = TARGET_LATENCY_S,
) -> dict[str, TurnPrice]:
    """Each benchmarked row's cost per turn at ``target``: its ``base`` variant, or with
    ``projected`` its cheapest measured variant."""
    prices: dict[str, TurnPrice] = {}
    for (config_key, variant), points in sorted(sweep._latest_runs(rows).items()):
        if not projected and variant != "base":
            continue
        bench = configs.CONFIGS[config_key]
        per_gpu_s, _ = sweep.throughput_at_speed(
            sweep._with_requests_per_gpu(points),
            target,
            throughput_key="requests_s_per_gpu",
        )
        if not per_gpu_s:
            continue
        price = TurnPrice(
            config_key,
            variant,
            per_gpu_s * 3600 * bench.gpus_per_engine,
            container_price(bench),
        )
        if config_key not in prices or price.usd < prices[config_key].usd:
            prices[config_key] = price
    return prices


def turns_by_step(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Agent turns each pool served for each rollout step, keyed by server class."""
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        step = row.get("rollout/step")
        if step is None:
            continue
        for key, value in row.items():
            if not (
                key.startswith("rollout/by_source/") and key.endswith("/sample_count")
            ):
                continue
            source = key[len("rollout/by_source/") : -len("/sample_count")]
            turns = row.get(f"rollout/by_source/{source}/turns_mean")
            if value and turns:
                out.setdefault(int(step), {})[source.split(":")[0]] = value * turns
    return out


Anchors = Sequence[tuple[int, Mapping[str, TurnPrice]]]


def usd_per_turn(config_key: str, step: int, anchors: Anchors) -> float:
    """``config_key``'s cost per turn at ``step``: linear between the anchors around it,
    the nearest anchor's outside them. Fails if an anchor lacks the row."""
    points = sorted(anchors, key=lambda anchor: anchor[0])
    for _, prices in points:
        if config_key not in prices:
            raise KeyError(f"no measured cost per turn for {config_key}")
    if step <= points[0][0]:
        return points[0][1][config_key].usd
    for (low, low_prices), (high, high_prices) in zip(points, points[1:], strict=False):
        if step <= high:
            share = (step - low) / (high - low)
            return (1 - share) * low_prices[config_key].usd + share * high_prices[
                config_key
            ].usd
    return points[-1][1][config_key].usd


def cumulative_cost(
    turns: Mapping[int, Mapping[str, float]],
    anchors: Anchors,
    pools: Mapping[str, str],
) -> dict[int, float]:
    """Rollout cost through each version: version ``v`` trained on rollout steps 0..v-1.
    Fails on a pool with no measured price rather than leaving it out."""
    total, out = 0.0, {0: 0.0}
    for step in range(max(turns) + 1 if turns else 0):
        for server, count in turns.get(step, {}).items():
            config_key = pools.get(server)
            if config_key is None:
                raise KeyError(f"no benchmark row serves pool {server}")
            total += count * usd_per_turn(config_key, step, anchors)
        out[step + 1] = total
    return out


def curve(
    points: Sequence[eval_source.EvalPoint],
    base: eval_source.EvalPoint,
    cost: Mapping[int, float],
    subset: str,
    hard: frozenset[str],
    k: int = K,
) -> tuple[list[float], list[float]]:
    """(cumulative cost in $k, pass@k) from the base model through each evaluated
    version the cost covers."""
    metric = f"pass@{k}"
    xs, ys = [0.0], [base.scores(subset, hard)[metric]]
    for point in sorted(points, key=lambda p: p.version):
        if point.n_samples >= k and point.version in cost:
            xs.append(cost[point.version] / 1000)
            ys.append(point.scores(subset, hard)[metric])
    return xs, ys


def draw(
    series: Sequence[eval_figures.Series],
    base: eval_source.EvalPoint,
    hard: frozenset[str],
    costs: Mapping[tuple[str, str], Mapping[int, float]],
) -> Any:
    """A panel per subset; each run as run, in its Figure 4 style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.4, 4.8))
    handles = []
    for ax, (subset, title) in zip(axes, PANELS, strict=True):
        for entry in series:
            cost = costs.get((entry.key, "as run"))
            if not cost:
                continue
            xs, ys = curve(entry.points, base, cost, subset, hard)
            style = eval_figures.style(entry)
            ax.plot(xs, ys, linewidth=1.8, markersize=5, **style)
            if ax is axes[0]:
                handles.append(
                    Line2D(
                        [],
                        [],
                        linewidth=1.8,
                        markersize=5,
                        label=FLEET_LABELS[entry.experiment.fleet],
                        **style,
                    )
                )
        ax.axhline(
            base.scores(subset, hard)[f"pass@{K}"],
            color=eval_figures.BASE_COLOR,
            linestyle=":",
            linewidth=1.2,
            zorder=1,
            gid="base-model",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cumulative rollout cost ($k)")
        ax.set_ylabel(f"pass@{K}")
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
    handles.append(
        Line2D(
            [],
            [],
            color=eval_figures.BASE_COLOR,
            linestyle=":",
            linewidth=1.2,
            label="Base model",
        )
    )
    fig.suptitle(
        f"Evaluation on SWE-bench Pro V2 against rollout cost, pass@{K}", fontsize=12
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.30, wspace=0.25)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    return fig


def read_bench(root: Path) -> list[dict[str, Any]]:
    return sweep.read_csv(sorted(root.rglob("*.csv")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        action="append",
        required=True,
        help="STEP:DIR, the benchmark's CSVs on the traces recorded at checkpoint STEP",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "stitch-figures"
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import wandb

    from cookbook.miles_disagg.eval_configs import swebench_pro_hetero as spec
    from cookbook.miles_disagg.figures import fetch, history
    from cookbook.miles_disagg.figures.experiments import BY_KEY

    benches = []
    for anchor in args.bench:
        step, _, directory = anchor.partition(":")
        benches.append((int(step), read_bench(Path(directory).expanduser())))
    prices = {
        scenario: [
            (step, turn_prices(rows, projected=scenario == "projected"))
            for step, rows in benches
        ]
        for scenario in ("as run", "projected")
    }
    pools = pool_configs()
    source = eval_source.EvalSource(spec, args.cache)
    series, base = eval_figures.collect(source, RUNS)
    hard = eval_source.hard_tasks()
    api = wandb.Api(timeout=60)
    costs: dict[tuple[str, str], dict[int, float]] = {}
    for key in RUNS:
        rows = history.stitch(
            [
                fetch.fetch_run(api, run_id, args.cache)[0]
                for run_id in BY_KEY[key].attempts
            ]
        )
        turns = turns_by_step(rows)
        for scenario, scenario_prices in prices.items():
            costs[(key, scenario)] = cumulative_cost(turns, scenario_prices, pools)

    out = args.out / "cost"
    out.mkdir(parents=True, exist_ok=True)
    fig = draw(series, base, hard, costs)
    for fmt in ("png", "svg"):
        fig.savefig(out / f"cost.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (out / "cost.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "scenario", "version", "rollout_usd"])
        for (key, scenario), cost in sorted(costs.items()):
            for version, usd in sorted(cost.items()):
                writer.writerow([key, scenario, version, f"{usd:.2f}"])
    (out / "SOURCES.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_p90_turn_latency_s": TARGET_LATENCY_S,
                "prices": {
                    scenario: {
                        str(step): {
                            key: {**vars(price), "usd_per_turn": price.usd}
                            for key, price in anchor.items()
                        }
                        for step, anchor in anchors
                    }
                    for scenario, anchors in prices.items()
                },
                "bench": args.bench,
                "pools": pools,
            },
            indent=2,
        )
        + "\n"
    )
    for (key, scenario), cost in sorted(costs.items()):
        print(f"{key} {scenario}: ${cost[max(cost)]:,.0f} through version {max(cost)}")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
