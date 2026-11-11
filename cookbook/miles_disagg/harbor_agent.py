"""Miles' Harbor agent function, with no environment variables in terminus' shell.

Miles binds terminus to OPENAI_API_KEY and OPENAI_API_BASE, and Harbor hands an agent's env to
`tmux new-session -e`, which tmux before 3.2 rejects: every CyberGym task (old images) failed to
start. terminus calls the model from the rollout worker, so the task's shell needs neither, and the
model should not see the session key there.
"""

import dataclasses

import harbor_agent_function as miles_harbor

for _name in ("terminus-2", "terminus-1", "terminus"):
    miles_harbor.HARNESS_BINDINGS[_name] = dataclasses.replace(
        miles_harbor.HARNESS_BINDINGS[_name], env=lambda session_url, api_key: {}
    )

run = miles_harbor.run

# A length control (the 9B's 2e-6 follow-up, arm B). Session server v2 assigns each sample the reward this function
# returns, so the penalty goes here; a --custom-rm-path function would never run (Miles calls it only for samples
# without a reward). A passing attempt loses up to LENGTH_MAX_PENALTY for output tokens above LENGTH_BUDGET; other
# attempts keep their reward, so failing fast never pays and a pass always beats a fail.
LENGTH_BUDGET = (
    27_000  # the 1e-6 seed-42 run's training output per attempt at steps 31-40 (26.7K)
)
LENGTH_MAX_PENALTY = 0.5


def length_shaped(reward: float, output_tokens: int | None) -> float:
    if reward < 1 or not output_tokens:
        return reward
    over = min(1.0, max(0.0, (output_tokens - LENGTH_BUDGET) / LENGTH_BUDGET))
    return reward - LENGTH_MAX_PENALTY * over


async def run_length_controlled(*args, **kwargs):
    out = await run(*args, **kwargs)
    tokens = (out.get("agent_metrics") or {}).get("n_output_tokens")
    out["reward"] = length_shaped(out["reward"], tokens)
    return out
