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
