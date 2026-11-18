"""Miles' Harbor agent function, with no environment variables in the agent's shell.

Miles binds terminus to OPENAI_API_KEY and OPENAI_API_BASE, and Harbor hands an agent's env to
`tmux new-session -e`, which tmux before 3.2 rejects: every CyberGym task (old images) failed to
start. terminus calls the model from the rollout worker, so the task's shell needs neither, and the
model should not see the session key there.

A row with agent_name "opencode" runs our OpenCode agent (opencode_agent.py, the evaluation
harness) inside the sandbox. It gets its session's public tunnel URL (session_tunnel.py) in its
OpenCode config, and nothing in its shell environment.
"""

import dataclasses

import harbor_agent_function as miles_harbor

from cookbook.miles_disagg import harbor_config, session_tunnel

OPENCODE_AGENT = "cookbook.miles_disagg.opencode_agent:RolloutOpenCode"


def _no_env(session_url: str, api_key: str) -> dict[str, str]:
    return {}


for _name in ("terminus-2", "terminus-1", "terminus"):
    miles_harbor.HARNESS_BINDINGS[_name] = dataclasses.replace(
        miles_harbor.HARNESS_BINDINGS[_name], env=_no_env
    )

_MILES_OPENCODE = miles_harbor.HARNESS_BINDINGS["opencode"]


def _opencode_kwargs(session_url, api_key, sampling_params, model, max_seq_len) -> dict:
    kwargs = _MILES_OPENCODE.kwargs(
        session_url, api_key, sampling_params, model, max_seq_len
    )
    for provider in kwargs["opencode_config"]["provider"].values():
        # The session server sends a reply's headers only with the whole reply; by default
        # OpenCode gives up on a reply after 5 minutes and sends the request again.
        provider["options"]["headerTimeout"] = False
    return kwargs


miles_harbor.HARNESS_BINDINGS[OPENCODE_AGENT] = dataclasses.replace(
    _MILES_OPENCODE, kwargs=_opencode_kwargs, env=_no_env
)

_MILES_ENVIRONMENT_CONFIG = miles_harbor._environment_config


def _environment_config():
    """Miles' environment config, with the session tunnels' hosts allowed from sandbox creation on (an agent-phase
    allowlist fails on Modal: an offline task's sandbox starts with no domain allowlist, which cannot be enabled later)."""
    config = _MILES_ENVIRONMENT_CONFIG()
    hosts = miles_harbor._allowed_hosts(session_tunnel.HOSTS_ENV)
    return config.model_copy(update={"extra_allowed_hosts": hosts}) if hosts else config


miles_harbor._environment_config = _environment_config


async def run(base_url, prompt, request_kwargs=None, metadata=None, **kwargs):
    if (metadata or {}).get("agent_name") == harbor_config.OPENCODE:
        base_url = session_tunnel.public_session_url(base_url)
        metadata = {**metadata, "agent_name": OPENCODE_AGENT}
    return await miles_harbor.run(base_url, prompt, request_kwargs, metadata, **kwargs)
