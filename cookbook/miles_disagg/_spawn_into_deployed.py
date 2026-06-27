"""Spawn Trainer.train on the already-deployed app via the Modal SDK.

This is the non-client-tied equivalent of the launch_train local_entrypoint:
it does a fire-and-forget `Cls.from_name(APP_NAME, "Trainer").train.spawn(...)`
against the DEPLOYED app, so it never creates an ephemeral app context (which is
what collides with / stops the deployed serving app). Run with plain `python`,
NOT `modal run`.
"""

import importlib
import sys

import modal


def main(experiment: str = "kimi_k2_6_nvfp4_disagg") -> None:
    run = importlib.import_module(f"cookbook.miles_disagg.configs.{experiment}")
    trainer = modal.Cls.from_name(run.APP_NAME, "Trainer")()
    call = trainer.train.spawn(experiment, run.miles.to_payload())
    print(f"Spawned train({experiment!r}) on {run.APP_NAME}: {call.object_id}")


if __name__ == "__main__":
    main(*sys.argv[1:])
