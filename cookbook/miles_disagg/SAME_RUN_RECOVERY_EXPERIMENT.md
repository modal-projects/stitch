# Same-run recovery experiment

This branch preserves an abandoned prototype for rewinding a live rollout pool
to an older saved checkpoint while retaining its `run_id`. It is a research
snapshot, not a supported Stitch design: reusing a weight version for different
bytes violates the append-only weight-lineage contract.

The matching source revisions are:

| Repository | Branch | Commit |
| --- | --- | --- |
| Miles | `feat/stitch-same-run-checkpoint-recovery` | `4b412fb46b76cdee0d6ed908bab9ec74279b5b9e` |
| SGLang | `feat/stitch-live-full-checkpoint` | `80ba2a61baa436054c6e5f5b1119994fa2f6da7d` |

The prototype completed an initial three-step run and resolved the saved resume
point correctly. Live NVFP4 checkpoint restoration exposed an unfinished
ModelOpt shared-expert reload lifecycle, so the end-to-end resume did not pass.
