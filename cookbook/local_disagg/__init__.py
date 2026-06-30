"""Minimal, dependency-free disaggregated-rollout harness.

No Modal, no slime/miles, no GPUs: a filesystem bulletin board, an in-memory
rollout pool, and a trainer-as-writer, wired with the *same* claim/advance/
reconcile primitives the real cookbooks use. It exists to exercise — and pin
down with tests — the pool-claim invariants quickly and locally.

See :mod:`cookbook.local_disagg.harness`.
"""
