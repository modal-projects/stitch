# tests

Mirrors `src/stitch/`. Includes the **in-memory core harness** (the port of the
old `cookbook/local_disagg`): it runs the real `Reconciler` / `AdmissionGate` /
pointer-rules path against fake `Store` / `Engine` instances — no Modal, no
sglang, no GPU. That harness is the Phase-1 gate: the core must pass with fakes
before any concrete instance exists.
