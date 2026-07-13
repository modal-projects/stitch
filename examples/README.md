# examples

Each subdirectory is one **experiment** — a config that owns the *whole* thing:
the concrete deployment (the Modal app: image + `Server` cls running
`stitch.service.serve()` + an engine + Flash + the `Trainer` cls + entrypoints),
model prep, run parameters, the ~2-line framework hook shim, and any consumer
facade (e.g. a Cognition-style `/hot_load`).

Examples are **customizable** — a different provider (k8s), image, model, or
downstream contract is a different example, not a core change. `_modal/` may hold
a shared Modal-app factory to keep examples DRY; it lives here, not in the
general library.
