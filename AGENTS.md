Use Google-style testing, i.e. test `src/stitch/foo.py` with `src/stitch/foo_test.py`, not `tests/foo.py`.

If jj is available, use that instead of git for version control.

Code legibility is highly valued, both for humans and future agents. Throughout the session, ensure that you are maintaining status quo or improving the legibility of the codebase.

## Session protocol (hard-won; do not skip)

Grounding:
- Reproduce a defect (temp Modal probe or failing test) before fixing it. Never commit a fix whose premise is only code-reading or a subagent audit claim. Grounding first tends to produce *simpler* fixes.
- Label every finding by confidence: confirmed-by-probe / code-read inference / hypothesis. Subagent audit output is a lead to verify, not a fact.

Modal:
- Preflight before any Modal work: confirm auth (`modal profile current`) and the target environment; pass `-e <env>` explicitly on every modal command, monitors included.
- Before an expensive GPU launch, run a cheap fail-fast check (CPU-only arg-parse/dry-run inside the image). Bundle image-build sanity checks into one probe; don't discover gotchas one rebuild at a time.
- Watch loops must assert positive progress (app exists, log lines grow) and must not redirect stderr to /dev/null — empty output is a failure signal, not "still booting". Smoke-test the monitor command manually once before trusting it. Long-running probes should print a machine-readable verdict line.

Verification:
- `uv run pytest` before every commit.
- A Modal-touching change is not done until exercised in Modal.