# Qwen3.6-35B-A3B mismatch study: run registry

The record of which run stands for each configuration, where its data lives, and what code
it ran. Update it whenever a run is launched, stopped or superseded.

## What every run shares

- **W&B:** project `nan-playground/fully-async-rl-modal`. The group is the recipe's
  `wandb_group`, and the run's config carries `run_id`. Filter on both: a group holds
  every attempt of its arm.
- **Modal:** environment `stitch-dev`, runtime `runc`. The app is `<APP_NAME>-<run>` and
  the Volume is `<APP_NAME>`. Run `rNN` keeps its files under `/rNN/` on that Volume
  (trainer checkpoints, HF exports `hf_checkpoints/weight_v*`, per-view weight updates).
- **Training:** Qwen3.6-35B-A3B from BF16, MiMo-V2.6 RL code tasks, B200 trainer (4 x 8
  GPUs, TP2/CP4/EP8). 1,344 concurrent sessions, 128 prompts x 8 samples per step, 500
  steps, a save every 10. Temperature 1, full-vocabulary sampling, unbounded staleness.
- **Arms:** each arm is vanilla GRPO plus only its algorithm (`test_config.py` pins the
  difference).
  - GRPO.
  - IcePop: masked importance weights in [0.5, 5].
  - Score centering (SC): top-128 candidates plus a tail model, no std normalization.
  - SC+MIS: score centering with IcePop's weights in [0.5, 5].

## Rungs

| Rung | Recipe prefix | Rollout fleet | What differs from the trainer |
|---|---|---|---|
| L0 | `qwen3_6_35b_a3b_b200_bf16_` | 42 x 1 B200, BF16 weights + BF16 KV, 32 sessions each | kernels only |
| L1 | `qwen3_6_35b_a3b_b200_nvfp4_` | 42 x 1 B200, NVFP4 W4A16 + FP8 KV, 32 sessions each | precision (same GPU) |
| L2 | `qwen3_6_35b_a3b_hetero_` | 8 pools: FP8 on H100/H200, NVFP4 on B200/B300, BF16 on A100/RTX PRO 6000/H100/H200 | GPU and precision |

L0 sizing comes from a one-replica calibration on 2026-10-03. At 32 sessions, B200 BF16
decodes 94 tok/s per request, against 90 for H200 FP8 at its 32.

## Code sets

| Code | Stitch | Miles | SGLang (modal-projects/sglang) |
|---|---|---|---|
| A | `abd5321` | `931466431e` | `cbc0988c10` (branch since rewritten by others) |
| B | `bfb6239` | `c0202f9621` | `25d7c62b2b` (`stitch-sglang-hetero`) |
| B+L | `bfb6239` + the ladder recipes (uncommitted at launch) | `c0202f9621` | `25d7c62b2b` |
| C | `ed38dde` (deployed from the working tree minutes before the commit; same content) | `0fe16fcb5e` | `25d7c62b2b` |
| D | `449cc3d` | `0fe16fcb5e` | `25d7c62b2b` |
| E | `a7526bb` plus the trainer AWS pin, deployed from the working tree before its commit | `0fe16fcb5e` | `25d7c62b2b` |
| F | E with the AWS pin removed (trainers on any cloud), plus the eval-only V2 grading code; working tree | `0fe16fcb5e` | `25d7c62b2b` |

B differs from A only in how top-logprob candidates for training are carried:
- the client reply omits them;
- session records keep them compactly;
- SGLang skips rendering them as OpenAI objects.

GRPO and IcePop request no candidates, so their runs compare across A and B.

E pins every study trainer to AWS (`ModalConfig.trainer_cloud`); the rollout pools may
still run on any provider, and nothing the trainer computes changes. F drops the pin
(07:53, 2026-10-04): AWS-only trainers sat queued for GPUs.

D adds only the replica's rollout source stamped in each response body, which tags
a one-pool fleet's samples with their view. It changes attribution metrics, nothing
the trainer optimizes.

C adds `--hf-export-weight-views`. A ladder run's trainer publishes only the view its fleet
serves, but every save exports all three, as an L2 save does. Only the HF export reads the
flag, so what each engine receives is the same as in B.

## Record runs (one per configuration)

| Rung | Arm | Recipe | Run | W&B run | Trainer call | Launched (UTC) | Code | Status |
|---|---|---|---|---|---|---|---|---|
| L2 | GRPO | `qwen3_6_35b_a3b_hetero_grpo` | r07 | [totxahfg](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/totxahfg) | `fc-01M40MGVDKA2J2NQWFWT6S3V51` | 2026-10-03 10:15 | C | stopped at 07:37 (2026-10-04) right after its step-120 save completed (tracker 119, three views); not resumed |
| L2 | IcePop | `qwen3_6_35b_a3b_hetero_icepop` | r06 | [j8nc4xng](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/j8nc4xng) | `fc-01M40MJSP0D4X8M35RWMN53W3M` | 2026-10-03 10:15 | C | running |
| L2 | SC | `qwen3_6_35b_a3b_hetero_score_centering` | r10 | | `fc-01M42V9RE182WVG65B6Q4X65P1` | 2026-10-04 06:52 | E | running |
| L2 | SC+MIS | `qwen3_6_35b_a3b_hetero_score_centering_mis` | | | | | | not relaunched yet |
| L1 | GRPO | `qwen3_6_35b_a3b_b200_nvfp4_grpo` | r02 | [gvo1ldkg](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/gvo1ldkg) | `fc-01M4091PER6VVB1CVBSDGSBSKP` | 2026-10-03 06:57 | C | stopped at v131 (06:34, 2026-10-04) after its step-130 save completed; not resumed |
| L1 | IcePop | `qwen3_6_35b_a3b_b200_nvfp4_icepop` | r02 | [quwkb52d](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/quwkb52d) | `fc-01M4090XWSFKYTS0KK8NDCVT3W` (attempt 1); `fc-01M42W8B46V98DX06H366V8BGR` queued on AWS and never ran; `fc-01M42YPP65QFXV73T9VQ23Q9G3` (attempt 2) | 2026-10-03 06:57 | C, then F from v100 | attempt 1 hit the 24 h limit at v102 (07:01, 2026-10-04); resumed from step 100 on code F (07:53) after an AWS-pinned attempt sat queued |
| L1 | SC | `qwen3_6_35b_a3b_b200_nvfp4_score_centering` | | | | | | paused (r01 stopped before its trainer got GPUs); launch after every IcePop trainer holds GPUs |
| L1 | SC+MIS | `qwen3_6_35b_a3b_b200_nvfp4_score_centering_mis` | | | | | | not launched |
| L0 | GRPO | `qwen3_6_35b_a3b_b200_bf16_grpo` | r02 | [7mpv4qdb](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/7mpv4qdb) | `fc-01M4099CZTA1TY6ZS51YJY4B90` | 2026-10-03 06:57 | C | stopped at v148 (06:39, 2026-10-04), 25 min before its 24 h attempt limit; last complete save step 140; not resumed |
| L0 | IcePop | `qwen3_6_35b_a3b_b200_bf16_icepop` | r04 | | `fc-01M42YQXTRXZVRNR8NS6QPEKJ5` | 2026-10-04 07:53 | F | waiting for trainer GPUs |
| L0 | SC | `qwen3_6_35b_a3b_b200_bf16_score_centering` | | | | | | not launched |
| L0 | SC+MIS | `qwen3_6_35b_a3b_b200_bf16_score_centering_mis` | | | | | | not launched |

Things to know when reading the record runs:

- **A resumed attempt is a new W&B run.** Miles mints a W&B run per trainer attempt (the
  recipes pass no `--wandb-run-id`), so a run resumed after its 24 h limit has one W&B
  run per attempt. The W&B column lists every attempt in order with the versions it
  covers. A resumed attempt retrains from its save, so where two attempts logged the
  same step, the later attempt is the record.

- **Ladder r02 runs (L0, L1): per-view metrics read `unknown`.** These fleets have one
  pool and so no router, and before the sidecar reported its own source, nothing tagged
  their samples. W&B files their per-view rollout metrics under `rollout/by_view/unknown/`
  (L1 `nvfp4`, L0 `bf16`), and the trainer logs no per-view or per-source mismatch split;
  `train/train_rollout_*/all` and the `lag_*` splits are complete. With one pool,
  `unknown` and `all` are the whole fleet. Runs deployed after the fix are tagged.
- **Ladder r02 runs (L0, L1):** these are the first runs to publish a single weight view. Miles
  still takes its views path with one view, so resume keeps the delta history and checks the
  new conversion against it. Every save exports `bf16`, `fp8` and `nvfp4` under
  `hf_checkpoints/weight_vN/`, the same layout as L2.

## Superseded runs (not for analysis)

| Arm | Run | W&B | Why it doesn't count |
|---|---|---|---|
| GRPO | r03 | none | old layout; stopped before v1 |
| GRPO | r04 | none | stopped before training to restore metrics |
| GRPO | r05 | none | trainer hung in its first optimizer step; restarted as r06 |
| IcePop | r01, r02 | none | Oct 1 attempts on the pre-clean layout |
| IcePop | r03 | [9985k4qd](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/9985k4qd), [u020ikox](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/u020ikox) | a different config, `qwen3_6_35b_a3b_hetero_icepop_advanced` (frozen; see below); stopped at iteration 159 |
| IcePop | r04 | none | stopped before training to restore metrics |
| SC | r01, r02 | none | Sep 30 to Oct 1 attempts on the pre-clean layout |
| SC | r03 | [1vtd813n](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/1vtd813n), [o58xf5g0](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/o58xf5g0) | a different config, `qwen3_6_35b_a3b_hetero_score_centering_advanced` (frozen; see below); stopped at step 130 |
| SC | r05, r06, r07 | none | rollout about 3x slow: SGLang rendered 128 candidates per token as OpenAI objects; fixed in code B |
| L1 GRPO | r01 | none | stopped at step 0 (06:30, 2026-10-03): would have exported only nvfp4; relaunched as r02 with all three exports |
| L1 IcePop | r01 | none | stopped at step 0 (06:30, 2026-10-03): same as L1 GRPO r01 |
| L0 GRPO | r01 | none | stopped in its first rollout (06:40, 2026-10-03): would have exported only bf16; relaunched as r02 with all three exports |
| L0 IcePop | r01 | none | stopped at the user's request before its trainer got GPUs (18:20, 2026-10-03); code D |
| L0 IcePop | r02 | none | stopped at the user's request at v0 (07:00, 2026-10-04), before its first publish; relaunched as r03 on code E |
| L0 IcePop | r03 | none | trainer queued for AWS GPUs from 07:06 and never ran; stopped 07:53 (2026-10-04) to relaunch without the AWS pin as r04 |
| L1 SC | r01 | none | stopped at the user's request before its trainer got GPUs (07:48, 2026-10-04): paused until every IcePop trainer holds GPUs |
| L2 GRPO | r06 | [b450kpxu](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/b450kpxu) | preempted at v58 (see below); stopped, rerun from scratch as r07 on code C |
| L2 IcePop | r05 | [mprwa9hg](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/mprwa9hg) | preempted at v55; stopped, rerun from scratch as r06 on code C |
| L2 SC | r08 | [3rp31dgx](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/3rp31dgx) | preempted at v25; stopped, to be rerun from scratch |
| L2 SC+MIS | r01 | [3ojxtmkv](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/3ojxtmkv) | preempted at v19; stopped, to be rerun from scratch |
| L2 SC | r09 | none | stopped before its trainer got GPUs (06:52, 2026-10-04); relaunched as r10 with the trainer pinned to AWS (code E) |
| GRPO | hetero-base-01 | [07obimhw](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/07obimhw) | Sep 30, older `qwen3_6_35b_a3b_mimo_code_heterogeneous` layout |
| TIS | hetero-tis-01 | [1tslnj1g](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/1tslnj1g) | Sep 30, older `qwen3_6_35b_a3b_mimo_code_heterogeneous` layout |

How the two frozen configs differ from the clean arms:
- `qwen3_6_35b_a3b_hetero_icepop_advanced`: B300 trainer, IcePop [0.2, 5], top-p 0.97 /
  top-k 4096, no std normalization, prompt-mean loss, frozen router.
- `qwen3_6_35b_a3b_hetero_score_centering_advanced`: top-p 0.97 / top-k 64.

The four preempted L2 runs (GRPO r06, IcePop r05, SC r08, SC+MIS r01) went down together
on 2026-10-03. At about 10:01 UTC, Modal terminated one node of each run's 4-node trainer
gang. Its system log reads "Container terminated due to preemption". A gang can't train
without a node, so each trainer attempt failed. Modal's retry policy restarts the whole
gang from the run's last save. The last saves were at iterations 49, 49, 19 and 9, and
IcePop r05 had resumed from checkpoint version 50 by 10:03. All four were stopped at 10:08,
and the arms are rerun from scratch. The ladder runs (L0, L1) weren't touched.
