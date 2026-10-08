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
  steps, a save every 10. Temperature 1, unbounded staleness, and full-vocabulary sampling
  except in the top-p arms.
- **Arms:** each arm is vanilla GRPO plus only its algorithm (`test_config.py` pins the
  difference).
  - GRPO.
  - IcePop: masked importance weights in [0.5, 5].
  - Score centering (SC): top-128 candidates plus a tail model, no std normalization.
  - SC+MIS: score centering with IcePop's weights in [0.5, 5].
  - Top-p arms (GRPO, SC+MIS): the same arm with samplers drawing from the top-p 0.97
    nucleus, capped at the top 64 tokens, and the trainer renormalized over that support
    (sampling-support replay). SC keeps logging 128 candidates, twice the cap, for ties.

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
| G | `aa6e15a` (F committed, plus the top-p recipes) | `0fe16fcb5e` | `25d7c62b2b` |

B differs from A only in how top-logprob candidates for training are carried:
- the client reply omits them;
- session records keep them compactly;
- SGLang skips rendering them as OpenAI objects.

GRPO and IcePop request no candidates, so their runs compare across A and B.

G adds only recipes. The arms that ran on F run the same code on G.

E pins every study trainer to AWS (`ModalConfig.trainer_cloud`); the rollout pools may
still run on any provider, and nothing the trainer computes changes. F drops the pin
(07:53, 2026-10-04): AWS-only trainers sat queued for GPUs.

D adds only the replica's rollout source stamped in each response body, which tags
a one-pool fleet's samples with their view. It changes attribution metrics, nothing
the trainer optimizes.

C adds `--hf-export-weight-views`. A ladder run's trainer publishes only the view its fleet
serves, but every save exports all three, as an L2 save does. Only the HF export reads the
flag, so what each engine receives is the same as in B.

The resumed attempts of the two SC+MIS + top-p r01 runs (from 2026-10-07) launched from the
working tree: G plus the eval-only commits through `92baab7` and opt-in request tracing and
trajectory dumps, off unless set (committed on 2026-10-08 in the commits after `b4857f9`). Their
training code is G's. On 2026-10-08 the branch took `dc573c6` and `3e3fed4` (the eval turn
limit and the training-signal rules), so from 07:37 that day their 24 h-limit resumes launch
from a frozen copy of the earlier tree (scratchpad worktree `wt_r01_live`), never from the
branch.

## Record runs (one per configuration)

| Rung | Arm | Recipe | Run | W&B run | Trainer call | Launched (UTC) | Code | Status |
|---|---|---|---|---|---|---|---|---|
| L2 | GRPO | `qwen3_6_35b_a3b_hetero_grpo` | r07 | [totxahfg](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/totxahfg) | `fc-01M40MGVDKA2J2NQWFWT6S3V51` | 2026-10-03 10:15 | C | stopped at 07:37 (2026-10-04) right after its step-120 save completed (tracker 119, three views); not resumed |
| L2 | IcePop | `qwen3_6_35b_a3b_hetero_icepop` | r06 | [j8nc4xng](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/j8nc4xng) (attempt 1, v0-v111), [ashnuuol](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/ashnuuol) (attempt 2, from v110) | `fc-01M40MJSP0D4X8M35RWMN53W3M` (attempt 1), `fc-01M435GY9QXRCYRNSSZT29JGSS` (attempt 2) | 2026-10-03 10:15 | C, then F from v110 | attempt 1 stopped after its step-110 save (09:47, 2026-10-04), before its 24 h limit; redeployed on code F and resumed from step 110 (09:57); stopped at 09:39 (2026-10-05) at the user's request, at v199 (last durable save: step 190), because its step-200 save would not land before its 24 h attempt limit; the experiment is over |
| L2 | SC | `qwen3_6_35b_a3b_hetero_score_centering` | r10 | [ccytlwt0](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/ccytlwt0) (attempt 1, v0-v45) (not found on W&B as of 2026-10-05 04:33) | `fc-01M42V9RE182WVG65B6Q4X65P1` (attempt 1), `fc-01M43WHQGHMW4GXVRY6S8Q32R9` (attempt 2) | 2026-10-04 06:52 | E, then F from v40 | collapsed from about step 33: the tail ratio `sc_tail_ratio` rose from 1.2 to 52 and reward fell to 0.07. Stopped at 16:11 (2026-10-04) at the user's request. Attempt 2 was a resume from step 40 at 16:32, at the user's request; it was stopped at 16:44, before its trainer got GPUs (no training, no W&B run), when the user chose a fresh rerun as r11 instead |
| L2 | SC | `qwen3_6_35b_a3b_hetero_score_centering` | r11 | [va49qscy](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/va49qscy) | `fc-01M43XCPMVYCP0QXW2QVA2WXK4` | 2026-10-04 16:44 | F | launched from scratch to replicate r10's collapse; trainer spawned 16:54, got GPUs 17:26. Reproduced it about 10 steps later: `sc_tail_ratio` passed 20 at step 40, the grad norm spiked at steps 44-45, and reward fell from about 0.36 to 0.10 and response length from 21K to 6.4K tokens by step 49. Stopped at 05:46 (2026-10-05) at the user's request, right after its step-60 save (tracker 59, 05:40) and exports (05:45) checked out |
| L2 | SC, top-p 0.97 / top-k 64 | `qwen3_6_35b_a3b_hetero_score_centering_advanced` (frozen) | r03 | [1vtd813n](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/1vtd813n), [o58xf5g0](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/o58xf5g0) (v0-v143) | `fc-01M3XRCN6YNJQQ7138V3265WG5` (last Oct 2 attempt), `fc-01M43WQAYK75BY3ZAW1D5SFQHD` (resume) | 2026-10-01 05:24 | Stitch 4e9b4b8/bae4aec with Miles c96c327f32 and SGLang cbc0988c10, then F from v130 | stopped on Oct 2 at the user's request. Its step-140 save has a shard that fails sha256, so the tracker stays at 129. A resume from step 130 at 16:35 (2026-10-04) rewound the pointers from v143 to v130. It was stopped at 17:01, before its trainer got GPUs, so nothing trained. The user moved the capacity to SC+MIS. The run can still be resumed from step 130 |
| L2 | SC+MIS | `qwen3_6_35b_a3b_hetero_score_centering_mis` | r02 | [tzkhwr7m](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/tzkhwr7m) (attempt 1, v0-v70), [ykc9v9ty](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/ykc9v9ty) (attempt 2, from v60), [12ruqi2d](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/12ruqi2d) (attempt 3, from v150; the 09:36 retry `o0wkdpgp` was preempted at 10:01 before logging a step) | `fc-01M43Y5YGKT4PF1XHC0ETQ78CH` | 2026-10-04 17:01 | F | launched from scratch, MIS [0.5, 5], full vocab (r01 was preempted at v19 on Oct 3); trainer spawned 17:07, got GPUs 18:07. Two trainer containers were preempted at 10:01 (2026-10-05) at v70; Modal's retry resumed from the step-60 save (attempt 2, steps 60-70 retrained), and the user chose to keep it. Trainer containers preempted again at 09:33 (2026-10-06), at v151, minutes before the planned step-149 handoff; Modal's retry resumes from the step-149 save (attempt 3, the planned 24 h handoff was cancelled so it would not cancel the retry); STOPPED at 09:46 (2026-10-07) at v223, as the user asked, once the step-220 save was durable. That save had stalled: the head host's (10.100.0.1) background upload of iteration 219 hung at 09:02 (rank shards 1, 2, 3, 6 and half the export missing on the Volume, tracker 209). With the user's OK it was recovered by hand from the head's local copy (the Oct 2 procedure): upload of the 118 missing files 09:29-09:36, then the Megatron plan check (32 files at exact size), every export index complete, and all 118 files byte-identical to the local copies (VERIFY_OK 09:45), then the export .complete markers and the tracker last (219 at 09:45:43). Step-220 eval launched 09:46 (8 engines, default sessions) |
| L2 | GRPO, top-p 0.97 / top-k 64 + replay | `qwen3_6_35b_a3b_hetero_grpo_top_p` | r01 | [wy4nvagw](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/wy4nvagw) | `fc-01M45JVXY7N2M97E89MT9KMGMP` | 2026-10-05 07:49 | G | launched from scratch with SC+MIS top-p r01. Its RTX PRO 6000 pool waited for capacity until 08:28, and the trainer spawned at 08:28. Preempted at step 2 (one trainer container, 09:32:40, 2026-10-05), before any save. Modal's retry restarted it from scratch (pointers back at v0), and I stopped the app at 09:35, which the user had not asked for. Rerun from scratch as r02 |
| L2 | GRPO, top-p 0.97 / top-k 64 + replay | `qwen3_6_35b_a3b_hetero_grpo_top_p` | r02 | [nttqm51w](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/nttqm51w) | `fc-01M45Q7NPNEDSA5ZTE3ABEJQCN` | 2026-10-05 09:38 | G (deployed from a clean worktree at `aa6e15a`) | rerun of r01 from scratch at the user's request. Collapsed: repeated format errors 21% at step 60 and 72% by step 69, and from step ~76 about half of each batch aborted (lost turns and agent connection errors), leaving the trainer rollout-bound (wait ratio 0.7–0.8). Stopped at 07:53 (2026-10-06) at the user's request at v~84, 2 h before its 24 h limit, and not resumed; last durable save step 79 (evals at steps 60 and 80 run separately) |
| L2 | SC+MIS, top-p 0.97 / top-k 64 + replay | `qwen3_6_35b_a3b_hetero_score_centering_mis_top_p` | r01 | [g0c583mu](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/g0c583mu) (attempt 1, v0-v~101), [fl29kozh](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/fl29kozh) (attempt 2, from v100), [pxajqlef](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/pxajqlef) (attempt 3, from v100), [srxgyliz](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/srxgyliz) (attempt 4, from v170), [qv8y7ue2](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/qv8y7ue2) (attempt 5, Modal's retry, from v190) | `fc-01M45JT2S61H0K7D81X81ZAD26` (attempt 1), `fc-01M483CJMYCVPPJVTFGB2DSV3H` (attempt 2, from step 99; taken over at 07:55 on 2026-10-06, 32 min before its 24 h limit, once the step-99 save was durable: EXPORT_OK, SAVE_OK and resolve_resume_point = 99; W&B id pending) | 2026-10-05 07:49 | G | launched from scratch with GRPO top-p r01; the candidate final recipe. Its RTX PRO 6000 pool waited for capacity until 08:27, and the trainer spawned at 08:27. After the 23:00 Modal outage its H100 BF16 pool fell from about 90 to 40–65 samples per step, its engines saturated (47 of 48 running requests, KV cache 85–98% full, prefix cache evicted) and that pool's TimeExceeded share rose to 28% by step 84; at the user's request the pool was live-resized 4 → 5 engines (min=max, `update_autoscaler`) at 03:36 (2026-10-06), so the fleet differs from the recipe from step ~85. After the resize that pool's TimeExceeded share fell from 28% (step 84) to 2% and 1% (steps 89–90), and its samples per step recovered from 37–64 to 84–97. Attempt 2's trainer was preempted at 09:33 (2026-10-06), at v102; Modal's retry resumes from the step-99 save again (attempt 3) ; attempt 3 (trainer call `fc-01M483CJMYCVPPJVTFGB2DSV3H`) reached step 178 but its step-180 save could not be made durable before its 24 h limit, so at 09:21 (2026-10-07) the run was resumed from the step-170 save (iteration 169; preflight READY) as attempt 4, trainer call `fc-01M4ATN4XRAZ7V2AZ54J7CZTS9`, W&B [srxgyliz](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/srxgyliz) (first step logged 10:32); steps 170-178 are retrained. Attempt 4's step-190 save (iteration 189) stalled on host 10.100.0.3 (rank shards 18, 19 missing; SAVE_STUCK 17:44); recovered by hand under the user's standing OK (Oct 2 procedure): the two shards uploaded 17:45-17:47 from that host's local copy, Megatron plan complete and both byte-identical (VERIFY_OK 17:50), then the head's export markers and tracker 189 at 17:50:51 after upload_markers re-checked ranks, dataset state and export indexes. Attempt 4 failed at 19:35:57 (2026-10-07) right after step 197's weight update (v198 published): train_async.py exited on FileNotFoundError opening /stitch/r01/events/attempt-10519e85c308/rollout_executor.jsonl for append (the RolloutExecutor's event logger), though the directory and file are on the Volume; a transient Volume-mount fault during an hour of Modal API InternalErrors, not a preemption or a training error. Modal's retry (same call, attempt 5, W&B qv8y7ue2 created 19:38) resumes from the step-190 save, retraining 190-197; the user was told and nothing was touched |
| L1 | GRPO | `qwen3_6_35b_a3b_b200_nvfp4_grpo` | r02 | [gvo1ldkg](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/gvo1ldkg) | `fc-01M4091PER6VVB1CVBSDGSBSKP` | 2026-10-03 06:57 | C | stopped at v131 (06:34, 2026-10-04) after its step-130 save completed; not resumed |
| L1 | IcePop | `qwen3_6_35b_a3b_b200_nvfp4_icepop` | r02 | [quwkb52d](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/quwkb52d) (attempt 1, v0-v102), [ke8q6lje](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/ke8q6lje) (attempt 2, from v100) (not found on W&B as of 2026-10-05 10:40), [uiz8nbyf](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/uiz8nbyf) (attempt 3, from v100 after a preemption) | `fc-01M4090XWSFKYTS0KK8NDCVT3W` (attempt 1); `fc-01M42W8B46V98DX06H366V8BGR` queued on AWS and never ran; `fc-01M42YPP65QFXV73T9VQ23Q9G3` (attempts 2 and 3: preempted at v102 on 10:01, 2026-10-04; Modal's retry resumed from step 100 at ~10:10) | 2026-10-03 06:57 | C, then F from v100 | attempt 1 hit the 24 h limit at v102 (07:01, 2026-10-04); resumed from step 100 on code F (07:53) after an AWS-pinned attempt sat queued; stopped at 06:48 (2026-10-05) at the user's request, at v159 (last durable save: step 150), when the plan narrowed to L2 arms plus one L0 run of the final recipe |
| L1 | SC | `qwen3_6_35b_a3b_b200_nvfp4_score_centering` | | | | | | not planned: the study runs SC only on L2 (user, 2026-10-04) |
| L1 | SC+MIS | `qwen3_6_35b_a3b_b200_nvfp4_score_centering_mis` | | | | | | not planned: the study runs SC only on L2 (user, 2026-10-04) |
| L0 | GRPO | `qwen3_6_35b_a3b_b200_bf16_grpo` | r02 | [7mpv4qdb](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/7mpv4qdb) | `fc-01M4099CZTA1TY6ZS51YJY4B90` | 2026-10-03 06:57 | C | stopped at v148 (06:39, 2026-10-04), 25 min before its 24 h attempt limit; last complete save step 140; not resumed |
| L0 | IcePop | `qwen3_6_35b_a3b_b200_bf16_icepop` | r04 | [ghnfty2y](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/ghnfty2y), [ylsplqu9](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/ylsplqu9) | `fc-01M42YQXTRXZVRNR8NS6QPEKJ5` (attempt 1 crashed at ~v17, 12:04 2026-10-04: `CUDA error: misaligned address` on rank 28 in the loss step, not a preemption; Modal's retry resumed from step 10) | 2026-10-04 07:53 | F | stopped at 06:48 (2026-10-05) at the user's request, at v86 (last durable save: step 80), when the plan narrowed to L2 arms plus one L0 run of the final recipe |
| L0 | SC | `qwen3_6_35b_a3b_b200_bf16_score_centering` | | | | | | not planned: the study runs SC only on L2 (user, 2026-10-04) |
| L0 | SC+MIS | `qwen3_6_35b_a3b_b200_bf16_score_centering_mis` | | | | | | not planned: the study runs SC only on L2 (user, 2026-10-04) |
| L0 | SC+MIS, top-p 0.97 / top-k 64 + replay | `qwen3_6_35b_a3b_b200_bf16_score_centering_mis_top_p` | r01 | [mw9z6ksn](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/mw9z6ksn) (attempt 1, v0-v~89), [pnsekp4q](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/pnsekp4q) (attempt 2, from v90), [elweshcy](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/elweshcy) (attempt 3, from v160; preempted before logging a step), [8mjqxo6r](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/8mjqxo6r) (attempt 4, Modal's retry, from v160) | `fc-01M45Q573X6MBNSM9T50S37DDN` (attempt 1), `fc-01M4834R2PZ6BT1HJ8MP1ZYRA3` (attempt 2), `fc-01M4APZMY27WPCPQ1TYHETYX5T` (attempt 3, from step 159) | 2026-10-05 09:40 | G (deployed from a clean worktree at `aa6e15a`) | the final recipe on homogeneous hardware, launched from scratch at the user's request. Attempt 1 was taken over at 07:51 (2026-10-06), 2 h before its 24 h limit, right after its step-89 save was durable (EXPORT_OK, SAVE_OK, and the trainer's own resolve_resume_point = iteration 89), with `launch --resume-from r01` at the user's request; attempt 2 resumes from step 89 on the same pool and code (W&B id pending). Attempt 2 hit the 24 h function timeout at 07:53 (2026-10-07) at v163, with no Modal retry; its last durable save was step 159 (EXPORT_OK 06:48, SAVE_OK 07:13, preflight READY). At the user's OK it was resumed with `launch --resume-from r01` at 08:03 on the same pool and code (configs, launcher, resume and app unchanged since `aa6e15a`); attempt 3's trainer spawned at 08:17 (W&B elweshcy). Its first batch took 68 min (rollout 160 collected 09:50); a trainer container was preempted at 10:01:29, ~10 min into training that step, so no step was logged. Modal's retry (same trainer call, attempt 4, W&B 8mjqxo6r created 10:07) resumes from the same step-160 save; the user was told and nothing was touched. Attempt 4's step-170 save (iteration 169) stalled like SC+MIS r02's: the head host's (10.100.0.1) upload hung (rank shards 2, 6 and part of the export missing; SAVE_STUCK 15:43). Recovered by hand under the user's standing OK for stuck saves (Oct 2 procedure): 65 files uploaded 15:44-15:49, Megatron plan and export indexes complete, all 65 byte-identical (VERIFY_OK 15:54), markers then tracker 169 at 15:55:08. Attempt 4's step-180 save (iteration 179) was slow, not lost: SAVE_STUCK at 21:15 (rank shards 0, 4, 5, 7 missing, all on the head host 10.100.0.1) came as the trainer's uploader timed out its first attempt after 3,850 s (21:14:47). A manual upload of the same local files ran 21:16-21:24 alongside its retry, which logged "270 data files (260.0 GB) durable in 4300s" at 21:22:17 and deleted the local copies; the manual upload was stopped. All 32 rank shards on the Volume at the exact sizes the Megatron plan needs (497.8 GB); the trainer publishes the markers and tracker at its next step, so no markers were uploaded by hand |

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

## SWE-bench Pro V2 eval

Each point is one checkpoint's BF16 export, scored on 641 of V2's 642 tasks with 8 samples
per task. The pool is 8 B300 engines with full-vocabulary sampling, and each sample is graded
in a fresh Sandbox. The eval Volume `stitch-swebench-pro-eval` is the ledger:
`swebench-pro-scale-v2/<recipe>/<run>/v000120/bf16/` (base: `swebench-pro-scale-v2/base/bf16/`) holds `metrics.json`,
`samples.jsonl` (with each verifier's output tail), `patches.jsonl`, `manifest.json` (code,
checkpoint, pool, sampler) and the dump. The launcher skips a point whose metrics are
complete. The 8-sample points ran on code `e3d1b94` and the 4-sample points on `69a75d6`,
both clean trees. `aborts.log` is copied in by hand
from the app's Modal logs: why each rerun attempt failed. A request that runs past the session
server's 1 h deadline counts as an infra failure, so it is rerun and not scored.

From code `55c2193` (2026-10-05), the session servers give up on a request after 600 s, and
the agent resends that turn, up to three times, inside the same episode. Only an episode
whose turn fails all three tries reruns from scratch. Each sample records its attempts'
abort reasons (`aborts` in `samples.jsonl` and `infra_failures.jsonl`), so `aborts.log` is no
longer needed, and the manifest records the SGLang commit and the request policy. Resending
one turn resamples only the lost reply, where an episode rerun resamples the whole
trajectory; so later points carry less rerun bias than the estimates below. On these points
`model_request_count` and the per-request durations include failed attempts.

From code `ea5e1ea` (2026-10-05 19:00), an eval app name that would pass Modal's 64
characters abbreviates score-centering as sc; every name that fit before is unchanged. The
final recipe on all-B200 BF16 needs it at every step, and the mixed-pool final recipe from
step 100. Its step-20 eval was refused at 15:59 for the 68-character name, before deploying.

From code `dc573c6` (branch `eval/turn-time-limit`, 2026-10-06 05:50; launched by
`scratchpad/eval_queue_runner3.py` from the worktree `eval_wt_rule`), the eval scores with a
**300 s turn time limit**: a turn still generating at 300 s fails its episode (exit status
`TurnTimeLimit`, reward 0), with no resend or rerun; other request failures are still resent up
to 3 times. Reason: the GPU pools' Flash path drops any response slower than ~343–356 s
(trace of GRPO + top-p step 60), so a resend could never complete such a turn, and resending
rejected exactly the long, mostly degenerate turns (an upward bias). Completed turns: 99.9th
percentile 100–190 s, 16 per million between 300 and 356 s, over 45 points. Every earlier point
is rescored by the same rule in the figure pipeline (`figures/eval_source.over_turn_time_limit`:
a final-attempt request ≥ 300 s, or a deadline abort); points from before `55c2193` keep no
abort reasons, so only their final attempt can be checked. The registry rows below keep the
recorded pass@1; under the rule, the cited points that move are SC+MIS r02 step 120 0.805 →
0.784, IcePop r06 step 180 0.734 → 0.722, IcePop 120/160 0.811/0.777 → 0.810/0.776, SC+MIS +
top-p (mixed) 20/60/80 0.749/0.783/0.803 → 0.747/0.782/0.802, SC+MIS + top-p (all-B200)
20/40/60 0.753/0.772/0.787 → 0.752/0.771/0.785, GRPO + top-p step 40 0.794 → 0.793, GRPO
all-B200 step 80 0.779 → 0.778; the rest move by ≤ 0.001. Full comparison:
`scratchpad/figs_rule/eval/points.csv` vs `scratchpad/figs/eval/points.csv`, and the exposure
table `scratchpad/eval_loss_impact.csv`.

Open question (2026-10-06 08:35): L0 SC+MIS + top-p step 80 (code `92baab7`) lost 413
requests at the 1800 s deadline in 371 episodes, 248 of which later passed, so the rule moves
it 0.795 → 0.698 (HARD-51 0.392 → 0.324). Its eval pool was healthy throughout (8 replicas,
56–78 tok/s per request, KV usage ≤ 0.62, no queue), the losses were steady 05:20–07:30, the
longest completed turn in those episodes has median 2.8K tokens, and all-B200 training hits
the 32K turn cap in only 0.03–0.07% of episodes; SC+MIS r02 step 140 (rule code, same hours)
hit the limit once. So the lost turns are not shown to be long generations; the rule may be
zeroing transport losses here (and in SC+MIS r02 step 120, 76 episodes, 0.805 → 0.784).
Undecided: rescored value vs a rerun. The same run's step 100, on the rule code (12:04–15:29),
had 5 episodes over the limit, which points to a fault in the step-80 eval run, not the checkpoint.

Extra off-cadence points (user, 2026-10-06 16:12, temporary): SC+MIS + top-p (mixed and all-B200)
steps 70/90/110 and SC+MIS r02 steps 70/90/110/130/150, all on the rule code `dc573c6`, queued
16:13 via `eval_queue3.txt` (runner3, at most 8 points at once).
L2 SC+MIS + top-p step 110 (first try, app `ap-kXderCzOBqe80DSUyHMlM1`) was stopped at 16:31: one
B300 replica's engine hung at 16:29:29 (no error logged) and the sidecar watchdog replaced it at
16:30:54. Its ~35 in-flight turns would have failed their episodes at the 300 s turn limit, so
the point was requeued from scratch. The rule cannot tell a hung replica from a slow turn.
`fl29kozh` (SC+MIS + top-p attempt 2) is no longer found on W&B (16:30); its rows are cached
and every step it logged was replaced by attempt 3.
L0 SC+MIS + top-p step 80 (user, 2026-10-06 16:40): dropped from Fig 4 and rerun on the rule code.
Its first result (`92baab7`) was archived to
`archive/swebench-pro-scale-v2/qwen3_6_35b_a3b_b200_bf16_score_centering_mis_top_p/r01/v000080-lost-requests-20261006`
(byte-identical copy verified) and removed from the live path. The rerun on the rule code `dc573c6`
finished 22:41 (2026-10-06) and its numbers replace the first result in the row above (pass@1 0.795 both times;
HARD-51 0.397 vs 0.392; 9 turn-limit exits). Extra off-cadence points go into Fig 4.
L2 GRPO r07 step 100 is left out of Fig 4 by an editorial choice (user, 2026-10-06 19:27), not as a
fault: `figures/eval_figures.HIDDEN_POINTS`. It stays on the eval Volume and in the rows above.
L2 GRPO r02 (top-p 0.97 / top-k 64 + replay) step 20 is also left out of Fig 4 by an editorial choice (user,
2026-10-06 ~21:45), in the same table; it stays on the eval Volume and in the rows above.
L2 SC+MIS + top-p step 90 (first try, 16:13-19:09) was archived to `.../r01/v000090-turn-limit-losses-20261006`
and is rerunning: 367 of 2564 episodes failed on the 300 s turn limit (neighbours 6-9), steady from 16:30
to 18:00 on its own eval pool; submitted episodes passed at a normal 0.814, so pass@1 0.697 reflects the
pool, not the checkpoint.

From 2026-10-06 19:18 (user), eval scores also fail an episode that ended without the agent's submit
command (`figures/eval_source.SCORING_RULE` = turn_time_limit_300s+require_submission), retroactively
for every point. Registry rows keep the recorded pass@1. The final recipe moves by <= 0.003; collapsing
points fall: IcePop 160/170/180 0.776/0.749/0.734 -> 0.641/0.480/0.254, SC 60 0.402 -> 0.250, GRPO 40
0.414 -> 0.326, GRPO all-B200 120 0.442 -> 0.299, SC+MIS 80 0.776 -> 0.749.

From code `92baab7` (2026-10-06 01:10), the session servers give up on a request after
1800 s instead of 600 s. A full 32K-token turn at about 45 tok/s per request takes about
12 min, and degenerating checkpoints write such turns: on 600 s, L2 GRPO + top-p r02 step
60 aborted 38 episodes in 4 min and L2 IcePop r06 step 180 a few, so both were stopped at
01:03 and rerun on `92baab7`. Earlier points on `55c2193`/`ea5e1ea` resent 0–8 requests
each at the 600 s deadline, inside the episode and never scored as failures.

L2 GRPO + top-p r02 step 40 was rerun on `ea5e1ea` (600 s deadline; 00:21–02:59, 2026-10-06):
0 reruns, 0 infra failures, 227K model requests with p99.9 104 s and 2 at the deadline. The first run (21:29–23:14, 2026-10-05) went
through the Modal outage with 76 samples scored as infra failures; it is archived on the eval
Volume under `archive/outage-20261005/` and is not used. Its step 60 was stopped twice (01:03 on
600 s, 02:54 on 1800 s): on that degenerating checkpoint, model requests hang from the start of
the eval (about 650 deadline hits in 2 h on healthy engines, against 13–18 on the other points
running then), cause not yet found. A third run (03:03, 2026-10-06, `92baab7`) was stopped at
03:50 for the same reason: 97 deadline hits in 03:40–03:49, so about 100 requests sent in its
first 10 min of serving never returned, while its 8 engines held about 160 requests between
them (no queue, KV cache 15–35%) against 309 episodes waiting on the model, and no engine had
been replaced. It is not a gateway-wide event: L2 IcePop r06 steps 150 and 170 started the
same minute with no hits, L2 SC+MIS top-p r01 step 80 (also top-p) had none, and L2 SC+MIS r02
step 120 and L2 IcePop r06 step 180 lost requests at a steady 1–14 per 10 min.

**Traced (2026-10-06):** the requests are lost in Modal's Flash path between the pool's
sidecar and the session server, once a response takes longer than about 350 s. The traced
rerun was a smoke of step 60, 100 tasks × 4 samples, into `smoke-100x4`, on 8 B300 from
04:13 to 04:53. It ran `92baab7` plus uncommitted tracing behind `STITCH_TRACE_REQUESTS=1`,
so each request carried one rid through the session server, the sidecar and SGLang.
- **What returned:** of 29,669 requests, 29,555 came back, every one that SGLang finished
  within 342.5 s.
- **What was lost:** all 114 that ran past about 355 s were lost.
  - In 102, the sidecar returned SGLang's 200 at 355.7–477.6 s, but the session server
    never received it.
  - In 12, the gateway closed the sidecar's connection at 358–443 s, mid-generation.
  - None of them returned an error. The session server waited until its 1800 s deadline:
    55 hit it before the stop, and the 1800.0 s cancels match the lost rids.
- **Not the session server:** a direct request to the same gateway was lost the same way;
  the sidecar answered at 377 s with 191 KB. The session server itself was healthy (event
  loop lag ≤ 1.5 s), and hop latency was ≤ 1 s at p99.
- **Why this checkpoint:** all 114 were first turns sent in the first 10 min, one per
  session. This checkpoint writes many maximum-length first turns that take 6–8 min;
  healthy checkpoints rarely write turns that long, hence their steady trickle.
- **Not Flash in general:** the same silent 400–420 s request came back through CPU-only
  Flash test apps on AWS, GCP and OCI, with and without `kv_aware_routing`, with 2 MB
  responses, and when dripping a byte every 20 s. So the cutoff belongs to the GPU pools'
  path. The window, 343–356 s, contains AWS NLB's 350 s idle timeout; that is a lead, not
  verified.
- **For Modal's Flash team** (gateway `x-request-id`, container, sidecar time received):
  - `63c66778-f8aa-46ab-8c6d-abc0014064c3`, ta-01M47PN0CA271FKA7YWNVKBRJR, 04:21:37
  - `f3c316d8-8866-4456-bc12-151abbd7116d`, ta-01M47PN0C0QPEKZWX2123FZ9ER, 04:21:42
  - `c4b41d84-f852-4a9f-bd5c-84d4a970578a`, ta-01M47PN0CMPH2J771FMA55SDWR, 04:22:16
    (connection closed at 399 s)
  - `6178da4e-a910-45bb-bdce-747eb95cb179`, ta-01M47PN0CMPEFXGMPSR2VM2RFR, 04:43:35 (the
    direct probe)

L2 SC+MIS r02 step 200 (`dc573c6`, 300 s turn limit) was stopped at 02:58 (2026-10-07), 12 min
into its episodes, and every-20-step queuing was paused at 03:15. It logged 20 turn-limit
failures in 02:51–02:58, against 1, 8 and 2 over the whole of steps 140, 160 and 180. The first
model request (02:46:53) never returned, and the oldest model request aged 30 s per 30 s up to
the limit, where healthy points hold it at 25–45 s for their first 5 min. Its 8 engines were
healthy (31–37 running, no queue, KV cache about 50%, 44–56 tok/s per request), and there were
no 502s, connection errors or disconnects. Training through step 202 shows no break (response
length 82–94K, repetition 1–3%, repeated format errors 10–15%, reward about 0.48). The signature
matches GRPO + top-p step 60 above, whose cause was maximum-length first turns. Nothing was
written to its eval directory beyond the manifest and task list. The rerun of the same
checkpoint and code (started 04:36) had no turn-limit failures in its first 25 min of
episodes, on healthy engines (1,300–1,500 tok/s each, about 45 tok/s per request, no queue).
So the first run's spike came from that eval pool, not from the checkpoint.

L2 SC+MIS r01 (top-p) step 160 (`dc573c6`) was stopped at 05:00 (2026-10-07), 30 min in.
Its turn-limit failures were 17 and rising, reaching 3–4 per minute after 04:54, against 1–2
per 10 min at the same stage of step 140. Every-20-step queuing was paused again at 05:01.

The engines were alive but saturated. Their KV cache was 74–98% full, and each re-prefilled
1–3M tokens in 3 min, so decode fell to about 12 tok/s per request. The final-recipe
checkpoints' contexts grow with training while the longest turn stays the same (p90 about
4.5K tokens), and turn-limit failures grow with the contexts:

| Run | Point | Median final context | Turns (median) | Turn-limit failures |
|---|---|---|---|---|
| Mixed | step 60 | 64K | 88 | 6 |
| Mixed | step 140 | 94K | 130 | 64 |
| All-B200 | step 60 | 69K | 94 | 14 |
| All-B200 | step 140 | 91K | 128 | 34 |

At 48 sessions per B300 engine, the 300 s limit increasingly measures eval load times context
length, not long turns.

Step 140 of both SC+MIS + top-p runs was re-evaluated at 16 × 24 (2026-10-07; the user asked for the 120 and 140 pairs too). The 8 × 48 points are archived under archive/swebench-pro-scale-v2/<experiment>/r01/v000140-8x48-20261007; all-B200 step 140 was 0.812 ± 0.012 / HARD-51 0.402 ± 0.052 at 8 × 48 (13 turn-limit fails) and is 0.824 ± 0.011 / 0.436 ± 0.051 at 16 × 24 (7). The mixed step-140 rerun hit an engine hang twice (08:31, 09:05: one engine's scheduler went silent at light load, the sidecar watchdog replaced it, its requests failed); rerun #3 (09:44-12:22) ran with SGLang's soft watchdog at 30 s (`EVAL_SOFT_WATCHDOG_TIMEOUT`, eval worktree, default off) so a repeat hang would log its batch and pool state; it finished with no stall (no post-startup watchdog dump, no failed health check, no engine lost), so the hang's cause is still unknown. The same hang then hit the all-B200 step-120 rerun (stopped 12:44 at 91%, 2342/2564): engine ta-01M4B0MTWKN7F53NVCVT2MXJJR logged nothing after a 12:42:26 prefill reusing a 219,648-token cached prefix at light load (5 running, KV 22%), and the sidecar watchdog replaced it at 12:43:38; so the hang is not specific to one checkpoint or pool. That point was relaunched at 12:46 with the soft watchdog on and finished at 15:26 with no stall; all-B200 step 120 was 0.818 ± 0.012 / HARD-51 0.441 ± 0.054 at 8 × 48 and is 0.828 ± 0.012 / 0.417 ± 0.056 at 16 × 24. Mixed step 120 (rerun 11:00-13:44 at 16 × 24, no soft watchdog, no hang) was 0.810 ± 0.012 / HARD-51 0.407 ± 0.058 at 8 × 48 and is 0.818 ± 0.012 / 0.436 ± 0.053. Mixed step 140 was 0.815 ± 0.012 / HARD-51 0.392 ± 0.052 at 8 × 48 and is 0.826 ± 0.012 / 0.426 ± 0.054 at 16 × 24. SC+MIS r02 step 220 at 8 × 48 was stopped at 10:19 (every engine's KV cache 97-99% full, 11 turn-limit fails in 4 min) and rerun at 16 × 24 (done 13:14, 2 turn-limit fails); its earlier points (to step 200) are 8 × 48.

From step 160 the SC+MIS + top-p evals run at 16 engines × 24 sessions (same 384 sessions, half the
load per engine; `eval_launch --sessions-per-engine`, uncommitted on the eval worktree, diff saved
beside each point's log). L2 step 160 at that setting: 1 turn-limit failure (step 140 at 8 × 48: 64),
pass@1 0.822 and HARD-51 0.466. Steps 120 and 140 of both pools ran at 8 × 48.

L0 SC+MIS + top-p r01 step 180 (2026-10-07/08): the auto run (21:57) died at 22:53, ~15% in,
when the local launcher's poll hit a Modal API "Deadline exceeded" and its cleanup stopped the
app (healthy engines). The launcher now retries transient poll errors (20 in a row; eval worktree
and main tree). Rerun #1 23:30-03:59, 16 × 24: 2,564/2,564. Its engines ran 27 requests past
200 s and none past 400 s (285K requests), so its ~40 turn-limit failures are real long turns of
this checkpoint (L0 step 160: 11; L2 step 180: 2), not lost requests: recorded pass@1 0.839,
0.822 under the 300 s rule (HARD-51 0.500 / 0.456). Task 336 (two samples) hit the limit on
every attempt. One episode's grading Sandbox (qutebrowser-ff1c025, sb-01M4CP633ZTVV25KNV25R3KVE5)
hung past the 1,200 s verifier cap for 49 min with the other 2,563 done; it was terminated by
hand at 03:31 and the sample took the normal infra retry. L2 step 200 (00:43-03:55, 16 × 24):
recorded pass@1 0.832, 0.828 under the rule, HARD-51 0.446; 3 samples rerun.

| Point | samples/task | pass@1 ± se | pass@2 | pass@4 | HARD-51 pass@1 ± se | HARD-51 pass@2 | HARD-51 pass@4 | ended by RepeatedFormatError | samples rerun / infra failures |
|---|---|---|---|---|---|---|---|---|---|
| Base | 8 | 0.744 ± 0.013 | 0.837 | 0.894 | 0.287 ± 0.048 | 0.389 | 0.501 | 0% | 63 / 0 |
| L0 GRPO r02, step 40 | 4 | 0.765 ± 0.013 | 0.855 | 0.910 | 0.363 ± 0.053 | 0.480 | 0.588 | 0% | 10 / 0 |
| L0 GRPO r02, step 80 | 8 | 0.779 ± 0.012 | 0.863 | 0.912 | 0.321 ± 0.049 | 0.432 | 0.538 | 0% | 70 / 0 |
| L0 GRPO r02, step 120 | 8 | 0.442 ± 0.016 | 0.544 | 0.624 | 0.047 ± 0.019 | 0.076 | 0.109 | 65% | 7 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 20 | 4 | 0.753 ± 0.014 | 0.843 | 0.902 | 0.333 ± 0.052 | 0.448 | 0.549 | 0% | 4 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 40 | 4 | 0.772 ± 0.013 | 0.859 | 0.910 | 0.328 ± 0.052 | 0.444 | 0.549 | 0% | 4 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 60 | 4 | 0.787 ± 0.013 | 0.873 | 0.924 | 0.368 ± 0.054 | 0.480 | 0.588 | 0% | 1 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 70 | 4 | 0.764 ± 0.013 | 0.861 | 0.913 | 0.319 ± 0.048 | 0.458 | 0.588 | 0% | 9 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 80 | 4 | 0.795 ± 0.013 | 0.875 | 0.916 | 0.397 ± 0.053 | 0.529 | 0.608 | 0% | 9 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 90 | 4 | 0.802 ± 0.012 | 0.887 | 0.945 | 0.402 ± 0.053 | 0.533 | 0.686 | 0% | 6 / 1 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 100 | 4 | 0.807 ± 0.012 | 0.884 | 0.925 | 0.397 ± 0.053 | 0.526 | 0.627 | 0% | 8 / 1 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 110 | 4 | 0.816 ± 0.012 | 0.895 | 0.936 | 0.368 ± 0.049 | 0.520 | 0.647 | 0% | 4 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 120 | 4 | 0.828 ± 0.012 | 0.901 | 0.939 | 0.417 ± 0.056 | 0.533 | 0.647 | 0% | 7 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 140 | 4 | 0.824 ± 0.011 | 0.909 | 0.952 | 0.436 ± 0.051 | 0.588 | 0.706 | 0% | 7 / 1 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 160 | 4 | 0.846 ± 0.011 | 0.914 | 0.958 | 0.417 ± 0.052 | 0.562 | 0.706 | 0% | 1 / 0 |
| L0 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 180 | 4 | 0.839 ± 0.011 | 0.919 | 0.955 | 0.500 ± 0.054 | 0.641 | 0.745 | 0% | 10 / 0 |
| L1 GRPO r02, step 40 | 4 | 0.767 ± 0.013 | 0.854 | 0.906 | 0.324 ± 0.052 | 0.438 | 0.569 | 0% | 14 / 0 |
| L1 GRPO r02, step 80 | 4 | 0.782 ± 0.013 | 0.857 | 0.903 | 0.363 ± 0.057 | 0.451 | 0.549 | 0% | 7 / 0 |
| L1 GRPO r02, step 120 | 8 | 0.411 ± 0.015 | 0.514 | 0.595 | 0.037 ± 0.020 | 0.054 | 0.067 | 57% | 124 / 0 |
| L2 GRPO r07, step 20 | 4 | 0.738 ± 0.013 | 0.847 | 0.906 | 0.333 ± 0.050 | 0.464 | 0.608 | 6% | 1 / 0 |
| L2 GRPO r07, step 40 | 4 | 0.414 ± 0.013 | 0.594 | 0.754 | 0.127 ± 0.028 | 0.222 | 0.373 | 62% | 84 / 0 |
| L2 GRPO r07, step 60 | 4 | 0.432 ± 0.013 | 0.617 | 0.763 | 0.132 ± 0.028 | 0.232 | 0.353 | 57% | 2 / 0 |
| L2 GRPO r07, step 80 | 4 | 0.289 ± 0.012 | 0.440 | 0.598 | 0.093 ± 0.025 | 0.163 | 0.294 | 76% | 9 / 0 |
| L2 GRPO r07, step 100 | 4 | 0.578 ± 0.014 | 0.741 | 0.855 | 0.245 ± 0.042 | 0.376 | 0.549 | 31% | 1 / 0 |
| L2 GRPO r07, step 120 | 8 | 0.256 ± 0.011 | 0.389 | 0.531 | 0.042 ± 0.017 | 0.071 | 0.111 | 76% | 27 / 0 |
| L2 GRPO r02 (top-p 0.97 / top-k 64 + replay), step 20 | 4 | 0.773 ± 0.013 | 0.861 | 0.920 | 0.402 ± 0.054 | 0.529 | 0.667 | 0% | 0 / 0 |
| L2 GRPO r02 (top-p 0.97 / top-k 64 + replay), step 40 | 4 | 0.794 ± 0.013 | 0.869 | 0.911 | 0.328 ± 0.053 | 0.435 | 0.549 | 0% | 0 / 0 |
| L2 GRPO r02 (top-p 0.97 / top-k 64 + replay), step 60 | 4 | 0.515 ± 0.014 | 0.680 | 0.814 | 0.201 ± 0.039 | 0.314 | 0.451 | 3% | 1 / 0 |
| L2 GRPO r02 (top-p 0.97 / top-k 64 + replay), step 80 | 4 | 0.483 ± 0.013 | 0.682 | 0.827 | 0.240 ± 0.044 | 0.353 | 0.431 | 0% | 59 / 0 |
| L2 SC r11 (full vocabulary), step 20 | 4 | 0.761 ± 0.013 | 0.853 | 0.905 | 0.348 ± 0.055 | 0.448 | 0.529 | 0% | 2 / 0 |
| L2 SC r11 (full vocabulary), step 40 | 4 | 0.725 ± 0.013 | 0.838 | 0.902 | 0.343 ± 0.050 | 0.474 | 0.588 | 12% | 12 / 0 |
| L2 SC r11 (full vocabulary), step 60 | 4 | 0.402 ± 0.014 | 0.560 | 0.704 | 0.123 ± 0.032 | 0.199 | 0.294 | 73% | 6 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 20 | 4 | 0.736 ± 0.014 | 0.839 | 0.899 | 0.299 ± 0.054 | 0.386 | 0.471 | 0% | 1 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 40 | 4 | 0.749 ± 0.013 | 0.845 | 0.897 | 0.348 ± 0.052 | 0.471 | 0.588 | 0% | 7 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 60 | 4 | 0.785 ± 0.013 | 0.866 | 0.914 | 0.338 ± 0.053 | 0.448 | 0.549 | 0% | 1 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 70 | 4 | 0.779 ± 0.013 | 0.865 | 0.910 | 0.348 ± 0.051 | 0.477 | 0.627 | 2% | 1 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 80 | 4 | 0.776 ± 0.013 | 0.865 | 0.910 | 0.353 ± 0.052 | 0.480 | 0.588 | 8% | 5 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 100 | 4 | 0.789 ± 0.012 | 0.879 | 0.928 | 0.377 ± 0.049 | 0.533 | 0.686 | 7% | 3 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 120 | 4 | 0.805 ± 0.012 | 0.887 | 0.936 | 0.382 ± 0.051 | 0.523 | 0.667 | 1% | 1 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 140 | 4 | 0.814 ± 0.012 | 0.891 | 0.931 | 0.402 ± 0.052 | 0.539 | 0.647 | 1% | 3 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 160 | 4 | 0.825 ± 0.012 | 0.898 | 0.936 | 0.441 ± 0.055 | 0.569 | 0.686 | 1% | 2 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 180 | 4 | 0.847 ± 0.011 | 0.909 | 0.942 | 0.461 ± 0.055 | 0.588 | 0.667 | 1% | 3 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 200 | 4 | 0.839 ± 0.012 | 0.906 | 0.942 | 0.422 ± 0.054 | 0.556 | 0.686 | 1% | 3 / 0 |
| L2 SC+MIS r02 (full vocabulary), step 220 | 4 | 0.837 ± 0.011 | 0.907 | 0.942 | 0.417 ± 0.052 | 0.559 | 0.647 | 1% | 2 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 20 | 4 | 0.749 ± 0.014 | 0.835 | 0.888 | 0.319 ± 0.051 | 0.435 | 0.569 | 0% | 0 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 40 | 4 | 0.765 ± 0.013 | 0.857 | 0.911 | 0.333 ± 0.052 | 0.451 | 0.588 | 0% | 1 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 60 | 4 | 0.783 ± 0.013 | 0.866 | 0.919 | 0.382 ± 0.053 | 0.507 | 0.608 | 0% | 3 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 70 | 4 | 0.789 ± 0.013 | 0.871 | 0.924 | 0.402 ± 0.054 | 0.529 | 0.667 | 0% | 2 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 80 | 4 | 0.803 ± 0.012 | 0.887 | 0.941 | 0.387 ± 0.049 | 0.546 | 0.686 | 0% | 2 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 100 | 4 | 0.807 ± 0.012 | 0.886 | 0.931 | 0.368 ± 0.053 | 0.490 | 0.588 | 0% | 1 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 110 | 4 | 0.814 ± 0.012 | 0.888 | 0.931 | 0.412 ± 0.055 | 0.536 | 0.627 | 0% | 2 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 120 | 4 | 0.818 ± 0.012 | 0.895 | 0.941 | 0.436 ± 0.053 | 0.575 | 0.686 | 0% | 2 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 140 | 4 | 0.826 ± 0.012 | 0.897 | 0.944 | 0.426 ± 0.054 | 0.556 | 0.667 | 0% | 2 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 160 | 4 | 0.822 ± 0.012 | 0.903 | 0.950 | 0.466 ± 0.051 | 0.624 | 0.765 | 0% | 3 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 180 | 4 | 0.831 ± 0.012 | 0.900 | 0.942 | 0.485 ± 0.053 | 0.634 | 0.765 | 0% | 3 / 0 |
| L2 SC+MIS r01 (top-p 0.97 / top-k 64 + replay), step 200 | 4 | 0.832 ± 0.012 | 0.904 | 0.944 | 0.446 ± 0.055 | 0.572 | 0.667 | 0% | 3 / 0 |
| L1 IcePop r02, step 40 | 4 | 0.779 ± 0.013 | 0.864 | 0.928 | 0.387 ± 0.052 | 0.523 | 0.667 | 0% | 12 / 0 |
| L1 IcePop r02, step 80 | 4 | 0.793 ± 0.013 | 0.878 | 0.922 | 0.392 ± 0.054 | 0.516 | 0.627 | 0% | 233 / 0 |
| L1 IcePop r02, step 120 | 4 | 0.798 ± 0.012 | 0.885 | 0.933 | 0.377 ± 0.050 | 0.523 | 0.627 | 0% | 84 / 0 |
| L2 IcePop r06, step 20 | 4 | 0.749 ± 0.014 | 0.841 | 0.892 | 0.309 ± 0.054 | 0.402 | 0.490 | 0% | 3 / 0 |
| L2 IcePop r06, step 40 | 4 | 0.780 ± 0.013 | 0.872 | 0.925 | 0.348 ± 0.049 | 0.490 | 0.627 | 0% | 14 / 0 |
| L2 IcePop r06, step 60 | 4 | 0.776 ± 0.013 | 0.868 | 0.919 | 0.412 ± 0.056 | 0.523 | 0.627 | 10% | 1 / 0 |
| L2 IcePop r06, step 80 | 4 | 0.795 ± 0.013 | 0.868 | 0.911 | 0.377 ± 0.051 | 0.516 | 0.647 | 4% | 8 / 0 |
| L2 IcePop r06, step 100 | 8 | 0.802 ± 0.012 | 0.874 | 0.917 | 0.392 ± 0.051 | 0.515 | 0.613 | 4% | 148 / 0 |
| L2 IcePop r06, step 120 | 4 | 0.811 ± 0.013 | 0.880 | 0.916 | 0.377 ± 0.054 | 0.493 | 0.588 | 0% | 14 / 0 |
| L2 IcePop r06, step 140 | 4 | 0.807 ± 0.013 | 0.878 | 0.922 | 0.333 ± 0.050 | 0.461 | 0.608 | 2% | 2 / 0 |
| L2 IcePop r06, step 150 | 4 | 0.798 ± 0.013 | 0.873 | 0.924 | 0.353 ± 0.056 | 0.448 | 0.549 | 6% | 1 / 0 |
| L2 IcePop r06, step 160 | 4 | 0.777 ± 0.013 | 0.859 | 0.902 | 0.319 ± 0.050 | 0.441 | 0.549 | 22% | 0 / 0 |
| L2 IcePop r06, step 170 | 4 | 0.749 ± 0.013 | 0.855 | 0.919 | 0.328 ± 0.052 | 0.444 | 0.549 | 40% | 0 / 0 |
| L2 IcePop r06, step 180 | 4 | 0.734 ± 0.013 | 0.847 | 0.908 | 0.294 ± 0.050 | 0.402 | 0.510 | 68% | 1 / 0 |
| L2 SC r03 (top-p 0.97 / top-k 64 sampling, frozen config), step 20 | 4 | 0.758 ± 0.013 | 0.852 | 0.910 | 0.319 ± 0.048 | 0.454 | 0.588 | 0% | 0 / 0 |
| L2 SC r03 (top-p 0.97 / top-k 64 sampling, frozen config), step 40 | 4 | 0.777 ± 0.013 | 0.865 | 0.913 | 0.363 ± 0.057 | 0.454 | 0.529 | 0% | 4 / 0 |
| L2 SC r03 (top-p 0.97 / top-k 64 sampling, frozen config), step 120 | 4 | 0.823 ± 0.012 | 0.900 | 0.947 | 0.397 ± 0.051 | 0.546 | 0.686 | 0% | 86 / 0 |

Grader re-check (2026-10-05): 100 stored patches from L2 IcePop r06 step 120 (50 passing,
50 failing, 99 tasks, 11 repositories) were re-graded in fresh Sandboxes through the eval's
own grading path (`grade_patch_in_fresh_sandbox`). All 100 verdicts matched, with no
timeouts or infrastructure retries, which puts the grader's flip rate below about 3% (95%
confidence) on the sampled tasks.

From 2026-10-04 ~17:00 points run 4 samples per task and report pass@1, pass@2 and pass@4. For the
8-sample points above, those columns use the same unbiased pass@k estimator.

L1 IcePop r02, step 80 had 233 reruns. One engine hung in its detokenizer and was replaced, and
162 samples had an attempt end on the 1 h deadline. Their reruns passed at 0.642. Scored 0
instead, pass@1 would be 0.752 (`deadline_rescore.py`), but that is the extreme case. Weighting
each task's samples by their model requests or wall time puts the rerun bias at +0.05 to
+0.08 pt, so 0.793 stands. On every other point the same estimate is within ±0.15 pt, except
L2 GRPO r07 step 40 at -0.35 to -0.40 pt: on that collapsed checkpoint the short episodes are
the ones that fail, so rerunning a long episode slightly understates pass@1.

Points with no result yet (2026-10-04):
- L1 IcePop r02, step 100: stopped at 15:16 after about 2,600 of 5,128 episodes. The driver
  container reached its 256 GiB memory limit, and Ray's OOM killer started killing agent workers.
  The 19 session servers grow with the tokens they serve (about 11 GiB each by then).
- L2 SC r03, step 120: its driver was preempted at 13:17 and the point restarted from scratch.
  It was stopped at 15:19, at about 1,500 episodes, because it was on course to hit the same memory
  limit at around 16:20.
- L1 GRPO r02, step 80: stopped at 16:04, at about 1,900 episodes. Its driver had reached 135 GiB
  and was growing about 60 GiB an hour, so it would have reached the limit before finishing
  (about 17:55).
- L2 GRPO r07, step 80: its driver was preempted at 16:45 with nearly every episode done (at least 5,048 of 5,128). The retry, which would rerun the point from scratch, got no L4 capacity for 10 minutes while the 8 B300 engines sat idle, so the point was stopped at 17:04.
- L2 IcePop r06, step 120: run at the user's request, then stopped at 17:07, at about 600 episodes. Its driver grew about 60 GiB an hour (107 GiB at 17:06) and would have reached the limit around 19:20, before finishing (about 20:20).

## Infrastructure smoke tests (not for analysis)

| Run | Recipe | App | Launched | Code | What it tests |
|---|---|---|---|---|---|
| smoke01 | `qwen3_6_35b_a3b_signal_smoke` (temporary, uncommitted) | `stitch-qwen36-signal-smoke-smoke01`, trainer `fc-01M49MEFRD71QCHQ8DW73W40SP` | 2026-10-06 ~22:10 (pools ready, trainer spawned) | branch `fix/training-signal` (on `dc573c6`), uncommitted, worktree `stitch-signal-fix` | the training-signal fixes end to end at small scale (user, 2026-10-06: smoke first, commit only if it works; no reruns of study runs): sidecar keep-alive through the router, fresh-Sandbox grading with the task start tree (both-ways grading on), the submission rule, the time-limit drop, per-pool abort logs. Final recipe (SC + MIS + top-p mask replay) on 2 B200 trainer nodes, 2 x H200 FP8 + 2 x B200 NVFP4 engines, 8 x 8 per step, 128 sessions, 6 steps, no saves; own volume, Sandbox app and W&B group `qwen36-signal-smoke` (W&B `8oiypmsj`). PASSED, finished 23:23: all 6 steps trained, every weight update applied on both pools within ~20 s, 0 aborted groups, fresh-grade disagreement 8% at step 0 then 0 (own Sandbox failing where the fresh grade passed), never-submitted 0-3.3% scored 0, fresh grading 11-24 s mean, every new metric present; the only aborts came from a Modal preemption of a router container. No turn passed 150 s on its own (max 94 s); the synthetic probe covered that. Apps stopped; fixes committed as `3e3fed4` on `fix/training-signal` (the smoke recipe itself is not committed) |
| mimo01 | `qwen3_6_35b_a3b_mimo_smoke` (temporary, uncommitted) | `stitch-qwen36-mimo-smoke-mimo01`, trainer `fc-01M4DA2TP90CDA7D31X2578NEN` | 2026-10-08 08:29 (pools ready, trainer spawned) | `491c790` plus the uncommitted MiMo leak fix (main tree) | the MiMo leak fix end to end at smoke01's scale (user, 2026-10-08: consolidate, then add the MiMo fix, heavily tested): pruned code tasks (`mimo-v2-6-rl-oss-pruned`: setup keeps only the history HEAD contains, drops broken refs and .keep/.promisor marks, gives tracked files the setup time), task setup inside the harness's 240 s, rewards above zero, fresh-Sandbox grading agreeing with the agent's own (both-ways on). Same recipe and scale as smoke01; own volume, Sandbox app and W&B group `qwen36-mimo-smoke` (W&B `gnslapqu`). PASSED, finished 09:43: all 6 steps trained, every weight update reached both pools, 0 aborted groups, 0 infra errors, 0 missing rewards; reward per step 0.30-0.67 (mean 0.445; smoke01 on the released tasks 0.33-0.63, mean 0.449); fresh-grade disagreement 0-7.8% (smoke01 0-8.1%); task setup mean 6-11 s, max 124 s (one episode; smoke01 max 15 s), inside the 240 s limit. Before it, the leak gate ran on every code task (2,698) with the harness's own sandbox, setup and grading: setup succeeded on all (p99 23 s, max 84 s); 2,682 images held later history or unreachable objects before setup and none after; file times uniform after setup; the 41 fixes the audit rebuilt from image history still score 1; an empty patch fails everywhere except 2 tasks that pass unfixed with the released setup too, and 4 images keep build output compiled from the fixed code; those 6 are excluded (`mimo_v2_6_code_excluded.json`). Apps stopped |

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
| SC | r03 | [1vtd813n](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/1vtd813n), [o58xf5g0](https://wandb.ai/nan-playground/fully-async-rl-modal/runs/o58xf5g0) | a different config, `qwen3_6_35b_a3b_hetero_score_centering_advanced` (frozen; see below); stopped at step 130; resumed 2026-10-04 (see the record table) |
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
