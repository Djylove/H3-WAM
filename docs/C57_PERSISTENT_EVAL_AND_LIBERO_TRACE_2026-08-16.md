# C57 fixed held-out evaluation and real LIBERO feedback trace

## Decision boundary

C57 is not a flattened history feature and this evaluation must not silently
turn it into one.  Training calls `forward_teacher_forced_history`, whose every
history chunk enters `H3LingBotPersistentKVPolicy.commit_executed_feedback`.
Deployment calls `LingBotPersistentRolloutSession.commit_real_feedback` through
`C57ExecutedFeedbackWire`; that path reaches the same model method.  Predicted
observation/action K/V is temporary and is rolled back before real feedback is
committed.

The actual LIBERO transaction is fixed to:

1. explicit episode reset;
2. prediction, with final-denoise K/V staged but not committed;
3. four actions accepted by `env.step`, followed by a real post-action image;
4. four more accepted actions and the second post-action image;
5. atomic action-8 commit of the two observations and eight normalized
   executed actions.  The first episode commit also owns its initial frame.

An episode tail shorter than eight actions is discarded only by the next
explicit reset.  Planned but unexecuted actions can never enter persistent K/V.
Every server command emits a `c57_persistent_trace` JSON line to the policy log.

## Pre-registered offline queue

`freeze_c57_heldout_eval_plan.py` first proves train/held-out episode
disjointness, then selects one maximum-history sample per episode with a fixed
hash seed.  The output records exact manifest SHA256 values and a distinct
deterministic flow seed per sample.  D0 and C57 therefore receive the same
current observation, target, timestep and noise.

All step200 through step5000 checkpoints are diagnostics.  Step5000 is the
only pre-registered decision checkpoint, preventing best-checkpoint peeking.
The offline gate is deliberately limited: at least 3% paired mean flow-loss
improvement and 55% paired sample wins can only authorize a traced closed-loop
LIBERO canary.  Offline loss alone cannot promote C57.

## Launch order (idle GPU only)

1. Build a held-out C57 sequence manifest from the frozen validation source
   with `--max-history-chunks 7`.  Seven chunks produce at most 15 observation
   frames plus 56 action rows, i.e. 536 tokens under the training contract;
   using the model constructor's nominal 15-window count here is invalid.
2. Freeze the selected manifest and plan with
   `freeze_c57_heldout_eval_plan.py`.
3. Export the `C57_EVAL_*` variables and run
   `run_c57_heldout_eval_queue.sh` on an idle node/GPU.  When sharing a node
   schedule, `arm_c57_eval_after_c58.sh` may wait for an audited C58 s10000
   checkpoint/report first.  The queue independently requires the selected
   physical GPU to remain free for two probes before every checkpoint restore.
   C56 is higher priority once it is executable: a live
   `train_c56b_fact_online.py` process prevents C57 from starting, as does the conjunction of C56 `GO_LONG.json`, the C58b
   online s10000 parent checkpoint, and its final strict-restore `READY.json`.
   `GO_LONG.json` by itself is not a reservation.  If the complete reservation
   appears during a C57 intermediate evaluation, the evaluator is terminated
   and retried later; its report is published only after a complete run.
   A waiting `watch_and_launch_c56b_after_c58b_final.sh` is not a training
   process and must not reserve the GPU.  Once C56 publishes its bit-exact
   s10000 restore, its reservation is released.  The queue also prepends the
   PyTorch cu130 runtime's own `nvidia/cu13/lib`; using the system cublas on n2
   was independently reproduced to fail even a minimal BF16 linear layer.

The n2 release watcher likewise waits for the old C58 s10000 train report,
strict restore report, and `COMPLETED` marker; a saved checkpoint alone cannot
release C57 evaluation.
4. `watch_finalize_launch_c57.sh` waits for the complete s5000 checkpoint,
   training report and heldout report.  `finalize_c57_lingbot_long5000.py`
   rechecks the fixed plan SHA, exact 80 sample/RNG pairs, checkpoint contract,
   all 5000 finite updates, and explicit strict restore.  If and only if the
   result is `GO_FRESH_LIBERO_CANARY`, the watcher launches the pre-registered
   four-suite task0/trial49 C57/D0 canary.  It uses identical environment and
   policy-noise seeds, no ensemble, no video/trajectory output and no new H3/KV
   cache.  Candidate logs must prove reset/predict/obs4/commit8 before the
   atomic `RESULTS.json` is published.

The n1 5000-step long training process is intentionally not stopped or shared
with online H3 inference.

## Read-only execution snapshot

The n2 evaluator queue and final watcher execute from the fixed composite
snapshot below, rather than from the shared mutable project checkout:

- main source commit: `71f93b0e351fa6380423177c11cae91323278dc0`;
- pinned StarWAM source commit: `cd76d96f273f81e228a05f40f9697fe2514e2356`;
- pinned DreamWAM source commit: `6e989facc0c452fd3488d75f60bc36411005558c`;
- n2 path:
  `/mnt/h3-wam/code-snapshots/h3-wam-71f93b0e351fa6380423177c11cae91323278dc0-starwam-cd76-min4-dreamwam-6e98-min3`;
- n2 mount contract: bind-remounted `ro`, with `PYTHONDONTWRITEBYTECODE=1`;
- queue entrypoint SHA256:
  `20c56dbd8da47e0e45e851bc6fcef0e0e57154ccbe9905585e29160a7b901911`;
- paired evaluator SHA256:
  `e2d73aa008fc2f93e754ecf76066c7d46bae479d4b14b0e94dd454594cb90f95`;
- final watcher SHA256:
  `c084c10c1592aeb4019da45f0dfaa35f28a578bff26b60f49ec2091be37cdae3`;
- StarWAM `action_dit.py`, `wan_block.py`, `scheduler.py`, and `loss.py`
  SHA256 values: `b6cd067c...`, `30334432...`, `5f9df0c8...`, and
  `00955545...`;
- DreamWAM `layers.py`, `experts.py`, and `mot.py` SHA256 values:
  `3cd38ad2...`, `9ba51dbb...`, and `5467d135...`.

The first Git-only snapshot correctly failed closed after C56 released because
the main repository archive excludes the pinned upstream repositories.  A
StarWAM-only composite then exposed its still-missing DreamWAM dependency in
the same way.  No evaluation report was published by either failed attempt.
The final composite copied only the seven source files actually imported by
this path from clean pinned repositories, passed a complete evaluator import
smoke, and was then bind-remounted read-only.  Both process `cwd` and
`PROJECT_ROOT` are this final snapshot path.  Step2000 subsequently completed
all 80 forwards with explicit strict restore, proving that the released queue
now consumes the backlog in order.  An incomplete or crashed C56 remains
fail-closed instead of silently stealing its reservation.

## Long-run budget and final relay audit

The frozen C57 train sequence manifest contains 200,779 windows (SHA256
`8f95005ac66fd89ca3a22a80d75480e9792b09f976e928f2eb70d4f08680049f`).
The actual global batch is eight ranks times ten accumulation samples = 80.
At step3237 the run had therefore consumed 258,960 samples, or 1.2898
effective epochs.  The pre-registered step5000 budget is 400,000 samples, or
1.9922 effective epochs.  The last-50-step mean at this audit was 6.6361
seconds, giving 11,699 seconds (3.25 hours) of training wall time remaining;
heldout catch-up and any authorized rollout are separate.

The step3000 and step3200 checkpoints both have the expected 2,099,867,657
bytes and no `.partial` sibling.  The final relay remains fail-closed and
ordered:

1. the queue must strictly restore and forward the fixed 80 samples for
   `c57_step05000.pt`, preserving plan SHA256
   `7d69a2aded4753985ac31c44f25ba0e88fab1fa47906621390df0fc5de07f73a`;
2. the watcher also waits for the complete 5000-step train report;
3. the finalizer rechecks checkpoint schema, all finite/nonzero updates,
   exact sample/RNG order and explicit strict-restore evidence;
4. only `PASS_C57_FINAL_OFFLINE_GATE / GO_FRESH_LIBERO_CANARY` can enter the
   fixed four-suite paired canary.  `NO_GO`, missing artifacts, mismatched
   hashes or restore failure launch nothing.

## Frozen heldout learning curve (same 80 samples and RNG)

| step | C57 mean | D0 mean | relative improvement | sample wins | decision |
|---:|---:|---:|---:|---:|---|
| 200 | 0.133945 | 0.076288 | -75.58% | 6.25% | NO_GO |
| 400 | 0.104021 | 0.076288 | -36.35% | 16.25% | NO_GO |
| 600 | 0.095226 | 0.076288 | -24.82% | 25.00% | NO_GO |
| 800 | 0.089742 | 0.076288 | -17.64% | 31.25% | NO_GO |
| 1000 | 0.086795 | 0.076288 | -13.77% | 35.00% | NO_GO |
| 1200 | 0.084842 | 0.076288 | -11.21% | 33.75% | NO_GO |
| 1400 | 0.083139 | 0.076288 | -8.98% | 36.25% | NO_GO |
| 1600 | 0.081978 | 0.076288 | -7.46% | 35.00% | NO_GO |
| 1800 | 0.081410 | 0.076288 | -6.71% | 33.75% | NO_GO |

The paired mean-loss curve is improving monotonically through s1800, while
sample wins fluctuate between 33.75% and 36.25% after s1000.  C57 remains
worse than D0.  These are diagnostics only; no intermediate checkpoint can be
selected for rollout.  The only effect decision remains s5000.
