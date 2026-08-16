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
   C56 is higher priority once it is executable: any live C56 process prevents
   C57 from starting, as does the conjunction of C56 `GO_LONG.json`, the C58b
   online s10000 parent checkpoint, and its final strict-restore `READY.json`.
   `GO_LONG.json` by itself is not a reservation.  If the complete reservation
   appears during a C57 intermediate evaluation, the evaluator is terminated
   and retried later; its report is published only after a complete run.
4. If and only if step5000 returns `GO_CLOSED_LOOP_CANARY`, invoke
   `launch_c57_traced_libero.sh` with the same task/trial/environment seeds as
   the frozen D0 control.

The n1 5000-step long training process is intentionally not stopped or shared
with online H3 inference.
