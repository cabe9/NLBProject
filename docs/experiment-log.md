# Experiment Log Archive

This file stores detailed historical run notes that were previously kept inline in `AGENTS.md`.

## Usage

- Treat this file as the canonical long-form archive for completed screens and run logs.
- Keep `AGENTS.md` focused on TL;DR state, current guardrails, and active direction.

## Historical Headline Ladder

- Prior STNDT-lite 4-layer constant-LR recipe (`mask_prob=0.6`,
  `heldin_loss_weight=1.0`, `use_mask_token=true`, identity+spike, 5-seed)
  local public-test: `0.3795 co-bps`.
- Prior STNDT-lite 4-layer constant-LR recipe (`mask_prob=0.5`,
  `heldin_loss_weight=1.0`, `use_mask_token=true`, identity+spike, 5-seed)
  local public-test: `0.3742 co-bps`.
- Prior STNDT-lite Screen C winner (3-layer CD reconcile: `mask_prob=0.5`,
  `heldin_loss_weight=1.0`, `use_mask_token=true`, identity+spike, 5-seed)
  local public-test: `0.3649 co-bps`.
- Prior STNDT-lite temporal-identity + spike-weighted 5-seed local public-test:
  `0.3413 co-bps`.
- Prior STNDT-lite 5-seed local public-test: `0.3302 co-bps`.
- Prior NDT-lite 7-seed local public-test: `0.3229 co-bps`.
- Lagged PCA public-test baseline: `~0.0268 co-bps`.

## Archived History

### Screen C (complete — gate cleared)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_cd_reconcile_screen.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_cd_reconcile_screen/`
- **Best full train/val (candidate 4):** `0.3492` co-bps (`mask_prob=0.5`,
  `heldin_loss_weight=1.0`, `use_mask_token=true`, `ensemble_size=5`, constant
  LR). Clears promotion gate (`>= 0.3259`) and identity+spike anchor floor
  (`>= 0.3239`).
- **3-seed sanity (complete):** `0.3427` train/val co-bps on the Screen C winner
  with `ensemble_size=3`
  (`results/benchmark_runs/stndt_lite_cd_reconcile_winner_3seed_sanity/`).
  Still above `0.3259` / `0.3239` floors; stable vs Screen C `0.3492` (5-seed).
- **Public-test (complete):** `0.3649` co-bps local public-test
  (`results/public_test/mc_maze_stndt_lite_cd_reconcile_winner_5seed/`).
  Config: `configs/benchmarks/mc_maze_stndt_lite_cd_reconcile_winner_5seed_public.yaml`.
  Beats prior headline `0.3413` by `+0.0236` co-bps.
- **Next gate:** no broad tuning without a new validation reason; do not
  public-test additional variants without explicit request.

### Screen D (complete — no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_cd_winner_neighborhood_screen.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_cd_winner_neighborhood_screen/`
- **Design:** 8 candidates, full train/val selection, one-knob neighborhood around
  Screen C winner; all candidates used `ensemble_size=5`.
- **Best full train/val:** `0.3503` co-bps (`mask_prob=0.6`, otherwise anchor).
- **Screen C reference:** `0.3492` co-bps. Delta `+0.0011` co-bps — below the
  `+0.003` interesting threshold (`0.3522` floor). **Do not promote**; keep
  Screen C winner as validated recipe and `0.3649` public-test headline.
- **Leaderboard order:** mask 0.6 (`0.3503`) > heldin 0.7 (`0.3499`) > anchor
  (`0.3492`) > identity 0.1 (`0.3491`) > identity 0.025 (`0.3489`) > spike 1.5
  (`0.3478`) > heldin 1.3 (`0.3475`) > mask 0.4 (`0.3417`).
- **Next step:** no public-test or broad tuning; if revisiting, try a different
  axis (e.g. schedule/depth) or require `>= 0.3522` train/val before sanity.
- **2026-05-23 local verification:** sanitized branch confirmed, `AGENTS.md`
  ignored/local-only, and Screen D artifacts re-read. Existing Screen D run
  matches the requested validation-only neighborhood; no public-test was run.
  Best remains mask 0.6 at `0.3503`, below the `0.3522` threshold, so no
  promotion and no public docs update.

### Screen E (complete - validation only)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_cd_schedule_depth_overnight_screen.yaml`
- **Runner:** `scripts/run_screen_e_schedule_depth_overnight.ps1`
- **Artifacts:** `results/benchmark_runs/stndt_lite_cd_schedule_depth_overnight_screen/`
- **Scheduled task:** `NLB Screen E Overnight`, one-time run at
  `2026-05-23 22:00 America/Los_Angeles`, with an 8-hour task/wrapper timeout.
- **Design:** narrow schedule/depth screen around the public-tested Screen C
  winner. Eight full train/val candidates: anchor, longer constant schedule,
  two cosine schedule variants, 2-layer constant, 4-layer constant, 4-layer
  longer constant, and 2-layer longer constant. All keep Screen C objective
  knobs (`mask_prob=0.5`, `heldin_loss_weight=1.0`, `use_mask_token=true`,
  `temporal_identity_scale=0.05`, `spike_loss_weight=1.0`,
  `contrast_loss_weight=0.0`) and `ensemble_size=5`.
- **Gate:** compare against Screen C reference `0.3492` and Screen D
  meaningful-improvement floor `0.3522`. Do not public-test unless a result
  clearly clears the gate and the user explicitly asks.
- **Next check:** after the task runs, inspect the latest `overnight_*.log`,
  `overnight_status_*.txt`, `metrics.csv`, and
  `full_val_candidate_leaderboard_*.txt`; update this local file with the
  leaderboard and promotion decision. Do not update public docs without a new
  public-test result.
- **2026-05-23 immediate launch:** user asked to start now instead of waiting
  for the 22:00 task. The scheduled task was disabled. First immediate attempt
  exited because `NLB_DATA_DIR` was not inherited by the background process;
  runner now defaults it to local `data/raw`. Relaunched at `2026-05-23
  01:01:56 America/Los_Angeles`; status file records experiment PID `41656`
  and logs under the Screen E artifact directory. Do not start another screen
  after it finishes.
- **2026-05-23 result:** complete in ~30 min; no public-test. Best full
  train/val was `0.352963` co-bps, clearing the `0.3522` floor by `+0.000763`
  and Screen C `0.3492` by `+0.003763`. Top candidates tied to 6 decimals:
  4-layer constant `max_epochs=90`/`patience=15` selected, and 4-layer constant
  `max_epochs=60`/`patience=10`; both scored `0.352963`. 3-layer anchor stayed
  `0.349179`; cosine variants underperformed; 2-layer variants dropped to
  `0.331180`. Treat this as a gate-clearing validation result, not a new
  public-test headline. Reasonable next step, only if user asks, is a sanity
  confirmation around 4-layer constant before any public-test.
- **Runtime calibration:** on this PC/GPU, Screen E candidate selection took
  about `28.5` minutes in the log, with individual 5-seed full train/val
  candidates around `2.1`-`3.7` minutes each. The 8-hour timeout was only a
  safety cap, not an expected duration. Similar narrow STNDT-lite 5-seed
  screens are likely closer to tens of minutes than overnight, unless the
  candidate count, epochs/patience, model width/depth, or data split changes
  substantially. Still use a timeout for unattended runs because early stopping
  and GPU state make exact runtime hard to guarantee.

### 4-Layer Sanity and Public-Test (complete - promoted)

- **Sanity config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_constant_sanity.yaml`
- **Sanity artifacts:** `results/benchmark_runs/stndt_lite_depth4_constant_sanity/`
- **Design:** 3-layer anchor seed `0`; 4-layer constant `60/10` seeds
  `0`, `101`, `202`; 4-layer constant `90/15` seeds `0`, `101`, `202`. All
  candidates kept Screen C objective knobs and `ensemble_size=5`.
- **Sanity result:** 4-layer `60/10` repeats were `0.352963`, `0.358860`,
  `0.363134`; mean `0.358319`, with 3/3 repeats above `0.3522`. 4-layer
  `90/15` repeats were `0.352963`, `0.358800`, `0.363133`; mean `0.358299`,
  also 3/3 above `0.3522`.
- **Promotion decision:** choose 4-layer `60/10` because `90/15` did not beat
  it by the `0.0005` mean tie-breaker. Sanity passed; exactly one public-test
  was justified.
- **Public-test config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_constant_5seed_public.yaml`
- **Public-test artifacts:** `results/public_test/mc_maze_stndt_lite_depth4_constant_5seed/`
- **Public-test result:** `0.3742113348156462` co-bps (`0.3742` rounded),
  improving the prior Screen C headline `0.3648954805894155` by
  `+0.0093158542262307`. This is a local public-test score against the frozen
  target, not a live leaderboard rank.
- **Docs:** public README/results/models docs were updated because the
  public-test headline improved.

### Screen G (complete - validation only, no public-test)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_objective_neighborhood_screen.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_depth4_objective_neighborhood_screen/`
- **Design:** narrow objective-neighborhood screen around the promoted 4-layer
  constant `60/10` recipe: anchor, `mask_prob=0.55`, `mask_prob=0.6`,
  `heldin_loss_weight=0.7`, `heldin_loss_weight=1.3`,
  `temporal_identity_scale=0.075`, `temporal_identity_scale=0.1`,
  `spike_loss_weight=1.5`.
- **Best full train/val:** `0.364878` co-bps with `mask_prob=0.6`; anchor was
  `0.352963`, so the delta is about `+0.011915` and clears anchor + `0.003`.
- **Leaderboard order:** mask 0.6 (`0.364878`) > mask 0.55 (`0.357588`) >
  spike 1.5 (`0.354359`) > anchor (`0.352963`) > identity 0.075
  (`0.352942`) > identity 0.1 (`0.352930`) > heldin 1.3 (`0.351236`) >
  heldin 0.7 (`0.350629`).
- **Next step:** do not public-test Screen G directly. If continuing, run a
  sanity confirmation around 4-layer `mask_prob=0.6` first; only consider a
  single public-test if that sanity passes.

### 4-Layer Mask 0.6 Sanity (complete - public-test eligible)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_mask06_sanity.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_depth4_mask06_sanity/`
- **Design:** validation-only sanity confirmation for the Screen G
  `mask_prob=0.6` candidate, using seeds `0`, `101`, `202`; all candidates
  used 4-layer constant LR, `max_epochs=60`, `patience=10`,
  `temporal_identity_scale=0.05`, `spike_loss_weight=1.0`,
  `contrast_loss_weight=0.0`, `heldin_loss_weight=1.0`, `use_mask_token=true`,
  and `ensemble_size=5`.
- **Seed scores:** seed 0 `0.364878`, seed 101 `0.362891`, seed 202
  `0.368330` train/val co-bps.
- **Sanity result:** mean `0.365366` co-bps; 3/3 seeds were above the
  per-seed floor `0.3583`, and the mean cleared the `0.3613` gate.
- **Decision:** `mask_prob=0.6` passes sanity and is eligible for exactly one
  future public-test, but do not run that public-test without explicit user
  approval. Current public-test headline remains the 4-layer `mask_prob=0.5`
  result at `0.3742` co-bps.

### 4-Layer Mask 0.6 Public-Test (complete - promoted)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_mask06_5seed_public.yaml`
- **Artifacts:** `results/public_test/mc_maze_stndt_lite_depth4_mask06_5seed/`
- **Public-test result:** `0.3795370600211466` co-bps (`0.3795` rounded),
  with vel R2 `0.8977512260497815` and psth R2 `0.6354028505647371`.
- **Comparison:** improves the previous 4-layer mask 0.5 public-test headline
  `0.3742113348156462` by `+0.0053257252055004` co-bps.
- **Decision:** promoted to current local public-test headline and public docs
  updated. This is still a local public-test score against the frozen target,
  not a live leaderboard rank. Do not run another nearby public-test without a
  fresh validation-gated result and explicit user approval.

### Screen H (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_screen_h_objective.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_depth4_screen_h_objective/`
- **Design:** narrow validation-only objective screen around the 4-layer
  `mask_prob=0.6` public-test headline. Candidates: anchor, `mask_prob=0.58`,
  `mask_prob=0.62`, `heldin_loss_weight=0.9`, `heldin_loss_weight=1.1`,
  `temporal_identity_scale=0.075`, `spike_loss_weight=1.25`, and
  `mask_prob=0.62` plus `spike_loss_weight=1.25`.
- **Best full train/val:** anchor candidate remained best at `0.364878`
  co-bps. Best non-anchor was `heldin_loss_weight=0.9` at `0.364161`; it did
  not beat the anchor, much less clear anchor + `0.003`.
- **Leaderboard order:** anchor (`0.364878`) > heldin 0.9 (`0.364161`) >
  mask 0.62 + spike 1.25 (`0.363530`) > mask 0.62 (`0.363440`) >
  identity 0.075 (`0.363342`) > spike 1.25 (`0.362589`) > heldin 1.1
  (`0.360529`) > mask 0.58 (`0.358895`).
- **Decision:** no sanity, no public-test, and no public docs update. The
  current public-test headline remains 4-layer `mask_prob=0.6` at `0.3795`
  co-bps. Do not repeat this exact objective neighborhood.

### Depth-4 Diagnostic Summary (complete)

- **Artifact:** `results/benchmark_runs/stndt_lite_depth4_diagnostic_summary.md`
- **Read:** train/val selection remained directionally useful from 4-layer
  `mask_prob=0.5` to `mask_prob=0.6`: sanity mean improved from `0.358319`
  to `0.365366`, and local public-test improved from `0.374211` to
  `0.379537`. Screen H then showed nearby objective tweaks around mask `0.6`
  did not beat the anchor.
- **Decision:** objective knobs around the current recipe are locally saturated;
  architecture/regularization was the right next axis.

### Screen I (complete - sanity passed, no public-test yet)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_screen_i_arch_reg.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_depth4_screen_i_arch_reg/`
- **Design:** validation-only architecture/regularization screen around the
  current 4-layer `mask_prob=0.6` recipe. Candidates: anchor, `dropout=0.03`,
  `dropout=0.08`, `d_model=224`, `d_model=256`, `n_layers=5`,
  `d_model=224/dropout=0.08`, and `n_layers=5/dropout=0.08`.
- **Best full train/val:** `n_layers=5`, `dropout=0.08` scored `0.370542`
  co-bps in the leaderboard; anchor was `0.364878`, so the delta was about
  `+0.005664` and cleared the `+0.003` screen gate.
- **Leaderboard order:** 5-layer/dropout 0.08 (`0.370542`) > 5-layer/dropout
  0.05 (`0.368785`) > width 256 (`0.366991`) > width 224 (`0.366095`) >
  anchor (`0.364878`) > width 224/dropout 0.08 (`0.363541`) > dropout 0.03
  (`0.361214`) > dropout 0.08 at 4 layers (`0.359851`).
- **Sanity config:** `configs/benchmarks/mc_maze_stndt_lite_depth5_dropout08_sanity.yaml`
- **Sanity artifacts:** `results/benchmark_runs/stndt_lite_depth5_dropout08_sanity/`
- **Sanity result:** seed 0 `0.370674`, seed 101 `0.371760`, seed 202
  `0.368157`; mean about `0.370197`. This clears the planned mean gate
  `0.3684`, and 3/3 seeds clear the per-seed floor `0.3654`.
- **Decision:** 5-layer/dropout 0.08 is public-test eligible, but do not run
  public-test without explicit user approval. Current public-test headline
  remains 4-layer `mask_prob=0.6` at `0.3795` co-bps until a public-test
  improves it.

### 5-Layer Dropout 0.08 Public-Test (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth5_dropout08_5seed_public.yaml`
- **Artifacts:** `results/public_test/mc_maze_stndt_lite_depth5_dropout08_5seed/`
- **Public-test result:** `0.37643077029988126` co-bps (`0.3764` rounded),
  with vel R2 `0.9025943636339236` and psth R2 `0.6390030921029497`.
- **Comparison:** below the current 4-layer mask 0.6 headline
  `0.3795370600211466` by about `-0.003106` co-bps, despite the stronger
  train/val sanity signal.
- **Decision:** no promotion and no public docs update. Keep the current
  public-test headline at 4-layer `mask_prob=0.6` (`0.3795` co-bps). Do not
  repeat nearby public-tests for depth-5/dropout 0.08 without a new validation
  reason and explicit user approval.

### Screen J (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_screen_j_train_robustness.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_depth4_screen_j_train_robustness/`
- **Design:** validation-only train-time robustness screen around the current
  4-layer `mask_prob=0.6` public-test headline. Candidates: anchor, longer
  `max_epochs=90`/`patience=15`, `weight_decay=0.0002`,
  `weight_decay=0.00005`, `dropout=0.04`, `dropout=0.06`,
  `learning_rate=0.0007`, and `learning_rate=0.0013`.
- **Best full train/val:** `learning_rate=0.0013` scored `0.369391` co-bps;
  anchor was `0.364878`, so the delta was about `+0.004513`.
- **Gate:** Screen J required anchor + `0.005` because the prior depth-5
  validation/public-test correlation was weaker. The best candidate did not
  clear that stricter gate.
- **Leaderboard order:** lr 0.0013 (`0.369391`) > weight_decay 0.00005
  (`0.364879`) > longer 90/15 (`0.364878`) > anchor (`0.364878`) >
  weight_decay 0.0002 (`0.364872`) > dropout 0.04 (`0.362700`) > dropout 0.06
  (`0.360578`) > lr 0.0007 (`0.353941`).
- **Decision:** no sanity, no public-test, and no public docs update. Current
  public-test headline remains 4-layer `mask_prob=0.6` at `0.3795` co-bps.
  If revisiting this axis, require a stronger validation margin or a different
  robustness design.

### 4-Layer LR 0.0013 Sanity (complete - no public-test)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_depth4_lr0013_sanity.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_depth4_lr0013_sanity/`
- **Design:** validation-only 3-seed sanity for Screen J's best train-time
  robustness candidate: 4-layer `mask_prob=0.6`, `dropout=0.05`,
  `learning_rate=0.0013`, constant LR, `max_epochs=60`, `patience=10`,
  `temporal_identity_scale=0.05`, `spike_loss_weight=1.0`,
  `heldin_loss_weight=1.0`, and `ensemble_size=5`.
- **Seed scores:** seed 0 `0.367423`, seed 101 `0.368702`, seed 202
  `0.370701` train/val co-bps.
- **Comparison to current mask 0.6 sanity:** mean about `0.368942`, above the
  current mask 0.6 sanity mean `0.365366`; all 3 seeds beat their matching
  mask 0.6 floors (`0.364878`, `0.362891`, `0.368330`).
- **Decision:** useful validation signal and plausible future anchor, but it
  does not clear the stronger `0.3704` mean threshold for spending another
  public-test after the depth-5 validation/public-test mismatch. No public-test
  and no public docs update.

### Diverse Ensemble Screen (paused by user - partial only)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_ensemble_screen.yaml`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/`
- **What was running:** validation-only prediction-averaging screen over three
  existing STNDT-lite recipes without architecture-code changes: current
  4-layer `mask_prob=0.6` anchor, 4-layer `learning_rate=0.0013`, and
  5-layer `dropout=0.08`. Mixed same-seed ensembles were evaluated by averaging
  predictions.
- **PID checked before stop:** `41488`.
- **Log path:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/ensemble_screen_stderr.log`
- **Completed members/scores before stop:** seed 0 singles: anchor `0.364878`,
  lr0013 `0.368539`, depth5/dropout08 `0.370552`; seed 0 mixes:
  anchor+lr0013 `0.370082`, anchor+depth5 `0.371782`, lr0013+depth5
  `0.372944`, all three `0.372816`; seed 101 anchor `0.362891`.
- **Current member when stop was requested:** `lr0013 seed=101` was in progress.
- **Partial result usability:** log-only directional signal, not usable for
  promotion or public-test gating because stability across seeds was not
  measured. No public-test was run.
- **Stop result:** `CloseMainWindow` was unavailable for the hidden background
  Python process; non-force `taskkill` did not stop the process tree, so
  force `taskkill /T /F` was used and terminated PID `41488` plus child
  processes. No complete `metrics.csv` was produced.
- **Partial summary:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/partial_status.md`
- **Rerun command:** from `C:\Users\david\NLBProject`, run
  `$env:NLB_DATA_DIR='data/raw'; C:\Users\david\.venvs\nlb-project\Scripts\python.exe -m nlb_project.cli.run_ensemble_screen --config configs/benchmarks/mc_maze_stndt_lite_diverse_ensemble_screen.yaml --log-level INFO`.

### Diverse Ensemble Screen Restart (complete - public-test eligible)

- **Restarted:** `2026-05-24T00:30:35 America/Los_Angeles`.
- **Wrapper:** `scripts/run_diverse_ensemble_screen_overnight.ps1`
- **Timeout:** `28800` seconds (8 hours); wrapper should force-stop the child
  process tree if it times out.
- **Wrapper PID:** `19508`; **experiment PID:** `23396`.
- **Status:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/diverse_ensemble_status_20260524T003035.txt`
- **Logs:** stdout
  `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/diverse_ensemble_20260524T003035.out.log`,
  stderr/progress
  `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/diverse_ensemble_20260524T003035.err.log`.
- **Command:** `powershell -ExecutionPolicy Bypass -File scripts\run_diverse_ensemble_screen_overnight.ps1 -TimeoutSeconds 28800`.
- **Discipline:** validation-only; do not public-test or start another job.
- **Finished:** `2026-05-24T01:20:02 America/Los_Angeles`; runtime about
  `49` minutes, well below the 8-hour timeout. No Python/NLB job remained.
- **Result:** best mixed ensemble was `lr0013_depth5`, averaging predictions
  from 4-layer `learning_rate=0.0013` and 5-layer `dropout=0.08`.
- **Seed scores:** seed 0 `0.373232`, seed 101 `0.373595`, seed 202
  `0.373819`; mean `0.373548` train/val co-bps.
- **Gate decision:** passes the diverse-ensemble gate: mean is clearly above
  `0.3704`, and 3/3 repeats are above `0.3704`. This is validation-only and
  makes exactly one future public-test worth considering, but do not public-test
  without explicit user approval.
- **Leaderboard:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/ensemble_diversity_leaderboard_20260524T081948Z.txt`
- **Metrics:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/metrics.csv`

### Diverse Ensemble Public-Test (complete - promoted)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_lr0013_depth5_public.yaml`
- **Artifacts:** `results/public_test/mc_maze_stndt_lite_diverse_lr0013_depth5_10member/`
- **Recipe:** one mixed prediction average from 4-layer `learning_rate=0.0013`
  STNDT-lite (`ensemble_size=5`, seed 0) and 5-layer `dropout=0.08`
  STNDT-lite (`ensemble_size=5`, seed 0), fitted on `train+val` and scored
  once against the frozen public-test target.
- **Public-test result:** `0.3830396802163914` co-bps (`0.3830` rounded),
  with vel R2 `0.9052949421304819` and psth R2 `0.6390176826899897`.
- **Comparison:** improves the prior 4-layer mask 0.6 headline
  `0.3795370600211466` by about `+0.003503` co-bps, but remains below the
  frozen `0.3862` reference by about `-0.003160`.
- **Decision:** promoted to current local public-test headline and public docs
  updated with conservative local-score wording. Do not public-test nearby
  diverse variants without a fresh validation-gated reason and explicit user
  approval.

### Diverse Weight Screen (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_weight_screen.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_diverse_weight_screen/`
- **Design:** validation-only integer-weighted prediction averages around the
  current mixed public-test headline. Trained the two public-test component
  recipes (`lr0013` and `depth5_dropout08`) for seeds `0`, `101`, and `202`,
  then scored weighted blends from 25/75 through 75/25. No public-test.
- **Gate:** required mean train/val co-bps `>= 0.3765` and at least 2/3 repeats
  above that floor before considering any future public-test.
- **Best result:** equal 50/50 blend remained best at mean `0.373094`
  train/val co-bps, with seed scores `0.373226`, `0.373466`, and `0.372590`.
  This is below both the prior diverse-screen mean `0.373548` and the new
  `0.3765` public-test-consideration gate.
- **Leaderboard order:** 50/50 (`0.373094`) > 60/40 (`0.372960`) > 40/60
  (`0.372957`) > 67/33 (`0.372721`) > 33/67 (`0.372716`) > 75/25
  (`0.372252`) > 25/75 (`0.372244`) > lr0013 single (`0.369682`) >
  depth5/dropout08 single (`0.369668`).
- **Decision:** no public-test, no public docs update, and do not repeat
  simple lr0013/depth5 blend-weight tuning. Current headline remains the
  `0.3830` local public-test mixed ensemble.

### Screen K Crossed Diversity Member (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_screen_k_crossed_member.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_diverse_screen_k_crossed_member/`
- **Design:** validation-only search for a third diverse member to add to the
  current public-test mixed ensemble (`lr0013` + `depth5_dropout08`). Candidate
  third members crossed depth, width, dropout, and LR: depth-5 LR `0.0013`
  variants, depth-5 LR `0.0007`, and width-256 LR `0.0013` variants. Ensembles
  evaluated the current pair, pair + candidate, and candidate paired with each
  current member. No public-test.
- **Gate:** required mean train/val co-bps `>= 0.3765` with at least 2/3 repeats
  above the gate before any future public-test consideration.
- **Best result:** `lr0013_depth5_width256_dropout06_lr0013` scored mean
  `0.373771` train/val co-bps, with seed scores `0.374973`, `0.371796`, and
  `0.374546`. This is only a small validation lift over the rerun current pair
  (`0.372956`) and below the `0.3765` promotion gate.
- **Leaderboard top:** width256/dropout06 three-member mix (`0.373771`) >
  width256 three-member mix (`0.373613`) > current pair (`0.372956`) >
  depth5+width256/dropout06 pair (`0.372396`) > depth5+width256 pair
  (`0.372300`). Depth-5 LR `0.0013` crossed variants collapsed as singles
  (`0.336`-`0.346`) and should not be repeated in this form.
- **Decision:** no public-test, no public docs update. Width-256 adds mild
  complementarity but not enough stability, especially on seed 101. Do not
  repeat this exact crossed-member screen; if continuing, try a different
  diversity source or a seed-101 robustness diagnosis before spending another
  public-test.

### Screen L Seed-101 Robust Member (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_screen_l_seed101_robust_member.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_diverse_screen_l_seed101_robust_member/`
- **Design:** validation-only robustness/diversity screen after Screen K showed
  width-256 complementarity but weak seed 101. Kept the current public-test
  mixed components fixed (`lr0013` and `depth5_dropout08`) and tested
  third-member variants intended to stabilize seed 101: 4-layer LR `0.0013`
  with lower weight decay, dropout `0.04`, identity `0.075`, spike weight
  `1.25`, 5-layer/dropout08 lower weight decay, plus width256/dropout06 as a
  control. No public-test.
- **Gate:** required mean train/val co-bps `>= 0.3765`, at least 2/3 repeats
  above the gate, and manual sanity that seed 101 was not the sole weak point.
- **Best result:** `lr0013_depth5_spike125` scored mean `0.374404` train/val
  co-bps, with seed scores `0.374160`, `0.374915`, and `0.374138`. It improved
  the rerun current pair (`0.373708`) and lifted seed 101, but still missed
  the `0.3765` public-test-consideration gate and had no repeats above `0.375`.
- **Leaderboard top:** spike125 three-member mix (`0.374404`) >
  width256/dropout06 three-member mix (`0.374128`) > identity075 three-member
  mix (`0.373948`) > lr0013+depth5_wd00005 pair (`0.373825`) > current pair
  (`0.373708`).
- **Decision:** no public-test, no public docs update. Spike-weight `1.25` is
  the best robustness/diversity clue so far, especially for seed 101, but the
  validation lift is still too small. Do not public-test this near variant
  without a stronger follow-up validation result.

### Screen M Training Dynamics (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_screen_m_training_dynamics.yaml`
- **Wrapper:** `scripts/run_screen_m_training_dynamics_overnight.ps1`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_diverse_screen_m_training_dynamics/`
- **Started:** `2026-05-25T00:08:42 America/Los_Angeles`.
- **Wrapper PID:** `24284`; **experiment PID:** `41792`.
- **Timeout:** `28800` seconds (8 hours); wrapper will force-stop the child
  process tree if it times out.
- **Status file:** `results/benchmark_runs/stndt_lite_diverse_screen_m_training_dynamics/training_dynamics_status_20260525T000842.txt`
- **Stdout:** `results/benchmark_runs/stndt_lite_diverse_screen_m_training_dynamics/training_dynamics_20260525T000842.out.log`
- **Stderr/progress:** `results/benchmark_runs/stndt_lite_diverse_screen_m_training_dynamics/training_dynamics_20260525T000842.err.log`
- **Command:** `powershell -ExecutionPolicy Bypass -File scripts\run_screen_m_training_dynamics_overnight.ps1 -TimeoutSeconds 28800`
- **Design:** validation-only screen over training-dynamics variants around the
  current public-test mixed components (`lr0013` and `depth5_dropout08`).
  Tests spike `1.25`, warmup, cosine+warmup, identity+spike, and depth5
  training-dynamics variants as possible diverse third members.
- **Gate:** no public-test unless a mixed ensemble clears mean train/val
  `>= 0.3765` with at least 2/3 repeats above the gate and no single lucky
  seed pattern.
- **Discipline:** do not public-test and do not start another job after this
  finishes.
- **Finished:** `2026-05-25T02:24:19 America/Los_Angeles`; runtime about
  `2h15m`, below the 8-hour timeout. No Python/NLB training job remained.
- **Best result:** `lr0013_depth5_spike125` scored mean `0.374660`
  train/val co-bps, with seed scores `0.374130`, `0.374569`, and `0.375281`.
  This improved the rerun current pair (`0.373647`) but missed the `0.3765`
  public-test-consideration gate and had 0/3 repeats above the gate.
- **Leaderboard top:** spike125 three-member mix (`0.374660`) >
  depth5+spike125 pair (`0.374410`) > lr0013+depth5+depth5_spike125
  (`0.374037`) > identity075+spike125 three-member mix (`0.373830`) >
  current pair (`0.373647`).
- **Decision:** no public-test and no public docs update. Warmup and
  cosine+warmup did not help; they underperformed the simple spike `1.25`
  third-member clue. The near-variant ensemble path appears plateaued around
  `0.374`-`0.375` validation, below the stricter gate needed to spend another
  public-test.

### Validation Residual Diagnostic (complete)

- **Script:** `scripts/diagnose_validation_residuals.py`
- **Artifacts:** `results/diagnostics/validation_residuals_20260525/`
- **Design:** compared saved validation HDF5 predictions from the current
  public-test validation pair, Screen K width best, Screen L spike125 best, and
  Screen M spike125 best against the train/val heldout target. Breakdowns cover
  overall Poisson residual loss, movement phase/time bins, velocity magnitude,
  heldout unit, trial type, and maze id.
- **Important caveat:** ensemble-screen prediction HDF5s store the headline
  seed-0 baseline/improved predictions, not all 3 sanity repeats. Treat this as
  a residual-shape diagnostic, not a promotion gate or public-test argument.
- **Finding:** Screen K width had the broadest seed-0 residual improvement over
  the current pair (`delta_vs_reference=+0.0000178` mean Poisson residual loss),
  especially pre-movement (`+0.0000236`), mid/high speed bins, and high-spike
  heldout units such as units 29 and 43. Spike125 variants were close but less
  broad in seed-0 residuals, despite having the better multi-seed leaderboard
  stability in Screen M.
- **Decision:** do not public-test. The diagnostic suggests the next worthwhile
  structural direction is not more near-neighbor ensembles but a targeted
  pre-movement/unit-calibration idea: preserve the stable spike125 signal while
  adding width-like capacity only where it helps, then gate strictly on full
  multi-seed train/val.

### Screen N Width+Spike POC (complete - no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_screen_n_width_spike_poc.yaml`
- **Artifacts:** `results/benchmark_runs/stndt_lite_diverse_screen_n_width_spike_poc/`
- **Design:** validation-only proof-of-concept from the residual diagnostic:
  combine width-like capacity with the stable spike `1.25` clue, without
  warmup/cosine. Tested width224/width256 variants with and without dropout
  `0.06` as third members added to the current `lr0013 + depth5_dropout08`
  pair. No public-test.
- **Gate:** mean train/val co-bps `>= 0.3765` and at least 2/3 repeats above
  that gate before any future public-test consideration.
- **Best result:** `lr0013_depth5_width224_spike125` scored mean `0.375146`
  train/val co-bps, with seed scores `0.374492`, `0.375341`, and `0.375604`.
  This is the best recent near-variant validation result and beats the rerun
  current pair (`0.373748`) by about `+0.001398`, but it still misses the
  `0.3765` gate and has 0/3 repeats above the gate.
- **Leaderboard top:** width224+spike three-member mix (`0.375146`) >
  width224/dropout06+spike three-member mix (`0.375114`) >
  width256/dropout06+spike three-member mix (`0.374580`) >
  spike125 three-member mix (`0.374510`) > width256+spike three-member mix
  (`0.374431`). Width256 singles were unstable, especially seed 101, while
  width224 was much more stable.
- **Decision:** no public-test and no public docs update. Width224+spike is a
  real clue but still not enough margin. Do not rerun simple width224/256
  ensemble variants; if continuing, either require a stronger structural reason
  or test one final width224-specific objective/capacity refinement with a
  stricter stop rule.

### Screen O Width224 Refinement (paused - incomplete)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_screen_o_width224_refinement.yaml`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/`
- **Started:** `2026-05-25T15:26:25 America/Los_Angeles`.
- **Paused/stopped:** user needed the computer on `2026-05-25T17:17 America/Los_Angeles`.
- **PIDs at pause request:** venv Python `43896`; child Python `29780`.
- **Progress log:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/width224_refinement_20260525T152625.err.log`
- **Partial status file:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/partial_pause_20260525T1717.md`
- **Command to rerun later:** `C:\Users\david\.venvs\nlb-project\Scripts\python.exe -m nlb_project.cli.run_ensemble_screen --config configs/benchmarks/mc_maze_stndt_lite_diverse_screen_o_width224_refinement.yaml --log-level INFO`
- **Design:** final validation-only width224 refinement around the current
  public-test mixed components (`lr0013` and `depth5_dropout08`) plus width224
  variants (`spike125`, dropout `0.04/0.06`, spike `1.15/1.35`, identity
  `0.075`, and weight decay `0.00005`). No public-test.
- **Gate:** mean train/val `>= 0.3765` and at least 2/3 repeats above gate;
  seed 101 should not be weak, and improvement over Screen N best should be
  meaningful before considering a public-test.
- **Completed before stop:** all seed 0 singles/mixes and seed 101 singles.
  Seed 101 mixed scoring had begun. `metrics.csv` was not emitted because the
  screen had not completed all configured seeds/ensembles.
- **Best partial seed 0:** `lr0013_depth5_width224_spike135` scored `0.374857`.
- **Useful seed 101 partials:** `lr0013_depth5_width224_spike125` scored
  `0.376369`; `lr0013_depth5_width224_dropout04_spike125` scored `0.375159`;
  `lr0013_depth5_width224_dropout06_spike125` scored `0.375588`;
  `lr0013_depth5_width224_spike115` scored `0.375270`;
  `lr0013_depth5_width224_spike135` scored `0.375865`;
  `lr0013_depth5_width224_identity075_spike125` scored `0.375009`.
- **Current member at stop point:** seed 101 mixed-ensemble scoring immediately
  after `lr0013_depth5_width224_identity075_spike125`; next configured mixed
  member was `lr0013_depth5_width224_wd00005_spike125`.
- **Partial results usability:** directional only. Do not promote or
  public-test from this incomplete run; rerun the full config later if computer
  time is available.

### Screen O Width224 Refinement Restart (running)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_diverse_screen_o_width224_refinement.yaml`
- **Wrapper:** `scripts/run_screen_o_width224_refinement_overnight.ps1`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/`
- **Started:** `2026-05-26T22:48:49 America/Los_Angeles`.
- **Timeout:** `28800` seconds (8 hours).
- **Wrapper PID:** `7524`; **experiment PID:** `16492`; **helper Python PID:** `7576`.
- **Status file:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/width224_refinement_status_20260526T224849.txt`
- **Stdout:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/width224_refinement_20260526T224849.out.log`
- **Stderr/progress:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/width224_refinement_20260526T224849.err.log`
- **Command:** `powershell -ExecutionPolicy Bypass -File scripts\run_screen_o_width224_refinement_overnight.ps1 -TimeoutSeconds 28800`
- **Plan while user sleeps:** let this validation-only run finish or time out;
  do not run public-test. If it finishes, inspect `metrics.csv` and the latest
  ensemble leaderboard, update this local memory, run targeted checks, and
  commit/push only public-safe artifacts if complete.
- **Finished:** `2026-05-27T01:39:40 America/Los_Angeles`; runtime about
  `2h51m`. No Python/NLB training job remained afterward.
- **Leaderboard:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/ensemble_diversity_leaderboard_20260527T083926Z.txt`
- **Metrics:** `results/benchmark_runs/stndt_lite_diverse_screen_o_width224_refinement/metrics.csv`
- **Best result:** `lr0013_depth5_width224_dropout06_spike125` scored mean
  `0.375097` train/val co-bps, with seed scores `0.373914`, `0.375696`, and
  `0.375681`.
- **Gate decision:** failed. Gate was mean `>= 0.3765` with at least 2/3
  repeats above `0.3765`; this had 0/3 repeats above gate and mean was
  `0.001403` below the gate.
- **Top aggregate order:** dropout06 three-member mix (`0.375097`) >
  spike115 three-member mix (`0.374592`) > spike135 three-member mix
  (`0.374569`) > spike125 three-member mix (`0.374479`) > dropout04
  three-member mix (`0.374452`).
- **Decision:** no public-test, no docs/headline update, and no additional
  overnight run. Dropout06 is the best width224 refinement but still below the
  public-test discipline gate; do not promote without a stronger validation
  result.

### Screen P Full-Capacity Long-Train (complete — no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_screen_p_full_capacity.yaml`
- **Wrapper:** `scripts/run_screen_p_full_capacity_overnight.ps1`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_screen_p_full_capacity/`
- **Started:** `2026-05-28T02:34:22 America/Los_Angeles` (successful run after
  config fixes; earlier launch attempts failed fast on config validation or
  `spatial_n_heads` mismatch).
- **Finished:** `2026-05-28T03:32:39 America/Los_Angeles`; runtime about `58m`.
- **Command:** `powershell -ExecutionPolicy Bypass -File scripts\run_screen_p_full_capacity_overnight.ps1 -TimeoutSeconds 28800`
- **Design:** validation-only full train/val candidate screen testing
  fuller-capacity STNDT-lite vs long-train control. Four candidates with anchor
  objective (`mask_prob=0.6`, `heldin_loss_weight=1.0`, `use_mask_token=true`,
  `temporal_identity_scale=0.05`, `spike_loss_weight=1.0`, `ensemble_size=5`):
  P1 `192x4` long-train control (`300/40`, cosine+warmup); P2 `224x6` (`300/40`);
  P3 `256x6` (`300/40`); P4 `256x6` (`500/60`). No public-test.
- **Leaderboard:** `results/benchmark_runs/stndt_lite_screen_p_full_capacity/full_val_candidate_leaderboard_20260528T102018Z.txt`
- **Metrics:** `results/benchmark_runs/stndt_lite_screen_p_full_capacity/metrics.csv`
- **Exact leaderboard rows:**
  - rank 1: `0.368445` — candidate 2 (`d_model=224`, `n_layers=6`, `n_heads=8`,
    `max_epochs=300`, `patience=40`, cosine+warmup)
  - rank 2: `0.366427` — candidate 3 (`d_model=256`, `n_layers=6`, `n_heads=8`,
    `max_epochs=300`, `patience=40`)
  - rank 3: `0.365529` — candidate 4 (`d_model=256`, `n_layers=6`, `n_heads=8`,
    `max_epochs=500`, `patience=60`)
  - rank 4: `0.362970` — candidate 1 (`d_model=192`, `n_layers=4`, `n_heads=4`,
    `max_epochs=300`, `patience=40`, long-train control)
- **Selected (full train/val re-fit):** `0.368454` co-bps (`summary.md` /
  `metrics.csv` improved row), matching candidate 2.
- **Gate decision:** failed. Best `0.368445` is `0.008055` below the `0.3765`
  promotion floor and `0.010055` below the `0.3785` sanity/public-test
  consideration mean. 3-seed sanity was not run; public-test was not run.
- **Comparison:** below current local public-test headline `0.3830` by about
  `-0.014555`; does not support the “lite vs full STNDT capacity” hypothesis on
  this objective stack.
- **Prediction artifacts:** yes — standard benchmark HDF5s were written
  (`baseline_predictions.h5` and `improved_predictions.h5`, each ~206 MB).
- **Decision:** negative validation only. No public-test, no README/results
  headline update. Do not repeat this capacity/long-train screen without a new
  structural reason. **Closed negative** — remaining Screen P steps (sanity,
  public-test) were not run and are moot.

### Screen Q Block/Span Masking (complete — no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_screen_q_block_mask.yaml`
- **Wrapper:** `scripts/run_screen_q_block_mask_overnight.ps1`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_screen_q_block_mask/`
- **Started:** `2026-06-01T18:35:54 America/Los_Angeles`.
- **Finished:** `2026-06-01T20:10:28 America/Los_Angeles`; runtime ~`1h35m`;
  exit code `0`.
- **Command:** `powershell -ExecutionPolicy Bypass -File scripts\run_screen_q_block_mask_overnight.ps1 -TimeoutSeconds 28800`
- **Design:** validation-only screen testing `mask_mode=block_time` (contiguous
  time-span masking across all neurons) vs `mask_mode=bernoulli` control on the
  4-layer `learning_rate=0.0013` anchor recipe. Candidates: bernoulli control;
  block_time with `span_length` 2, 4, 8. No public-test.
- **Leaderboard:** `results/benchmark_runs/stndt_lite_screen_q_block_mask/full_val_candidate_leaderboard_20260602T030458Z.txt`
- **Metrics:** `results/benchmark_runs/stndt_lite_screen_q_block_mask/metrics.csv`
- **Exact leaderboard rows:**
  - rank 1: `0.368724` — Q1 bernoulli control
  - rank 2: `0.344828` — Q3 block_time span 4
  - rank 3: `0.344509` — Q2 block_time span 2
  - rank 4: `0.342474` — Q4 block_time span 8
- **Selected (full train/val re-fit):** `0.367252` co-bps (`metrics.csv`
  improved row), bernoulli control as expected.
- **Stage A gate:** failed. Best block_time (`0.344828`) is `0.023896` below
  bernoulli (`0.368724`); required `>= +0.003`. Stages B/C not run; no
  public-test.
- **Comparison:** does not support block/span masking on this STNDT-lite stack;
  bernoulli remains the validated masking mode. Headline remains `0.3830`
  local public-test (mixed ensemble).
- **Decision:** negative validation only. No README/results headline update.
  Do not repeat block masking on this backbone without a new structural reason.

### Screen R1 Unit Affine Calibration (complete — no promotion)

- **Config:** `configs/benchmarks/mc_maze_stndt_lite_screen_r1_unit_calibration.yaml`
- **Wrapper:** `scripts/run_screen_r1_unit_calibration_overnight.ps1`
- **Artifacts/logs:** `results/benchmark_runs/stndt_lite_screen_r1_unit_calibration/`
- **Started:** `2026-06-02T22:17:37 America/Los_Angeles` (paused/resumed mid-run).
- **Finished:** `2026-06-02T23:40:40 America/Los_Angeles`; exit code `0`.
- **Design:** validation-only screen for per-unit affine held-out calibration
  (`a_unit * logit + b_unit`, identity init, L2 toward identity) vs R0 control
  without calibration; reg strengths {10, 1, 0.1}.
- **Leaderboard:** `full_val_candidate_leaderboard_20260603T063453Z.txt`
- **Exact leaderboard rows:**
  - rank 1: `0.369113` — R0 control (`unit_calibration=false`)
  - rank 2: `0.366648` — R1c reg 0.1
  - rank 3: `0.365973` — R1b reg 1.0
  - rank 4: `0.365423` — R1a reg 10.0
- **Gate:** failed. Best R1 (`0.366648`) is `0.002465` below R0; required
  `>= +0.0015`. R2 hidden-state adapter not run; no public-test.
- **Decision:** negative validation. Headline remains `0.3830`. Do not repeat
  readout affine calibration on this backbone without a new structural reason.
- **Post-run audit (2026-06-03):** valid negative — full-val selection protocol
  confirmed; R0 selected with `unit_calibration=false`; all four candidates shared
  identical anchor/data/eval path (only calibration knobs differed); logs had no
  WARN/ERROR/NaN; pause/resume affected R0 training only and R0 still matched the
  Screen Q bernoulli anchor (`0.369113` vs `0.368724`). Secondary calibration-curve
  check on R1 artifacts was not run. R2 not justified.

- **Script:** `scripts/diagnose_validation_residuals.py --mode absolute`
- **Export helper (optional refresh):** `scripts/export_headline_val_predictions.py`
- **Artifacts:** `results/diagnostics/headline_residuals_20260602/`
- **Prediction source:** `results/benchmark_runs/stndt_lite_diverse_ensemble_screen/predictions/improved_predictions.h5`
  (headline `lr0013_depth5` mixed ensemble, seed 0, val split).
- **Design:** absolute Poisson residual breakdown by movement phase, speed
  tertile, and held-out unit; bias and loss-share disproportion metrics; automated
  Screen R go/no-go gate.
- **Key findings:**
  - Phase/speed slices are nearly uniform: disproportion ranges ~0.97–1.04 (max
    `low_speed` 1.038); no slice >= 1.10 threshold.
  - Highest absolute-loss units are high-rate (28, 29, 43) but **under-**
    disproportionate (0.78–0.86): they account for 32% of loss vs 38% of spikes.
  - Small positive mean bias (~0.00026) → slight over-prediction overall.
- **Screen R gate:** **NO-GO** (slice signal fail, unit signal fail). See
  `go_no_go.md` in artifact dir.
- **Decision:** do not design or run Screen R phase/unit loss reweighting on this
  evidence. May 2025 comparative diagnostic suggested width helped pre-movement
  in seed-0 deltas, but absolute analysis shows no concentrated error budget to
  target on the current headline ensemble.

### Headline Calibration/Dispersion Step 1.5 (complete — calibration path)

- **Script:** `scripts/diagnose_calibration_dispersion.py`
- **Artifacts:** `results/diagnostics/headline_calibration_dispersion_20260602/`
- **Prediction source:** same headline `lr0013_depth5` val ensemble as Step 1.
- **Design:** calibration curves (pred-rate bins vs mean target) overall, by
  high/mid/low-rate unit group, and units 29/43/17; dispersion check (Var/target
  vs Poisson mean) binned by predicted rate.
- **Calibration:** signal **yes** — 5 overall bins with >= 5% relative error;
  unit 29 over-predicts at high predicted rates (max |error| ~0.010 in bin 9);
  unit 43 under-predicts at highest bin (~+0.010).
- **Dispersion:** signal **no** — median Var/mean ratio 0.976; 0 bins >= 1.25.
  Poisson/readout variance model is adequate; NB/ZINB not supported as first test.
- **Recommendation:** `calibration_adapter_or_unit_scale_bias` if pursuing a
  structural change; not phase-weighted loss, not NB head.

### LFADS baseline track (setup — 2026-06-03)

- **Not a STNDT-lite screen.** Separate `lfads-torch` env and scripts under
  `scripts/prepare_lfads_mc_maze.py`, `scripts/run_lfads_mc_maze_smoke.py`,
  `scripts/export_lfads_rates.py`, `scripts/evaluate_lfads_outputs.py`,
  `docs/lfads_baseline_plan.md`.
- **Headline unchanged:** STNDT-lite `0.3830` co-bps (5 ms). LFADS reference HDF5 is 20 ms.
- **Smoke train:** `results/lfads_smoke/20260603T074245Z/` (1 epoch, checkpoint saved).
- **Evaluation bridge (2026-06-03):** export + `nlb_tools.evaluate` on smoke subset;
  plumbing OK (`co-bps` negative on 1-epoch smoke — not a baseline score).
- **No public-test** until full val baseline + explicit approval.
