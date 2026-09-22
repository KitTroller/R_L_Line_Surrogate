# R-L line DeepONet — lab log

Running log of decisions, tests and findings. Newest entries at the top of each section.
Findings are numbered F1, F2, ... and marked with how many seeds back them.

---

## Status

| stage | what | state |
|---|---|---|
| A | steady-state current + decaying homogeneous mode | **closed 2026-09-21** — linear branch, 1.0e-5 pu chained error (3 seeds) |
| B | slow envelope, amplitude halves/doubles over 0.5 s | folded into C as the `drift` event (a slow ramp stays under the current limit; `exp(r t)` did not) |
| C | faults + noise: step, dip + linear recovery, phase jump, drift; band-limited noise in dv | **designed and prototype-checked 2026-09-21** — guide §11, to implement |

---

## 2026-09-21 — Stage A closed; Stage C designed

Everything below is on `line_stage_A.npz` (seed 0 data, 4250 train / 750 val windows) and scored on
`line_stage_A_traj.npz` (300 held-out trajectories x 40 windows). "Chained error" is
`predicted_error_rms`, the deployed number. Medians over seeds 0, 1, 2 unless stated.

### Results table

| config | chained error | per-window error (assisted) | compounding | epochs |
|---|---|---|---|---|
| tanh branch, w_phys 0 | 1.97e-3 | 8.3e-4 | 2.37 | early stop ~220-480 |
| tanh branch, w_phys 0.3 (v1) | 9.65e-4 | 4.8e-4 | 2.03 | early stop ~460-540 |
| tanh branch, w_phys 1 | 8.90e-4 | 4.5e-4 | 1.98 | early stop ~390-640 |
| tanh branch + hard IC | 6.94e-4 | 4.7e-4 | 1.31 | early stop ~450-510 |
| **linear branch** | **1.03e-5** (0.90-1.26e-5) | 7.5e-6 | 1.37 | 3000 cap, still improving |
| linear branch + hard IC | 1.26e-5, 1.90e-5; seed 2 failed (2.0e-3, F13) | 1.2e-5 | 1.25 | 3000 cap / 285 |

### Findings

- **F7 — the physics term halves the error, and 0.3 is enough (3 seeds).** w_phys 0 / 0.3 / 1 gives
  1.97e-3 / 9.65e-4 / 8.90e-4. The same ~2x as the PLL, not the larger gain predicted for a linear
  system.
- **F8 — hard IC fixes the handover, not the window (tanh, 3 seeds).** i = i0 + (t/T_w)*G(...) makes
  the IC error 9e-9 instead of 7.5e-4 (F3). The per-window error does not change (4.7e-4 vs 4.8e-4)
  but compounding drops 2.0 -> 1.3, so the chained error improves 1.4x. F2's sawtooth was the soft IC.
- **F9 — the tanh branch cannot memorise (overfit test, 32 train windows, 1 seed each).** Train MSE
  stays above 1e-6 at lr 3e-3 (9.6e-7) and gets *worse* at lr 3e-4 (5.5e-6); weight decay made no
  difference; validation stays near 4e-4. This answers F5: the plateau was the tanh branch, not the
  optimiser.
- **F10 — a linear branch (identity activations) is ~94x better (3 seeds).** Chained error 1.03e-5,
  worst trajectory 2.0e-5, compounding 1.37, validation MSE 7e-11. Reason: the true operator is
  *linear* in (i0, dv) — i(t) = i0*exp(-t/tau) + integral of the forcing — so the ideal branch is a
  linear map. A deep linear branch has no more expressive power than one matrix, but one affine layer
  trained to 1.6e-4 (scratch, 1 seed, early stop 249): depth helps optimisation only. All three runs
  hit the 3000-epoch cap still improving 8-18% per 500 epochs; more epochs would not change the
  ranking.
- **F11 — Stage A inputs span only ~5 directions.** SVD of the 126 branch inputs (i0 + 125 dv
  samples) over 5000 windows: 99.9998% of the variance in 4 directions, the 5th singular value is
  2.7e-6 of the first. Every Stage A input is a mix of i0, cos and sin of the 50 Hz carrier and two
  small frequency-offset corrections. Three consequences:
  - The linear branch trained on the **32-window overfit set** reaches validation MSE 3.6e-8 and a
    chained error of 2.6e-4 — better than every tanh model trained on 4250 windows.
  - It **extrapolates along directions it has seen** (amplitude x2: 4.7e-5; DC offset x3: 3.2e-5;
    tanh 7.2e-2 and 2.5e-2) but **not along ones it has not** (+5% 5th harmonic: 6.7e-2; 3rd:
    2.8e-2). Scratch test, 1 seed.
  - The 94x is a result on a 5-direction input family. **Stage A is saturated**: it can no longer
    separate architectures. Stage C exists to add the directions deployment will present.
- **F12 — hard IC adds nothing on top of the linear branch.** Two good seeds 1.26e-5 and 1.90e-5 vs
  0.90-1.26e-5 without it, plus the failed seed (F13). The linear branch already gets i0 nearly
  right, so there is little handover error left to remove. Hard IC keeps one property worth
  remembering for the handoff: exact continuity at every window join, by construction.
- **F13 — early stopping killed runs, not the architecture.** With patience 40, linear seed 2 stopped
  at epoch 170 (7.5e-4) and all three linear + hard-IC runs at 96-137 (~1.1e-3). At lr 3e-3 the
  linear branch passes through a noisy phase where validation jumps to 100-360x its best. Scratch
  reruns of the failures: patience 150 -> 1.29e-5 (linear s2) and 1.93e-5 (linear + hard IC s0);
  lr 1e-3 alone was not enough (3.2e-5, and 3.95e-4 still stopped at 176). Two fixes applied:
  - `patience: 150` is now the default.
  - The improvement test is relative, `best * (1 - 1e-4)`. The old absolute `best - 1e-9` asked for
    a 2% gain per epoch at a loss of 5e-8, so good runs were stopped while still improving.
  - Still fragile: linear + hard IC seed 2 stopped at 285 because epoch 135 was one lucky low value
    (2.0e-6) and the typical loss afterwards was 5x higher. More patience only means more chances to
    get lucky. **Before Stage C** (noisier losses): add `min_epochs`, or early-stop on a smoothed
    validation loss (e.g. the median of the last 10 epochs) while still saving the best raw
    checkpoint.
- **F14 — the trunk is a continuous function of t (v1, 1 seed).** RMS error at the 125 training
  times 7.79e-4, at the 125 never-seen midpoints 7.78e-4. The model can be queried at any time in
  the window; nothing is interpolated.

### Housekeeping

- 2026-09-18: the spike guard (abort when validation > 50x best) killed two runs that SOAP would
  have recovered (its spikes reach ~70x). The trainer now aborts on a non-finite loss only; the two
  runs (w_phys 0 seed 1, hard IC seed 1) were rerun and are in the table.
- New model flags, both in the run tag: `model.hard_ic` (`_hic`) and `model.branch_act`
  (`_blinear`). Old checkpoints load with the defaults (`False`, `tanh`).
- `plot_sweep.py` takes lists for `match=` (all must appear) and `exclude=` (none may appear), and
  refuses to plot runs that differ in anything but `x` and the seed.
- Overfit runs go to `sweeps_overfit/` and `runs_overfit/` (`runs_overfit/` is git-ignored) so they
  cannot overwrite main records — the tag does not include `lr`.

### Stage C design (details and code: guide §11)

- **One rule from physics: inductor current cannot step.** A "step" is a ramp, at most 1 pu per
  5 ms (supervisor). The amplitude stays under the current limit (1.0 pu assumed; to confirm).
- **Every event is built from one primitive**, a 0 -> 1 ramp with an exact derivative.
  A(t) = I0 + a1*ramp1 + a2*ramp2 and phi(t) = phi0 + dphi*ramp_p; then
  i = A cos(theta), di/dt = A' cos(theta) - A (omega_g + phi') sin(theta), dv = ell*di/dt + R*i.
  Still analytic, still no integrator. Events: clean, drift (replaces Stage B), step, dip + linear
  recovery, phase jump.
- **F15 — sharp ramp corners cost 10^7; rounded corners cost nothing.** New test T11, the linear
  floor: the best linear map from branch inputs to current, by least squares, scored on held-out
  windows. No linear branch can beat it, and it needs no training. Stage A: 2e-16. Stage C with
  sharp (piecewise-linear) ramps: **8.1e-7**, all of it in windows containing a corner (2.7e-6 vs
  5e-9). At a sharp corner dv jumps, so what happens *between* two samples changes the current and
  the samples do not show it. Rounding each corner over c (slope = difference of two raised-cosine
  steps, value = its integral, still exact): c = 0.2 ms -> 1.9e-11, **1 ms -> 1.3e-14**,
  2 ms -> 5.7e-16. A raised cosine over the whole ramp: 1.6e-10. Chosen: straight ramps (the
  supervisor's linear recovery) with 1 ms rounded corners. The ramp is then C2, dv has no jumps, and
  the physics residual can be zero everywhere. Training agreed (prototype, 1 seed, 1500 epochs,
  linear branch): sharp corners 5.5e-4 chained error, raised cosine 1.2e-4.
- **Starts on the sample grid anyway** (integer sample indices x dt): simulator events sit on time
  steps, the per-window `transition` flag becomes integer arithmetic, and `corner=0` stays a fair
  comparison.
- **F16 — the handover point must be a training point (prototype, 1 seed).** With trunk features up
  to 1 kHz (`model.F=20 model.max_freq=6283`) the prediction at t = S*dt — one step past the
  training grid, where the `t_ext` handover reads it — is 19-38x worse than on the grid. Per-window
  error improved 3x, yet the chained error got *worse* (compounding 16-24x). Fix: Stage C
  evaluation sets let neighbouring windows **share their boundary sample** (stride S-1; 40 windows
  still fit in 5000 samples) and the handover takes the prediction at the last training sample.
  Same checkpoints, nothing retrained: 5.6e-4 -> **6.3e-5** without noise, 9.6e-4 -> 1.0e-4 with
  noise. F=4 is barely affected (x1-2), which is why Stage A never showed it. The flag travels in
  the eval set's metadata; scoring a shared-boundary set with the old handover gives compounding
  57-195x.
- **Prototype training on the final design (1 seed, linear branch unless stated, shared
  boundary):** F=4 1.5e-4, **F=20 6.3e-5** (clean 2.6e-5, step 5.4e-5, dip 8.1e-5, phase jump
  9.3e-5); with noise F=4 1.7e-4, F=20 1.0e-4. tanh branch (F=4): 9.1e-4, overfitting with a 28x
  train/val gap. So: linear still wins (6x at equal trunk), the trunk needs features to 1 kHz, noise
  costs ~1.6x, and phase jumps and dips set the number.
- **Noise = band-limited white noise in dv, made backwards** as a sum of K random steady-state tones
  (a multisine), each exactly like the Stage A carrier. Not per-sample (zero-order-hold) noise:
  that current has a kink at *every* sample, so the physics loss would fight the data. The band is
  100 Hz-1 kHz: below it the line's admittance explodes (1/R = 200 at DC), above it the current
  response is tiny and the trunk would need very high Fourier features. This is noise the current
  really responds to, not measurement noise to be ignored (a different problem; ask the supervisor
  whether sensor noise matters).
- **Windows are cut from long event trajectories** (5 per trajectory, half placed over the event
  onset), and each window gets its own B*exp(-t/tau) so i0 keeps its wide spread. Evaluation stays
  300 trajectories x 40 aligned windows (seed + 10000), scored per event type, and per window as
  "in a transition" (overlaps a fast ramp) vs steady.
- **The guide's code was checked before it was written down**: every guided-tier block of §11,
  pasted into a copy of this repo, passes T6-T11 — derivative vs a 1e-8 s finite difference 4e-9;
  an independent exponential integrator reproduces every event type to 3.8e-7; the multisine's dv
  equals the sum of its tones to 2e-15; per-window kernel 3e-17; linear floor 2e-14 — and trains
  end to end through `CLI_Train.py` with per-event keys in the record. Directions needed for
  99.99% of the branch-input variance: Stage A ~4, Stage C 13, Stage C + noise 27 (sharp corners
  had inflated it to 43). 37% of training windows overlap a fast ramp.

### Open questions for the supervisor

1. Which block is the deliverable, and where is its boundary: the line in abc (linear), the line in
   the PLL's dq frame (nonlinear once omega varies), or the converter?
2. The current limit (assumed 1.0 pu) and the fastest realistic current change (assumed 1 pu in 5 ms).
3. A sample of his simulator's waveforms around a fault, to check the event shapes and ranges.
4. Interface: abc or dq, per phase, time step, per-unit base.
5. Does relative accuracy at low current matter (F4), or is absolute error what counts?
6. Should the model be robust to sensor noise, i.e. see noisy dv but predict the clean current?

---

## 2026-09-17 — pipeline v1 complete and tested

The full pipeline exists and ran end to end on the laptop: seeded dataset generation, training,
checkpoint + JSON record, recurrent rollout evaluation, and plots.

```
src/Line_Physics.py            ODE  ell*di/dt + R*i = dv, closed-form current and exact di/dt
src/Line_Dataset_Generator.py  LHS sampling, training windows, evaluation trajectories, npz I/O
src/Line_Operator.py           Unstacked DeepONet + Single_PINN benchmark
src/Line_Residual.py           trunk Fourier features, physics residual via autograd
src/Line_Trainer.py            split, normalisation, SOAP training, early stop, checkpoint, record
src/Line_Evaluate.py           load_checkpoint, batched rollout (chained / assisted), metrics
src/CLI_Dataset.py             seed is mandatory; writes {name}.npz and {name}_traj.npz
src/CLI_Train.py               key=value overrides; list values = sweep; list=true; index=K
src/analysis/                  style.py, plot_run.py, plot_sweep.py
```

### How it was tested

- **Generator.** Adding the homogeneous term B*exp(-t/tau) changes i and leaves dv unchanged to
  1.8e-10. Central-difference check of the stored di/dt: 1.6e-4 relative, which is exactly the
  O(dt^2) truncation floor (dt^2/6)*A*w^3 — not a formula error (the old sign bug gave 3e-2 and an
  exponential shape).
- **Residual.** Exact solution substituted into `didt + i/tau - dv/ell`: 1.5e-5 pu/s on a
  157 pu/s scale (machine zero in float32). The wrong sign gave 4.4e+2.
- **Checkpoint.** Save -> load round trip reproduces the network output exactly (max diff 0.0), and
  re-running the rollout from the reloaded file reproduces the JSON record exactly.
- **Rollout.** Assisted arm equals predicting each window in isolation (8e-7). Window 0 is identical
  in both arms. The handover must use the prediction at t = S*dt (the `t_ext` point), not the last
  sample: on the same model, `i_hat[:, -1]` gave compounding 6.59x, `t_ext` gave 1.57x.
- **CLIs.** Refuse: missing/non-int seed, unknown argument, overwriting a dataset, a typo'd config
  key (struct mode), sweeping a key not in the run tag (runs would overwrite each other), `index=`
  out of range, a missing eval set (checked before training). A crashing config does not stop the
  sweep and gives exit code 1. Paths do not depend on the working directory.
- **Plots.** `plot_run.py` recomputes the rollout and checks it against the record.

### Reference run

`python src/CLI_Dataset.py seed=0` then `python src/CLI_Train.py` (default config), M1 Max, MPS.
The first run and an independent re-run gave identical numbers.

| | |
|---|---|
| tag | `line_stage_A_n5000_W40_F4_mf628_wp0.3_s0sp0_h64` |
| params | 25,408 |
| wall time | 59.3 s, early stop at 497 / 800, best epoch 457 |
| val MSE(i) / train MSE(i) | 7.9e-7 / 6.8e-7 |
| assisted_error_rms (true i0 every window) | 5.8e-4 pu |
| **predicted_error_rms (chained, 40 windows)** | **1.11e-3 pu** |
| predicted_error_max (worst trajectory) | 2.56e-3 pu |
| compounding | 1.90x |

Figures: `graphs/line_stage_A_n5000_W40_F4_mf628_wp0.3_s0sp0_h64/01..04`.

### First findings — ONE seed, provisional

- **F1 — error does not compound over 40 handovers.** The chained error peaks in the first ~100 ms
  (while the B mode is still alive) and then settles near 1.1e-3 instead of growing. Consistent with
  the physics: a handover error decays with tau = 63.7 ms, where the PLL's integrator kept it.
- **F2 — the handover is visible as a sawtooth.** Every 12.5 ms the chained error jumps, then
  decays inside the window (RMS over all windows: 1.45e-3 at the start, 1.03e-3 at the end).
- **F3 — the initial condition is only soft.** Given the exact i0, the network's own
  i_hat(t=0) still differs from it by 7.5e-4 RMS. Visible in figure 04 as jumps in the *assisted*
  error at every handover. A hard-IC ansatz, i(t) = i0 + (t/T_w)*G(...), would make this exactly
  zero and is the obvious candidate for shrinking F2.
- **F4 — the error is roughly absolute, not relative.** About 8e-4 pu for every amplitude below
  0.45 pu, rising above 0.5 pu. That is 0.1% of a 1 pu current but 1.6% of a 0.05 pu one — MSE
  weights absolute error. The worst trajectory (#172) is a ~1 pu one. Whether low-current relative
  accuracy matters for the EMT use is a question for the supervisor.
- **F5 — not data-limited; the plateau is optimisation, not approximation.** Train and validation
  MSE are nearly equal, and the operator is linear, so a DeepONet can represent it exactly — yet
  training stops at MSE ~8e-7. Open question: optimiser, precision, soft IC (F3), or capacity.
- **F6 — why this trains ~1000x faster than the PLL (measured).**

  | | s/epoch |
  |---|---|
  | line, MPS | 0.12 |
  | PLL 45k params, MPS | 7.4 (x62: 40x more rows per epoch — 200,000 windows vs 5,000 — and 1.6x per step) |
  | PLL 45k params, HPC CPU | 45-53 (~6.5x slower than MPS) |
  | PLL L4_w128, HPC CPU | 150 (x ~1000 epochs = the ~40 h jobs) |

  Laptop benchmark: MPS is 4.2x faster than the laptop CPU at every size up to 174k params, so a
  GPU should help the big PLL jobs substantially compared with HPC CPU nodes. Not yet measured on
  an HPC GPU.

### Known limits of v1

- Stage A only: no envelope, no faults, no noise.
- One seed behind every number above.
- Run tags do not include `lr`, `epochs`, `patience`, `w_deriv` or `val_frac`, so two runs that
  differ only in those overwrite each other (`CLI_Train.py` refuses this inside one sweep, not
  across invocations).

---

## Settled decisions

- `Xc = 0.1` is a per-unit **reactance**: ell = Xc/omega_b = 3.183e-4 s, tau = 63.66 ms (not 20 s).
- Data is generated **backwards**: prescribe i(t), differentiate analytically, dv = ell*di/dt + R*i.
  No integrator.
- `B*exp(-t/tau)` spans the kernel of the line operator: it changes i without changing dv, which is
  what makes i0 an independent branch input.
- Train on **independent windows** (S = 125, dt = 100 us). Trajectories (300 x 40 windows) exist
  only to measure rollout, generated with seed + 10000 so they are held out.
- Rollout handover at t = S*dt (`t_ext`).
- Imports are one-way: `Line_Trainer -> Line_Evaluate`, never back.
- The line's branch is **linear** (`model.branch_act=linear`) in every stage: its operator is linear
  in (i0, dv) (F10). Not a rule for nonlinear blocks.
- Early stopping: relative improvement `best * (1 - 1e-4)`, `patience: 150` (F13).
- Stage B is the `drift` event of Stage C.
- Stage C ramps are straight with 1 ms rounded corners, starts on the sample grid (F15).
- Stage C noise is a multisine in dv (100 Hz-1 kHz, RMS up to 0.005 pu per trajectory), never
  per-sample noise.
- Before training on a new dataset, compute its linear floor (T11). A run far above it is an
  optimisation problem; a run near it is a data problem.
- Stage C evaluation sets share the boundary sample between windows; the handover is the prediction
  at the last training sample (F16).
- Stage C trunk: `model.F=20 model.max_freq=6283` (features every 50 Hz up to 1 kHz) — prototype
  evidence, to confirm on 3 seeds.
