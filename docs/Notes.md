# R-L line DeepONet — lab log

Running log of decisions, tests and findings. Newest entries at the top of each section.
Findings are numbered F1, F2, ... and marked with how many seeds back them.

---

## Status

| stage | what | state |
|---|---|---|
| A | steady-state current + decaying homogeneous mode | **v1 pipeline complete and tested, 2026-09-17** |
| B | slow envelope, amplitude halves/doubles over 0.5 s | not started |
| C | faults (fast forced decay, profiles from the supervisor's simulator) | waiting on profiles |

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
