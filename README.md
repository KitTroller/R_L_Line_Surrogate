# R-L Line Surrogate

A physics-informed DeepONet surrogate of the converter-side R-L line of a grid-following
converter, built to be chained window by window like a step of an EMT solver.

This is the first rung of a ladder toward a neural surrogate of the full converter
(line + SRF-PLL + current and power controllers). The line alone has a closed-form solution,
which is exactly why it comes first: every part of the pipeline can be checked against ground
truth before that stops being possible.

**Status (2026-09-21):** Stage A (steady state) closed. With a linear branch the chained error
over 0.5 s is 1.0e-5 pu (3 seeds), 94x better than the tanh branch. Stage C (faults and noise)
is designed and prototype-checked; implementation is next. See `docs/Notes.md`.

---

## The system

Per phase, with dv = u - v the voltage across the inductor:

```
ell * di/dt + R * i = dv          R = 0.005 pu,  Xc = 0.1 pu,  ell = Xc / omega_b = 3.183e-4 s
                                  tau = ell / R = 63.66 ms,  f0 = 50 Hz
```

`Xc` is a per-unit reactance, so the time constant is 63.66 ms, not the 20 s you get by reading
it as henries.

## How the data is made

Backwards, with no numerical integrator. A current is prescribed in closed form:

```
i(t) = I0 * exp(r*t) * cos(omega_g*t + phi)  +  B * exp(-t/tau)
dv(t) = ell * di/dt + R * i                    (di/dt written out analytically)
```

- `r = 0` in Stage A (steady state). A nonzero `r` is the Stage B envelope.
- `B * exp(-t/tau)` is the homogeneous solution: it changes `i` but leaves `dv` unchanged, so the
  initial current carries information the forcing does not.
- Parameters are Latin-hypercube sampled (`config/Line_Initial_Conditions.yml`).

Training uses **independent 12.5 ms windows** (S = 125 samples at dt = 100 us). A separate set of
continuous 0.5 s trajectories (40 windows each, a different seed) is used only to evaluate the
chained rollout.

## The model

An unstacked DeepONet (`src/Line_Operator.py`):

- **branch**: `[i0, dv(t_1..t_S)]` -> 64-dimensional coefficients
- **trunk**: `[t, sin/cos Fourier features]` -> 64-dimensional basis
- output `i(t)` = their inner product

Loss = MSE on `i` + `w_phys` x the normalised physics residual `di/dt + i/tau - dv/ell`
(autograd through the trunk), optimised with SOAP.

Two switches, both part of the run tag:

| flag | effect | tag |
|---|---|---|
| `model.branch_act=linear` | identity activations in the branch. The line's operator is linear in `(i0, dv)`, so this is the right inductive bias here — not for a nonlinear block. | `_blinear` |
| `model.hard_ic=true` | `i = i0 + (t/T_w) * G(...)`: the initial condition holds exactly | `_hic` |

At deployment the operator is **chained**: the predicted current at the end of one window
(t = S*dt) becomes the next window's `i0`.

## Repository layout

```
config/
  Line_Constants.yml            physics constants, sampling rate, windows
  Line_Initial_Conditions.yml   sampling ranges
  Line_DeepONet_Models.yml      architecture defaults
  Line_Training.yml             training defaults (every CLI override lands here)
src/
  Line_Physics.py               the ODE and the closed-form current
  Line_Dataset_Generator.py     datasets and npz I/O
  Line_Operator.py              DeepONet and a plain-MLP benchmark
  Line_Residual.py              Fourier features and the physics residual
  Line_Trainer.py               training loop, checkpoint, JSON record
  Line_Evaluate.py              checkpoint loading, rollout, metrics
  CLI_Dataset.py                build a seeded dataset pair
  CLI_Train.py                  train one config or a sweep
  analysis/                     plots
docs/
  Notes.md                      lab log: decisions, tests, findings
  Line_Generator_Build.html     step-by-step build guide (open in a browser)
graphs/                         figures, one folder per run
sweeps/                         one JSON record per run (tracked)
sweeps_overfit/                 records of the overfit tests, kept apart (tracked)
data/, runs/, runs_overfit/     datasets and checkpoints (not tracked, regenerable)
```

## Quickstart

Tested with Python 3.14 on an Apple M1 Max (MPS). CUDA and CPU are picked automatically.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Build the data.** The seed is mandatory.

```bash
python src/CLI_Dataset.py seed=0
```

Writes `data/line_stage_A.npz` (5000 training windows) and `data/line_stage_A_traj.npz`
(300 evaluation trajectories, seed + 10000).

**2. Train.** A few minutes on an M1 Max.

```bash
python src/CLI_Train.py model.branch_act=linear epochs=3000
```

Without `model.branch_act=linear` you get the v1 tanh branch (about one minute, ~100x less
accurate).

Any config key can be overridden with `key=value`, nested keys with dots:

```bash
python src/CLI_Train.py w_phys=0.1 model.F=2 epochs=200
```

**3. Sweep.** Any list-valued key becomes a sweep axis. Quote brackets (zsh globs them).

```bash
python src/CLI_Train.py 'w_phys=[0,0.3,1]' 'seed=[0,1,2]' list=true
python src/CLI_Train.py 'w_phys=[0,0.3,1]' 'seed=[0,1,2]'
```

The first prints the 9 configurations without training. On an LSF cluster, add
`index=$LSB_JOBINDEX` to the same command in an array job (`#BSUB -J name[1-9]`) to run one
configuration per job.

**4. Plot.**

```bash
python src/analysis/plot_run.py                  # newest checkpoint; or tag=<run tag>
python src/analysis/plot_sweep.py x=w_phys       # metric vs a config key, every seed shown
python src/analysis/plot_sweep.py x=branch_act 'match=[line_stage_A_n5000,_wp0.3_]' exclude=_hic
```

`match=` keeps runs whose tag contains every listed string, `exclude=` drops runs containing any.
The sweep plot refuses runs that differ in anything other than `x` and the seed.

Runs that differ only in `lr`, `epochs` or `patience` get the same tag and would overwrite each
other; send them elsewhere with `results_dir=... runs_dir=...` (as the overfit tests do).

## What a run produces

- `runs/<tag>.pth` — weights, architecture config, training config, normalisation statistics,
  history.
- `sweeps/<tag>.json` — the record: configuration, epochs run, wall time, losses and rollout
  metrics.
- `graphs/<tag>/` — training history, rollout error, error vs amplitude, truth vs predicted.

The rollout metrics:

| metric | meaning |
|---|---|
| `assisted_error_rms` | every window is given the true `i0` — the operator alone |
| `predicted_error_rms` | windows chained on the model's own predictions — what is deployed |
| `predicted_error_max` | the worst trajectory |
| `compounding` | predicted / assisted — the cost of chaining |

## Stage A results

Median of seeds 0, 1, 2; 25,408 parameters. Errors in pu, RMS over the 300 held-out trajectories.

| branch | predicted error, 0.5 s chained | assisted error | compounding |
|---|---|---|---|
| tanh (v1) | 9.7e-4 | 4.8e-4 | 2.0x |
| tanh + hard IC | 6.9e-4 | 4.7e-4 | 1.3x |
| **linear** | **1.0e-5** | 7.5e-6 | 1.4x |

The linear result needs `epochs=3000` and `patience=150` (the default). The caveat that matters:
Stage A inputs span only ~5 directions, so this is accuracy on a narrow family of waveforms. A
model trained on it still fails on a harmonic it never saw. Stage C exists to fix that. Findings
F1-F14 with their evidence are in `docs/Notes.md`.

## Roadmap

- **Stage A** — steady state: closed (2026-09-21).
- **Stage B** — slowly varying amplitude: folded into Stage C as the `drift` event.
- **Stage C** — events built from straight ramps with 1 ms rounded corners (exact derivatives):
  clean, drift, step, dip + linear recovery, phase jump; plus band-limited white noise in `dv`
  generated as a multisine. Evaluation windows share their boundary sample so the handover point
  is a training point. Each event type scored separately. A prototype of the design reached
  6.3e-5 pu chained error (1 seed). Build guide §11; findings F15-F16.
- **Next rung** — the line in the PLL's dq frame, or the PLL itself, where the operator is no longer
  linear. Candidate branch: a linear path plus a zero-initialised tanh MLP.

## References

- S. Wang, P. Perdikaris, "Long-time integration of parametric evolution equations with
  physics-informed DeepONets", *J. Comput. Phys.* 475 (2023) 111855.
- I. Ventura et al., "Physics-Informed Neural Network Models for EMT Simulators",
  *IEEE Trans. Power Syst.* (2026).
- I. Karampinis, P. Ellinas, J. Vorwerk, S. Chatzivasileiadis, "Neural Operators for Power Systems:
  A Physics-Informed Framework for Modeling Power System Components", arXiv:2511.05216.
- M. K. Bakhshizadeh, S. Ghosh, L. Kocewiak, G. Yang, "Transient stability assessment of Type-4
  wind turbines based on an improved reduced order model".
