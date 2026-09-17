# R-L Line Surrogate

A physics-informed DeepONet surrogate of the converter-side R-L line of a grid-following
converter, built to be chained window by window like a step of an EMT solver.

This is the first rung of a ladder toward a neural surrogate of the full converter
(line + SRF-PLL + current and power controllers). The line alone has a closed-form solution,
which is exactly why it comes first: every part of the pipeline can be checked against ground
truth before that stops being possible.

**Status: v1 (2026-09-17)** — Stage A (steady state) pipeline complete and tested.

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
data/, runs/                    datasets and checkpoints (not tracked, regenerable)
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

**2. Train.** About one minute on an M1 Max.

```bash
python src/CLI_Train.py
```

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
```

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

## v1 reference result

Default configuration, seed 0, one run (see `docs/Notes.md` for the tests and caveats):

| | |
|---|---|
| parameters | 25,408 |
| training | 59 s, early stop at epoch 497 |
| assisted error (RMS) | 5.8e-4 pu |
| predicted error over 0.5 s, 40 windows (RMS) | **1.1e-3 pu** |
| compounding | 1.9x |

## Roadmap

- **Stage A** — steady state: done (v1).
- **Stage B** — slowly varying amplitude (`envelope_rate` on).
- **Stage C** — faults: fast forced current decay, from converter simulator profiles.
- **Next rung** — line coupled to the SRF-PLL, where the problem becomes nonlinear and three-phase.

## References

- S. Wang, P. Perdikaris, "Long-time integration of parametric evolution equations with
  physics-informed DeepONets", *J. Comput. Phys.* 475 (2023) 111855.
- I. Ventura et al., "Physics-Informed Neural Network Models for EMT Simulators",
  *IEEE Trans. Power Syst.* (2026).
- I. Karampinis, P. Ellinas, J. Vorwerk, S. Chatzivasileiadis, "Neural Operators for Power Systems:
  A Physics-Informed Framework for Modeling Power System Components", arXiv:2511.05216.
- M. K. Bakhshizadeh, S. Ghosh, L. Kocewiak, G. Yang, "Transient stability assessment of Type-4
  wind turbines based on an improved reduced order model".
