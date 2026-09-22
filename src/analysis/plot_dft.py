"""Where is the current's power, where do the trunk's Fourier combs sit, and which comb won?

    python src/analysis/plot_dft.py                                  # Stage A eval trajectories
    python src/analysis/plot_dft.py data=line_stage_C_traj.npz       # any trajectory file, later
    python src/analysis/plot_dft.py branch=tanh                      # score tanh runs instead
    python src/analysis/plot_dft.py sweeps=sweeps_comb_gpu           # the HPC runs (hpc/exp01_comb.txt)

Writes two figures into graphs/:
    dft_<stem>.png    the spectrum of the current over full 0.5 s trajectories, the RMS of
                      everything above a frequency f, and the four combs of the 2x2 test
    comb_<stem>.png   the 2x2 test itself: per-window and chained error for every
                      (lowest feature, top) cell found in sweeps/, every seed shown
                      (comb_<stem>_<dir>.png when sweeps=<dir> is given)

The comb is build_ff_trunk_input's: features at max_freq*k/F, k = 1..F. So the LOWEST feature
is also the SPACING (max_freq/F), and the TOP is max_freq. The PLL found the lowest feature
controls (its notes F51, F56). This separates the two on the line.

The file is read with numpy, not through Line_Dataset_Generator, so it works while the
pipeline modules are being edited.

Reading the Stage A curve: the flat plateau near 1e-6 pu above ~1000 rad/s is not physics.
Line_Simulator builds its time grid in float32; over 0.5 s the sample times are off by up to
2.7e-8 s, which moves the current by ~1.7e-6 pu RMS. That is the floor of the Stage A eval set
(harmless at 1e-5). Stage C builds its grid in float64 (guide §11.6).
"""
import json, sys
from collections import defaultdict
from statistics import median

import numpy as np
from omegaconf import OmegaConf

from style import ROOT, C, save, plt

# The four cells of the 2x2: (F, max_freq). Lowest feature 157 or 314 rad/s x top 628 or 6283.
COMBS = [(4, 628), (40, 6283), (2, 628), (20, 6283)]
# Record keys that must be equal across the compared runs (F, max_freq and seed may vary).
SAME = ["dataset", "eval_dataset", "S", "dt", "w_phys", "w_deriv", "split_seed", "lr",
        "batch_size", "arch", "n_layers", "width", "hard_ic", "branch_act"]
DEFAULTS = {"hard_ic": False, "branch_act": "tanh"}


def load_trajectories(name):
    """(n_runs, N) full trajectories of the current, stitched back from the eval windows."""
    z = np.load(ROOT / "data" / name, allow_pickle=False)
    meta = json.loads(z["meta_json"].item())
    W, S, dt = meta["W"], meta["S"], meta["dt"]
    i = z["i"].astype(np.float64).reshape(-1, W, S)          # rows are ordered (run, segment)
    if meta.get("shared_boundary", False):                   # window k+1 starts ON window k's last sample
        i = np.concatenate([i[:, 0], i[:, 1:, 1:].reshape(len(i), -1)], axis=1)
    else:
        i = i.reshape(len(i), W * S)
    return i, dt


def spectrum(x, dt):
    """Mean power spectrum over runs (Hann window: a 0.5 s run is not periodic) -> f [rad/s], P."""
    x = x - x.mean(axis=1, keepdims=True)
    X = np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)
    return np.fft.rfftfreq(x.shape[1], dt) * 2 * np.pi, (np.abs(X) ** 2).mean(axis=0)


def rms_above(f, P, rms_total):
    """RMS [pu] of the part of the signal above each frequency: total RMS x sqrt(power share above)."""
    share_above = 1 - np.cumsum(P) / P.sum()
    return rms_total * np.sqrt(np.clip(share_above, 0, None))


def fig_dft(i, dt, stem):
    f, P = spectrum(i, dt)
    rms_total = np.sqrt(((i - i.mean(axis=1, keepdims=True)) ** 2).mean())
    above = rms_above(f, P, rms_total)
    cum = np.cumsum(P) / P.sum()
    marks = {q: f[np.searchsorted(cum, q)] for q in (0.90, 0.99, 0.9999)}

    # Window-leakage floor: a PURE carrier at the same RMS through the same pipeline. Where the
    # data's curve sits on this line, it is showing the Hann window, not the signal.
    t = np.arange(i.shape[1]) * dt
    phases = np.linspace(0, np.pi, 8, endpoint=False)
    tone = np.sqrt(2) * rms_total * np.cos(2 * np.pi * 50 * t[None, :] + phases[:, None])
    ft, Pt = spectrum(tone, dt)
    leak = rms_above(ft, Pt, rms_total)

    nyq = np.pi / dt
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 9.4), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 1.6]})
    keep = f > 0
    a = ax[0]
    a.plot(f[keep], P[keep] / P.max(), color=C["predicted"])
    a.set_yscale("log"); a.set_ylim(1e-16, 2)
    a.set_title("Power spectrum of the current (mean over trajectories, normalised to the peak)")
    a.set_ylabel("power / peak")

    a = ax[1]
    a.plot(f[keep], above[keep], color=C["predicted"], label="this data")
    a.plot(ft[ft > 0], leak[ft > 0], color=C["muted"], ls="--", lw=1.1,
           label="a pure 50 Hz tone at the same RMS (window leakage floor)")
    for e in (1e-4, 1e-5):
        a.axhline(e, color=C["muted"], lw=0.8, ls=":")
        a.text(f[keep][0] * 1.05, e * 1.15, f"{e:.0e} pu", color=C["muted"], fontsize=8)
    a.set_yscale("log"); a.set_ylim(1e-8, 2)
    a.set_title("RMS of everything above f — what a comb stopping at f leaves the MLP to build")
    a.set_ylabel("RMS above f [pu]")
    a.legend(loc="lower left")

    for a in ax[:2]:
        for q, fq in marks.items():
            a.axvline(fq, color=C["assisted"], lw=0.9, ls=":")
        a.axvline(nyq, color=C["ink"], lw=0.9)
    # one label for all three percentiles: on a line spectrum they sit almost on top of each other
    ax[0].text(0.02, 0.97, "dotted: frequency below which lies\n"
               + "\n".join(f"  {q * 100:g}% of the power:  {fq:.0f} rad/s" for q, fq in marks.items()),
               transform=ax[0].transAxes, va="top", fontsize=8.5, color=C["assisted"])
    ax[0].text(nyq, 1.3, "Nyquist\n" + f"{nyq:.0f} ", color=C["ink"], fontsize=8, va="top", ha="right")

    a = ax[2]
    for row, (F, mf) in enumerate(COMBS):
        feats = mf * np.arange(1, F + 1) / F
        a.scatter(feats, np.full(F, row), s=10, color=C["muted"], zorder=3)
        a.scatter(feats[:1], [row], s=46, color=C["predicted"], zorder=4)          # the lowest feature
        a.text(nyq * 1.08, row, f"F={F}, mf={mf}", va="center", fontsize=8.5, color=C["ink"])
    for fq in marks.values():
        a.axvline(fq, color=C["assisted"], lw=0.9, ls=":")
    a.set_yticks(range(len(COMBS)), [f"lowest {mf / F:.0f}" for F, mf in COMBS])
    a.set_ylim(-0.7, len(COMBS) - 0.3); a.grid(axis="y", visible=False)
    a.set_title("The four combs of the 2x2 test (large dot = lowest feature = spacing)")
    a.set_xscale("log"); a.set_xlim(10, nyq * 1.02)
    a.set_xlabel("frequency [rad/s]")

    fig.suptitle(f"data/{stem}_traj  ·  {i.shape[0]} trajectories of {i.shape[1] * dt:g} s  ·  "
                 f"resolution {f[1]:.1f} rad/s", x=0.01, ha="left", fontsize=9, color=C["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, f"dft_{stem}.png")

    print(f"\npower below   f_90% {marks[0.90]:.0f}   f_99% {marks[0.99]:.0f}   f_99.99% {marks[0.9999]:.0f} rad/s"
          f"   (resolution {f[1]:.1f})")
    for fr in (314, 628, 1257, 3142, 6283):
        k = np.searchsorted(f, fr)
        print(f"  RMS of the current above {fr:5d} rad/s: {above[k]:.1e} pu   (leakage floor {leak[k]:.1e})")


def fig_comb(stem, eval_name, branch, sweeps="sweeps"):
    recs = [json.loads(p.read_text()) for p in sorted((ROOT / sweeps).glob("*.json"))]
    recs = [r for r in recs if r.get("eval_dataset") == eval_name and r["status"] == "ok"]
    for r in recs:
        for k, v in DEFAULTS.items():
            r.setdefault(k, v)
    recs = [r for r in recs if r["branch_act"] == branch and not r["hard_ic"]
            and (r["F"], round(r["max_freq"])) in {(F, mf) for F, mf in COMBS}]
    if not recs:
        print(f"\nno finished {branch}-branch runs of the 2x2 on {eval_name} in {sweeps}/ yet -- comb figure skipped")
        return
    # the runs must differ only in the comb and the seed
    varies = {k: sorted({str(r.get(k)) for r in recs}) for k in SAME}
    varies = {k: v for k, v in varies.items() if len(v) > 1}
    if varies:
        sys.exit(f"these runs also differ in {varies} -- they are not one experiment")

    cells = defaultdict(list)                                      # (lowest, top) -> records
    for r in recs:
        cells[(round(r["max_freq"] / r["F"]), round(r["max_freq"]))].append(r)
    lows = sorted({lo for lo, _ in cells})
    tops = sorted({tp for _, tp in cells})

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = [("assisted_error_rms", "Per-window error (true i0 every window)", C["assisted"]),
              ("predicted_error_rms", "Chained error over 0.5 s (includes the handover, F16)", C["predicted"])]
    for a, (metric, title, col) in zip(ax, panels):
        for j, lo in enumerate(lows):
            xs, meds = [], []
            for k, tp in enumerate(tops):
                ys = [r[metric] for r in cells.get((lo, tp), [])]
                if not ys:
                    continue
                x0 = k + (j - (len(lows) - 1) / 2) * 0.12
                a.scatter([x0] * len(ys), ys, s=20, color=col, alpha=0.5, linewidths=0,
                          marker="o" if j == 0 else "s", zorder=3)
                xs.append(x0); meds.append(median(ys))
            a.plot(xs, meds, color=col, ls="-" if j == 0 else "--", marker="o" if j == 0 else "s",
                   ms=6, zorder=4, label=f"lowest feature {lo} rad/s")
            for x0, m in zip(xs, meds):
                a.annotate(f"{m:.2e}", (x0, m), textcoords="offset points", xytext=(6, 4), fontsize=8,
                           color=C["ink"])
        a.set_xticks(range(len(tops)), [f"top {tp}" for tp in tops])
        a.set_xlim(-0.5, len(tops) - 0.5)
        a.set_yscale("log"); a.set_title(title); a.set_ylabel(metric)
        a.legend(loc="best")
    n = sorted({len(v) for v in cells.values()})
    fig.suptitle(f"The 2x2 comb test on {stem}  ·  {sweeps}/  ·  {branch} branch  ·  dots = seeds, line = median, "
                 f"n = {'/'.join(map(str, n))} per cell", x=0.01, ha="left", fontsize=9, color=C["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"comb_{stem}.png" if sweeps == "sweeps" else f"comb_{stem}_{sweeps}.png")

    print(f"\n{'lowest':>7s} {'top':>6s} {'F':>3s} {'seeds':>5s} {'per-window (median)':>20s} "
          f"{'chained (median)':>17s} {'epochs run':>14s}")
    for lo in lows:
        for tp in tops:
            rs = cells.get((lo, tp), [])
            if not rs:
                print(f"{lo:7d} {tp:6d}   -   missing")
                continue
            print(f"{lo:7d} {tp:6d} {rs[0]['F']:3d} {len(rs):5d} "
                  f"{median(r['assisted_error_rms'] for r in rs):20.3e} "
                  f"{median(r['predicted_error_rms'] for r in rs):17.3e} "
                  f"{','.join(str(r['epochs_run']) for r in sorted(rs, key=lambda r: r['seed'])):>14s}")


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    unknown = set(cli) - {"data", "branch", "sweeps"}
    if unknown:
        sys.exit(f"unknown argument(s) {sorted(unknown)} -- allowed: data=, branch=, sweeps=")
    data = cli.get("data", "line_stage_A_traj.npz")
    if not (ROOT / "data" / data).exists():
        sys.exit(f"data/{data} not found")
    stem = data.removesuffix(".npz").removesuffix("_traj")

    i, dt = load_trajectories(data)
    fig_dft(i, dt, stem)
    sweeps = cli.get("sweeps", "sweeps")
    if not (ROOT / sweeps).is_dir():
        sys.exit(f"{sweeps}/ not found -- pull it from the cluster first")
    fig_comb(stem, data, cli.get("branch", "linear"), sweeps)
