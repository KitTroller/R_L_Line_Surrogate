"""Diagnostics for ONE trained model.

    python src/analysis/plot_run.py                  # newest checkpoint in runs/
    python src/analysis/plot_run.py tag=<run tag>    # a specific one

Writes into graphs/<tag>/:
    01_history.png             data loss and physics residual per epoch, best epoch marked
    02_rollout_error.png       where the error lives: across the 0.5 s horizon, inside one
                               window, and on the median and the worst trajectory
    03_error_vs_amplitude.png  is the error absolute (flat) or relative (rising with |i|)?
    04_truth_vs_predicted.png  the current itself: full horizon, a zoom on the first windows,
                               and the error on that zoom -- median and worst trajectory

The eval set is the one recorded inside the checkpoint, and the rollout is re-run through
Line_Evaluate, so the numbers here are the numbers in the run's JSON record -- checked below.
"""
import json, sys
import torch
from omegaconf import OmegaConf

from style import ROOT, C, save, plt
from Line_Dataset_Generator import Dataset_Generator
from Line_Evaluate import load_checkpoint, rollout


def newest_tag():
    runs = sorted((ROOT / "runs").glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit("no checkpoints in runs/ -- train one with src/CLI_Train.py")
    return runs[-1].stem


def fig_history(ck, tag):
    h = ck["history"]
    ep = range(1, len(h["train"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for a, key, title in [(ax[0], "i", "Data loss  MSE(i)"),
                          (ax[1], "physics_residual", "Physics residual  mean((res/s)^2)")]:
        a.plot(ep, [e[key] for e in h["train"]], color=C["train"], label="train")
        a.plot(ep, [e[key] for e in h["val"]], color=C["val"], label="validation")
        a.axvline(ck["best_ep"], color=C["muted"], lw=1)
        a.text(ck["best_ep"], 0.97, f"best epoch {ck['best_ep']} ", transform=a.get_xaxis_transform(),
               ha="right", va="top", color=C["muted"], fontsize=9)
        a.set_yscale("log"); a.set_title(title); a.set_xlabel("epoch")
    ax[0].legend(loc="lower left")
    fig.suptitle(tag, x=0.01, ha="left", fontsize=9, color=C["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, f"{tag}/01_history.png")


def fig_rollout(truth, pred, asst, dt, W, tag):
    R, _, S = truth.shape
    e_p = (pred - truth).reshape(R, W * S)
    e_a = (asst - truth).reshape(R, W * S)
    rms_p = e_p.pow(2).mean(1).sqrt()                                  # per trajectory
    t_ms = torch.arange(W * S) * dt * 1e3
    tl_ms = torch.arange(S) * dt * 1e3

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a = ax[0][0]
    a.plot(t_ms, e_p.pow(2).mean(0).sqrt(), color=C["predicted"], label="predicted (chained)")
    a.plot(t_ms, e_a.pow(2).mean(0).sqrt(), color=C["assisted"], label="assisted (true i0)")
    a.set_title("RMS error over the horizon, all trajectories"); a.set_xlabel("t [ms]"); a.set_ylabel("error [pu]")
    a.set_ylim(bottom=0)

    a = ax[0][1]
    a.plot(tl_ms, (pred - truth).pow(2).mean((0, 1)).sqrt(), color=C["predicted"])
    a.plot(tl_ms, (asst - truth).pow(2).mean((0, 1)).sqrt(), color=C["assisted"])
    a.set_title("RMS error inside one window, all windows"); a.set_xlabel("local t [ms]"); a.set_ylabel("error [pu]")
    a.set_ylim(bottom=0)

    order = torch.argsort(rms_p)
    for a, r, name in [(ax[1][0], int(order[R // 2]), "median"), (ax[1][1], int(order[-1]), "worst")]:
        a.plot(t_ms, e_p[r], color=C["predicted"], lw=1)
        a.plot(t_ms, e_a[r], color=C["assisted"], lw=1)
        a.axhline(0, color=C["muted"], lw=0.8)
        a.set_title(f"{name.capitalize()} trajectory (#{r}), predicted RMS {float(rms_p[r]):.2e}")
        a.set_xlabel("t [ms]"); a.set_ylabel("error [pu]")
    # one legend for the figure: the same two series in every panel, and no panel has room for it
    fig.legend(*ax[0][0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.suptitle(tag, x=0.01, ha="left", fontsize=9, color=C["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, f"{tag}/02_rollout_error.png"), rms_p


def fig_amplitude(truth, rms_p, tag):
    amp = truth[:, -1].abs().amax(1)            # last window: the decaying B mode is gone, pure carrier amplitude
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(amp, rms_p, s=14, color=C["predicted"], alpha=0.7, linewidths=0)
    x = torch.tensor([float(amp.min()), float(amp.max())])
    for rel in (1e-3, 1e-2):                     # reference: error proportional to amplitude
        ax.plot(x, rel * x, color=C["muted"], lw=1)
        ax.text(float(x[-1]), rel * float(x[-1]), f" {rel:.1%} of |i|", color=C["muted"], va="center", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("current amplitude [pu]"); ax.set_ylabel("predicted RMS error [pu]")
    ax.set_title("Error vs amplitude, one dot per trajectory")
    return save(fig, f"{tag}/03_error_vs_amplitude.png")


def fig_truth_vs_predicted(truth, pred, asst, dt, W, rms_p, tag, n_zoom=4):
    """The PLL's 03_prediction_vs_truth, for a 50 Hz current: 25 cycles over 0.5 s make the two
    traces indistinguishable at full scale, so each column pairs the full horizon with a zoom
    into the first `n_zoom` windows (where the decaying B mode makes error largest) and the
    error on that same zoom. Thin vertical lines mark the handovers."""
    R, _, S = truth.shape
    t_ms = torch.arange(W * S) * dt * 1e3
    z = n_zoom * S
    order = torch.argsort(rms_p)
    picks = [(int(order[R // 2]), "median"), (int(order[-1]), "worst")]

    fig, ax = plt.subplots(3, 2, figsize=(11, 8.6), sharex="row")
    for col, (r, name) in enumerate(picks):
        tr, pr, asr = (x[r].reshape(-1) for x in (truth, pred, asst))
        a = ax[0][col]
        a.plot(t_ms, tr, color=C["truth"], lw=2.4, label="truth")
        a.plot(t_ms, pr, color=C["predicted"], lw=0.9, label="predicted (chained)")
        a.set_title(f"{name.capitalize()} trajectory #{r}: full 0.5 s")
        a.set_ylabel("i [pu]")

        a = ax[1][col]
        a.plot(t_ms[:z], tr[:z], color=C["truth"], lw=2.4)
        a.plot(t_ms[:z], pr[:z], color=C["predicted"], lw=1.1)
        a.set_title(f"First {n_zoom} windows")
        a.set_ylabel("i [pu]")

        a = ax[2][col]
        a.plot(t_ms[:z], (pr - tr)[:z], color=C["predicted"], lw=1.1)
        a.plot(t_ms[:z], (asr - tr)[:z], color=C["assisted"], lw=1.1, label="assisted (true i0)")
        a.axhline(0, color=C["muted"], lw=0.8)
        a.set_title(f"Error, first {n_zoom} windows  ·  0.5 s RMS {float(rms_p[r]):.2e}")
        a.set_ylabel("error [pu]"); a.set_xlabel("t [ms]")

        for row in (1, 2):                                   # handovers, on the zoomed rows only
            for k in range(1, n_zoom):
                ax[row][col].axvline(k * S * dt * 1e3, color=C["grid"], lw=1.4, zorder=0)

    h, l = ax[0][0].get_legend_handles_labels()
    h2, l2 = ax[2][0].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="upper right", ncol=3)
    fig.suptitle(tag, x=0.01, ha="left", fontsize=9, color=C["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, f"{tag}/04_truth_vs_predicted.png")


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    tag = cli.get("tag") or newest_tag()
    model, ck = load_checkpoint(ROOT / "runs" / f"{tag}.pth")
    traj, tm = Dataset_Generator.load_dataset(ck["train_cfg"]["eval_dataset"])
    W, S, dt = tm["W"], tm["S"], tm["dt"]

    truth = traj["i"].float().view(-1, W, S)
    pred = rollout(model, ck, traj, W, feedback=True)
    asst = rollout(model, ck, traj, W, feedback=False)

    fig_history(ck, tag)
    _, rms_p = fig_rollout(truth, pred, asst, dt, W, tag)
    fig_amplitude(truth, rms_p, tag)
    fig_truth_vs_predicted(truth, pred, asst, dt, W, rms_p, tag)

    rec_path = ROOT / "sweeps" / f"{tag}.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        here = float(rms_p.mean())
        same = abs(here - rec["predicted_error_rms"]) <= 1e-6 * max(1.0, abs(here))
        print(f"predicted_error_rms {here:.4e}  record {rec['predicted_error_rms']:.4e}  "
              f"{'matches' if same else 'DOES NOT MATCH -- the eval set or the checkpoint changed since training'}")
