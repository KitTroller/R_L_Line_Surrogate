"""Compare runs across a sweep: one metric against one record key, every seed visible.

    python src/analysis/plot_sweep.py x=w_phys
    python src/analysis/plot_sweep.py x=w_phys metric=compounding
    python src/analysis/plot_sweep.py x=F match=line_stage_A_n5000        # only tags containing this
    python src/analysis/plot_sweep.py x=hard_ic 'match=[line_stage_A_n5000,_wp0.3_]'   # ALL must appear
    python src/analysis/plot_sweep.py x=w_phys match=line_stage_A_n5000 exclude=_hic   # none may appear

Writes graphs/sweep_<x>_<metric>.png and prints every run, best first.

Refuses to plot runs that differ in anything other than `x` and the seed: a w_phys plot that
silently mixes two datasets or two widths looks exactly as clean as a correct one.
"""
import json, sys
from collections import defaultdict
from statistics import median
from omegaconf import OmegaConf

from style import ROOT, C, save, plt

# Record keys that define "the same experiment". Anything here that varies (other than x and
# seed) means the runs are not comparable on one axis.
SAME = ["dataset", "eval_dataset", "S", "dt", "F", "max_freq", "w_phys", "w_deriv", "split_seed",
        "lr", "batch_size", "arch", "n_layers", "width", "hard_ic", "branch_act"]
# Value a key had before it existed in the record, so older records still compare correctly.
DEFAULTS = {"hard_ic": False, "branch_act": "tanh"}


def as_list(v):
    """match=foo -> ['foo'];  'match=[a,b]' -> ['a', 'b'];  absent -> []."""
    if v is None:
        return []
    if isinstance(v, (str, int, float, bool)):
        return [str(v)]
    return [str(s) for s in v]


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    x = cli.get("x") or sys.exit("pass x=<record key>, e.g. x=w_phys")
    metric = cli.get("metric", "predicted_error_rms")
    match, exclude = as_list(cli.get("match")), as_list(cli.get("exclude"))

    recs = [json.loads(p.read_text()) for p in sorted((ROOT / "sweeps").glob("*.json"))]
    recs = [r for r in recs if all(m in r["tag"] for m in match) and not any(e in r["tag"] for e in exclude)]
    for r in recs:
        for k, v in DEFAULTS.items():
            r.setdefault(k, v)
    bad = [r["tag"] for r in recs if r["status"] != "ok"]
    if bad:
        print(f"!! {len(bad)} diverged run(s) excluded: {', '.join(bad)}")
    recs = [r for r in recs if r["status"] == "ok"]
    if not recs:
        sys.exit("no ok runs match")
    for key in (x, metric):
        if key not in recs[0]:
            sys.exit(f"'{key}' is not a record key. Available: {sorted(recs[0])}")

    varies = {k: sorted({str(r.get(k)) for r in recs}) for k in SAME if k != x}
    varies = {k: v for k, v in varies.items() if len(v) > 1}
    if varies:
        sys.exit(f"these runs also differ in {varies} -- narrow them with match= and/or exclude=")

    groups = defaultdict(list)
    for r in recs:
        groups[r[x]].append(r)
    xs = sorted(groups, key=lambda v: (v is None, v))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, xv in enumerate(xs):
        ys = [r[metric] for r in groups[xv]]
        jitter = [(j - (len(ys) - 1) / 2) * 0.06 for j in range(len(ys))]
        ax.scatter([k + d for d in jitter], ys, s=22, color=C["predicted"], alpha=0.55, linewidths=0, zorder=3)
    meds = [median(r[metric] for r in groups[xv]) for xv in xs]
    ax.plot(range(len(xs)), meds, color=C["predicted"], marker="o", ms=6, zorder=4)
    for k, m in enumerate(meds):
        ax.annotate(f"{m:.2e}", (k, m), textcoords="offset points", xytext=(7, 5), ha="left", va="bottom",
                    fontsize=9, color=C["ink"])
    ax.set_xticks(range(len(xs)), [str(v) for v in xs])
    ax.set_xlabel(x); ax.set_ylabel(metric)
    if metric != "compounding":
        ax.set_yscale("log")
    n_seeds = sorted({len(g) for g in groups.values()})
    ax.set_title(f"{metric} vs {x}")
    fig.suptitle(f"{' + '.join(match) or 'all runs in sweeps/'}"
                 f"{'  minus ' + ', '.join(exclude) if exclude else ''}  ·  dots = seeds, line = median, "
                 f"n = {'/'.join(map(str, n_seeds))} per point", x=0.01, ha="left", fontsize=9, color=C["muted"])
    suffix = "".join(f"_{m.strip('_')}" for m in match)
    save(fig, f"sweep_{x}_{metric}{suffix}.png")

    print(f"\n{'tag':62s} {x:>10s} {'seed':>4s} {metric:>20s} {'comp':>6s} {'epochs':>7s} {'s':>6s}")
    for r in sorted(recs, key=lambda r: r[metric]):
        print(f"{r['tag']:62s} {str(r[x]):>10s} {r['seed']:>4d} {r[metric]:20.4e} "
              f"{r.get('compounding', float('nan')):6.2f} {r['epochs_run']:7d} {r.get('seconds', float('nan')):6.0f}")
