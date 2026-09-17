"""Train one config, or a sweep. Any list-valued key becomes a sweep axis (cartesian product).

    python src/CLI_Train.py                                          # config/Line_Training.yml as-is
    python src/CLI_Train.py w_phys=0.1 model.F=2 epochs=200          # overrides, dotted for nested keys
    python src/CLI_Train.py 'w_phys=[0,0.1,0.3]' 'seed=[0,1]'        # 6 runs, one after another
    python src/CLI_Train.py 'w_phys=[0,0.1,0.3]' 'seed=[0,1]' list=true               # print the 6, train nothing
    python src/CLI_Train.py 'w_phys=[0,0.1,0.3]' 'seed=[0,1]' index=$LSB_JOBINDEX     # HPC array: run only #k

QUOTE anything with brackets -- zsh treats [ ] as a glob and fails with "no matches found".
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must precede EVERY torch import in the process

import itertools, sys, traceback
sys.stdout.reconfigure(line_buffering=True)            # HPC logs show epochs live, not at exit
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent

# Keys that appear in the run tag (see Line_trainer.fit). Sweeping any OTHER key gives every run the
# same tag, and each run silently overwrites the previous one's .pth and .json. Keep in sync with fit().
TAGGED = {"dataset", "seed", "split_seed", "w_phys", "arch",
          "model.F", "model.max_freq", "model.hidden_dim", "model.n_layers", "model.width"}


def _dotted(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _dotted(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def expand(cfg):
    """Yield (config, {axis: value}) for every point of the sweep. No list-valued keys -> exactly one."""
    plain = OmegaConf.to_container(cfg, resolve=True)
    axes = {k: v for k, v in _dotted(plain) if isinstance(v, list)}
    for combo in itertools.product(*axes.values()):      # product() of zero lists yields one empty tuple
        c = OmegaConf.create(plain)
        for k, v in zip(axes, combo):
            OmegaConf.update(c, k, v)
        yield c, dict(zip(axes, combo))


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    index = cli.pop("index", None)
    list_only = cli.pop("list", False)

    base = OmegaConf.load(ROOT / "config" / "Line_Training.yml")
    OmegaConf.set_struct(base, True)                       # a typo like wphys=0.3 raises instead of being ignored
    try:
        cfg = OmegaConf.merge(base, cli)
    except Exception as e:
        sys.exit(f"bad override: {e}")

    runs = list(expand(cfg))
    untagged = sorted(set(runs[0][1]) - TAGGED)
    if len(runs) > 1 and untagged:
        sys.exit(f"sweeping {untagged}, which is not in the run tag: every run would overwrite the last. "
                 f"Add it to the tag in Line_trainer.fit and to TAGGED here.")

    for k, (_, point) in enumerate(runs, 1):
        print(f"  [{k}/{len(runs)}] {point or '(single run)'}")
    if list_only:
        sys.exit(0)
    if index is not None:
        if type(index) is not int or not 1 <= index <= len(runs):
            sys.exit(f"index={index!r}: must be an int in 1..{len(runs)} (1-based, like LSB_JOBINDEX)")
        runs = [runs[index - 1]]

    # pre-flight: a missing eval set is otherwise discovered AFTER the whole training run
    for c, _ in runs:
        for key in ("dataset", "eval_dataset"):
            if not (ROOT / "data" / c[key]).exists():
                sys.exit(f"{key}: data/{c[key]} not found -- build it with src/CLI_Dataset.py")

    from Line_Trainer import Line_trainer                  # late import: list=true never loads torch

    done, crashed = [], 0
    for c, point in runs:
        print(f"\n=== {point or 'single run'} ===")
        try:
            done.append((c, Line_trainer(c).fit()))
        except Exception:
            crashed += 1
            print(f"!! {point} crashed:")
            traceback.print_exc(file=sys.stdout)            # same stream: trace stays under its header

    print(f"\n{'tag':62s} {'status':9s} {'epochs':>9s} {'pred_rms':>10s} {'comp':>6s}")
    for c, r in done:
        ep = f"{r['epochs_run']}/{c.epochs}" + (" CAP" if r["epochs_run"] >= c.epochs else "")
        pr = f"{r['predicted_error_rms']:.3e}" if "predicted_error_rms" in r else "-"
        cp = f"{r['compounding']:.2f}" if "compounding" in r else "-"
        print(f"{r['tag']:62s} {r['status']:9s} {ep:>9s} {pr:>10s} {cp:>6s}")
    if crashed:
        print(f"\n{crashed} run(s) crashed -- tracebacks above")
    sys.exit(1 if crashed else 0)
