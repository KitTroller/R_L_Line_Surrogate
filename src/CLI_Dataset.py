"""Build a dataset pair. The seed is mandatory: an unseeded dataset can never be regenerated.

    python src/CLI_Dataset.py seed=0 envelope_off=true                 # Stage A -> line_stage_A*.npz
    python src/CLI_Dataset.py seed=0 envelope_off=false                # Stage B -> line_stage_B*.npz
    python src/CLI_Dataset.py seed=0 name=line_stage_A_big n_runs=20000 n_traj=500
    python src/CLI_Dataset.py seed=0 overwrite=true

Without envelope_off= the default comes from config/Line_Initial_Conditions.yml.
The default file name follows the envelope state, so a file name can never lie about its stage.

Writes two files into data/:
    {name}.npz        independent training windows            (seed)
    {name}_traj.npz   continuous trajectories, rollout only   (seed + EVAL_SEED_OFFSET, so held out)
"""
import sys
from pathlib import Path
from omegaconf import OmegaConf

from Line_Dataset_Generator import Dataset_Generator

ALLOWED = {"seed", "name", "n_runs", "n_traj", "overwrite", "envelope_off"}
EVAL_SEED_OFFSET = 10_000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    cli = OmegaConf.to_container(OmegaConf.from_cli())
    unknown = set(cli) - ALLOWED
    if unknown:
        sys.exit(f"unknown argument(s) {sorted(unknown)} -- allowed: {sorted(ALLOWED)}")
    if type(cli.get("seed")) is not int:
        sys.exit("refusing to build an unreproducible dataset: pass seed=<int>")
    if "envelope_off" in cli and type(cli["envelope_off"]) is not bool:
        sys.exit("envelope_off must be true or false")

    seed = cli["seed"]
    g = Dataset_Generator(seed=seed)
    g_eval = Dataset_Generator(seed=seed + EVAL_SEED_OFFSET)
    if "envelope_off" in cli:                     # BOTH generators: train and eval must be the same stage
        g.envelope_off = g_eval.envelope_off = cli["envelope_off"]
    if "n_runs" in cli:
        g.n_runs = int(cli["n_runs"])
    stage = "A" if g.envelope_off else "B"

    name = cli.get("name", f"line_stage_{stage}")
    train_file, traj_file = f"{name}.npz", f"{name}_traj.npz"
    for f in (train_file, traj_file):
        if (DATA_DIR / f).exists() and cli.get("overwrite") is not True:
            sys.exit(f"data/{f} already exists and checkpoints may have been trained on it. "
                     f"Pick another name=, or pass overwrite=true if you really mean it.")

    print(f"stage {stage}: envelope {'OFF' if g.envelope_off else 'ON'}")
    records, meta = g.build_training_set()
    g.save_dataset(records, meta, train_file)
    traj, traj_meta = g_eval.build_trajectories(n_traj=int(cli.get("n_traj", 300)))
    g_eval.save_dataset(traj, traj_meta, traj_file)

    print(f"\ntrain on it with:\n  python src/CLI_Train.py dataset={train_file} eval_dataset={traj_file}")
