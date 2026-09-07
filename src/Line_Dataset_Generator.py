from omegaconf import OmegaConf
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
from pathlib import Path
import json

from Line_Physics import Line_Physics, Line_Simulator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "Line_Initial_Conditions.yml")
line_constants = OmegaConf.load(CONFIG_DIR / "Line_Constants.yml")

_F64 = {"t_local", "params"}
_INT = {"run_id", "segment_id"}

class Dataset_Generator():
    def __init__(self, initial_conditions_config=initial_conditions_config,line_constants=line_constants, seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.init_cond = initial_conditions_config
        self.line_constants = line_constants
        self.line_simulator = Line_Simulator()
        self.W = line_constants.Windows
        self.sensors = line_constants.sensors
        self.S = self.line_simulator.S
        self.n_runs = line_constants.n_runs
        self.t = self.line_simulator.t
        
    def _lhs(self, n, samples):  # returns "samples" dimentional array of n-dimentional points
        return LatinHypercube(d=n, seed=self._rng).random(samples)
    
    def get_init_condition_space(self, variables=5, n_runs=None, envelope_off=True):
        if n_runs is None:
            n_runs = self.n_runs
        samples = self._lhs(variables, n_runs)
        
        LABELS = ["frequency_offset", "envelope_rate", "ic_magnitude", "ic_angle", "beta"]
        
        bounds = np.array([self.init_cond.ranges[var] for var in LABELS])
        low_bound = bounds[:, 0]
        high_bound = bounds[:, 1]
        samples = low_bound + (high_bound - low_bound) * samples
        if envelope_off:
            samples[:, 1] = 0  # shuting off the envelope
        B = samples[:, 2] * samples[:, 4]
        sample_space = np.concatenate([samples, B[:,None]], axis=1)  # B is beta rescaled based on I magnitude of that run
        return torch.as_tensor(sample_space)
    
    def build_meta(self):
        sim = self.line_simulator
        return {
            "dt": float(sim.dt), "N": int(self.sensors), "W": int(self.W),
            "S": int(self.S), "n_runs": int(self.n_runs),
            "columns": ["frequency_offset", "envelope_rate", "ic_magnitude", "ic_angle", "beta", "B"],
            "ranges": OmegaConf.to_container(self.init_cond.ranges, resolve=True),
            "lhs_seed": self.seed,
        }
    
    def build_training_set(self):
        parameters = self.get_init_condition_space()
        dv, i = self.line_simulator.batch(parameters[:, 0:1], parameters[:, 1:2], parameters[:, 2:3], parameters[:, 3:4],  parameters[:, 5:6])  # final arg is B not beta (scaled according to I0)
        i0 = i[:, 0]
        records = {
            "i":        i,                                      # (n,S)  the target
            "delta_v":  dv,                                     # (n,S)  the branch function
            "i0":       i0,                                     # (n,)   the branch scalar
            "t_local":  self.line_simulator.t.squeeze(0),       # (S,)   the trunk input
            "params":   torch.as_tensor(parameters),            # (n,5)  provenance
        }
        return records, self.build_meta()
    
    def build_trajectories(self):  # Produces a baseline for recurrent rollout testing
        W = self.W
        S = self.S
        
        n__full_trajectories = 300
        horizon_simulation = Line_Simulator()
        horizon_simulation.S = S * W
        horizon_simulation.t = (torch.arange(horizon_simulation.S) * horizon_simulation.dt).reshape(-1,horizon_simulation.S)
        parameters = self.get_init_condition_space(n_runs=n__full_trajectories)
        dv, i = horizon_simulation.batch(parameters[:, 0:1], parameters[:, 1:2], parameters[:, 2:3], parameters[:, 3:4],  parameters[:, 5:6])
        win = lambda x: x.unfold(1, S, S).reshape(-1, S)  # (n_traj*W, S)
        return {
            "i": win(i), "delta_v": win(dv),
            "i0": win(i)[:, 0],
            "t_local": self.line_simulator.t.squeeze(0),
            "run_id":     torch.arange(n__full_trajectories).repeat_interleave(W),
            "segment_id": torch.arange(W).repeat(n__full_trajectories),
        }
        
    def save_dataset(self, records, meta, path="../data/line_stageA.npz"):
        out = {}
        for k, v in records.items():
            a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            out[k] = a.astype(np.int32 if k in self._INT else np.float64 if k in self._F64 else np.float32)
        out["meta_json"] = np.array(json.dumps(meta))
        p = Path(__file__).resolve().parent.parent / "data" / path
        p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p, **out)
        print(f"saved {p}  {p.stat().st_size/1e6:.1f} MB  {out['i'].shape}")

    @staticmethod
    def load_dataset(path="line_stageA.npz", as_torch=True):
        p = Path(__file__).resolve().parent.parent / "data" / path
        z = np.load(p, allow_pickle=False)
        meta = json.loads(z["meta_json"].item())
        return {k: (torch.from_numpy(z[k]) if as_torch else z[k]) for k in z.files if k != "meta_json"}, meta

if __name__ == "__main__":
    g = Dataset_Generator(seed=0)
    rec, meta = g.build_training_set()
    t  = rec["t_local"] * 1e3   # ms
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # 1 - one window. dv is ~10x smaller than i, so twin axis.
    a = ax[0][0]; a2 = a.twinx()
    for k in range(3):
        a.plot(t, rec["i"][k], lw=1.0)
        a2.plot(t, rec["delta_v"][k], lw=1.0, ls="--")
    a.set_xlabel("t [ms]"); a.set_ylabel("i [pu] solid"); a2.set_ylabel("delta_v [pu] dashed")
    a.set_title("dv leads i by ~87 deg,  |dv| ~ 0.1 |i|")

    # 2 - T2 as a picture: B moves i, must NOT move delta_v
    p = g.get_init_condition_space(n_runs=1)
    args = (p[:,0:1], p[:,1:2], p[:,2:3], p[:,3:4])
    dv0, i0_ = g.line_simulator.batch(*args, torch.zeros(1,1))
    dv1, i1_ = g.line_simulator.batch(*args, torch.full((1,1), 0.3))
    a = ax[0][1]
    a.plot(t, dv0[0], lw=2.6, label="delta_v, B=0")
    a.plot(t, dv1[0], lw=1.0, label="delta_v, B=0.3")
    a.plot(t, (i1_-i0_)[0], lw=1.0, ls=":", label="i(B=0.3) - i(B=0)")
    a.legend(fontsize=8); a.set_xlabel("t [ms]")
    a.set_title(f"T2  max|d(delta_v)| = {(dv1-dv0).abs().max():.1e}   want 0")

    # 3 - T2b: is the stored didt really di/dt?
    i_, didt = g.line_simulator.trajectory(p[:,2:3], p[:,3:4], p[:,0:1], p[:,1:2], torch.full((1,1), 0.3))
    num = (i_[:,2:] - i_[:,:-2]) / (2*meta["dt"])          # central difference
    e   = (num - didt[:,1:-1])
    a = ax[1][0]; a.plot(t[1:-1], e[0], lw=.9)
    a.set_xlabel("t [ms]"); a.set_ylabel("error [pu/s]")
    a.set_title(f"T2b  rel {e.abs().max()/didt.abs().max():.1e}   want ~1e-8")

    # 4 - phase portrait over a FULL trajectory: the spiral is the B mode
    tr = g.build_trajectories()
    a = ax[1][1]
    a.plot(tr["i"][:40].reshape(-1), tr["delta_v"][:40].reshape(-1), lw=.5)
    a.set_xlabel("i [pu]"); a.set_ylabel("delta_v [pu]")
    a.set_title("one trajectory: spirals onto the steady-state ellipse")

    for row in ax:
        for a in row: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("../graphs/stageA_sanity.png", dpi=1024)