from omegaconf import OmegaConf
from pathlib import Path
import os, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1") # Soap does not run on mps...sad
import torch
import torch.nn as nn
from pytorch_optimizer import SOAP
import numpy as np

from Line_Dataset_Generator import Dataset_Generator
from Line_Physics import Line_Physics
from Line_Operator import Unstacked_DeepONet, Single_PINN
from Line_Residual import compute_i
from Line_Evaluate import rollout_metrics, rollout, load_checkpoint

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
line_training = OmegaConf.load(CONFIG_DIR/"Line_Training.yml")
ARCHS = {"deepONet": Unstacked_DeepONet, "pinn": Single_PINN}
DEVICE = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class Line_trainer():
    def __init__(self, line_training=line_training):
        self.cfg = line_training
        self.split_seed = line_training.split_seed
        self.seed = line_training.seed
        torch.manual_seed(line_training.seed)
        self.n_layers = line_training.model.n_layers
        self.width = line_training.model.width
        self.lr = line_training.lr
        self.n_eval_runs = line_training.n_eval_traj
        self.batch_size = line_training.batch_size
        self.epochs = line_training.epochs
        self.model_specs = line_training.model
        self.device = DEVICE
        self.load_dataset = Dataset_Generator.load_dataset
        self.dataset_path = line_training.dataset
        self.results_dir = Path(__file__).resolve().parent.parent / line_training.results_dir
        self.runs_dir = Path(__file__).resolve().parent.parent / line_training.runs_dir
        self.data, self.meta = self.load_dataset(self.dataset_path)

        self.w_phys = line_training.w_phys
        self.w_deriv = line_training.w_deriv
        self.arch = line_training.arch
        self.patience = line_training.patience
        self.best_ep = -1
        self.history = {}
        i, dv, i0, didt_target = self.data["i"], self.data["delta_v"], self.data["i0"], self.data["di/dt"]
        n = i0.shape[0]
        g = torch.Generator().manual_seed(line_training.split_seed)
        permutations = torch.randperm(n, generator=g)
        n_val  = int(round(n * line_training.val_frac))
        validation_mask, training_mask = permutations[:n_val], permutations[n_val:]
        
        self.i0_mean, self.i0_s_deviation = i0[training_mask].mean(), i0[training_mask].std()
        self.s_deviation_dv = dv[training_mask].std()
        self.s_deviation_didt, self.mean_didt = didt_target[training_mask].std(), didt_target[training_mask].mean()
        physics = Line_Physics()
        ell, tau = physics.Lc, physics.t_constant
        self.s   = (dv[training_mask]/ell - i[training_mask]/tau).pow(2).mean().sqrt().item()   # 'round 157 pu/s
        branch = torch.cat([((i0-self.i0_mean)/self.i0_s_deviation).unsqueeze(-1), dv/self.s_deviation_dv], dim=-1)
        pack = lambda m: tuple(t[m].to(self.device) for t in (branch, dv, i, didt_target))
        
        self.training = pack(training_mask) # ready training data in form: branch, dv, i network input, output, forcing input perfect! ----------------------
        self.validation = pack(validation_mask) # same for validation data ------------------------------------------- will need the data for rollout
        
        self.n_tr, self.n_val = len(training_mask), len(validation_mask)
        self.t_local = self.data["t_local"].float().to(self.device)
        
        ov = dict(line_training.model)
        ov["S_win"] = self.meta["S"]
        self.model = ARCHS[line_training.arch](ov=ov).to(self.device)
        self.opt = SOAP(self.model.parameters(), lr=line_training.lr, betas=(0.95,0.95), weight_decay=0.01, precondition_frequency=10)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=max(3, line_training.patience // 3))
    
    def batches(self, tensors, n, batch_size, shuffle, drop_last):
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        stop = n - (n % batch_size) if drop_last else n
        for i in range(0, stop, batch_size):
            b = idx[i:i + batch_size]
            yield tuple(t[b] for t in tensors)
            
            
    def save_checkpoint(self, path): # nice and supremely organised
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":       {k: v.cpu() for k, v in self.model.state_dict().items()},
            "arch_cfg":         self.model.config(),                             # rebuilds the net
            "train_cfg":        OmegaConf.to_container(self.cfg, resolve=True),  # replays the run
            "data_meta":        self.meta,
            "i0_mean":          self.i0_mean,
            "i0_s_deviation":   self.i0_s_deviation,
            "s_deviation_dv":   self.s_deviation_dv,
            "s_deviation_didt": self.s_deviation_didt,
            "mean_didt":        self.mean_didt,
            "s":                self.s,
            "t_local":          self.t_local.cpu(),
            "history":          self.history,
            "best_ep":          self.best_ep,
        }, path)
    
    def _write_record(self, rec, tag):
        p = self.results_dir / f"{tag}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2)); tmp.replace(p)   # atomic: no half-files
    
    def _epoch(self, split, train):
        self.model.train(train)
        losses = {"i": 0.0, "physics_residual": 0.0, "derivative": 0.0, "total": 0.0}
        nb = 0
        n = self.n_tr if train else self.n_val
        for branch, dv, target, didt_target in self.batches(split, n, self.cfg.batch_size, shuffle=train, drop_last=train):
            B = branch.shape[0]
            t_query = self.t_local.view(1, -1, 1).expand(B, -1 , 1).requires_grad_(True)
            out = compute_i(self.model, t_query, branch_input=branch, delta_V=dv.unsqueeze(-1))
            
            loss_i = nn.functional.mse_loss(out["i"], target.unsqueeze(-1))
            loss_physics = (out["residual"] / self.s).pow(2).mean()
            loss_total = loss_i + self.w_phys * loss_physics
            if self.cfg.w_deriv:
                loss_derivative = self.cfg.w_deriv * nn.functional.mse_loss((out["di/dt"]- self.mean_didt) / self.s_deviation_didt, (didt_target - self.mean_didt) / self.s_deviation_didt)
                loss_total = loss_total + loss_derivative # might not be a good idea with noisy data might inject white noise straight into loss function
                losses["derivative"] += loss_derivative.detach().cpu().tolist()
            if train:
                self.opt.zero_grad()
                loss_total.backward()
                self.opt.step()
            losses["i"] += loss_i.item()
            losses["physics_residual"] += loss_physics.item()
            losses["total"]+= loss_total.item()
            nb += 1  # number of items inside a batch
        return {k: v / nb for k, v in losses.items()}
        
        
    def fit(self):
        best, best_state, self.best_ep, bad = float("inf"), None, -1, 0
        self.history = {"train": [], "val": []}
        status = "ok"
        
        tag = (f"{Path(self.dataset_path).stem}_n{self.meta['n_runs']}_W{self.meta['W']}_F{self.model.F}" f"_mf{self.model.max_freq:g}_wp{self.w_phys:g}_s{self.seed}sp{self.split_seed}"
           + (f"_h{self.model_specs.hidden_dim}" if self.model_specs.hidden_dim is not None else "")
           + (f"_L{self.model_specs.n_layers}" if self.model_specs.n_layers is not None else "")
           + (f"_w{self.model_specs.width}" if self.model_specs.width is not None else "")
           + ("" if self.arch == "deepONet" else f"_{self.arch}"))
        
        
        for ep in range(1, self.epochs + 1):
            training_losses = self._epoch(self.training, train=True)
            validation_losses = self._epoch(self.validation, train=False)
            
            if (not np.isfinite(validation_losses["total"])) or (ep > 3 and validation_losses["total"] > 50 * best):
                status = "diverged"
                print(f"[{tag}] DIVERGED at epoch {ep}: val total {validation_losses['total']:.3e}" f"vs best {best:.3e} -- aborting")
                break
            self.history["train"].append(training_losses)
            self.history["val"].append(validation_losses)
            self.scheduler.step(validation_losses["total"])
            improved = validation_losses["total"] < best - 1e-9
            if improved:
                best, self.best_ep, bad = validation_losses["total"], ep, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            print(f"[{ep:4d}] train i {training_losses['i']:.3e} physics {training_losses['physics_residual']:.3e} "
                f"derivative {training_losses['derivative']:.3e} | "
                f"[{ep:4d}] validation i {validation_losses['i']:.3e} physics {validation_losses['physics_residual']:.3e} "
                f"derivative {validation_losses['derivative']:.3e}{'  *' if improved else ''}")
            if bad >= self.patience:
                print(f"early stop at {ep} (best {self.best_ep})"); break
                
        rec = {"tag": tag, "status": status, "dataset": self.dataset_path,
        "W": self.meta["W"], "S": self.meta["S"], "dt": self.meta["dt"],
        "window_s": self.meta["S"] * self.meta["dt"],
        "F": self.model.F, "max_freq": self.model.max_freq, "w_phys": self.w_phys,
        "seed": self.seed, "split_seed": self.split_seed, "lr": self.lr,
        "arch": self.arch,
        "n_layers": self.n_layers, "width": self.width,
        "params": sum(p.numel() for p in self.model.parameters()),
        "batch_size": self.batch_size, "device": str(self.device),
        "n_eval_runs": self.n_eval_runs,
        "epochs_run": len(self.history["val"])}
        
        if status == "ok" and best_state is not None:
            self.model.load_state_dict(best_state)
            out_dir = self.runs_dir / f"{tag}.pth"
            self.save_checkpoint(path=out_dir)
            h = self.history["val"][self.best_ep - 1]
            rec.update({"best_epoch": self.best_ep, "ckpt": str(out_dir),
                    "val_i": h["i"], "physics_residual": h["physics_residual"], "derivative": h["derivative"], "total": h["total"],
                    "train_i": self.history["train"][self.best_ep - 1]["i"]})
            ev_model, ck = self.load_checkpoint(out_dir)
            traj, traj_meta = Dataset_Generator.load_dataset(self.cfg.eval_dataset)
            assert (traj_meta["S"], traj_meta["dt"]) == (self.meta["S"], self.meta["dt"]), f"eval set S={traj_meta['S']} dt={traj_meta['dt']} != training S={self.meta['S']} dt={self.meta['dt']}"
            rec.update(rollout_metrics(ev_model, ck, traj, W=traj_meta["W"]))
            rec["eval_dataset"] = self.cfg.eval_dataset
            print(f"[{tag}] predicted_error_rms {rec['predicted_error_rms']:.4e} compounding {rec['compounding']:.2f}x predicted_error_max {rec['predicted_error_max']:.4e}")
        else:
            rec["status"] = "diverged"                    # also covers best_state=None
            print(f"[{tag}] DIVERGED -- no checkpoint written ")
        self._write_record(rec, tag)
        return rec