from omegaconf import OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
from pytorch_optimizer import SOAP

from Line_Dataset_Generator import Dataset_Generator
from Line_Physics import Line_Physics
from Line_Operator import Unstacked_DeepONet, Single_PINN

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
line_training = OmegaConf.load(CONFIG_DIR/"Line_Training.yml")
ARCHS = {"deeponet": Unstacked_DeepONet, "pinn": Single_PINN}


class Line_trainer():
    def __init__(self, line_training=line_training):
        torch.manual_seed(line_training.seed)
        self.device = line_training.device
        self.data, self.meta = self.load_dataset(line_training.dataset)
        self.load_dataset = Dataset_Generator.load_dataset
        i, dv, i0 = self.data["i"], self.data["delta_v"], self.data["i0"]
        n = i0.shape[0]
        g = torch.Generator().manual_seed(line_training.split_seed)
        permutations = torch.randperm(n, generator=g)
        n_val  = int(round(n * line_training.val_frac))
        validation_mask, training_mask = permutations[:n_val], permutations[n_val:]
        
        self.i0_mean, self.i0_s_deviation = i0[training_mask].mean(), i0[training_mask].std()
        self.s_deviation_dv = dv[training_mask].std()
        ell, tau = Line_Physics().Lc, Line_Physics().t_constant
        self.s   = (dv[training_mask]/ell - i[training_mask]/tau).pow(2).mean().sqrt().item()   # 'round 157 pu/s
        branch = torch.cat([((i0-self.i0_mean)/self.i0_s_deviation).unsqueeze(-1), dv/self.s_deviation_dv], dim=-1)
        pack = lambda m: tuple(t[m].to(self.device) for t in (branch, dv, i))
        self.training = pack(training_mask) # ready training data in form: branch, dv, i network input, output, forcing input perfect!
        self.validation = pack(validation_mask) # same for validation data
        self.n_tr, self.n_val = len(self.training), len(self.validation)
        self.t_local = self.data["t_local"].to(self.device)
        
        ov = dict(line_training.model)
        ov["S_win"] = self.meta["S"]
        self.model = ARCHS[line_training.arch](ov=ov).to(self.device)
        self.opt = SOAP(self.model.parameters(), lr=line_training.lr, betas=(0.95,0.95), weight_decay=0.01, precondition_frequency=10)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=max(3, line_training.patience // 3))