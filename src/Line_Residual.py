import torch
from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "Line_Initial_Conditions.yml")
line_constants = OmegaConf.load(CONFIG_DIR / "Line_Constants.yml")

f_0 = line_constants.f_0
OMEGA_BASE = 2 * torch.pi * f_0

def build_ff_trunk_input(t, F, max_F_freq):
    feats = [t]
    for k in range(1, F + 1):
        w = max_F_freq * k / F
        feats += [torch.cos(w * t), torch.sin(w * t)]
    return torch.cat(feats, dim=-1)
        
def compute_i(model, t_query, branch_input, delta_V):
    Xc = line_constants.Xc
    Rc = line_constants.Rc
    Lc = Xc / OMEGA_BASE
    trunk_input = build_ff_trunk_input(t_query, model.F, model.max_freq)
    i = model.forward(branch_input, trunk_input)
    didt = torch.autograd.grad(i, t_query, grad_outputs=torch.ones_like(i), create_graph=True)[0]
    residual = didt + (Rc / Lc) * i - delta_V / Lc
    return {"i": i, "di/dt": didt, "residual": residual}