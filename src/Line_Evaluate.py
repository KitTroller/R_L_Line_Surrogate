import torch
from Line_Residual import build_ff_trunk_input
from Line_Operator import Unstacked_DeepONet, Single_PINN

ARCHS_BY_NAME = {"Unstacked_DeepONet": Unstacked_DeepONet, "Single_PINN": Single_PINN}


@torch.no_grad()
def rollout(model, ck, traj, W, feedback=True, device="cpu"):
    """Chain the operator across W consecutive windows of every trajectory (Run, Window, Sensor)"""
    t = ck["t_local"]
    S = t.shape[0]
    t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])]).view(1, S + 1, 1)
    dv = traj["delta_v"].float().view(-1, W, S).to(device)   # (R,W,S)
    i0_true = traj["i0"].float().view(-1, W).to(device)                             # TRUE only for w=0
    R  = dv.shape[0]
    trunk  = build_ff_trunk_input(t_ext.expand(R, -1, 1), model.F, model.max_freq)
    mu, sd, s_deviation_dv = ck["i0_mean"], ck["i0_s_deviation"], ck["s_deviation_dv"]
    i0, preds = i0_true[:, 0], []
    for w in range(W):
        if not feedback:
            i0 = i0_true[:, w]
        branch = torch.cat([((i0 - mu)/sd).unsqueeze(-1), dv[:, w] / s_deviation_dv], dim=-1)
        i_hat  = model(branch, trunk)[..., 0]        # (R,S + 1)
        preds.append(i_hat[:, :S])
        i0 = i_hat[:, S]                            # its own output, not the truth
    return torch.stack(preds, dim=1)                     # (R,W,S)
    

def rollout_metrics(model, checkpoint, traj, W, device="cpu"):
    S = checkpoint["t_local"].shape[0]
    truth = traj["i"].float().view(-1, W, S).to(device)
    per_run = lambda pred: (pred - truth).pow(2).mean(dim=(1, 2)).sqrt()
    assisted = per_run(rollout(model, checkpoint, traj, W, feedback=False, device=device))
    predicted = per_run(rollout(model, checkpoint, traj, W, feedback=True, device=device))
    return {
        "assisted_error_rms": float(assisted.mean()),
        "predicted_error_rms": float(predicted.mean()),
        "predicted_error_max": float(predicted.max()),
        "compounding": float(predicted.mean() / assisted.mean())
    }
    
    
def load_checkpoint(path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cls = ARCHS_BY_NAME[ck["arch_cfg"]["arch"]]
        model = cls(cfg=ck["arch_cfg"])          # rebuilt from the checkpoint
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()
        ck["t_local"] = ck["t_local"].to(device)
        return model, ck