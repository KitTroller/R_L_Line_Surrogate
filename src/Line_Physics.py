from omegaconf import OmegaConf
from pathlib import Path
import torch
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
line_constants = OmegaConf.load(CONFIG_DIR/"Line_Constants.yml")
f_0 = line_constants.f_0
OMEGA_BASE = 2 * torch.pi * f_0

class Line_Physics():
    
    def __init__(self, line_constants=line_constants):
        self.Rc = line_constants.Rc
        self.Xc = line_constants.Xc
        self.Lc = self.Xc / OMEGA_BASE
        self.t_constant = self.Lc / self.Rc
        
    def line_ODEs(self, ic, delta_V): # Delta_V is the forcing function and ic is the single state variable
        Rc = self.Rc
        Lc = self.Lc
        didt = - (Rc  / Lc) * ic + (delta_V) / Lc
        return didt
    
    def get_delta_V(self, ic, didt):
        Lc = self.Lc
        Rc = self.Rc
        delta_V = didt * Lc + ic * Rc
        return delta_V
    
class Line_Simulator():
    
    def __init__(self, line_constants=line_constants):
        self.physics = Line_Physics()
        self.time_horizon = line_constants.time_horizon
        self.n_runs = line_constants.n_runs
        self.S = line_constants.sensors
        self.dt = self.time_horizon / self.S
        self.W = line_constants.Windows
        self.t = (torch.arange(self.S) * self.dt).reshape(1, self.S)
    
    def trajectory(self, init_magnitude, init_phase, freq_offset, envelope_rate, B):  # We are adding an exponential decay on top of the steady state sinusoid. B is the magnitude of that and envelope rate how fast it decays
        omega_g = OMEGA_BASE + 2 * torch.pi * freq_offset
        t = self.t
        phase = omega_g * t + init_phase
        A = init_magnitude * torch.exp(envelope_rate * t)
        h = torch.exp(-t/self.physics.t_constant)
        i = A * torch.cos(phase) + B * h
        didt =  envelope_rate * A * torch.cos(phase) - A * omega_g * torch.sin(phase) - (B / self.physics.t_constant) * h
        return i, didt
    
    def batch(self, i0_magnitude, i0_phase, freq_offset, envelope_rate, B):
        i, didt = self.trajectory(i0_magnitude, i0_phase, freq_offset, envelope_rate, B)
        delta_V = self.physics.get_delta_V(i, didt)
        return delta_V, i