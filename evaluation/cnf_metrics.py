import torch
from torchdyn.core import NeuralODE
from models.cnf import CNF

def cnf_test_metrics(model, x1, device, base):
    cnf = NeuralODE(
        CNF(model, likelihood_estimator="exact"), sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        x1_with_ll = torch.cat([x1, torch.zeros(x1.shape[0], 1, device=device)], dim=-1)
        x0_with_ll = cnf.trajectory(x1_with_ll, t_span=torch.linspace(1, 0, 2, device=device))[-1]
        nfe = cnf.vf.nfe
        log_probs = base.log_prob(x0_with_ll[..., :-1]) + x0_with_ll[..., -1]

        avg_lp = log_probs.mean().item()
        
    return avg_lp, nfe