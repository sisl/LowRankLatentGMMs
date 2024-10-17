import torch

# torchdyn imports
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.nn import Augmenter


def autograd_trace(x_out, x_in, **kwargs):
    """
    Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd.
    Code from torchdyn library: https://github.com/DiffEqML/torchdyn
    """
    trJ = 0.0
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ


def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd.
    Code from torchdyn library: https://github.com/DiffEqML/torchdyn
    """
    jvp = torch.autogradgrad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ


class CNF(torch.nn.Module):
    """
    Continuous normalizing flow class.
    Code from torchdyn library: https://github.com/DiffEqML/torchdyn
    """
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x, *args, **kwargs):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(True)  # first dimension reserved to divergence propagation
            x_out = self.net(torch.cat([x_in, t * torch.ones(x.shape[0], 1).type_as(x_in)], dim=-1))
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph
    

def compute_log_probs(model, x, t, device, base):
    # Return 
    cnf = DEFunc(CNF(model))
    nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
    cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
    with torch.no_grad():
        aug_traj = (
            cnf_model[1].to(device).trajectory(
                Augmenter(1, 1)(x).to(device), t_span=torch.linspace(t, 0, 101).to(device),
            )
        )[-1].cpu()
        log_probs = base.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]

    return log_probs