"""
Per-frame network: one reading in, 120 states out.

The obvious first thing to try, and worth keeping for two reasons. It is the
honest measure of how much a single sample can tell you - no history, no
dynamics - and its posterior is the emission model that structured decoding
runs on top of, which is what actually performs.

Predicting tilt and angle with two independent heads was tried and dropped: it
is strictly worse than one joint head, because the two errors are correlated
(a misread tilt drags the angle with it) and a factorised output cannot
represent that.

    uv run net_mlp.py            benchmark this architecture on its own
    uv run net_mlp.py --smoke    tiny settings
    uv run net_mlp.py --check    self-check only
"""

import numpy as np
import torch.nn as nn

import hmm
from nets import mlp, model_loglik, predict_last, train
from statespace import N_STATES


class JointMLP(nn.Module):
    """One per-frame net over the 120 joint states."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = mlp(N_STATES, hidden)

    def forward(self, x):
        return self.net(x)


def fit(s, frac):
    """Train once per noise level; history length does not affect this model."""
    return train(
        JointMLP(), s.split_tr, s.split_va, s.norm, frac * s.signal,
        steps=s.cfg["frame_steps"], batch=1024, per_frame=True,
    )


def per_frame(s, frac):
    """Read the newest frame and nothing else."""
    model = fit(s, frac)
    return lambda win: predict_last(model, win.X)


def with_filter(s, frac):
    """The same network, its posteriors handed to the HMM forward filter.

    This is the pairing that matters: the network supplies the emission model,
    the filter supplies the dynamics the network was never told about.
    """
    model = fit(s, frac)

    def predict(win):
        return hmm.decode_with(
            win,
            lambda c: model_loglik(model, win.X[c], s.log_prior),
            s.const_rates(len(win.X)), s.log_pi,
        )
    return predict


METHODS = {
    "Per-frame MLP": per_frame,
    "Per-frame MLP + HMM filter": with_filter,
}


def demo():
    """A one-frame window in must give one prediction per frame out."""
    import torch

    net = JointMLP(hidden=16)
    assert net(torch.zeros(4, 1, 3)).shape == (4, 1, N_STATES)
    assert net(torch.zeros(4, 7, 3)).shape == (4, 7, N_STATES), "any length works"
    # the net is per-frame, so shuffling time must not change per-frame outputs
    x = torch.randn(2, 5, 3)
    out, flipped = net(x), net(x.flip(1))
    assert np.allclose(out.flip(1).detach().numpy(), flipped.detach().numpy(),
                       atol=1e-6), "per-frame model must ignore ordering"
    print("net_mlp: ok")


if __name__ == "__main__":
    from evaluate import main

    main(METHODS, "net_mlp", check=demo)
