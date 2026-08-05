"""
Causal GRU over the history window: learn the dynamics instead of being told.

This is the natural "just use a sequence model" answer, and it is the direct
comparison against `hmm`, which is handed the transition rules for free. The
GRU has to discover from data that the stick cannot teleport, that tilts
persist, and that usage rates vary between sessions.

It is also the one method here that genuinely needs a buffer of readings: a
recursive filter carries a 120-number belief vector and never re-reads the
past, whereas this re-runs over the whole window each time.

The MLP trunk in front matters. A single linear layer cannot carve 120 Voronoi
cells out of 3 inputs, so without it the recurrent net never even learns the
emission, let alone the dynamics.

    uv run net_gru.py            benchmark this architecture on its own
    uv run net_gru.py --smoke    tiny settings
    uv run net_gru.py --check    self-check only
"""

import torch
import torch.nn as nn

from nets import predict_last, train
from population import window_split
from statespace import N_STATES


class GRUNet(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, N_STATES)

    def forward(self, x):
        return self.head(self.rnn(self.trunk(x))[0])


def causal_gru(s, frac):
    """Trained per history length, since the window it sees is what it learns on.

    Caches by W so a sweep does not retrain the same model twice.
    """
    trained = {}

    def predict(win):
        if win.W not in trained:
            if win.W == 1:
                # a one-frame GRU has no recurrence to use; train it as such
                trained[win.W] = train(
                    GRUNet(), s.split_tr, s.split_va, s.norm, frac * s.signal,
                    steps=s.cfg["frame_steps"], batch=1024, per_frame=True,
                )
            else:
                trained[win.W] = train(
                    GRUNet(),
                    window_split(s.states_tr, s.tables_tr, win.W, warmup=s.warmup,
                                 align=s.align),
                    window_split(s.states_va, s.tables_va, win.W, warmup=s.warmup,
                                 align=s.align),
                    s.norm, frac * s.signal, steps=s.cfg["seq_steps"], batch=256,
                )
        return predict_last(trained[win.W], win.X)

    return predict


METHODS = {"GRU (causal)": causal_gru}


def demo():
    """The GRU must be causal: the newest output may not depend on the future."""
    torch.manual_seed(0)
    net = GRUNet(hidden=16).eval()
    x = torch.randn(1, 6, 3)
    full = net(x)[:, -1]
    # truncating after the last frame changes nothing; altering the past does
    assert torch.allclose(full, net(x[:, :6])[:, -1])
    tampered = x.clone()
    tampered[:, 0] += 5.0
    assert not torch.allclose(full, net(tampered)[:, -1]), "past must matter"
    # and a prefix's own prediction is unaffected by frames appended later
    assert torch.allclose(net(x)[:, 2], net(x[:, :3])[:, -1], atol=1e-6), "not causal"
    assert net(torch.zeros(2, 5, 3)).shape == (2, 5, N_STATES)
    print("net_gru: ok")


if __name__ == "__main__":
    from evaluate import main

    main(METHODS, "net_gru", check=demo)
