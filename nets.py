"""
Shared torch plumbing for the learned approaches.

The architectures themselves live in their own runnable modules - `net_mlp.py`
and `net_gru.py` - so each can be benchmarked on its own. What they have in
common sits here: the training loop, the newest-frame readout, and the bridge
that lets a network's output be used as an emission model by `hmm`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from population import flatten, observe
from statespace import N_STATES, score

DEVICE = torch.device("cpu")  # models are tiny; mps/cuda overhead dominates


def mlp(out_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(3, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def train(model, split_tr, split_va, norm, sigma, steps, batch, lr=3e-3, seed=0,
          per_frame=False):
    """Train to a gradient-step budget, keeping the best-validation weights.

    Noise is resampled every epoch, so the network sees each state under fresh
    perturbation - the augmentation that makes it robust rather than a lookup
    table. Validation noise is held fixed so model selection is not itself
    noisy, and validation scores the newest frame only, which is what
    deployment reads.
    """
    torch.manual_seed(seed)
    if per_frame:
        # frames are i.i.d. for this model: flatten so a batch is frames, not
        # sequences (otherwise an "epoch" is a handful of gradient steps)
        split_tr, split_va = flatten(split_tr), flatten(split_va)
    states_tr, states_va = split_tr[0], split_va[0]
    # budget in gradient steps, so per-frame and windowed models both converge
    epochs = max(1, round(steps / max(1, len(states_tr) // batch)))
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    y_tr = torch.from_numpy(states_tr.astype(np.int64)).to(DEVICE)
    y_va_last = states_va[:, -1]
    X_va = torch.from_numpy(observe(split_va, sigma, 10_000, norm)).to(DEVICE)

    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        X_tr = torch.from_numpy(observe(split_tr, sigma, epoch, norm)).to(DEVICE)
        perm = torch.randperm(len(X_tr))
        model.train()
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = model(X_tr[idx])
            F.cross_entropy(out.reshape(-1, N_STATES), y_tr[idx].reshape(-1)).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_va).argmax(-1)[:, -1].cpu().numpy()
        acc = score(pred, y_va_last)["joint"]
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_last(model, X: np.ndarray, chunk=8192) -> np.ndarray:
    """Model prediction for the newest frame of each window."""
    out = []
    for i in range(0, len(X), chunk):
        logits = model(torch.from_numpy(X[i : i + chunk]).to(DEVICE))
        out.append(logits.argmax(-1)[:, -1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def model_loglik(model, X: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Learned emission: turn the net's posterior back into a likelihood.

    The network is trained on the state distribution the trajectories produce,
    so its output already carries that prior. Dividing it out (subtracting in
    logs) leaves p(reading | state), which is what the filter wants - otherwise
    the prior gets counted twice.
    """
    logits = model(torch.from_numpy(X).to(DEVICE))
    return (F.log_softmax(logits, -1).cpu().numpy() - log_prior).astype(np.float64)


def demo():
    """Train a tiny net on a clean lookup and check the shared pieces line up."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = nn.Sequential(mlp(N_STATES, 32))
    centres = rng.normal(size=(N_STATES, 3)).astype(np.float32)
    states = rng.integers(0, N_STATES, size=512)
    X = centres[states][:, None, :]  # (n, 1, 3) one-frame windows

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xt, yt = torch.from_numpy(X), torch.from_numpy(states.astype(np.int64))[:, None]
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(net(xt).reshape(-1, N_STATES), yt.reshape(-1)).backward()
        opt.step()
    net.eval()

    pred = predict_last(net, X)
    assert score(pred, states)["joint"] > 95, "should memorise a clean lookup"

    log_prior = np.full(N_STATES, -np.log(N_STATES))
    ll = model_loglik(net, X, log_prior)
    assert ll.shape == (len(X), 1, N_STATES)
    assert (ll.argmax(-1)[:, 0] == pred).all(), "likelihood must rank like the net"
    print("nets: ok")


if __name__ == "__main__":
    demo()
