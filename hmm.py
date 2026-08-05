"""
Structured decoding: treat the joystick as a hidden Markov model.

The stick cannot teleport. From any state only about seven of the 120 are
reachable in one step - hold, rotate one notch either way, or change tilt - and
that constraint is worth more than any amount of extra network. This module is
the whole of it: a transition model, an emission model, and a forward filter.

The transition rules mirror `loader.filter_legal_tilt` / `filter_legal_rotation`
exactly, so this is those legality rules with probabilities attached. A hard
finite-state-automaton constraint is the special case where every legal move
carries equal weight.

Rates are per trajectory, because real usage is not one fixed set of rates.
"""

import numpy as np

from statespace import GROUND, N_ANGLES, N_STATES, N_TILTS


def apply_T(v, p_rot, p_tilt, reverse=False):
    """v @ T (or v @ T.T), with a different T for every row of v.

    Applies the transition structure directly instead of building a matrix, so
    each trajectory can carry its own rates at no extra cost - and it is cheaper
    than the matmul anyway, since only ~7 of 120 targets are reachable.
    """
    a = v.reshape(-1, N_TILTS, N_ANGLES)
    pr = np.asarray(p_rot).reshape(-1, 1, 1)
    pt = np.asarray(p_tilt).reshape(-1, 1, 1)
    out = (1.0 - pr - pt) * a
    out += (pr / 2) * (np.roll(a, 1, axis=2) + np.roll(a, -1, axis=2))
    if reverse:
        # backward message: mass flows against the arrows
        out[:, :GROUND, :] += pt[:, :, :] * a[:, GROUND, :][:, None, :]
        out[:, GROUND, :] += (pt[:, :, 0] / 4) * a[:, :GROUND, :].sum(1)
    else:
        out[:, :GROUND, :] += (pt / 4) * a[:, GROUND, :][:, None, :]
        out[:, GROUND, :] += pt[:, :, 0] * a[:, :GROUND, :].sum(1)
    return out.reshape(-1, N_STATES)


def transition_matrix(p_rotate: float, p_tilt: float) -> np.ndarray:
    """(120, 120) explicit transition matrix. Reference for `apply_T`."""
    T = np.zeros((N_STATES, N_STATES))
    for tilt in range(N_TILTS):
        for angle in range(N_ANGLES):
            s = tilt * N_ANGLES + angle
            T[s, s] += 1 - p_rotate - p_tilt
            T[s, tilt * N_ANGLES + (angle + 1) % N_ANGLES] += p_rotate / 2
            T[s, tilt * N_ANGLES + (angle - 1) % N_ANGLES] += p_rotate / 2
            if tilt == GROUND:
                for t2 in range(4):
                    T[s, t2 * N_ANGLES + angle] += p_tilt / 4
            else:
                T[s, GROUND * N_ANGLES + angle] += p_tilt
    assert np.allclose(T.sum(1), 1)
    return T


def gaussian_loglik(X, table, sigma, norm) -> np.ndarray:
    """Physics emission: -||b - table[s]||^2 / 2 sigma^2.

    `table` is either (120, 3) - one reference map for every unit, all that is
    available without per-unit calibration - or (n, 120, 3), each window's own
    map, whether that is the true one (oracle) or a fitted one.
    """
    mu, sd = norm
    tbl = (table - mu) / sd
    scale = max(sigma / sd, 5e-3)  # floor keeps it finite at zero sensor noise
    if tbl.ndim == 2:
        d2 = ((X[..., None, :] - tbl) ** 2).sum(-1)
    else:
        d2 = ((X[:, :, None, :] - tbl[:, None, :, :]) ** 2).sum(-1)
    return -d2 / (2 * scale**2)


def standardised_loglik(X, tbl_std, scale) -> np.ndarray:
    """Emission for maps already in standardised units, one per window."""
    return -((X[:, :, None, :] - tbl_std[:, None]) ** 2).sum(-1) / (
        2 * max(scale, 5e-3) ** 2
    )


def filter_last(loglik, rates, log_pi):
    """Causal forward recursion; prediction for the newest frame.

    Smoothing would give an identical answer here - the backward message at the
    last frame is uniform - so there is no separate offline variant when the
    question is 'what is the state right now'. Note the recursion carries only a
    120-number belief vector, so a deployed decoder needs no reading buffer at
    all; the window matters only at a cold start.
    """
    pr, pt = rates[:, 0], rates[:, 1]
    alpha = log_pi + loglik[:, 0]
    for t in range(1, loglik.shape[1]):
        m = alpha.max(-1, keepdims=True)
        alpha = np.log(np.maximum(apply_T(np.exp(alpha - m), pr, pt), 1e-300))
        alpha += m + loglik[:, t]
    return alpha.argmax(-1)


def posterior(loglik, rates, log_pi) -> np.ndarray:
    """Normalised forward-backward posterior marginals (n, seq, 120)."""
    pr, pt = rates[:, 0], rates[:, 1]
    n, seq, n_states = loglik.shape
    log_alpha = np.empty((n, seq, n_states))
    log_alpha[:, 0] = log_pi + loglik[:, 0]
    for t in range(1, seq):
        m = log_alpha[:, t - 1].max(-1, keepdims=True)
        log_alpha[:, t] = (
            np.log(np.maximum(apply_T(np.exp(log_alpha[:, t - 1] - m), pr, pt), 1e-300))
            + m + loglik[:, t]
        )
    log_beta = np.zeros((n, seq, n_states))
    for t in range(seq - 2, -1, -1):
        v = loglik[:, t + 1] + log_beta[:, t + 1]
        m = v.max(-1, keepdims=True)
        log_beta[:, t] = (
            np.log(np.maximum(apply_T(np.exp(v - m), pr, pt, reverse=True), 1e-300)) + m
        )
    g = log_alpha + log_beta
    g -= g.max(-1, keepdims=True)
    g = np.exp(g)
    return g / g.sum(-1, keepdims=True)


def estimate_rates(loglik, rates, log_pi, prior):
    """Estimate a trajectory's own [p_rotate, p_tilt] from its readings.

    Hard-EM: decode with the current rates, count what the decoded path did,
    shrink those counts towards the population prior. With a few dozen frames
    there are only a few dozen transitions to learn two numbers from, so the
    Dirichlet prior is doing real work - without it a quiet session reads as
    "this user never moves" and the filter then refuses to let them.
    """
    path = posterior(loglik, rates, log_pi).argmax(-1)
    tilt, angle = path // N_ANGLES, path % N_ANGLES
    rotated = (angle[:, 1:] != angle[:, :-1]).sum(1)
    tilted = (tilt[:, 1:] != tilt[:, :-1]).sum(1)
    held = np.maximum(path.shape[1] - 1 - rotated - tilted, 0)
    counts = np.stack([rotated, tilted, held], 1) + prior
    return (counts / counts.sum(1, keepdims=True))[:, :2]


# --------------------------------------------------------------------------
# methods: run this file on its own to benchmark structured decoding
# --------------------------------------------------------------------------
def decode_with(win, loglik_fn, rates, log_pi):
    """Filter each chunk of a window and return newest-frame predictions."""
    return np.concatenate(
        [filter_last(loglik_fn(c), rates[c], log_pi) for c in win.chunks]
    )


def lookup(s, frac):
    """No training, no history, no calibration: nearest entry of the mean map.

    The floor everything else has to beat.
    """
    def predict(win):
        return np.concatenate([
            gaussian_loglik(win.X[c], s.ref_table, win.sigma, s.norm).argmax(-1)[:, -1]
            for c in win.chunks
        ])
    return predict


def mean_unit_filter(s, frac):
    """Physics emission from the mean unit, population-average rates."""
    def predict(win):
        return decode_with(
            win,
            lambda c: gaussian_loglik(win.X[c], s.ref_table, win.sigma, s.norm),
            s.const_rates(len(win.X)), s.log_pi,
        )
    return predict


def oracle_map(s, frac):
    """Ceiling on the map alone: each unit's true map, average rates."""
    def predict(win):
        true_tbl = s.tables_te[win.sw[1]]
        return decode_with(
            win,
            lambda c: gaussian_loglik(win.X[c], true_tbl[c], win.sigma, s.norm),
            s.const_rates(len(win.X)), s.log_pi,
        )
    return predict


def oracle_all(s, frac):
    """Full ceiling: true map and this session's true usage rates."""
    def predict(win):
        true_tbl = s.tables_te[win.sw[1]]
        return decode_with(
            win,
            lambda c: gaussian_loglik(win.X[c], true_tbl[c], win.sigma, s.norm),
            s.rates_te[win.traj], s.log_pi,
        )
    return predict


METHODS = {
    "Lookup vs mean unit (no training)": lookup,
    "Mean-unit physics + HMM filter": mean_unit_filter,
    "Oracle map, mean rates": oracle_map,
    "Oracle map + oracle rates": oracle_all,
}


def demo():
    rng = np.random.default_rng(0)
    # the structural operator must match the explicit matrix, both directions
    for pr, pt in [(0.4, 0.2), (0.1, 0.7), (0.0, 0.0), (0.55, 0.45)]:
        T = transition_matrix(pr, pt)
        v = rng.random((6, N_STATES))
        assert np.allclose(apply_T(v, np.full(6, pr), np.full(6, pt)), v @ T)
        assert np.allclose(
            apply_T(v, np.full(6, pr), np.full(6, pt), reverse=True), v @ T.T
        )
    # and every row may carry its own rates
    prs, pts = rng.uniform(0, 0.5, 4), rng.uniform(0, 0.4, 4)
    v = rng.random((4, N_STATES))
    ref = np.stack([v[i] @ transition_matrix(prs[i], pts[i]) for i in range(4)])
    assert np.allclose(apply_T(v, prs, pts), ref)

    # a confident, self-consistent observation sequence decodes to itself
    log_pi = np.full(N_STATES, -np.log(N_STATES))
    truth = np.array([[10, 11, 12, 13]])
    ll = np.full((1, 4, N_STATES), -50.0)
    for t, s in enumerate(truth[0]):
        ll[0, t, s] = 0.0
    rates = np.array([[0.5, 0.1]])
    assert filter_last(ll, rates, log_pi)[0] == truth[0, -1]
    assert posterior(ll, rates, log_pi).argmax(-1).tolist() == truth.tolist()

    # rate estimation recovers "this session rotated every step"
    est = estimate_rates(ll, rates, log_pi, prior=np.array([2.0, 1.0, 2.0]))
    assert est[0, 0] > est[0, 1], "should read as rotation-heavy"
    print("hmm: ok")


if __name__ == "__main__":
    from evaluate import main

    main(METHODS, "hmm", check=demo)
