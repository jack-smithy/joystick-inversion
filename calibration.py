"""
Self-calibration: work out which joystick this is, and how it is being used,
from the readings alone.

Two things about a deployed unit are unknown and unmeasured: its own forward
map (build tolerances move every entry) and the rates at which its user
rotates, tilts and holds. Both are recoverable without ever touching the
device, because both are low-dimensional:

  * The map deviation is nearly rank-10. Tolerance enters through ~20 physical
    parameters, so a unit's 360-dimensional departure from the population mean
    lies close to a small subspace learned from training units. That leaves
    about ten numbers to fit per joystick.
  * The usage rates are two numbers, counted off a decoded path.

Both fits are unsupervised - no labels, no reference measurement - and both are
shrunk towards the population, which is what keeps them from doing harm when
the evidence is thin. See `fit_tolerance` for why that matters more than it
looks.
"""

import numpy as np

import hmm
from statespace import N_STATES


def tolerance_basis(tables_tr: np.ndarray, k: int = 10):
    """Low-rank basis for how units deviate from the mean unit.

    Learned from training units only. Also returns the population variance of
    each coefficient, which is the prior a single unit's fit is shrunk towards.
    """
    ref = tables_tr.mean(0)
    dev = (tables_tr - ref).reshape(len(tables_tr), -1)
    _, _, Vt = np.linalg.svd(dev, full_matrices=False)
    basis = Vt[:k]
    # Coefficient variances stay in raw units even when the caller rescales the
    # basis: dev/sd = sum_j c_j (V_j/sd) has the *same* c_j. Rescaling these too
    # would inflate the prior ~1/sd^2 and switch the shrinkage off entirely.
    return ref, basis.reshape(k, N_STATES, 3), (dev @ basis.T).var(0)


def fit_tolerance(gamma, X, ref, basis, coef_var, noise_var):
    """MAP estimate of one unit's tolerance coefficients.

    Given soft state assignments, solve for the coefficients that best explain
    these readings, then rebuild the unit's map. All terms collapse over states
    first, so cost does not grow with how many frames were used.

    The shrinkage matters more than it looks. A short trajectory visits maybe a
    dozen of the 120 states, so a plain least-squares fit is badly determined in
    the directions those states do not constrain, and extrapolating it to
    unvisited states is *worse than not calibrating at all*. Shrinking towards
    the population prior makes the fit degrade gracefully to the mean unit when
    evidence is thin.
    """
    w = gamma.sum(1)                                    # (n, 120) state occupancy
    G = np.einsum("nts,nta->nsa", gamma, X)             # (n, 120, 3) summed readings
    M = np.einsum("ns,jsa,lsa->njl", w, basis, basis)   # (n, k, k)
    v = np.einsum("jsa,nsa->nj", basis, G - w[..., None] * ref)
    damp = np.diag(noise_var / np.maximum(coef_var, 1e-30))
    c = np.linalg.solve(M + damp, v[..., None])[..., 0]
    return ref + np.einsum("nj,jsa->nsa", c, basis)


def self_calibrate(X, ref, basis, coef_var, scale, tol_scale, rates, log_pi,
                   rate_prior, iters=3, fit_rates=True):
    """EM over both unknowns: this unit's map and this session's usage rates.

    Alternates between inferring states, refitting the map, and recounting the
    rates, tightening the assumed sigma once the map is no longer a guess.
    Returns (per-trajectory map, per-trajectory rates), both in standardised
    units. Neither is measured - both are read off the readings themselves.
    """
    tbl = np.repeat(ref[None], len(X), axis=0)
    r = rates.copy()
    for i in range(iters):
        # the first pass does not know the unit yet, so keep sigma wide
        s = np.sqrt(scale**2 + (tol_scale**2 if i == 0 else 0.01 * tol_scale**2))
        ll = hmm.standardised_loglik(X, tbl, s)
        if fit_rates:
            r = hmm.estimate_rates(ll, r, log_pi, rate_prior)
        tbl = fit_tolerance(
            hmm.posterior(ll, r, log_pi), X, ref, basis, coef_var,
            max(scale, 5e-3) ** 2,
        )
    return tbl, r


# --------------------------------------------------------------------------
# methods: run this file on its own to benchmark self-calibration
# --------------------------------------------------------------------------
_STARTUP_CACHE: dict = {}


def _fit_at_startup(s, frac):
    """Fit each test unit's map and rates from its reserved opening frames.

    Scored windows start after these, so nothing measured here was used to
    calibrate - the same separation a device gets between its first seconds and
    everything after.

    Cached because the ablation rows below reuse the same fit; it is the same
    start-up procedure either way, only what the decoder is then allowed to use
    differs.
    """
    from population import ALPHA0, observe

    key = (id(s), frac)
    if key in _STARTUP_CACHE:
        return _STARTUP_CACHE[key]

    ref, basis, coef_var = tolerance_basis(s.tables_tr)
    calib = (s.states_te[:, : s.warmup],
             np.arange(len(s.states_te)) % len(s.tables_te), s.tables_te)
    X_cal = observe(calib, frac * s.signal, 40_000, s.norm)
    ref_n = (ref - s.norm[0]) / s.signal
    tbl, rates = self_calibrate(
        X_cal, ref_n, basis / s.signal, coef_var, frac, s.tol_scale,
        s.const_rates(len(X_cal)), s.log_pi, ALPHA0,
    )
    truth = (s.tables_te[calib[1]] - s.norm[0]) / s.signal
    print(f"  start-up fit @ {frac:.0%} noise: map error "
          f"{np.abs(tbl - truth).mean():.4f} signal std (uncalibrated "
          f"{np.abs(ref_n - truth).mean():.4f}); rate error "
          f"{np.abs(rates - s.rates_te).mean():.4f} (population mean "
          f"{np.abs(s.pop_rates - s.rates_te).mean():.4f})")
    _STARTUP_CACHE[key] = (ref, tbl, rates)
    return _STARTUP_CACHE[key]


def startup_full(s, frac):
    """Both unknowns fitted at start-up: this unit's map and this user's rates."""
    _, tbl, rates = _fit_at_startup(s, frac)
    scale = np.sqrt(frac**2 + 0.01 * s.tol_scale**2)

    def predict(win):
        return hmm.decode_with(
            win,
            lambda c: hmm.standardised_loglik(win.X[c], tbl[win.traj[c]], scale),
            rates[win.traj], s.log_pi,
        )
    return predict


def startup_map_only(s, frac):
    """Ablation: fitted map, population-average rates."""
    _, tbl, _ = _fit_at_startup(s, frac)
    scale = np.sqrt(frac**2 + 0.01 * s.tol_scale**2)

    def predict(win):
        return hmm.decode_with(
            win,
            lambda c: hmm.standardised_loglik(win.X[c], tbl[win.traj[c]], scale),
            s.const_rates(len(win.X)), s.log_pi,
        )
    return predict


def startup_rates_only(s, frac):
    """Ablation: mean-unit map, fitted rates. The cheap half of the fit."""
    ref, _, rates = _fit_at_startup(s, frac)

    def predict(win):
        return hmm.decode_with(
            win,
            lambda c: hmm.gaussian_loglik(win.X[c], ref, win.sigma, s.norm),
            rates[win.traj], s.log_pi,
        )
    return predict


METHODS = {
    "Start-up calibrated (map + rates)": startup_full,
    "Start-up calibrated (map only, mean rates)": startup_map_only,
    "Mean-unit physics + fitted rates": startup_rates_only,
}


def demo():
    """Recover a known unit and a known usage pattern from clean readings."""
    from statespace import GROUND, N_ANGLES

    rng = np.random.default_rng(0)
    # a toy population: smooth random deviations around a mean map
    ref_true = rng.normal(size=(N_STATES, 3))
    comps = rng.normal(size=(3, N_STATES, 3))
    coefs = rng.normal(size=(200, 3)) * np.array([1.0, 0.6, 0.3])
    tables = ref_true + np.einsum("nj,jsa->nsa", coefs, comps)

    ref, basis, coef_var = tolerance_basis(tables, k=3)
    assert np.allclose(ref, tables.mean(0))

    # one held-out unit, observed over a long walk with no sensor noise.
    # The walk must be legal under the transition model - one rotation step at
    # a time - or the decoder rightly refuses to believe it.
    held = ref_true + comps[0] * 2.0
    path = GROUND * N_ANGLES + np.arange(60) % N_ANGLES
    X = held[path][None].astype(np.float64)
    log_pi = np.full(N_STATES, -np.log(N_STATES))
    rates = np.array([[0.4, 0.2]])

    tbl, r = self_calibrate(
        X, ref, basis, coef_var, scale=1e-3, tol_scale=1.0, rates=rates,
        log_pi=log_pi, rate_prior=np.array([2.0, 1.0, 2.0]),
    )
    before = np.abs(ref - held)[path].mean()
    after = np.abs(tbl[0] - held)[path].mean()
    assert after < before / 2, f"calibration should sharpen the map ({before}->{after})"
    assert r.shape == (1, 2) and (r >= 0).all() and (r.sum(1) <= 1).all()
    print(f"calibration: ok (map error on visited states {before:.3f} -> {after:.3f})")


if __name__ == "__main__":
    from evaluate import main

    main(METHODS, "calibration", check=demo)
