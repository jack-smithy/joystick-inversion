"""
The shared evaluation harness: identical data, identical scoring, for every
approach.

Each algorithm module exports a `METHODS` dict and calls `main(METHODS, name)`,
so any of them can be run on its own and produce real numbers:

    uv run hmm.py            structured decoding only
    uv run net_mlp.py        the per-frame network only
    uv run benchmarks.py     everything, plus the report

A *method* is a factory `fn(setup, noise_frac) -> (window -> predictions)`.
The factory runs once per noise level, which is where anything expensive lives
(training a network, fitting a calibration); the returned closure then runs per
history length. Predictions are state indices for the **newest frame** of each
window, because that is what a deployed decoder has to answer.

Keeping this in one place is what makes the comparison fair: every method sees
the same joysticks, the same trajectories and the same noise draw.
"""

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

from population import (
    CACHE, WINDOWS, cached_states, make_split, normalisation, observe,
    population_rates, system_tables, window_readings, window_split,
)
from statespace import N_STATES, score

DEFAULT_CFG = dict(
    n_train=6000, n_val=1500, n_test=1500, seq_len=64, frame_steps=8000, seq_steps=3000,
    n_sys_train=800, n_sys_val=200, n_sys_test=200, warmup=32, align=16,
)
SMOKE_CFG = dict(
    n_train=600, n_val=200, n_test=200, seq_len=32, frame_steps=800, seq_steps=300,
    n_sys_train=60, n_sys_val=20, n_sys_test=20, warmup=16, align=8,
)
NOISE_FRACS = [0.0, 0.02, 0.05, 0.10, 0.20]
SMOKE_NOISE = [0.0, 0.10]
SMOKE_WINDOWS = [1, 3, 8]


@dataclass
class Setup:
    """Everything the methods share. Built once, then held constant."""

    cfg: dict
    tables_tr: np.ndarray
    tables_va: np.ndarray
    tables_te: np.ndarray
    norm: tuple
    signal: float
    ref_table: np.ndarray  # the mean training unit: the best map without calibration
    tol_scale: float  # across-unit spread, in signal std
    states_tr: np.ndarray
    states_va: np.ndarray
    states_te: np.ndarray
    rates_te: np.ndarray  # true usage rates - oracle rows only
    split_tr: tuple
    split_va: tuple
    pop_rates: np.ndarray  # population averages: what a decoder may assume
    log_pi: np.ndarray
    log_prior: np.ndarray

    @property
    def warmup(self) -> int:
        return self.cfg["warmup"]

    @property
    def align(self) -> int:
        """Longest history swept: windows line up on their last frame so that
        every W predicts the same targets from the same readings."""
        return self.cfg["align"]

    @property
    def ref_n(self) -> np.ndarray:
        """The mean unit's map, standardised."""
        return (self.ref_table - self.norm[0]) / self.signal

    def const_rates(self, n: int) -> np.ndarray:
        return np.repeat(self.pop_rates[None], n, axis=0)


@dataclass
class Window:
    """One (noise level, history length) evaluation slice."""

    frac: float
    sigma: float
    W: int
    sw: tuple
    X: np.ndarray
    y: np.ndarray  # true state of the newest frame
    chunks: list = field(default_factory=list)

    @property
    def traj(self) -> np.ndarray:
        return self.sw[3]


def build_setup(cfg: dict) -> Setup:
    tables_tr = system_tables("train", cfg["n_sys_train"])
    tables_va = system_tables("val", cfg["n_sys_val"])
    tables_te = system_tables("test", cfg["n_sys_test"])
    assert not (
        set(map(tuple, tables_tr.reshape(len(tables_tr), -1)))
        & set(map(tuple, tables_te.reshape(len(tables_te), -1)))
    ), "train and test joysticks overlap"

    norm = normalisation(tables_tr)
    states_tr, _ = cached_states(cfg["n_train"], cfg["seq_len"], 1, cfg["n_sys_train"])
    states_va, _ = cached_states(cfg["n_val"], cfg["seq_len"], 2, cfg["n_sys_val"])
    states_te, rates_te = cached_states(cfg["n_test"], cfg["seq_len"], 3,
                                        cfg["n_sys_test"])
    prior = np.bincount(states_tr.ravel(), minlength=N_STATES).astype(np.float64)
    prior /= prior.sum()
    log_prior = np.log(prior + 1e-12)

    return Setup(
        cfg=cfg,
        tables_tr=tables_tr, tables_va=tables_va, tables_te=tables_te,
        norm=norm, signal=norm[1],
        ref_table=tables_tr.mean(0),
        tol_scale=float(np.sqrt(tables_tr.var(0).mean())) / norm[1],
        states_tr=states_tr, states_va=states_va, states_te=states_te,
        rates_te=rates_te,
        split_tr=make_split(states_tr, tables_tr),
        split_va=make_split(states_va, tables_va),
        pop_rates=population_rates(states_tr),
        log_pi=log_prior, log_prior=log_prior,
    )


def chunks(n: int, size: int = 4000) -> list:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def test_readings(s: Setup, frac: float) -> np.ndarray:
    """(n_traj, seq_len, 3) test readings, drawn once per noise level."""
    return observe(make_split(s.states_te, s.tables_te), frac * s.signal, 20_000,
                   s.norm)


def make_window(s: Setup, frac: float, W: int, obs=None) -> Window:
    obs = test_readings(s, frac) if obs is None else obs
    sw = window_split(s.states_te, s.tables_te, W, warmup=s.warmup,
                      align=s.align)
    X = window_readings(obs, W, warmup=s.warmup, align=s.align)
    return Window(frac=frac, sigma=frac * s.signal, W=W, sw=sw, X=X,
                  y=sw[0][:, -1], chunks=chunks(len(X)))


def sweep(s: Setup, methods: dict, noise_fracs, windows, verbose=True) -> list:
    """Run every method at every noise level and history length."""
    results = []
    for frac in noise_fracs:
        if verbose:
            print(f"\n=== noise {frac:.0%} of signal std "
                  f"(sigma={frac * s.signal:.2e}) ===")
        ready = {name: factory(s, frac) for name, factory in methods.items()}
        obs = test_readings(s, frac)  # one draw shared by every history length
        for W in windows:
            win = make_window(s, frac, W, obs)
            for name, predict in ready.items():
                results.append(dict(model=name, noise=frac, window=W,
                                    **score(predict(win), win.y)))
            if verbose:
                row = {r["model"]: r["joint"] for r in results
                       if r["noise"] == frac and r["window"] == W}
                print(f"  W={W:<3d} " + "  ".join(
                    f"{k.split(' (')[0][:24]} {v:5.1f}" for k, v in row.items()))
    return results


def print_summary(results: list, windows) -> None:
    """Joint accuracy as methods x history length, averaged over noise levels."""
    names = list(dict.fromkeys(r["model"] for r in results))
    width = max(len(n) for n in names)
    print(f"\n{'method'.ljust(width)}  " + "  ".join(f"W={w:<4d}" for w in windows))
    for n in names:
        cells = []
        for w in windows:
            vals = [r["joint"] for r in results if r["model"] == n and r["window"] == w]
            cells.append(f"{np.mean(vals):6.2f}" if vals else "     -")
        print(f"{n.ljust(width)}  " + "  ".join(cells))
    print("\n(mean over noise levels; see BENCHMARKS.md for the full breakdown)")


def main(methods: dict, name: str, check=None) -> list:
    """Shared CLI so every algorithm module can be run on its own."""
    ap = argparse.ArgumentParser(description=f"benchmark: {name}")
    ap.add_argument("--smoke", action="store_true", help="tiny settings")
    ap.add_argument("--check", action="store_true", help="self-check only, no sweep")
    args = ap.parse_args()

    if args.check:
        if check is None:
            print(f"{name}: no self-check defined")
        else:
            check()
        return []

    cfg = SMOKE_CFG if args.smoke else DEFAULT_CFG
    fracs = SMOKE_NOISE if args.smoke else NOISE_FRACS
    windows = SMOKE_WINDOWS if args.smoke else WINDOWS
    s = build_setup(cfg)
    results = sweep(s, methods, fracs, windows)
    print_summary(results, windows)

    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"results_{name}{'_smoke' if args.smoke else ''}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results
