"""
The data the benchmark decodes: a population of joysticks, the way people use
them, and the readings that come out.

Three sources of variation live here, and keeping them in one place is the
point - every one of them is something a deployed decoder does *not* know:

  * **Which unit.** Each `make_dataset` seed is a different physical joystick,
    its forward map perturbed by build tolerances.
  * **How it is being used.** Rotate/tilt/hold rates are drawn per unit and
    again per session, and recorded nowhere a decoder can read.
  * **Sensor noise.** Resampled on every call, so it can be treated as
    augmentation during training.

Everything is cached to `.bench_cache/`, since generating trajectories through
`loader.make_trajectory` is slow enough to be worth doing once.
"""

from pathlib import Path

import numpy as np

from joystick import make_dataset
from loader import make_trajectory
from parameters import magnetization_values
from statespace import N_ANGLES, N_STATES

CACHE = Path(".bench_cache")

# Usage patterns: (rotate, tilt, hold) mixes. ALPHA0 sets the population - mean
# (0.4, 0.2, 0.4) - but now with real spread. KAPPA controls how tightly one
# unit's sessions cluster around that unit's own habit.
ALPHA0 = np.array([2.0, 1.0, 2.0])
KAPPA = 20.0

# Bounded history: a decoder sees the newest W frames and nothing else.
WINDOWS = [1, 2, 3, 5, 8, 16]
MAIN_WINDOW = 5  # the "few recent readings" case, used for headline tables
WARMUP = 32  # frames reserved for start-up calibration; scoring uses what follows
STRIDE = 4  # subsample window start positions

# Disjoint seed blocks -> a joystick can never appear in two splits.
SYS_SEED_BASE = dict(train=0, val=10_000, test=20_000)


# --------------------------------------------------------------------------
# joysticks
# --------------------------------------------------------------------------
def clean_table(seed: int = 0) -> np.ndarray:
    """(120, 3) B reading for each state of one joystick; row = tilt*24 + angle.

    `seed` draws that unit's manufacturing tolerances (magnet/sensor positions
    to 0.1 mm, orientations to 0.1 deg), so a different seed is a different
    physical joystick.
    """
    X, y = make_dataset(
        magnetizations=magnetization_values(), seed=seed, n_steps=N_ANGLES
    )
    assert np.array_equal(y.tilt_idx * N_ANGLES + y.angle_idx, np.arange(N_STATES))
    return X.to_numpy(np.float32)


def system_tables(split: str, n_systems: int) -> np.ndarray:
    """(n_systems, 120, 3) forward maps, one per joystick, cached to disk."""
    CACHE.mkdir(exist_ok=True)
    base = SYS_SEED_BASE[split]
    path = CACHE / f"tables_{split}_n{n_systems}.npy"
    if path.exists():
        return np.load(path)
    tables = np.stack([clean_table(base + i) for i in range(n_systems)])
    np.save(path, tables)
    return tables


def normalisation(tables_tr: np.ndarray) -> tuple:
    """(mean, std) for standardising readings, from *training* units only.

    Standardising with each unit's own statistics would be a per-unit
    calibration, and would quietly hand back the tolerance this benchmark
    exists to pose.
    """
    return tables_tr.reshape(-1, 3).mean(0), float(tables_tr.std())


# --------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------
def usage_patterns(n_traj: int, n_units: int, seed: int) -> np.ndarray:
    """(n_traj, 2) per-trajectory [p_rotate, p_tilt]. Never told to a decoder.

    Real users are not one Markov chain with fixed rates. Each unit gets its own
    habitual mix of rotating / tilting / holding still, and each session
    deviates from that habit, so the dynamics vary both between joysticks and
    between sessions on the same joystick.
    """
    rng = np.random.default_rng([seed, 7])
    base = rng.dirichlet(ALPHA0, size=n_units)  # per-unit habit
    unit_of = np.arange(n_traj) % n_units
    p = np.stack([rng.dirichlet(KAPPA * base[u]) for u in unit_of])
    return p[:, :2]  # [p_rotate, p_tilt]; the remainder is p_hold


def population_rates(states: np.ndarray) -> np.ndarray:
    """The only rates a decoder may use: averages counted off training data."""
    return np.array([
        (states[:, 1:] % N_ANGLES != states[:, :-1] % N_ANGLES).mean(),
        (states[:, 1:] // N_ANGLES != states[:, :-1] // N_ANGLES).mean(),
    ])


def cached_states(n_traj: int, seq_len: int, seed: int, n_units: int):
    """(n_traj, seq_len) states and (n_traj, 2) true rates, via make_trajectory."""
    CACHE.mkdir(exist_ok=True)
    stem = f"n{n_traj}_L{seq_len}_s{seed}_u{n_units}"
    s_path, p_path = CACHE / f"states_{stem}.npy", CACHE / f"rates_{stem}.npy"
    if s_path.exists():
        return np.load(s_path), np.load(p_path)

    X, y = make_dataset(magnetizations=magnetization_values(), seed=0, n_steps=N_ANGLES)
    p = usage_patterns(n_traj, n_units, seed)
    rows = [
        make_trajectory(X, y, seq_len, float(p[i, 0]), float(p[i, 1]),
                        seed=[seed, i], start_angle=None)
        for i in range(n_traj)
    ]
    states = np.stack([
        (r["tilt_idx"].to_numpy() * N_ANGLES + r["angle_idx"].to_numpy()) for r in rows
    ]).astype(np.int16)
    np.save(s_path, states)
    np.save(p_path, p)
    return states, p


# --------------------------------------------------------------------------
# splits and readings
# --------------------------------------------------------------------------
def make_split(states: np.ndarray, tables: np.ndarray) -> tuple:
    """Whole trajectories bound to joysticks, for training."""
    return states, np.arange(len(states)) % len(tables), tables


def window_split(states, tables, W, stride=STRIDE, warmup=WARMUP, align=None):
    """Cut trajectories into W-frame windows -> (states, unit, tables, trajectory).

    Each window is an independent short episode: the decoder gets these W
    readings and nothing else. Windows start after `warmup` frames, which are
    reserved for start-up calibration - so nothing scored here overlaps the
    readings a warm-started decoder calibrated on. It also skips the
    ground-start frames, which would otherwise flatter the early windows.

    Windows are aligned on their *last* frame, not their first, so every W
    predicts exactly the same set of targets and differs only in how much
    context precedes them. Without that, sweeping W would quietly change the
    evaluation set as well as the history, and a method that ignores history
    would appear to vary with it.
    """
    idx = window_indices(states.shape[1] - warmup, W, stride, align)
    v = states[:, warmup:][:, idx]  # (n_traj, n_windows, W)
    traj = np.repeat(np.arange(v.shape[0]), v.shape[1])
    sys = traj % len(tables)  # a trajectory keeps its unit; tolerances are hardware
    return v.reshape(-1, W), sys, tables, traj


def window_indices(length: int, W: int, stride=STRIDE, align=None) -> np.ndarray:
    """(n_windows, W) offsets into a post-warmup trajectory, aligned on the end."""
    align = max(WINDOWS) if align is None else align
    assert W <= align, f"W={W} needs align >= {W}"
    ends = np.arange(align - 1, length, stride)
    return ends[:, None] + np.arange(-W + 1, 1)


def window_readings(obs, W, stride=STRIDE, warmup=WARMUP, align=None) -> np.ndarray:
    """Whole-trajectory readings -> (n_windows, W, 3), matching `window_split`.

    Readings are drawn once per trajectory and then sliced, so a given instant
    carries the same noise no matter which window it lands in - as it would on
    real hardware, where there is one sensor sample per instant, not one per
    window. Sweeping W then varies context alone.
    """
    idx = window_indices(obs.shape[1] - warmup, W, stride, align)
    return obs[:, warmup:][:, idx].reshape(-1, W, obs.shape[-1])


def flatten(split: tuple) -> tuple:
    """Split -> one frame per row, for the per-frame models."""
    states, sys, tables = split[0], split[1], split[2]
    return states.reshape(-1, 1), np.repeat(sys, states.shape[1]), tables


def observe(split: tuple, sigma: float, seed: int, norm: tuple):
    """Split -> noisy standardised readings from each trajectory's own joystick."""
    states, sys, tables = split[0], split[1], split[2]
    rng = np.random.default_rng(seed)
    B = tables[sys[:, None], states] + rng.normal(0.0, sigma, size=(*states.shape, 3))
    mu, sd = norm
    return ((B - mu) / sd).astype(np.float32)


def demo():
    tables = system_tables("test", 4)
    assert tables.shape == (4, N_STATES, 3)
    # different seeds really are different joysticks
    assert not np.allclose(tables[0], tables[1])
    norm = normalisation(tables)

    states, rates = cached_states(6, 24, seed=99, n_units=4)
    assert states.shape == (6, 24) and rates.shape == (6, 2)
    assert (rates.sum(1) <= 1).all(), "rotate + tilt must leave room for holding"

    sw = window_split(states, tables, W=3, stride=1, warmup=8, align=3)
    assert sw[0].shape[1] == 3
    assert (sw[0][:, -1] == states[:, 8:][:, 2:].reshape(-1)).all(), "newest frame"

    # every history length must see the same targets *and* the same readings,
    # so the W sweep measures context and nothing else
    obs = observe(make_split(states, tables), 0.05 * norm[1], 3, norm)
    cut = dict(stride=2, warmup=4, align=8)
    tgt = {W: window_split(states, tables, W, **cut)[0][:, -1] for W in (1, 2, 5, 8)}
    read = {W: window_readings(obs, W, **cut) for W in (1, 2, 5, 8)}
    assert all(np.array_equal(t, tgt[1]) for t in tgt.values()), "targets must match"
    assert all(np.allclose(r[:, -1], read[1][:, -1]) for r in read.values()), (
        "the newest reading must be identical across history lengths"
    )

    clean = observe(sw, 0.0, 0, norm)
    noisy = observe(sw, 0.05 * norm[1], 0, norm)
    assert clean.shape == (len(sw[0]), 3, 3)
    assert np.abs(noisy - clean).mean() > 0, "noise must actually be added"
    # a window's readings come from its own unit's map
    one = ((tables[sw[1][0], sw[0][0]] - norm[0]) / norm[1]).astype(np.float32)
    assert np.allclose(clean[0], one)
    print("population: ok")


if __name__ == "__main__":
    demo()
