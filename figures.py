"""
Figures for the write-up. Reads the cached sweep, writes figures/*.png + *.svg.

    uv run figures.py

Colour is assigned per method and held fixed across every chart, so a reader
who learns "the filter is yellow" on one slide is not re-taught on the next.
Ceilings are drawn as a grey dashed reference rather than a sixth colour -
an oracle is not a method you can ship, and should not compete for attention.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calibration import tolerance_basis  # noqa: E402
from population import system_tables  # noqa: E402
from report import MAIN_WINDOW  # noqa: E402
from statespace import N_ANGLES, N_STATES  # noqa: E402

OUT = Path("figures")
CACHE = Path(".bench_cache")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"
CEILING = "#8a8984"

# Validated categorical order (light mode, adjacent pairs): see dataviz palette.
# Entity -> colour, fixed everywhere.
COLOR = {
    "Lookup vs mean unit (no training)": "#2a78d6",   # slot 1 blue
    "Per-frame MLP": "#eb6834",                       # slot 2 orange
    "GRU (causal)": "#1baf7a",                        # slot 3 aqua
    "Per-frame MLP + HMM filter": "#eda100",          # slot 4 yellow
    "Start-up calibrated (map + rates)": "#e87ba4",   # slot 5 magenta
}
SHORT = {
    "Lookup vs mean unit (no training)": "Lookup, no training",
    "Per-frame MLP": "Per-frame MLP",
    "GRU (causal)": "GRU",
    "Per-frame MLP + HMM filter": "MLP + HMM filter",
    "Start-up calibrated (map + rates)": "+ start-up calibration",
}
ORACLE = "Oracle map + oracle rates"
REF_NOISE = 0.10


def style():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12.5, "axes.labelsize": 10.5,
        "axes.titleweight": "semibold", "axes.titlepad": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "figure.constrained_layout.use": True,
    })


def finish(ax, title, xlabel, ylabel, subtitle=None):
    # title and subtitle are drawn as axes-relative text so the two never
    # collide the way set_title + a floating label does
    ax.text(0, 1.10 if subtitle else 1.03, title, transform=ax.transAxes,
            fontsize=12.5, fontweight="semibold", color=INK, va="bottom")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=INK_2, va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.png + .svg")


def load():
    d = json.loads((CACHE / "results.json").read_text())
    return d["matched"], d["mismatch"], d["windows"], d["noise_fracs"]


def pick(rows, model, **kw):
    out = [r for r in rows
           if r["model"] == model and all(r[k] == v for k, v in kw.items())]
    return out


# --------------------------------------------------------------------------
def fig_history(rows, windows):
    """The headline: how much history is actually worth having."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for model, colour in COLOR.items():
        ys = [pick(rows, model, noise=REF_NOISE, window=w)[0]["joint"] for w in windows]
        ax.plot(windows, ys, color=colour, marker="o", label=SHORT[model],
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ceil = [pick(rows, ORACLE, noise=REF_NOISE, window=w)[0]["joint"] for w in windows]
    ax.plot(windows, ceil, color=CEILING, linestyle=(0, (4, 3)), linewidth=1.6,
            label="Oracle (knows unit + user)", zorder=2)

    ax.axvline(MAIN_WINDOW, color=GRID, linewidth=1.2, zorder=1)
    ax.annotate(f"{MAIN_WINDOW} frames", xy=(MAIN_WINDOW, 87.4), fontsize=9,
                color=INK_2, ha="left", va="bottom", xytext=(4, 0),
                textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows)
    ax.set_xticklabels(windows)
    ax.set_ylim(54, 88)
    finish(ax, "Five frames of history is nearly all it is worth",
           "frames of history available (W)", "state accuracy (%)",
           f"held-out joysticks, {REF_NOISE:.0%} sensor noise")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    save(fig, "01_history")


def fig_noise(rows, noise_fracs):
    """How each approach degrades as the sensor gets worse."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = [f * 100 for f in noise_fracs]
    for model, colour in COLOR.items():
        ys = [pick(rows, model, noise=f, window=MAIN_WINDOW)[0]["joint"]
              for f in noise_fracs]
        ax.plot(xs, ys, color=colour, marker="o", label=SHORT[model],
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ceil = [pick(rows, ORACLE, noise=f, window=MAIN_WINDOW)[0]["joint"]
            for f in noise_fracs]
    ax.plot(xs, ceil, color=CEILING, linestyle=(0, (4, 3)), linewidth=1.6,
            label="Oracle (knows unit + user)", zorder=2)
    finish(ax, "Sensor noise, not tolerance, sets the difficulty",
           "sensor noise (% of signal std)", "state accuracy (%)",
           f"held-out joysticks, W={MAIN_WINDOW} frames of history")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    save(fig, "02_noise")


def fig_unknowns(rows):
    """What each thing the decoder does not know actually costs."""
    order = [
        ("Mean-unit physics + HMM filter", "knows neither"),
        ("Start-up calibrated (map only, mean rates)", "fits the unit's map"),
        ("Mean-unit physics + fitted rates", "fits the user's rates"),
        ("Start-up calibrated (map + rates)", "fits both"),
        ("Oracle map, mean rates", "told the true map"),
        ("Oracle map + oracle rates", "told map and rates"),
    ]
    vals = [pick(rows, m, noise=REF_NOISE, window=MAIN_WINDOW)[0]["joint"]
            for m, _ in order]
    base = vals[0]
    # Plot the gain over "knows neither", not the absolute score: the spread is
    # ~2 points on a ~81 point base, so absolute bars would either be six
    # identical stripes or need a truncated axis, where length stops meaning
    # magnitude. The delta has a true zero, so the bars are honest.
    deltas = [v - base for v in vals][1:]
    labels = [lab for _, lab in order][1:]
    is_oracle = [m.startswith("Oracle") for m, _ in order][1:]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ypos = np.arange(len(deltas))
    ax.barh(ypos, deltas, color=[CEILING if o else "#2a78d6" for o in is_oracle],
            height=0.6, zorder=3)
    for y, (d, v) in enumerate(zip(deltas, vals[1:])):
        ax.text(d + 0.04, y, f"{d:+.1f}  ({v:.1f}%)", va="center", fontsize=9.5,
                color=INK_2)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(deltas) + 0.75)
    finish(ax, "What the decoder does not know, priced",
           f"accuracy gained over knowing neither (points)", "",
           f"{REF_NOISE:.0%} sensor noise, W={MAIN_WINDOW}; baseline {base:.1f}%. "
           "Grey = oracle, not available in the field")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    save(fig, "03_unknowns")


def fig_tilt_vs_angle(rows):
    """Tilt is the whole problem; angle was never in doubt."""
    models = list(COLOR) + [ORACLE]
    tilt = [pick(rows, m, noise=REF_NOISE, window=MAIN_WINDOW)[0]["tilt"]
            for m in models]
    angle = [pick(rows, m, noise=REF_NOISE, window=MAIN_WINDOW)[0]["angle"]
             for m in models]
    names = [SHORT.get(m, "Oracle").replace(" + ", "\n+ ").replace(
        "Lookup, no training", "Lookup\n(no training)").replace(
        "Per-frame MLP", "Per-frame\nMLP") for m in models]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(models))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, tilt, w, label="tilt (5 classes)",
           color="#2a78d6", zorder=3)
    ax.bar(x + w / 2 + 0.01, angle, w, label="angle (24 classes)",
           color="#eb6834", zorder=3)
    for xi, (t, a) in enumerate(zip(tilt, angle)):
        ax.text(xi - w / 2 - 0.01, t + 1, f"{t:.0f}", ha="center", fontsize=9,
                color=INK_2)
        ax.text(xi + w / 2 + 0.01, a + 1, f"{a:.0f}", ha="center", fontsize=9,
                color=INK_2)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 108)
    finish(ax, "Tilt is the bottleneck - angle was never the problem",
           "", "accuracy (%)",
           f"{REF_NOISE:.0%} sensor noise, W={MAIN_WINDOW}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "04_tilt_vs_angle")


def fig_geometry(cfg_n_sys=800):
    """Why tilt is hard, measured in field space."""
    tables = system_tables("train", cfg_n_sys)
    table, signal = tables.mean(0), tables.std()
    D = np.linalg.norm(table[:, None, :] - table[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    tilt = np.arange(N_STATES) // N_ANGLES
    angle = np.arange(N_STATES) % N_ANGLES

    def nearest(mask):
        return np.median(np.where(mask, D, np.inf).min(1)) / signal

    bars = [
        ("rotation neighbour\n(same tilt, 15 deg)",
         nearest(tilt[:, None] == tilt[None, :])),
        ("tilt neighbour\n(same angle, 4 deg)",
         nearest(angle[:, None] == angle[None, :])),
    ]
    spread = float(np.sqrt(tables.var(0).mean())) / signal

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(bars))
    ax.bar(x, [v for _, v in bars], 0.5, color="#2a78d6", zorder=3)
    for xi, (_, v) in zip(x, bars):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=10, color=INK_2)
    ax.axhline(spread, color="#eb6834", linewidth=1.8, linestyle=(0, (4, 3)),
               zorder=4)
    ax.text(0.5, spread + 0.012, f"unit-to-unit spread  {spread:.3f}",
            ha="center", fontsize=9.5, color="#eb6834")
    ax.set_xticks(x, [n for n, _ in bars], fontsize=9.5)
    finish(ax, "Why tilt is hard: a 4 degree tilt barely moves the magnet",
           "", "median distance to nearest such state\n(signal std)",
           "states are packed ~2x more tightly along tilt than along rotation")
    save(fig, "05_geometry")


def fig_tolerance_rank(cfg_n_sys=800):
    """Tolerance is low-rank, which is what makes self-calibration possible."""
    tables = system_tables("train", cfg_n_sys)
    dev = (tables - tables.mean(0)).reshape(len(tables), -1)
    s = np.linalg.svd(dev, compute_uv=False) ** 2
    cum = np.cumsum(s) / s.sum() * 100
    k = np.arange(1, 21)

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(k, cum[:20], color="#2a78d6", marker="o", markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3)
    ax.axvline(10, color=GRID, linewidth=1.2, zorder=1)
    ax.annotate(f"10 components\nexplain {cum[9]:.1f}%", xy=(10, cum[9]),
                xytext=(11.2, 55), textcoords="data", fontsize=9.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.9,
                                shrinkA=0, shrinkB=4))
    ax.set_ylim(0, 104)
    ax.set_xticks([1, 5, 10, 15, 20])
    finish(ax, "A joystick's build error is only ~10 numbers",
           "principal components of the unit-to-unit deviation",
           "variance explained (%)",
           "the 360-dim forward map moves through only ~20 physical parameters")
    save(fig, "06_tolerance_rank")


def fig_mismatch(mismatch, noise_fracs):
    """Training noise is a guess; what does guessing wrong cost?"""
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    xs = [f * 100 for f in noise_fracs]
    for model, colour, lab in [
        (f"Per-frame MLP (trained @ 5%)", "#eb6834", "per-frame MLP"),
        (f"MLP + HMM filter (trained @ 5%)", "#eda100", "MLP + HMM filter"),
    ]:
        ys = [r["joint"] for f in noise_fracs
              for r in mismatch if r["model"] == model and r["noise"] == f]
        ax.plot(xs, ys, color=colour, marker="o", label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.axvline(5, color=GRID, linewidth=1.2, zorder=1)
    ax.annotate("trained here", xy=(5, 45), xytext=(5, 0),
                textcoords="offset points", fontsize=9, color=INK_2)
    finish(ax, "One model, trained at 5% noise, tested everywhere",
           "sensor noise at test time (% of signal std)", "state accuracy (%)",
           f"W={MAIN_WINDOW}; the filter absorbs most of the mismatch")
    ax.legend(loc="lower left")
    save(fig, "07_noise_mismatch")


def main():
    style()
    rows, mismatch, windows, noise_fracs = load()
    print("writing figures/")
    fig_history(rows, windows)
    fig_noise(rows, noise_fracs)
    fig_unknowns(rows)
    fig_tilt_vs_angle(rows)
    fig_geometry()
    fig_tolerance_rank()
    fig_mismatch(mismatch, noise_fracs)
    print("done")


if __name__ == "__main__":
    main()
