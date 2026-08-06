from pathlib import Path

import magpylib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage
from sklearn.metrics import confusion_matrix

from constants import TILT_NAMES
from train import Predictions, bin_error

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdbd3"
SERIES = ("#2a78d6", "#eb6834")  # blue, orange

# magnitude reads as one hue light -> dark; polarity needs two opposed hues either
# side of a neutral midpoint, so that zero reads as nothing
MAGNITUDE = "Blues"
POLARITY = LinearSegmentedColormap.from_list(
    "blue_neutral_orange",
    (SERIES[0], "#eeede8", SERIES[1]),
)


def _style(ax: Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """
    Recessive chrome: hairline solid grid, no box, labels in muted ink.
    """
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)

    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.6)
    for side, spine in ax.spines.items():
        # polar axes name their spines differently, and need theirs kept
        spine.set_visible(side not in ("top", "right"))
        spine.set_color(GRID)
        spine.set_linewidth(0.6)


def plot_error_histogram(ax: Axes, predictions: Predictions) -> None:
    """
    Where the angular error actually sits. Log counts, because the interesting part
    is the tail rather than the bulk.
    """
    ax.hist(predictions.angle_error, bins=60, color=SERIES[0], linewidth=0)
    ax.set_yscale("log")

    # staggered heights: the two markers sit close together at this scale
    for value, label, height in (
        (np.median(predictions.angle_error), "median", 0.95),
        (np.percentile(predictions.angle_error, 95), "p95", 0.82),
    ):
        ax.axvline(value, color=INK, linewidth=0.8)
        ax.annotate(
            f"{label} {value:.1f}",
            xy=(value, height),
            xycoords=("data", "axes fraction"),
            color=INK,
            fontsize=7,
            ha="left",
            va="top",
            xytext=(4, 0),
            textcoords="offset points",
        )

    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax, "Angular error", "error / deg", "predictions (log)")


def plot_bin_distance(ax: Axes, predictions: Predictions, n_steps: int = 24) -> None:
    """
    Snapped to rest positions, how far out are the misses? Near-misses and
    near-antipodal failures are very different problems.
    """
    delta = bin_error(predictions, n_steps=n_steps)
    counts = np.bincount(delta, minlength=n_steps // 2 + 1)

    ax.bar(np.arange(len(counts)), counts, color=SERIES[0], width=0.75, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(counts)))

    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax, "Miss distance", "rotation steps out", "predictions (log)")


def plot_error_by_state(ax: Axes, predictions: Predictions, n_steps: int = 24) -> None:
    """
    Is the error spread evenly around the circle, or concentrated on some states?

    Drawn on the dial the states actually live on, so a run of bad states reads as an
    arc rather than as a stretch of the x-axis. Expects a polar axes.
    """
    states = np.arange(n_steps)
    rows = [predictions.angle_error[predictions.angle_idx_true == i] for i in states]

    # repeat the first point so the line closes rather than leaving a gap at state 0
    theta = np.concatenate((states, states[:1])) * (2 * np.pi / n_steps)

    for summarize, colour, label in (
        (np.median, SERIES[0], "median"),
        (lambda r: np.percentile(r, 95), SERIES[1], "p95"),
    ):
        values = [summarize(r) if len(r) else np.nan for r in rows]
        ax.plot(theta, values + values[:1], color=colour, linewidth=2, label=label)

    # a dial: state 0 at the top, and cw (the +1 step) going clockwise
    ax.set_theta_zero_location("N")  # ty: ignore
    ax.set_theta_direction(-1)  # ty: ignore

    ax.set_xticks(states[::3] * (2 * np.pi / n_steps), [str(i) for i in states[::3]])

    # keep the radial scale off the data, and the legend off the plot entirely
    ax.set_rlabel_position(247.5)  # ty: ignore
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=MUTED,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax, "Error by rotation state / deg")


def _heatmap(ax: Axes, matrix: np.ndarray, labels: list[str] | None) -> AxesImage:
    """
    Row-normalized confusion, i.e. each row is 'given this true state, what was read'.
    """
    image = ax.imshow(matrix, cmap=MAGNITUDE, vmin=0, vmax=1)

    if labels is not None:
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)

        # only readable on the small matrix, and only worth it off the diagonal
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value >= 0.005:
                    ax.annotate(
                        f"{value:.2f}".lstrip("0"),
                        xy=(j, i),
                        color="#ffffff" if value > 0.5 else MUTED,
                        fontsize=7,
                        ha="center",
                        va="center",
                    )

    return image


def plot_tilt_confusion(ax: Axes, predictions: Predictions) -> None:
    matrix = confusion_matrix(
        y_true=predictions.tilt_true,
        y_pred=predictions.tilt_pred,
        normalize="true",
    )
    _heatmap(ax, matrix, labels=TILT_NAMES)
    _style(ax, "Tilt confusion", "read as", "actually")


def plot_rotation_confusion(ax: Axes, predictions: Predictions) -> None:
    matrix = confusion_matrix(
        y_true=predictions.angle_idx_true,
        y_pred=predictions.angle_idx_pred,
        normalize="true",
    )
    image = _heatmap(ax, matrix, labels=None)
    bar = plt.colorbar(image, ax=ax, fraction=0.046)
    bar.outline.set_linewidth(0)  # ty: ignore
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=3, width=0.6)
    _style(ax, "Rotation confusion", "read as", "actually")


def plot_field_correlation(ax: Axes, transitions: pd.DataFrame) -> None:
    """
    How much the six input features duplicate each other. Strong off-diagonal
    structure means the network has less to work with than six numbers suggests.
    """
    columns = ["Bx_start", "By_start", "Bz_start", "Bx_end", "By_end", "Bz_end"]
    matrix = transitions[columns].corr().to_numpy()

    image = ax.imshow(matrix, cmap=POLARITY, vmin=-1, vmax=1)
    labels = [c.replace("B", "").replace("_", " ") for c in columns]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.annotate(
                f"{value:+.2f}",
                xy=(j, i),
                color=INK if abs(value) < 0.6 else "#ffffff",
                fontsize=6,
                ha="center",
                va="center",
            )

    bar = plt.colorbar(image, ax=ax, fraction=0.046)
    bar.outline.set_linewidth(0)  # ty: ignore
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=3, width=0.6)
    _style(ax, "Field correlation", "", "")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_evaluation(
    predictions: Predictions,
    transitions: pd.DataFrame,
    directory: Path | str = "plots",
    n_steps: int = 24,
) -> list[Path]:
    """
    Write the evaluation figures, grouped so each sheet answers one question.

    Args:
        predictions (Predictions): Output of `run.evaluate`.
        transitions (pd.DataFrame): The test transition table, for the feature
            correlation figure.
        directory (Path | str, optional): Written to, created if missing.
        n_steps (int, optional): Rotation discretization. Defaults to 24.

    Returns:
        list[Path]: The files written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths = []

    # how wrong is the angle, continuously and snapped to rest positions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    plot_error_histogram(axes[0], predictions)
    plot_bin_distance(axes[1], predictions, n_steps=n_steps)
    paths.append(_save(fig, directory / "error.png"))

    # what gets read as what
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    plot_tilt_confusion(axes[0], predictions)
    plot_rotation_confusion(axes[1], predictions)
    paths.append(_save(fig, directory / "confusion.png"))

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    plot_error_by_state(ax, predictions, n_steps=n_steps)
    paths.append(_save(fig, directory / "rotation_error.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    plot_field_correlation(ax, transitions)
    paths.append(_save(fig, directory / "correlation.png"))

    return paths


def show_system(*collections):
    magpylib.show(collections)


def plot_loops(Bfields: list[np.ndarray], labels: list[str] | None = None):
    if labels is not None:
        assert len(Bfields) == len(labels)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")

    for i, B in enumerate(Bfields):
        if labels is not None:
            ax.plot(B[:, 0], B[:, 1], B[:, 2], label=labels[i])
        else:
            ax.plot(B[:, 0], B[:, 1], B[:, 2])
    return fig, ax
