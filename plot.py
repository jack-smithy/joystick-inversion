import magpylib
import matplotlib.pyplot as plt
import numpy as np


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
