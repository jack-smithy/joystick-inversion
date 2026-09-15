from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import FIELD_COLUMNS, TILT_NAMES, mT_TO_T


@dataclass
class Predictions:
    """
    Raw per-sample predictions, so every metric and plot shares one inference pass.
    """

    tilt_true: np.ndarray
    tilt_pred: np.ndarray
    angle_idx_true: np.ndarray
    angle_idx_pred: np.ndarray
    n_steps: int = 24

    @property
    def angle_error(self) -> np.ndarray:
        """
        Per-sample angular error in degrees.

        Quantized: the model picks one of `n_steps` rest positions, so this is the miss
        distance in degrees and can only take multiples of `360 / n_steps`. It is kept
        because the per-state spread is the diagnostic that shows the antipodal tail.
        """
        return bin_error(self, self.n_steps) * (360 / self.n_steps)


def bin_error(predictions: Predictions, n_steps: int = 24) -> np.ndarray:
    """
    How many rotation steps out each prediction is, the short way round the circle.
    """
    delta = np.abs(predictions.angle_idx_pred - predictions.angle_idx_true)
    return np.minimum(delta, n_steps - delta)


def state_index(y: torch.Tensor, n_steps: int = 24) -> torch.Tensor:
    """
    Loader label (angle_idx, tilt) -> the single joint state the model classifies.

    Tilt and rotation are not independent to predict, and there are only
    `5 * n_steps` = 120 of them, so one softmax over the lot beats two heads that can
    disagree. Decode with `state // n_steps` and `state % n_steps`.
    """
    return y[:, 1] * n_steps + y[:, 0]


def make_dataloader(
    df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = True,
    noise: float = 0.0,
    seed: int | None = None,
) -> DataLoader:
    """
    Torch loader over a transition table from `make_transitions`: predict the state
    the joystick ends up in from the length-2 field trajectory that got it there.

    Fields are handed over in mT, which keeps them O(1) for the network.

    Args:
        df (pd.DataFrame): Transition table from `joystick.make_transitions`.
        batch_size (int, optional): Defaults to 32.
        shuffle (bool, optional): Defaults to True.
        noise (float, optional): Std of Gaussian sensor noise in mT, drawn once
            per sample (not re-drawn each epoch). Defaults to 0.0.
        seed (int | None, optional): Seed for that noise. Defaults to None.

    Returns:
        DataLoader: yields X (batch, 2, 6) = (B_start, B_end) in mT, each timestep
            holding both 3-D sensors, y (batch, 2) = (angle_idx_end, tilt_end) as int64
            class labels.
    """
    B = df[[f"{column}_{when}" for when in ("start", "end") for column in FIELD_COLUMNS]]

    # copy=True: pandas hands back negative-stride views that torch rejects
    # (N, 12) -> (N, 2, 6): both sensors' 3-D readings, per timestep
    X = torch.tensor(B.to_numpy(copy=True), dtype=torch.float32).reshape(
        -1, 2, len(FIELD_COLUMNS)
    )
    X /= mT_TO_T
    y = torch.tensor(
        df[["angle_idx_end", "tilt_end"]].to_numpy(copy=True), dtype=torch.long
    )

    if noise:
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        X += noise * torch.randn(X.shape, generator=generator)

    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    from joystick import make_dataset, make_transitions

    t = make_transitions(make_dataset())
    loader = make_dataloader(t, batch_size=16)

    assert len(loader.dataset) == len(t)  # ty: ignore

    X, y = next(iter(loader))
    assert X.shape == (16, 2, len(FIELD_COLUMNS)) and X.dtype == torch.float32
    assert y.shape == (16, 2) and y.dtype == torch.long
    assert y[:, 0].max() < 24 and y[:, 1].max() < len(TILT_NAMES)

    # the trajectory really is (B_start, B_end) in mT, in that order, and each timestep
    # carries both sensors in FIELD_COLUMNS order
    clean = next(iter(make_dataloader(t, batch_size=len(t), shuffle=False)))[0]
    for step, when in enumerate(("start", "end")):
        want = t[[f"{c}_{when}" for c in FIELD_COLUMNS]].to_numpy() / mT_TO_T
        assert np.allclose(clean[:, step], want, atol=1e-6), when

    # noise is applied on top, at the requested scale
    noisy = next(iter(make_dataloader(t, len(t), shuffle=False, noise=0.1, seed=0)))[0]
    assert not torch.allclose(noisy, clean)
    assert abs((noisy - clean).std().item() - 0.1) < 0.01

    print(X.shape, y.shape)

    # the joint state is a bijection: every (angle, tilt) gets its own class, and
    # decoding by // and % lands back on the pair it came from
    all_y = next(iter(make_dataloader(t, batch_size=len(t), shuffle=False)))[1]
    state = state_index(all_y, n_steps=24)
    assert state.min() >= 0 and state.max() < 24 * len(TILT_NAMES)
    assert torch.equal(state // 24, all_y[:, 1]) and torch.equal(state % 24, all_y[:, 0])
    assert state.unique().numel() == 24 * len(TILT_NAMES)

    print("ok")
