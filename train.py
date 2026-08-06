from dataclasses import dataclass

import numpy as np
import numpy.linalg as LA
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import TILT_NAMES, mT_TO_T


@dataclass
class Predictions:
    """
    Raw per-sample predictions, so every metric and plot shares one inference pass.
    """

    tilt_true: np.ndarray
    tilt_pred: np.ndarray
    angle_idx_true: np.ndarray
    angle_idx_pred: np.ndarray
    angle_error: np.ndarray  # deg, per sample


def bin_error(predictions: Predictions, n_steps: int = 24) -> np.ndarray:
    """
    How many rotation steps out each prediction is, the short way round the circle.
    """
    delta = np.abs(predictions.angle_idx_pred - predictions.angle_idx_true)
    return np.minimum(delta, n_steps - delta)


def process_angle(y: np.ndarray) -> np.ndarray:
    """
    y  -> (sin(y), cos(y))
    """
    y_rad = np.deg2rad(y)
    sin_y, cos_y = np.sin(y_rad), np.cos(y_rad)
    return np.concatenate((sin_y[None], cos_y[None]), axis=0).T


def unprocess_angle(y: np.ndarray) -> np.ndarray:
    """
    (sin(y), cos(y)) -> y
    """
    norm = np.linalg.norm(y, axis=1, keepdims=True)
    y_normalized = y / np.clip(norm, 1e-8, None)
    return np.rad2deg(np.arctan2(y_normalized[:, 0], y_normalized[:, 1]))


def angle_index(y: np.ndarray, n_steps: int = 24) -> np.ndarray:
    """
    (sin(y), cos(y)) -> nearest discrete rotation state

    The joystick only rests at `n_steps` rotations, so a continuous prediction can be
    snapped back to one. Angles come out of `unprocess_angle` in (-180, 180], hence the
    modulo, which also handles the wrap either side of 0.

    Args:
        y (np.ndarray): (n, 2) of (sin, cos). Need not be normalized.
        n_steps (int, optional): Rotation discretization. Defaults to 24.

    Returns:
        np.ndarray: (n,) rotation state indices in [0, n_steps).
    """
    degrees = unprocess_angle(y)
    return np.round(degrees / (360 / n_steps)).astype(int) % n_steps


def angular_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    (sin(y_pred), cos(y_pred)), (sin(y_true), cos(y_true)) -> per-sample dy in degrees

    The mean of this hides a long tail, so the spread is usually the interesting part.
    """
    y_true = y_true / LA.norm(y_true, axis=1, keepdims=True)
    y_pred = y_pred / LA.norm(y_pred, axis=1, keepdims=True)

    dot = np.sum(y_true * y_pred, axis=1)
    dot = np.clip(dot, -1, 1)

    return np.rad2deg(np.arccos(dot))


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
        DataLoader: yields X (batch, 2, 3) = (B_start, B_end) in mT,
            y (batch, 2) = (angle_idx_end, tilt_end) as int64 class labels.
    """
    B = df[["Bx_start", "By_start", "Bz_start", "Bx_end", "By_end", "Bz_end"]]

    # copy=True: pandas hands back negative-stride views that torch rejects
    # (N, 6) -> (N, 2, 3): one 3-D field reading per timestep
    X = torch.tensor(B.to_numpy(copy=True), dtype=torch.float32).reshape(-1, 2, 3)
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
    assert X.shape == (16, 2, 3) and X.dtype == torch.float32
    assert y.shape == (16, 2) and y.dtype == torch.long
    assert y[:, 0].max() < 24 and y[:, 1].max() < len(TILT_NAMES)

    # the trajectory really is (B_start, B_end) in mT, in that order
    clean = next(iter(make_dataloader(t, batch_size=len(t), shuffle=False)))[0]
    B_start = t[["Bx_start", "By_start", "Bz_start"]].to_numpy() / mT_TO_T
    assert np.allclose(clean[:, 0], B_start, atol=1e-6)
    assert np.allclose(
        clean[:, 1], t[["Bx_end", "By_end", "Bz_end"]].to_numpy() / mT_TO_T, atol=1e-6
    )

    # noise is applied on top, at the requested scale
    noisy = next(iter(make_dataloader(t, len(t), shuffle=False, noise=0.1, seed=0)))[0]
    assert not torch.allclose(noisy, clean)
    assert abs((noisy - clean).std().item() - 0.1) < 0.01

    print(X.shape, y.shape)

    # every rest position bins back to itself, and the wrap either side of 0 lands on 0
    states = np.arange(24)
    assert np.array_equal(angle_index(process_angle(states * 15.0)), states)
    assert np.array_equal(angle_index(process_angle(np.array([359.0, 1.0]))), [0, 0])
    assert np.array_equal(angle_index(process_angle(np.array([352.6, 7.4]))), [0, 0])
    assert np.array_equal(angle_index(process_angle(np.array([337.6, 22.4]))), [23, 1])

    print("ok")
