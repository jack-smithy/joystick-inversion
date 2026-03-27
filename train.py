from typing import Literal

import numpy as np
import numpy.linalg as LA
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from constants import DIRECTIONS
from utils import timed


def filter(
    df: pd.DataFrame, col: Literal["tilt", "angle"]
) -> tuple[np.ndarray, np.ndarray]:
    X = df[["Bx", "By", "Bz"]]
    y = df[col]

    if col == "tilt":
        y = y.astype(np.int8)

    return X.to_numpy(), y.to_numpy()


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


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    (sin(y_pred), cos(y_pred)), (sin(y_true), cos(y_true)) -> dy
    """
    y_true /= LA.norm(y_true, axis=1, keepdims=True)
    y_pred /= LA.norm(y_pred, axis=1, keepdims=True)

    dot = np.sum(y_true * y_pred, axis=1)
    dot = np.clip(dot, -1, 1)

    return np.mean(np.rad2deg(np.arccos(dot)))


@timed()
def train_tilt(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model: LGBMClassifier,
) -> LGBMClassifier:

    X_train, y_train = filter(df=df_train, col="tilt")
    X_val, y_val = filter(df=df_val, col="tilt")

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    res = classification_report(y_val, y_pred, target_names=DIRECTIONS)

    print(res)

    return model


@timed()
def train_angle(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model: MultiOutputRegressor,
):
    X_train, y_train = filter(df=df_train, col="angle")
    X_val, y_val = filter(df=df_val, col="angle")

    y_train = process_angle(y=y_train)
    y_val = process_angle(y=y_val)

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    dy = cosine_similarity(y_pred=y_pred, y_true=y_val)  # ty: ignore
    print(f"\n Mean angular error: {dy:.5f}deg\n")

    return model


def val_angle(df: pd.DataFrame, model: MultiOutputRegressor):
    X, y = filter(df=df, col="angle")
    y = process_angle(y=y)

    y_pred = model.predict(X)

    dy = cosine_similarity(y_true=y, y_pred=y_pred)  # ty: ignore
    print(f"\n Mean angular error: {dy:.5f}deg\n")

    return model


def val_tilt(df: pd.DataFrame, model: LGBMClassifier) -> LGBMClassifier:

    X, y = filter(df=df, col="tilt")

    y_pred = model.predict(X)
    res = classification_report(y_true=y, y_pred=y_pred, target_names=DIRECTIONS)

    print(res)

    return model
