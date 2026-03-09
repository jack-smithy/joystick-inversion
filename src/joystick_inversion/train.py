import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

labels = {
    "south": 0,
    "north": 1,
    "east": 2,
    "west": 3,
    "ground": 4,
    "push": 5,
}

# Inverse map: {0: "south", 1: "north", ...}
label_names = {v: k for k, v in labels.items()}
class_names = [label_names[i] for i in range(len(labels))]  # ordered list


def train_tilt(
    data: pd.DataFrame,
    model: LGBMClassifier,
    include_push: bool = False,
) -> LGBMClassifier:

    X = data[["Bx", "By", "Bz"]]
    y = data["tilt"].astype(np.int8)

    if not include_push:
        data = data[data["tilt"] != 5]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=42,
    )

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    res = classification_report(y_val, y_pred, target_names=class_names)
    print(res)

    return model


def process_angle(y: pd.Series) -> pd.Series:
    y *= np.pi / 180
    sin_y, cos_y = y.map(np.sin), y.map(np.cos)
    return pd.Series({"sin_y": sin_y, "cos_y": cos_y})


def unprocess_angle(y: pd.Series) -> pd.Series:
    return np.arctan2(y["sin_y"], y["cos_y"]) * 180 / np.pi


def train_angle(
    data: pd.DataFrame,
    model: LGBMRegressor,
    include_push: bool = False,
):
    if not include_push:
        data = data[data["tilt"] != 5]

    X = data[["Bx", "By", "Bz"]]
    y = data["angle"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=42,
    )

    model = model.fit(X_train, y_train)

    return model
