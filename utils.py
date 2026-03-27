from collections.abc import Callable
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import ParamSpec, TypeVar

import pandas as pd

from constants import DIRECTION_MAP, DIRECTIONS, mT_TO_T

P = ParamSpec("P")
R = TypeVar("R")


def timed() -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = perf_counter() - start
                print(f"[{func.__name__}]: {elapsed:.6f}s")

        return wrapper

    return decorator


def load_measurement_data(path: Path) -> pd.DataFrame:
    """
    Load measurement data from files and return df in same format as simulated data
    """
    if "sensor2" in path.__str__():
        raise ValueError("Sensor 2 data not trained yet")

    dataset = {
        "Bx": [],
        "By": [],
        "Bz": [],
        "angle": [],
        "tilt": [],
    }

    for file in path.iterdir():
        for direction in DIRECTIONS:
            if direction in file.name:
                with open(file, "r") as f:
                    data = pd.read_csv(f)

                tilt = [DIRECTION_MAP[direction] for _ in range(data.shape[0])]

                dataset["tilt"].extend(tilt)
                dataset["angle"].extend(data["angle_deg"])
                dataset["Bx"].extend(data["Bx[mT]"] * mT_TO_T)
                dataset["By"].extend(data["By[mT]"] * mT_TO_T)
                dataset["Bz"].extend(data["Bz[mT]"] * mT_TO_T)

    return pd.DataFrame(dataset)
