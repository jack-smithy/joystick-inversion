from pathlib import Path

import numpy as np


def load_data(path: Path) -> np.ndarray:
    fields = []
    for file in path.iterdir():
        data = np.load(file)
        fields.append(data["B"])
    return np.concat(fields, axis=0)
