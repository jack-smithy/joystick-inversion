import numpy as np
from pathlib import Path

DIRECTIONS = [
    "north",
    "south",
    "east",
    "west",
    "zero",
]

DIRECTION_MAP = {
    "south": 0,
    "north": 1,
    "east": 2,
    "west": 3,
    "zero": 4,
}


def load_real_data(path: Path):
    fields = []
    directions = []
    for file in path.iterdir():
        for dir in DIRECTIONS:
            if dir in file.__str__():
                data = np.load(file)["B"]
                fields.append(data)
                dirs = np.tile(np.array((DIRECTION_MAP[dir])), (data.shape[0],))
                directions.append(dirs)

    return fields, directions


if __name__ == "__main__":
    path = Path("data/calibrated")
    fields, directions = load_real_data(path=path)

    print(len(fields), len(directions))
