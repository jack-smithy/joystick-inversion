import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from joystick import make_dataset
from parameters import calibration_values, magnetization_values


def make_pairs(X: pd.DataFrame, y: pd.DataFrame):
    sweep = pd.concat((X, y), axis=1)
    # ponytail: infer n_steps from data instead of threading a parameter through
    n_steps = int(sweep["angle_idx"].max()) + 1
    pairs = []
    for i, start in sweep.iterrows():
        for j, end in sweep.iterrows():
            pairs.append((start, end))

    legal_pairs = [
        p for p in pairs if filter_legal_tilt(p) and filter_legal_rotation(p, n_steps)
    ]
    return legal_pairs


def filter_legal_tilt(pair) -> bool:
    start, end = pair

    # if we are not in ground state
    if start["tilt"] != "ground":
        # check we are returning to ground
        if end["tilt"] == "ground":
            return True
        # or check we are staying in the same location
        if end["tilt"] == start["tilt"]:
            return True

    # if we are in ground state, any tilt direction is fine
    if start["tilt"] == "ground":
        return True

    return False


def filter_legal_rotation(pair, n_steps: int) -> bool:
    start, end = pair
    delta = (end["angle_idx"] - start["angle_idx"]) % n_steps

    # no rotation: no-op or pure tilt, both fine here (tilt filter governs)
    if delta == 0:
        return True

    # rotating: exactly one step cw/ccw, and tilt must not change in the same transition
    return delta in (1, n_steps - 1) and start["tilt"] == end["tilt"]


def make_trajectory(
    X: pd.DataFrame,
    y: pd.DataFrame,
    length: int,
    p_rotate: float = 0.25,
    p_tilt: float = 0.25,
    seed=None,
    start_angle: int | None = 0,
) -> pd.DataFrame:
    """Random walk of legal transitions, starting at ground and start_angle
    (None = uniform random, for state-space coverage). Each step rotates one
    step (cw/ccw uniformly) with probability p_rotate, makes a tilt move with
    probability p_tilt (from ground: uniform over the 4 directions; tilted:
    back to ground), and stays put with the remainder."""
    assert p_rotate + p_tilt <= 1
    rng = np.random.default_rng(seed)
    sweep = pd.concat((X, y), axis=1)
    n_steps = int(sweep["angle_idx"].max()) + 1
    lookup = sweep.set_index(["tilt", "angle_idx"], drop=False)

    tilt = "ground"
    angle = int(rng.integers(n_steps)) if start_angle is None else start_angle
    rows = []
    for _ in range(length):
        rows.append(lookup.loc[(tilt, angle)])
        u = rng.random()
        if u < p_rotate:
            angle = (angle + rng.choice((-1, 1))) % n_steps
        elif u < p_rotate + p_tilt:
            if tilt == "ground":
                tilt = rng.choice(("south", "north", "east", "west"))
            else:
                tilt = "ground"

    return pd.DataFrame(rows).reset_index(drop=True)


class TrajectoryDataset(Dataset):
    """On-the-fly trajectories. Item i is fully determined by (seed, i), so it
    is reproducible across runs, shuffling, and dataloader workers."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_trajectories: int,
        seq_len: int,
        p_rotate: float = 0.25,
        p_tilt: float = 0.25,
        seed: int = 0,
    ):
        self.X, self.y = X, y
        self.n_trajectories, self.seq_len = n_trajectories, seq_len
        self.p_rotate, self.p_tilt, self.seed = p_rotate, p_tilt, seed

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, i):
        traj = make_trajectory(
            self.X,
            self.y,
            self.seq_len,
            self.p_rotate,
            self.p_tilt,
            seed=[self.seed, i],  # independent stream per item
            start_angle=None,  # random start so short sequences still cover all angles
        )
        features = torch.tensor(traj[["Bx", "By", "Bz"]].to_numpy(np.float32))
        target = torch.tensor(traj[["tilt_idx", "angle_idx"]].to_numpy(np.int64))
        return features, target


def main():
    mk = lambda tilt, idx: pd.Series({"tilt": tilt, "angle_idx": idx})
    assert filter_legal_rotation((mk("ground", 0), mk("ground", 1)), 4)  # one step cw
    assert filter_legal_rotation(
        (mk("ground", 0), mk("ground", 3)), 4
    )  # one step ccw (wrap)
    assert not filter_legal_rotation((mk("ground", 0), mk("ground", 2)), 4)  # two steps
    assert not filter_legal_rotation(
        (mk("ground", 0), mk("north", 1)), 4
    )  # tilt+rotation combo

    X, y = make_dataset(
        magnetizations=magnetization_values(),
        seed=0,
        n_steps=24,
    )

    traj = make_trajectory(X, y, length=50, p_rotate=0.5, p_tilt=0.2, seed=0)
    steps = [r for _, r in traj.iterrows()]
    assert all(
        filter_legal_tilt((a, b)) and filter_legal_rotation((a, b), 4)
        for a, b in zip(steps, steps[1:])
    )
    print(traj[["tilt", "angle"]].head(25).to_string(), "\n")

    ds = TrajectoryDataset(
        X, y, n_trajectories=32, seq_len=16, p_rotate=0.4, p_tilt=0.3
    )
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    Xb, yb = next(iter(loader))
    assert Xb.shape == (8, 16, 3) and Xb.dtype == torch.float32
    assert yb.shape == (8, 16, 2) and yb.dtype == torch.int64

    ds2 = TrajectoryDataset(
        X, y, n_trajectories=32, seq_len=16, p_rotate=0.4, p_tilt=0.3
    )
    assert torch.equal(ds[3][0], ds2[3][0]) and torch.equal(ds[3][1], ds2[3][1])
    print("batch X:", tuple(Xb.shape), "y:", tuple(yb.shape))

    # pairs = make_pairs(X, y)

    # random_idxs = np.random.randint(0, len(pairs), size=(10,))

    # for p in random_idxs:
    #     start, end = pairs[p]
    #     print(f"start: tilt={start['tilt']} angle={start['angle']}")
    #     print(f"end:   tilt={end['tilt']} angle={end['angle']}")
    #     print([k for k, v in start.items()])
    #     print("\n")


# dataset = BackMappingDataset(seed=0, n_steps=24, n_sims=1)

# loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

# for batch in loader:
#     print(batch[0], batch[1])


if __name__ == "__main__":
    main()
