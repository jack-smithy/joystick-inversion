import magpylib as magpy
import numpy as np
import pandas as pd

from constants import DIRECTION_MAP, FIELD_COLUMNS
from parameters import N_MAGNETS, Parameters, parameter_factory
from utils import timed

GROUND = DIRECTION_MAP["zero"]


def direction_from_index(index: int, M: float) -> tuple:
    """
    Index from optimization procedure for joystick design mapped to vector representing
    the magnetization
    """
    if index == 0:
        return (M, 0, 0)
    elif index == 1:
        return (-M, 0, 0)
    elif index == 2:
        return (0, M, 0)
    elif index == 3:
        return (0, -M, 0)
    elif index == 4:
        return (0, 0, M)
    elif index == 5:
        return (0, 0, -M)
    else:
        raise ValueError("Index must be in the range [0, 5] for 6 directions")


def setup_magnets(parameters: Parameters) -> magpy.Collection:
    """
    Build the magnets at positions/with orientations according to the joystick design optimization
    1) Place magnet with magnetization M at optimum design position
    2) Rotate magnet by phi about z, then by theta about y, anchored at the magnet

    theta is nominally zero and only ever carries a small tolerance, so its ordering
    relative to phi is a second-order effect.
    """
    w = parameters.magnet_size

    magnets = []
    for i in range(N_MAGNETS):
        position = parameters.magnet_position[i]

        magnet = magpy.magnet.Cuboid(
            position=position,
            dimension=(w, w, w),
            polarization=direction_from_index(
                index=int(parameters.magnet_direction[i]),
                M=parameters.magnet_polarization[i],
            ),
        )
        magnet.rotate_from_angax(
            angle=parameters.magnet_phi[i],
            axis="z",
            anchor=position,
        )
        magnet.rotate_from_angax(
            angle=parameters.magnet_theta[i],
            axis="y",
            anchor=position,
        )

        magnets.append(magnet)

    return magpy.Collection(*magnets)


def setup_sensors(parameters: Parameters) -> magpy.Collection:
    """
    Build both sensors at the positions from the optimum design procedure, each rotated
    by the -45deg mounting angle plus its own orientation tolerances.

    Sensor 2 is not optional. Sensor 1 alone cannot separate every state: tilting south at
    one rotation produces almost exactly the field of tilting east at another, and the two
    sit closer together than either wanders across manufacturing tolerance. A second
    sensor at a different position sees a different projection of the same array and
    breaks every such pair. See `FIELD_COLUMNS`.

    Attention: Infineon 3-D sensors are left-handed. Right-handed sensors are chosen here
    instead, which would need the field z-component flipped (B[2] = -B[2]) to match the
    real part. That flip is not applied anywhere yet.
    """
    sensors = []
    for position in parameters.sensor_position:
        sensor = magpy.Sensor(position=position, handedness="right")

        sensor.rotate_from_angax(
            angle=-45 + parameters.sensor_phi,
            axis="z",
            anchor=position,
            start=0,  # type: ignore
        )
        sensor.rotate_from_angax(
            angle=parameters.sensor_theta,
            axis="y",
            anchor=position,
            start=0,  # type: ignore
        )

        sensors.append(sensor)

    return magpy.Collection(*sensors)


def make_sensor_readings(
    magnets: magpy.Collection,
    sensors: magpy.Collection,
    parameters: Parameters,
    n_steps: int = 24,
):
    """
    Sweep the joystick through every state and read the field at both sensors.

    Returns (5 * n_steps, n_sensors, 3) in tesla.

    The path is 5 blocks of `n_steps` rotations: south, north, east, west, ground.
    Each block's tilt is applied to the path from its start index onwards, so the
    rotations accumulate down the path and every delta below is the difference between
    two blocks, not an absolute tilt. Deltas about the same axis add (magpylib rotates
    in the global frame), which is what makes the arithmetic work out.
    """
    s, n, e, w = parameters.tilt_angle

    angles = np.linspace(start=0, stop=360, num=n_steps, endpoint=False)

    # repeat x5 for ground  + (tilt x 4)
    angles = np.tile(angles, 5)

    # spinny spinny
    magnets.rotate_from_angax(angle=angles, axis="z", anchor=(0, 0, 0), start=0)

    # (block, axis, delta) -> net tilt reached by that block
    tilts = [
        (0, "x", s),  # 0. south   -> (+s,  0)
        (1, "x", -(s + n)),  # 1. north   -> (-n,  0)
        (2, "x", n),  # 2. east    -> ( 0, +e), cancelling the x tilt on the way
        (2, "y", e),  #    |
        (3, "y", -(e + w)),  # 3. west    -> ( 0, -w)
        (4, "y", w),  # 4. ground  -> ( 0,  0)
    ]
    for block, axis, delta in tilts:
        magnets.rotate_from_angax(
            angle=delta,
            axis=axis,
            anchor=(0, 0, 0),
            start=n_steps * block,  # type: ignore
        )

    B = magnets.getB(sensors)

    return B


def make_positions(
    n_steps: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce matching input arguments for the simulation. Be careful
    if you edit this as it needs to match `make_sensor_readings`.

    Args:
        n_steps (int, optional): Rotation discretization. Defaults to 24.

    Returns:
        tuple[np.ndarray, np.ndarray]: Corresponding tilts and angles.
    """
    angles = np.linspace(start=0, stop=360, num=n_steps, endpoint=False)
    angles = np.tile(angles, 5)

    states = np.ones((n_steps * 5,))
    for i in range(5):
        states[n_steps * i : n_steps * (i + 1)] *= i

    return states, angles


@timed()
def make_dataset(
    n_steps: int = 24,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Create a full training/ validaton data set.

    Args:
        n_steps (int, optional): Rotation discretization. Defaults to 24.
        seed (int | None, optional): Random seed for tolerances. Defaults to None,
            which gives the nominal joystick with no tolerances applied.

    Returns:
        pd.DataFrame: Full simulation dataset
    """

    # init system parameters
    generator = np.random.default_rng(seed=seed) if seed is not None else None
    params = parameter_factory(generator=generator)

    # init joystick simulation
    sensors = setup_sensors(parameters=params)
    magnets = setup_magnets(parameters=params)

    # simulate whole sweeep
    B = make_sensor_readings(
        magnets=magnets,
        sensors=sensors,
        parameters=params,
        n_steps=n_steps,
    )

    # get corresponding input states
    # (the bit we are trying to predict)
    states, angles = make_positions(n_steps=n_steps)

    # (n, n_sensors, 3) -> one column per sensor axis, in FIELD_COLUMNS order
    dataset = dict(zip(FIELD_COLUMNS, B.reshape(len(B), -1).T))
    dataset["tilt"] = states
    dataset["angle"] = angles

    return pd.DataFrame(dataset)


def legal_moves(
    tilt: int,
    angle_idx: int,
    n_steps: int = 24,
) -> dict[str, tuple[int, int]]:
    """
    The joystick's action set, and where each action lands.

    From any state it can rotate one step clockwise or anticlockwise, or do nothing.
    From the ground state it can tilt in one of 4 directions; from a tilted state it
    can return to ground. Rotation wraps; tilting never changes the angle.

    Args:
        tilt (int): Current tilt state.
        angle_idx (int): Current rotation step.
        n_steps (int, optional): Rotation discretization. Defaults to 24.

    Returns:
        dict[str, tuple[int, int]]: action -> (tilt_end, angle_idx_end).
    """
    moves = {
        "noop": (tilt, angle_idx),
        "cw": (tilt, (angle_idx + 1) % n_steps),
        "ccw": (tilt, (angle_idx - 1) % n_steps),
    }

    if tilt == GROUND:
        moves |= {
            f"tilt_{name}": (end, angle_idx)
            for name, end in DIRECTION_MAP.items()
            if end != GROUND
        }
    else:
        moves["ground"] = (GROUND, angle_idx)

    return moves


def make_transitions(df: pd.DataFrame, n_steps: int = 24) -> pd.DataFrame:
    """
    Every legal single-step move of the joystick, with the sensor reading at both ends.

    Args:
        df (pd.DataFrame): A single sweep from `make_dataset` (one row per state).
        n_steps (int, optional): Rotation discretization. Defaults to 24.

    Returns:
        pd.DataFrame: One row per legal transition.
    """
    step = 360 / n_steps

    out = pd.DataFrame(
        [
            (tilt, i, *end, action)
            for tilt in range(len(DIRECTION_MAP))
            for i in range(n_steps)
            for action, end in legal_moves(tilt, i, n_steps).items()
        ],
        columns=[
            "tilt_start",
            "angle_idx_start",
            "tilt_end",
            "angle_idx_end",
            "transition",
        ],
    )

    # look up B by (tilt, angle index) at each end of the transition
    lut = df[FIELD_COLUMNS].set_axis(
        pd.MultiIndex.from_arrays(
            [df["tilt"].astype(int), (df["angle"] / step).round().astype(int)]
        )
    )
    assert not lut.index.duplicated().any(), (
        "df must be a single sweep with one row per state"
    )

    out = out.join(lut.add_suffix("_start"), on=["tilt_start", "angle_idx_start"])
    out = out.join(lut.add_suffix("_end"), on=["tilt_end", "angle_idx_end"])

    out["angle_start"] = out["angle_idx_start"] * step
    out["angle_end"] = out["angle_idx_end"] * step

    return out[
        [f"{column}_start" for column in FIELD_COLUMNS]
        + ["tilt_start", "angle_start", "angle_idx_start"]
        + [f"{column}_end" for column in FIELD_COLUMNS]
        + ["tilt_end", "angle_end", "angle_idx_end", "transition"]
    ]


def make_datasets(
    n_repeats: int,
    seed: int,
    n_steps: int = 24,
):
    dfs = []
    for i in range(n_repeats):
        data = make_dataset(
            n_steps=n_steps,
            seed=seed + i,
        )
        dfs.append(data)
    return pd.concat(dfs)


def make_transitions_datasets(
    n_repeats: int,
    seed: int,
    n_steps: int = 24,
) -> pd.DataFrame:
    """
    `make_datasets` for transitions: one joystick unit per repeat, each with its own
    tolerances from `seed + i`, so the same transition appears once per unit with a
    different pair of field readings.

    Args:
        n_repeats (int): Number of joystick units to simulate.
        seed (int): Base random seed; unit i uses `seed + i`.
        n_steps (int, optional): Rotation discretization. Defaults to 24.

    Returns:
        pd.DataFrame: `make_transitions` output stacked over units, plus a `unit` column.
    """
    dfs = []
    for i in range(n_repeats):
        data = make_dataset(n_steps=n_steps, seed=seed + i)
        dfs.append(make_transitions(df=data, n_steps=n_steps).assign(unit=i))
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    n = 24

    # Every block of the sweep must reach the tilt it is labelled with, whatever the
    # tolerances. The tilts accumulate down the rotation path, so this is the check that
    # the deltas in `make_sensor_readings` still cancel correctly.
    for seed in (None, 0, 1, 2, 3):
        params = parameter_factory(
            generator=np.random.default_rng(seed) if seed is not None else None
        )
        magnets = setup_magnets(parameters=params)
        B = make_sensor_readings(
            magnets=magnets,
            sensors=setup_sensors(parameters=params),
            parameters=params,
            n_steps=n,
        )
        assert B.shape == (5 * n, 2, 3), B.shape

        south, north, east, west = params.tilt_angle

        # The assembly carries its own design rotations and the z-spin, so read the tilt
        # as each block's orientation relative to the ground block at the same spin
        # angle: that leaves exactly the tilt rotation and nothing else.
        orientation = magnets[0].orientation
        rest = orientation[GROUND * n].inv()

        for block, axis, angle in (
            (DIRECTION_MAP["south"], "x", south),
            (DIRECTION_MAP["north"], "x", -north),
            (DIRECTION_MAP["east"], "y", east),
            (DIRECTION_MAP["west"], "y", -west),
            (GROUND, "x", 0.0),
        ):
            got = (orientation[block * n] * rest).as_matrix()
            want = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
            assert np.allclose(got, want, atol=1e-9), (seed, block, angle)

        # loose units guard, not a physics claim: catches a polarization or position
        # wired in with the wrong scale
        peak = np.abs(B).max()
        assert 1e-3 < peak < 1.0, f"fields are not in tesla any more: {peak}"

    print(f"tilt angles reached, s/n/e/w = {np.round(params.tilt_angle, 3)}")
    print(f"peak |B| = {peak * 1e3:.1f} mT")

    t = make_transitions(make_dataset(), n_steps=n)
    print(t.head(10))

    # # 5*n states x (noop, cw, ccw), + 4 tilts out of ground, + 1 return to ground
    # assert len(t) == 5 * n * 3 + n * 4 + 4 * n, len(t)
    # assert not t.isna().any().any(), "unmatched (tilt, angle) in the B lookup"

    # counts = t.groupby(["tilt_start", "angle_idx_start"]).size()
    # from_ground = counts.index.get_level_values(0) == GROUND
    # assert (counts[from_ground] == 7).all()  # + 4 ways to tilt
    # assert (counts[~from_ground] == 4).all()  # + 1 way back to ground

    # same_angle = t["angle_idx_start"] == t["angle_idx_end"]
    # same_tilt = t["tilt_start"] == t["tilt_end"]

    # for name, delta in (("cw", 1), ("ccw", -1)):
    #     rot = t[t["transition"] == name]
    #     assert same_tilt[rot.index].all()
    #     assert (rot["angle_idx_end"] == (rot["angle_idx_start"] + delta) % n).all()

    # tilting = t[t["transition"].str.startswith("tilt_")]
    # assert (tilting["tilt_start"] == GROUND).all()
    # assert (tilting["tilt_end"] != GROUND).all()
    # assert same_angle[tilting.index].all()

    # grounding = t[t["transition"] == "ground"]
    # assert (grounding["tilt_start"] != GROUND).all()
    # assert (grounding["tilt_end"] == GROUND).all()
    # assert same_angle[grounding.index].all()

    # noop = t[t["transition"] == "noop"]
    # assert same_tilt[noop.index].all() and same_angle[noop.index].all()

    # assert (t["angle_start"] == t["angle_idx_start"] * 360 / n).all()

    # print(t.head())
    # print(t["transition"].value_counts())
    # print("ok")

    m = make_transitions_datasets(n_repeats=3, seed=0, n_steps=n)
    assert len(m) == 3 * len(t) and m["unit"].nunique() == 3

    graph = ["tilt_start", "angle_idx_start", "tilt_end", "angle_idx_end", "transition"]
    unit_0, unit_1 = (g.reset_index(drop=True) for _, g in list(m.groupby("unit"))[:2])

    # same state graph every unit, but each unit reads a different field
    assert unit_0[graph].equals(unit_1[graph])
    assert not np.allclose(unit_0["Bx1_start"], unit_1["Bx1_start"])

    # both sensors are wired through, and they see genuinely different fields --
    # if they did not, sensor 2 would add nothing and the degenerate pairs would remain
    assert not np.allclose(unit_0["Bx1_start"], unit_0["Bx2_start"])

    print(m.shape)
    print("ok")
