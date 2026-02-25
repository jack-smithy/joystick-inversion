import magpylib as magpy
import numpy as np
import pandas as pd

from joystick_inversion.parameters import (
    magnetization_factory,
    parameter_factory,
    calibration_values,
)


def direction_from_index(index, M):
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


def setup_magnets(parameters, magnetizations) -> magpy.Collection:
    """
    Build the magnets at positions/with orientations according to the joystick design optimization
    1) Place magnet with magnetization M at optimum design position
    2) Rotate magnet in xy-plane by angle obtained from optimum design procedure
    """
    w = parameters[26]

    # Magnet 1:
    idx1 = int(np.clip(round(parameters[18]), 0, 5))
    pol1 = direction_from_index(idx1, magnetizations[0])
    cub1 = magpy.magnet.Cuboid(
        position=(parameters[4], 0, parameters[0]),
        dimension=(w, w, w),
        polarization=pol1,
    )
    cub1.rotate_from_angax(
        angle=parameters[14],
        axis="z",
        anchor=(parameters[4], 0, parameters[0]),
    )

    # Magnet 2:
    idx2 = int(np.clip(round(parameters[19]), 0, 5))
    pol2 = direction_from_index(idx2, magnetizations[1])
    cub2 = magpy.magnet.Cuboid(
        position=(parameters[5], 0, parameters[1]),
        dimension=(w, w, w),
        polarization=pol2,
    )
    cub2.rotate_from_angax(
        angle=parameters[15],
        axis="z",
        anchor=(parameters[5], 0, parameters[1]),
    )

    # Magnet 3:
    cub3 = magpy.magnet.Cuboid(
        position=(0, parameters[6], parameters[2]),
        dimension=(w, w, w),
        polarization=(0, 0, -magnetizations[2]),
    )
    cub3.rotate_from_angax(
        angle=parameters[16],
        axis="z",
        anchor=(0, parameters[6], parameters[2]),
    )

    # Magnet 4:
    cub4 = magpy.magnet.Cuboid(
        position=(0, parameters[7], parameters[3]),
        dimension=(w, w, w),
        polarization=(0, 0, magnetizations[3]),
    )
    cub4.rotate_from_angax(
        angle=parameters[17],
        axis="z",
        anchor=(0, parameters[7], parameters[3]),
    )

    return magpy.Collection(cub1, cub2, cub3, cub4)


def setup_sensor(parameters: np.ndarray, two_sensors: bool = False) -> magpy.Sensor:
    """
    Build sensors 1 and 2 at positions from optimum design procedure.
    Attention: Infineon 3-D sensors are left-handed.
    Here right-handed sensors are chosen nonetheless, but when evaluating the field at sensor position,
    the sign of the field z-component is flipped: B1[2] = -B1[2]. This approach is still valid
    when rotating the sensor in the xy-plane first, but loses correctness when small angle errors
    (tolerances) out of the xy-plane are considered.

    (Jack): I think so far we only have data from one sensor. guessing its sensor 1
    """
    sensor_1_x = parameters[8]
    sensor_1_y = parameters[9]
    sensor_1_z = parameters[10]

    # sensor_2_x = parameters[11]
    # sensor_2_y = parameters[12]
    # sensor_2_z = parameters[13]

    sensor_1 = magpy.Sensor(
        position=(sensor_1_x, sensor_1_y, sensor_1_z),
        handedness="right",
    )

    sensor_1.rotate_from_angax(
        angle=-45,
        axis="z",
        anchor=(sensor_1_x, sensor_1_y, sensor_1_z),
        start=0,  # type: ignore
    )

    # sensors = [sensor_1]

    # if two_sensors:
    #     sensor_2 = magpy.Sensor(
    #         position=(sensor_2_x, sensor_2_y, sensor_2_z),
    #         handedness="right",
    #     )

    #     sensor_2.rotate_from_angax(
    #         angle=-45,
    #         axis="z",
    #         anchor=(sensor_2_x, sensor_2_y, sensor_2_z),
    #         start=0,  # type: ignore
    #     )

    #     sensors.append(sensor_2)

    return sensor_1


def make_sensor_readings(
    magnets: magpy.Collection,
    sensors: magpy.Sensor,
    parameters: np.ndarray,
    n_steps: int = 24,
):
    angles = np.linspace(start=0, stop=360, num=n_steps, endpoint=False)
    # repeat x6 for ground + push + (tilt x 4)
    angles = np.tile(angles, 6)

    # spinny spinny
    magnets.rotate_from_angax(angle=angles, axis="z", anchor=(0, 0, 0), start=0)  # type: ignore

    # 0. south
    magnets.rotate_from_angax(
        angle=parameters[22],
        axis="x",
        anchor=(0, 0, 0),
        start=n_steps * 0,  # type: ignore
    )

    # 1. north
    magnets.rotate_from_angax(
        angle=-parameters[23] * 2,
        axis="x",
        anchor=(0, 0, 0),
        start=n_steps * 1,  # type: ignore
    )

    # 2. east (through ground state)
    magnets.rotate_from_angax(
        angle=parameters[24],
        axis="x",
        anchor=(0, 0, 0),
        start=n_steps * 2,  # type: ignore
    ).rotate_from_angax(
        angle=parameters[24],
        axis="y",
        anchor=(0, 0, 0),
        start=n_steps * 2,  # type: ignore
    )

    # 3. west
    magnets.rotate_from_angax(
        angle=-parameters[25] * 2,
        axis="y",
        anchor=(0, 0, 0),
        start=n_steps * 3,  # type: ignore
    )

    # 4. ground
    magnets.rotate_from_angax(
        angle=parameters[25],
        axis="y",
        anchor=(0, 0, 0),
        start=n_steps * 4,  # type: ignore
    )

    # 5. push
    magnets.move(
        displacement=(0, 0, -2e-3),
        start=n_steps * 5,  # type: ignore
    )

    B = magnets.getB(sensors)

    return B


def make_positions(n_simulations: int, n_steps: int = 24):
    angles = np.linspace(start=0, stop=360, num=n_steps, endpoint=False)
    angles = np.tile(angles, 6)

    states = np.ones((n_steps * 6,))
    for i in range(6):
        states[n_steps * i : n_steps * (i + 1)] *= i

    if n_simulations > 1:
        angles = np.tile(angles, n_simulations)
        states = np.tile(states, n_simulations)

    return states, angles


def make_dataset(n_simulations):
    calibration = calibration_values()
    parameters = parameter_factory(calibration=calibration)
    magnetizations = magnetization_factory()

    sensor = setup_sensor(parameters=parameters)

    Bxi, Byi, Bzi = [], [], []
    for _ in range(n_simulations):
        magnets = setup_magnets(
            parameters=parameters,
            magnetizations=magnetizations,
        )

        bx, by, bz = make_sensor_readings(
            magnets=magnets,
            sensors=sensor,
            parameters=parameters,
        ).T

        Bxi.append(bx)
        Byi.append(by)
        Bzi.append(bz)

    Bx = np.concat(Bxi)
    By = np.concat(Byi)
    Bz = np.concat(Bzi)

    states, angles = make_positions(n_simulations=n_simulations)

    dataset = {
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
        "tilt": states,
        "angle": angles,
    }

    return pd.DataFrame(dataset)
