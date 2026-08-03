from joystick import make_dataset, setup_magnets, setup_sensor
from parameters import calibration_values, magnetization_values, parameter_factory
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import magpylib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from magpylib import Collection
from dataclasses import dataclass

sensor = setup_sensor(parameters=parameter_factory())
magnets = setup_magnets(
    parameters=parameter_factory(),
    magnetizations=magnetization_values(),
)


@dataclass
class State:
    angle: int
    tilt: int

    def rotate_cw(self):
        self.angle = (self.angle + 1) % 24

    def rotate_ccw(self):
        self.angle = (self.angle - 1) % 24


def make_readings_pair(
    sensor, magnets: Collection, init_state: State, final_state: State
):
    # setup magnets into initial state
    rot_angle = init_state.angle * 15
    magnets.rotate_from_angax(rot_angle, axis="z", anchor=(0, 0, 0))

    if init_state.tilt == 0:
        magnets.rotate_from_angax(4, "x", anchor=(0, 0, 0))
    if init_state.tilt == 1:
        magnets.rotate_from_angax(-4, "x", anchor=(0, 0, 0))
    if init_state.tilt == 2:
        magnets.rotate_from_angax(4, "y", anchor=(0, 0, 0))
    if init_state.tilt == 3:
        magnets.rotate_from_angax(-4, "y", anchor=(0, 0, 0))

    return magnets.getB(sensor)


X, y = make_dataset(
    calibration=calibration_values(),
    magnetizations=magnetization_values(),
    n_simulations=1,
    seed=2,
)

_LEGAL_ANGLES = {0: [1, 2, 3, 4], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4]}


class SensorReadout:
    def __init__(self, n_simulations: int = 1, n_steps: int = 24, seed: int = 0):
        self.n_steps = n_steps
        self.X, self.y = make_dataset(
            calibration=calibration_values(),
            magnetizations=magnetization_values(),
            n_simulations=n_simulations,
            n_steps=n_steps,
            seed=seed,
        )

    def get_next_valid_state(self, state: tuple[int, int], seed):
        tilt, angle = state
        next_states = []  # no-op is fine
        next_states.append((tilt, (angle + 1) % self.n_steps))  # rotate cw
        next_states.append((tilt, (angle - 1) % self.n_steps))  # rotate ccw
        for aa in _LEGAL_ANGLES[angle]:
            next_states.append((tilt, aa))
        return next_states


def main():
    sensor = SensorReadout(1, 24, 0)
    state = (0, 0)
    next_state = sensor.get_next_valid_state(state, 0)
    print(next_state)


if __name__ == "__main__":
    main()
