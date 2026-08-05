"""
The joystick state space, shared by every approach in the benchmark.

A state is one of `N_STATES` combinations of tilt and rotation, flattened to a
single index so it can be a classification target and an HMM state at once:

    state = tilt_idx * N_ANGLES + angle_idx

Row order matches `joystick.make_dataset`, so a state index also indexes
straight into a unit's 120-entry forward map. Deliberately free of heavy
imports - the physics, torch and pandas layers all sit on top of this.
"""

import numpy as np

N_ANGLES, N_TILTS = 24, 5
N_STATES = N_TILTS * N_ANGLES
GROUND = 4  # joystick.state_dict = {0: south, 1: north, 2: east, 3: west, 4: ground}


def split_state(state):
    """state index -> (tilt, angle)."""
    return state // N_ANGLES, state % N_ANGLES


def join_state(tilt, angle):
    """(tilt, angle) -> state index."""
    return tilt * N_ANGLES + angle


def score(pred, target) -> dict:
    """Accuracy (%) of predicted state indices, split by what went wrong.

    `joint` is the one that matters - tilt and angle both right. The separate
    columns are worth keeping because they are so lopsided: tilt is much harder
    than angle, and a headline number alone hides that.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    tilt_ok = pred // N_ANGLES == target // N_ANGLES
    angle_ok = pred % N_ANGLES == target % N_ANGLES
    return {
        "tilt": float(tilt_ok.mean()) * 100,
        "angle": float(angle_ok.mean()) * 100,
        "joint": float((tilt_ok & angle_ok).mean()) * 100,
    }


def demo():
    assert join_state(*split_state(np.arange(N_STATES))).tolist() == list(
        range(N_STATES)
    )
    perfect = score(np.arange(N_STATES), np.arange(N_STATES))
    assert perfect == {"tilt": 100.0, "angle": 100.0, "joint": 100.0}
    # right angle, wrong tilt: angle scores, joint does not
    off_tilt = score(np.array([join_state(0, 3)]), np.array([join_state(1, 3)]))
    assert off_tilt == {"tilt": 0.0, "angle": 100.0, "joint": 0.0}
    print("statespace: ok")


if __name__ == "__main__":
    demo()
