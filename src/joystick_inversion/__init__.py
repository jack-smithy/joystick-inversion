from joystick_inversion.train import train_tilt, train_angle
from joystick_inversion.system import make_dataset
from joystick_inversion.parameters import calibration_values, magnetization_values
from lightgbm import LGBMClassifier


def main() -> None:
    df = make_dataset(
        calibration=calibration_values(),
        magnetizations=magnetization_values(),
        n_simulations=10,
        seed=1,
    )

    tilt_model = train_tilt(data=df, model=LGBMClassifier(random_state=42))

    angle_model = train_angle(data=df, model=LGBMClassifier(random_state=42))
