from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path

from joystick import make_dataset
from parameters import calibration_values, magnetization_values
from train import train_angle, train_tilt, val_tilt, val_angle
from utils import load_measurement_data

SEED = 1


def main() -> None:
    print("Creating training data")
    df_train = make_dataset(
        calibration=calibration_values(),
        magnetizations=magnetization_values(),
        n_simulations=10,
        seed=2,
    )

    print("Creating validation data")
    df_val = make_dataset(
        calibration=calibration_values(),
        magnetizations=magnetization_values(),
        n_simulations=4,
        seed=42,  # seed must be different to train, otherwise data will be identical
    )

    model_tilt = LGBMClassifier(random_state=SEED)
    model_tilt = train_tilt(
        df_train=df_train,
        df_val=df_val,
        model=model_tilt,
    )

    model_angle = LGBMRegressor(random_state=SEED)
    model_angle = MultiOutputRegressor(model_angle)
    model_angle = train_angle(
        df_train=df_train,
        df_val=df_val,
        model=model_angle,
    )

    path = Path("data/sensor1")
    df_meas = load_measurement_data(path=path)
    val_tilt(df=df_meas, model=model_tilt)
    val_angle(df=df_meas, model=model_angle)


if __name__ == "__main__":
    main()
