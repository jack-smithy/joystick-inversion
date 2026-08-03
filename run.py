from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from joystick import make_dataset
from parameters import calibration_values, magnetization_values
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 1
LABELS = ["south", "north", "west", "east", "ground"]


def main() -> None:
    ### this part is fixed - do not touch ###################
    print("Creating data")
    X, y = make_dataset(
        calibration=calibration_values(),
        magnetizations=magnetization_values(),
        n_simulations=50,
        seed=2,
    )
    ########################################################

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.75, random_state=SEED
    )

    y_train_tilt, y_val_tilt = y_train["tilt"], y_val["tilt"]

    param_grid = {
        "hidden_layer_sizes": [(16, 32, 32), (32, 64), (64,)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],  # L2 regularization
        "learning_rate_init": [1e-3, 1e-2],
    }
    search = GridSearchCV(
        MLPClassifier(solver="adam", random_state=SEED),
        param_grid,
        cv=5,
        n_jobs=-1,
    )
    search = search.fit(X_train, y_train_tilt)
    print(f"best params = {search.best_params_}")

    model = search.best_estimator_
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    acc_train = accuracy_score(y_true=y_train_tilt, y_pred=y_pred_train)  # type: ignore
    acc_val = accuracy_score(y_true=y_val_tilt, y_pred=y_pred_val)  # type: ignore

    print(f"train accuracy = {acc_train:.3f}, val accuracy = {acc_val:.3f}")


if __name__ == "__main__":
    main()
