import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from constants import TILT_NAMES
from joystick import make_transitions_datasets
from plot import plot_evaluation
from train import (
    Predictions,
    angle_index,
    angular_error,
    bin_error,
    make_dataloader,
)
from utils import timed

SEED = 1
N_STEPS = 24
N_UNITS_TRAIN = 8
N_UNITS_TEST = 4
HIDDEN = 128
# 40 underfits badly (~10deg, train err == test err); past ~150 the gap opens up and
# the extra capacity goes into memorizing the training units
EPOCHS = 150

# sensor noise std in mT; fields at the sensor are a few mT peak-to-peak
NOISE_LEVELS = (0.0, 0.01, 0.05, 0.1, 0.5)


def mlp(n_out: int) -> nn.Sequential:
    """
    (batch, 2, 3) field trajectory -> (batch, n_out)

    A GRU over the same input scored worse on tilt (0.92 vs 0.97) for no gain on angle:
    the state is near enough determined by the current reading, with the one predecessor
    resolving what is left, so there is nothing for recurrence to integrate.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(2 * 3, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, n_out),
    )


def angle_target(y: torch.Tensor) -> torch.Tensor:
    """
    angle index -> (sin(angle), cos(angle))
    """
    theta = y[:, 0] * (2 * torch.pi / N_STEPS)
    return torch.stack((theta.sin(), theta.cos()), dim=1)


def train(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    target: str,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for X, y in loader:
            y = y[:, 1] if target == "tilt" else angle_target(y)

            loss = loss_fn(model(X), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def evaluate(
    model_tilt: nn.Module,
    model_angle: nn.Module,
    loader: DataLoader,
) -> Predictions:
    model_tilt.eval()
    model_angle.eval()

    tilt_true, tilt_pred, idx_true, angle_true, angle_pred = [], [], [], [], []

    for X, y in loader:
        tilt_true.append(y[:, 1].numpy())
        tilt_pred.append(model_tilt(X).argmax(dim=1).numpy())

        idx_true.append(y[:, 0].numpy())
        angle_true.append(angle_target(y).numpy())
        angle_pred.append(model_angle(X).numpy())

    angle_true, angle_pred = np.concatenate(angle_true), np.concatenate(angle_pred)

    return Predictions(
        tilt_true=np.concatenate(tilt_true),
        tilt_pred=np.concatenate(tilt_pred),
        angle_idx_true=np.concatenate(idx_true),
        # snap the continuous prediction to the nearest rest position
        angle_idx_pred=angle_index(angle_pred, n_steps=N_STEPS),
        angle_error=angular_error(y_true=angle_true, y_pred=angle_pred),
    )


def metrics(predictions: Predictions) -> dict[str, float]:
    """
    Headline scores, one row's worth.

    `angle acc` is how often the rotation state is read correctly; `within 1` allows a
    single-step miss, which separates near-misses from real confusions. The error
    median and p95 say what the mean cannot.
    """
    delta = bin_error(predictions, n_steps=N_STEPS)

    return {
        "tilt acc": float((predictions.tilt_true == predictions.tilt_pred).mean()),
        "angle acc": float((delta == 0).mean()),
        "within 1": float((delta <= 1).mean()),
        "err mean": float(predictions.angle_error.mean()),
        "err med": float(np.median(predictions.angle_error)),
        "err p95": float(np.percentile(predictions.angle_error, 95)),
    }


def report(predictions: Predictions) -> None:
    """
    Per-class detail, for when the headline numbers average something interesting away.
    """
    print(
        classification_report(
            y_true=predictions.tilt_true,
            y_pred=predictions.tilt_pred,
            target_names=TILT_NAMES,
            zero_division=0,
        )
    )

    delta = bin_error(predictions, n_steps=N_STEPS)

    print(f"{'tilt state':<12}{'angle acc':>10}{'err med':>10}{'worst miss':>12}")
    for tilt, name in enumerate(TILT_NAMES):
        rows = predictions.tilt_true == tilt
        print(
            f"{name:<12}"
            f"{(delta[rows] == 0).mean():>10.3f}"
            f"{np.median(predictions.angle_error[rows]):>10.2f}"
            f"{delta[rows].max():>9d} steps"
        )


@timed()
def main() -> None:
    print("Simulating joystick units")
    df_train = make_transitions_datasets(
        n_repeats=N_UNITS_TRAIN,
        seed=2,
        n_steps=N_STEPS,
    )
    # different seeds -> unseen units, so the test set is a generalization check
    df_test = make_transitions_datasets(
        n_repeats=N_UNITS_TEST,
        seed=100,
        n_steps=N_STEPS,
    )

    noise = 0.1
    torch.manual_seed(SEED)

    # train and test at the same noise level
    loader_train = make_dataloader(df_train, batch_size=64, noise=noise, seed=SEED)
    loader_test = make_dataloader(
        df_test,
        batch_size=256,
        shuffle=False,
        noise=noise,
        seed=SEED + 1,
    )

    model_tilt = train(
        model=mlp(len(TILT_NAMES)),
        loader=loader_train,
        loss_fn=nn.CrossEntropyLoss(),
        target="tilt",
    )
    model_angle = train(
        model=mlp(2),
        loader=loader_train,
        loss_fn=nn.MSELoss(),
        target="angle",
    )

    predictions = evaluate(model_tilt, model_angle, loader_test)
    scores = metrics(predictions)

    print(f"\n{'noise/mT':>10}" + "".join(f"{name:>11}" for name in scores))
    print(f"{noise:>10.2f}" + "".join(f"{value:>11.3f}" for value in scores.values()))
    print()

    report(predictions)

    for path in plot_evaluation(predictions, transitions=df_test, n_steps=N_STEPS):
        print(f"evaluation plot -> {path}")


if __name__ == "__main__":
    main()
