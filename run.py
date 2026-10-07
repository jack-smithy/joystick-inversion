import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from constants import FIELD_COLUMNS, TILT_NAMES
from joystick import make_datasets
from plot import plot_evaluation
from train import (
    Predictions,
    bin_error,
    make_dataloader,
    state_index,
)
from utils import timed
from tqdm import tqdm

SEED = 1
N_STEPS = 24
# one class per (tilt, rotation) pair: 5 * 24 = 120
N_STATES = len(TILT_NAMES) * N_STEPS
# both 3-D sensors, one reading
N_FEATURES = len(FIELD_COLUMNS)
# 512 units scored the same as 128 (0.933 state acc), so more data buys nothing here
N_UNITS_TRAIN = 128
# per-unit accuracy ranges 0.85-0.98, so a small test population is mostly luck
N_UNITS_TEST = 16
# 512 wide / 3 deep scored no better
HIDDEN = 256
# with cosine decay. Constant lr: 50 -> 200 epochs stayed at ~0.917, cosine at 100 gets 0.933
EPOCHS = 100

# Sensor noise std in mT. The signal std across states is 22.5 mT, so 0.5 mT is ~2% of
# signal. The old 0.1 mT was 0.44%, low enough that it perturbed nothing and every model
# architecture scored the same.
NOISE = 0.5


def mlp(n_out: int) -> nn.Sequential:
    """
    (batch, 6) field reading -> (batch, n_out)

    Tilt, not the model, is the ceiling (~0.93 at 0.5 mT). Every miss is a tilt
    state read as `zero` or the reverse. They bunch at four angles 90 degrees apart,
    where tilting moves the field only 1-2.5 mT on a given unit, while units differ
    from each other by 5-8 mT per axis. One reading can't tell which unit it came
    from; the transition model's start reading supplied that, which is how it reached
    0.97 tilt.
    """
    return nn.Sequential(
        nn.Linear(N_FEATURES, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, n_out),
    )


def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    with tqdm(range(epochs)) as bar:
        for _ in bar:
            for X, y in loader:
                loss = loss_fn(model(X), state_index(y, n_steps=N_STEPS))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_description(f"loss={loss.item():.4f}")
            scheduler.step()

    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Predictions:
    model.eval()

    true, pred = [], []

    for X, y in loader:
        true.append(state_index(y, n_steps=N_STEPS).numpy())
        pred.append(model(X).argmax(dim=1).numpy())

    true, pred = np.concatenate(true), np.concatenate(pred)

    # the joint class carries both fields; split it back out so every per-tilt and
    # per-rotation metric below still works
    return Predictions(
        tilt_true=true // N_STEPS,
        tilt_pred=pred // N_STEPS,
        angle_idx_true=true % N_STEPS,
        angle_idx_pred=pred % N_STEPS,
        n_steps=N_STEPS,
    )


def metrics(predictions: Predictions) -> dict[str, float]:
    """
    Headline scores, one row's worth.

    `state acc` is the one that matters: how often all 120 states are read exactly right,
    which is what the model is actually trained on. `tilt acc` and `angle acc` decompose
    it, and are both upper bounds on it. `within 1` allows a single-step rotation miss —
    it sits on top of `angle acc`, so the two being equal means there are no near-misses
    at all, only gross ones. `err mean` is a quantized miss distance (see
    `Predictions.angle_error`), i.e. how far out the misses land, not a precision.

    The median and p95 of that error were dropped: the classifier is exact well over 95%
    of the time, so both read 0.0 at every noise level and said nothing.
    """
    delta = bin_error(predictions, n_steps=N_STEPS)
    tilt_hit = predictions.tilt_true == predictions.tilt_pred

    return {
        "state acc": float((tilt_hit & (delta == 0)).mean()),
        "tilt acc": float(tilt_hit.mean()),
        "angle acc": float((delta == 0).mean()),
        "within 1": float((delta <= 1).mean()),
        "err mean": float(predictions.angle_error.mean()),
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

    print(f"{'tilt state':<12}{'angle acc':>10}{'err mean':>10}{'worst miss':>12}")
    for tilt, name in enumerate(TILT_NAMES):
        rows = predictions.tilt_true == tilt
        print(
            f"{name:<12}"
            f"{(delta[rows] == 0).mean():>10.3f}"
            f"{predictions.angle_error[rows].mean():>10.2f}"
            f"{delta[rows].max():>9d} steps"
        )


@timed()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)

    args = parser.parse_args()

    save_dir = f"results/{args.run_name}"

    print("Simulating joystick units")
    df_train = make_datasets(
        n_repeats=N_UNITS_TRAIN,
        seed=2,
        n_steps=N_STEPS,
        zero_offset=False,
    )
    # different seeds -> unseen units, so the test set is a generalization check.
    # No zero_offset: a per-unit label shift is invisible in a single reading, so
    # angle acc collapses to the share of units whose offset happens to be 0 (~0.375)
    df_test = make_datasets(
        n_repeats=N_UNITS_TEST,
        seed=100,
        n_steps=N_STEPS,
        zero_offset=False,
    )

    torch.manual_seed(SEED)

    # train and test at the same noise level
    loader_train = make_dataloader(
        df_train,
        batch_size=64,
        noise=NOISE,
        seed=SEED,
        n_steps=N_STEPS,
    )

    loader_test = make_dataloader(
        df_test,
        batch_size=256,
        shuffle=False,
        noise=NOISE,
        seed=SEED + 1,
        n_steps=N_STEPS,
    )

    model = train(model=mlp(N_STATES), loader=loader_train)

    predictions = evaluate(model, loader_test)
    scores = metrics(predictions)

    print(f"\n{'noise/mT':>10}" + "".join(f"{name:>11}" for name in scores))
    print(f"{NOISE:>10.2f}" + "".join(f"{value:>11.3f}" for value in scores.values()))
    print()

    report(predictions)

    for path in plot_evaluation(
        predictions, states=df_test, n_steps=N_STEPS, directory=save_dir
    ):
        print(f"evaluation plot -> {path}")


if __name__ == "__main__":
    main()
