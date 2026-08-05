"""
Predicting joystick state (tilt, angle) from a short history of B-field
readings, on joysticks the model has never seen.

This module runs *everything* and writes BENCHMARKS.md. Each approach can also
be run on its own, which is usually the faster way to iterate:

    uv run hmm.py          structured decoding, and the oracle ceilings
    uv run net_mlp.py      the per-frame network
    uv run net_gru.py      the causal sequence network
    uv run calibration.py  start-up self-calibration
    uv run benchmarks.py   all of the above, plus the report

    <module>.py --check    fast self-check, no sweep
    <module>.py --smoke    tiny settings

Layout:

    statespace.py   what a state is, and how predictions are scored
    population.py   joysticks, usage patterns, trajectories, readings
    evaluate.py     the shared harness every method is run through
    nets.py         torch plumbing shared by the two architectures
    report.py       turning results into BENCHMARKS.md

Three things are deliberately withheld from every method, because a deployed
decoder would not have them: which unit it is running on, how that user moves,
and any history beyond the last W frames. Only rows labelled *oracle* get the
first two, and they exist to price what the others are missing.
"""

import argparse
import json

import torch

import calibration
import hmm
import net_gru
import net_mlp
from evaluate import (
    DEFAULT_CFG, NOISE_FRACS, SMOKE_CFG, SMOKE_NOISE, SMOKE_WINDOWS, build_setup,
    make_window, sweep,
)
from nets import model_loglik, predict_last, train
from net_mlp import JointMLP
from population import CACHE, MAIN_WINDOW, WINDOWS
from report import TRAIN_FRAC, write_report
from statespace import score

# Ordered so the report reads as a progression: no learning, then learning,
# then dynamics, then calibration, then the ceilings.
ALL_METHODS = {
    "Lookup vs mean unit (no training)": hmm.METHODS[
        "Lookup vs mean unit (no training)"],
    "Per-frame MLP": net_mlp.METHODS["Per-frame MLP"],
    "GRU (causal)": net_gru.METHODS["GRU (causal)"],
    "Mean-unit physics + HMM filter": hmm.METHODS["Mean-unit physics + HMM filter"],
    "Per-frame MLP + HMM filter": net_mlp.METHODS["Per-frame MLP + HMM filter"],
    "Start-up calibrated (map + rates)": calibration.METHODS[
        "Start-up calibrated (map + rates)"],
    "Start-up calibrated (map only, mean rates)": calibration.METHODS[
        "Start-up calibrated (map only, mean rates)"],
    "Mean-unit physics + fitted rates": calibration.METHODS[
        "Mean-unit physics + fitted rates"],
    "Oracle map, mean rates": hmm.METHODS["Oracle map, mean rates"],
    "Oracle map + oracle rates": hmm.METHODS["Oracle map + oracle rates"],
}


def noise_mismatch(s, noise_fracs):
    """Train once at one noise level, test across the sweep.

    The realistic case: sensor noise is not known when the model is trained, so
    what matters is how gracefully it degrades when the guess is wrong.
    """
    print(f"\n=== noise mismatch: trained at {TRAIN_FRAC:.0%}, W={MAIN_WINDOW} ===")
    model = train(
        JointMLP(), s.split_tr, s.split_va, s.norm, TRAIN_FRAC * s.signal,
        steps=s.cfg["frame_steps"], batch=1024, per_frame=True,
    )
    rows = []
    for frac in noise_fracs:
        win = make_window(s, frac, MAIN_WINDOW)
        raw = score(predict_last(model, win.X), win.y)
        filtered = score(
            hmm.decode_with(
                win,
                lambda c: model_loglik(model, win.X[c], s.log_prior),
                s.const_rates(len(win.X)), s.log_pi,
            ),
            win.y,
        )
        rows += [
            dict(model=f"Per-frame MLP (trained @ {TRAIN_FRAC:.0%})", noise=frac,
                 window=MAIN_WINDOW, **raw),
            dict(model=f"MLP + HMM filter (trained @ {TRAIN_FRAC:.0%})", noise=frac,
                 window=MAIN_WINDOW, **filtered),
        ]
        print(f"  test @ {frac:>4.0%}:  per-frame {raw['joint']:6.2f}   "
              f"+filter {filtered['joint']:6.2f}")
    return rows


def run(smoke: bool):
    cfg = dict(SMOKE_CFG if smoke else DEFAULT_CFG)
    noise_fracs = SMOKE_NOISE if smoke else NOISE_FRACS
    windows = SMOKE_WINDOWS if smoke else WINDOWS

    s = build_setup(cfg)
    print(
        f"joysticks: {len(s.tables_tr)} train / {len(s.tables_va)} val / "
        f"{len(s.tables_te)} test (disjoint seeds); across-unit spread "
        f"{s.tol_scale:.3f} signal std"
    )
    print(
        f"usage rates: population mean p_rotate={s.pop_rates[0]:.3f} "
        f"p_tilt={s.pop_rates[1]:.3f}; per-trajectory spread "
        f"{s.rates_te[:, 0].std():.3f} / {s.rates_te[:, 1].std():.3f} "
        f"(p_rotate ranges {s.rates_te[:, 0].min():.2f}-{s.rates_te[:, 0].max():.2f})"
    )

    results = sweep(s, ALL_METHODS, noise_fracs, windows)
    mismatch = noise_mismatch(s, noise_fracs)
    return results, mismatch, cfg, noise_fracs, windows, s.signal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild BENCHMARKS.md from cached results, no training")
    args = ap.parse_args()
    torch.set_num_threads(8)

    if args.report_only:
        d = json.loads((CACHE / "results.json").read_text())
        write_report(d["matched"], d["mismatch"], d["cfg"], d["noise_fracs"],
                     d["windows"], d["signal"])
    else:
        out = run(args.smoke)
        if args.smoke:
            print("\nsmoke run: BENCHMARKS.md not written")
            raise SystemExit
        write_report(*out)
    print("\nwrote BENCHMARKS.md")
