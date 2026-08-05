# BENCHMARKS

Predicting joystick state (tilt, angle) from a short history of magnetic field readings `[Bx, By, Bz]`, on joysticks the model has never seen.

## Setup

- `n_steps = 24` -> 5 tilts x 24 angles = **120 discrete states** per joystick.
- **800 training joysticks / 200 validation / 200 test**, each a different `make_dataset(seed=...)` draw of build tolerances, from disjoint seed blocks. No unit appears in two splits, so a model cannot memorise the map of a unit it is scored on.
- **Bounded history.** Every method is causal and scored on the newest frame of a W-frame window, cold-started with no carried state. W is swept over [1, 2, 3, 5, 8, 16]; 5 frames is the headline case.
- **Unknown, varying usage.** Rotate/tilt/hold rates are no longer fixed or known. Each unit draws a habitual mix from a Dirichlet, each trajectory deviates from its unit's habit, and the true rates are recorded nowhere a decoder can reach. A decoder may use the population average (countable from training trajectories) or try to infer the session's rates from the readings.
- Trajectories from `loader.make_trajectory`, length 64, 6000 train / 1500 val / 1500 test, each trajectory drawn entirely from one unit and cut into windows after a 32-frame warm-up (trajectories start at `ground`, which would flatter the early frames).
- Noise: isotropic Gaussian per axis, sigma as a fraction of the signal std (0.0229), resampled every epoch; train and test matched.
- Inputs standardised by constants from the *training* units only - standardising per unit would be a per-unit calibration and would leak away the tolerance this benchmark is about. Metric: accuracy (%) on the newest frame, `joint` = tilt and angle both correct.

## The methods

Nothing below is given the true forward map of the unit it is scored on, or the true usage rates of the session it is scored on, except the rows explicitly labelled *oracle*.

| method | needs training | unit map used | usage rates used |
|---|---|---|---|
| Lookup vs mean unit | no | mean unit | none (1 frame) |
| Per-frame MLP | yes | learned | none (1 frame) |
| GRU (causal) | yes | learned | learned from data |
| Mean-unit physics + HMM filter | no | mean unit | population mean |
| Per-frame MLP + HMM filter | yes | learned | population mean |
| Start-up calibrated (map + rates) | no | fitted at start-up | fitted at start-up |
| Oracle map + oracle rates | no | **true, per unit** | **true, per session** |

Dropped after the earlier sweeps, all consistently dominated: a naive MLP with independent tilt/angle heads (worse than one joint head, because the two errors are correlated); a transformer encoder (below the recurrent net at several times the training cost); Viterbi decoding (maximises whole-path probability, the wrong objective when each frame is scored separately); and the bidirectional/offline variants, which cannot run on a live device and which collapse to their causal versions anyway when the target is the newest frame.

## What manufacturing tolerance does to the problem

Across units the reading for a *fixed* state moves by 0.084 signal std per axis, which is 0.47x the median spacing between neighbouring states. Three facts shape everything below:

1. **The lookup table stops being exact, but not by much.** Decoding a held-out unit against the mean training unit with *zero* sensor noise costs 0.9% of frames (99.1% correct). The spread is still well under the state spacing, so a population of joysticks remains separable by one fixed decoder.
2. **The error is a bias, not noise.** A unit's offset is fixed for as long as you own the joystick, so averaging frames does not remove it - unlike sensor noise. It only becomes the binding constraint once sensor noise is small.
3. **The deviation is low-rank.** A unit's 360-dimensional deviation from the mean map is nearly rank-10 (99.8% of the variance). There are only about ten unknown numbers per joystick, which is what makes self-calibration from a handful of readings possible at all.

## Why tilt is the bottleneck, not angle

A tilt is a 4 degree deflection, so it barely moves the magnet; a rotation step is 15 degrees. In field space the states are therefore packed much more tightly along tilt than along angle:

| nearest state that differs by | median distance (signal std) |
|---|---|
| rotation (same tilt) | 0.447 |
| tilt (same angle) | 0.241 |
| anything | 0.180 |

For 80 of the 120 states the closest confusable state is a pure tilt error at the same angle, and the closest pair in the whole map is only 0.010 signal std apart - indistinguishable from one sample at any realistic noise. That is why every model scores worse on tilt than on angle, and why a few frames of history help so much: tilt persists, so evidence accumulates.

## Results

### Joint accuracy vs frames of history (W)

**Noise 0% of signal std**

| method | W=1 | W=2 | W=3 | W=5 | W=8 | W=16 |
|---|---|---|---|---|---|---|
| Lookup vs mean unit (no training) | 99.28 | 99.28 | 99.28 | 99.28 | 99.28 | 99.28 |
| Per-frame MLP | 99.63 | 99.63 | 99.63 | 99.63 | 99.63 | 99.63 |
| GRU (causal) | 99.61 | 99.59 | 99.85 | 99.91 | 99.92 | 99.95 |
| Mean-unit physics + HMM filter | 99.27 | 99.53 | 99.64 | 99.69 | 99.69 | 99.71 |
| Per-frame MLP + HMM filter | 99.71 | 99.88 | 99.95 | 99.97 | 100.00 | 100.00 |
| Start-up calibrated (map + rates) | 99.97 | 99.97 | 99.97 | 99.99 | 99.99 | 100.00 |
| Oracle map + oracle rates | 99.99 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

**Noise 2% of signal std**

| method | W=1 | W=2 | W=3 | W=5 | W=8 | W=16 |
|---|---|---|---|---|---|---|
| Lookup vs mean unit (no training) | 91.65 | 91.65 | 91.65 | 91.65 | 91.65 | 91.65 |
| Per-frame MLP | 93.71 | 93.71 | 93.71 | 93.71 | 93.71 | 93.71 |
| GRU (causal) | 93.60 | 96.53 | 97.03 | 97.31 | 97.44 | 97.69 |
| Mean-unit physics + HMM filter | 93.25 | 96.63 | 97.36 | 97.59 | 97.71 | 97.79 |
| Per-frame MLP + HMM filter | 93.72 | 96.89 | 97.49 | 97.65 | 97.73 | 97.76 |
| Start-up calibrated (map + rates) | 94.36 | 97.35 | 97.99 | 98.16 | 98.20 | 98.20 |
| Oracle map + oracle rates | 94.48 | 97.44 | 98.13 | 98.32 | 98.35 | 98.36 |

**Noise 5% of signal std**

| method | W=1 | W=2 | W=3 | W=5 | W=8 | W=16 |
|---|---|---|---|---|---|---|
| Lookup vs mean unit (no training) | 76.41 | 76.41 | 76.41 | 76.41 | 76.41 | 76.41 |
| Per-frame MLP | 81.87 | 81.87 | 81.87 | 81.87 | 81.87 | 81.87 |
| GRU (causal) | 81.93 | 87.64 | 89.63 | 90.24 | 90.84 | 91.60 |
| Mean-unit physics + HMM filter | 81.65 | 88.32 | 90.27 | 91.40 | 91.69 | 91.93 |
| Per-frame MLP + HMM filter | 81.84 | 88.53 | 90.29 | 91.47 | 91.79 | 91.97 |
| Start-up calibrated (map + rates) | 82.53 | 88.87 | 91.08 | 92.25 | 92.59 | 92.79 |
| Oracle map + oracle rates | 82.64 | 89.29 | 91.47 | 92.67 | 93.04 | 93.20 |

**Noise 10% of signal std**

| method | W=1 | W=2 | W=3 | W=5 | W=8 | W=16 |
|---|---|---|---|---|---|---|
| Lookup vs mean unit (no training) | 58.15 | 58.15 | 58.15 | 58.15 | 58.15 | 58.15 |
| Per-frame MLP | 67.57 | 67.57 | 67.57 | 67.57 | 67.57 | 67.57 |
| GRU (causal) | 67.45 | 74.88 | 77.72 | 79.77 | 80.89 | 82.08 |
| Mean-unit physics + HMM filter | 67.77 | 75.57 | 78.75 | 80.89 | 82.20 | 82.61 |
| Per-frame MLP + HMM filter | 67.40 | 75.59 | 78.83 | 81.03 | 82.19 | 82.68 |
| Start-up calibrated (map + rates) | 68.27 | 76.73 | 79.89 | 82.39 | 83.81 | 84.33 |
| Oracle map + oracle rates | 68.71 | 77.11 | 80.39 | 82.92 | 84.48 | 85.16 |

**Noise 20% of signal std**

| method | W=1 | W=2 | W=3 | W=5 | W=8 | W=16 |
|---|---|---|---|---|---|---|
| Lookup vs mean unit (no training) | 40.31 | 40.31 | 40.31 | 40.31 | 40.31 | 40.31 |
| Per-frame MLP | 50.63 | 50.63 | 50.63 | 50.63 | 50.63 | 50.63 |
| GRU (causal) | 50.73 | 59.43 | 62.15 | 65.72 | 67.76 | 69.75 |
| Mean-unit physics + HMM filter | 50.87 | 59.56 | 63.21 | 66.24 | 68.67 | 70.16 |
| Per-frame MLP + HMM filter | 50.84 | 59.55 | 63.21 | 66.20 | 68.69 | 70.13 |
| Start-up calibrated (map + rates) | 51.11 | 60.60 | 64.33 | 68.09 | 70.63 | 72.08 |
| Oracle map + oracle rates | 51.47 | 61.03 | 64.87 | 68.81 | 71.56 | 73.28 |

### Detail at W=5

| method | 0% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| Lookup vs mean unit (no training) | 99.28 | 91.65 | 76.41 | 58.15 | 40.31 |
| Per-frame MLP | 99.63 | 93.71 | 81.87 | 67.57 | 50.63 |
| GRU (causal) | 99.91 | 97.31 | 90.24 | 79.77 | 65.72 |
| Mean-unit physics + HMM filter | 99.69 | 97.59 | 91.40 | 80.89 | 66.24 |
| Per-frame MLP + HMM filter | 99.97 | 97.65 | 91.47 | 81.03 | 66.20 |
| Start-up calibrated (map + rates) | 99.99 | 98.16 | 92.25 | 82.39 | 68.09 |
| Oracle map + oracle rates | 100.00 | 98.32 | 92.67 | 82.92 | 68.81 |

Tilt and angle separately, at W=5 and 10% noise:

| method | tilt | angle | joint |
|---|---|---|---|
| Lookup vs mean unit (no training) | 60.29 | 81.31 | **58.15** |
| Per-frame MLP | 73.48 | 82.76 | **67.57** |
| GRU (causal) | 81.92 | 94.44 | **79.77** |
| Mean-unit physics + HMM filter | 83.09 | 94.45 | **80.89** |
| Per-frame MLP + HMM filter | 83.25 | 94.61 | **81.03** |
| Start-up calibrated (map + rates) | 84.36 | 95.28 | **82.39** |
| Oracle map + oracle rates | 84.81 | 95.55 | **82.92** |

### What each unknown costs (W=5)

Same decoder throughout; only what it is told about the unit and the session changes. Reading down a column shows the price of not knowing the hardware, the price of not knowing the usage pattern, and how much of each a start-up fit gets back.

| decoder knows | 0% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| Mean-unit physics + HMM filter | 99.69 | 97.59 | 91.40 | 80.89 | 66.24 |
| Mean-unit physics + fitted rates | 99.71 | 97.73 | 91.77 | 81.89 | 67.49 |
| Start-up calibrated (map only, mean rates) | 99.99 | 98.12 | 91.89 | 81.49 | 66.52 |
| Start-up calibrated (map + rates) | 99.99 | 98.16 | 92.25 | 82.39 | 68.09 |
| Oracle map, mean rates | 100.00 | 98.07 | 92.05 | 81.80 | 66.83 |
| Oracle map + oracle rates | 100.00 | 98.32 | 92.67 | 82.92 | 68.81 |

### Noise mismatch

Trained once at 5% noise, then tested across the sweep at W=5 - the realistic case, where true sensor noise is not known at training time.

| method | 0% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| Per-frame MLP (trained @ 5%) | 86.23 | 86.13 | 81.87 | 63.97 | 39.67 |
| MLP + HMM filter (trained @ 5%) | 95.05 | 94.76 | 91.51 | 77.52 | 54.36 |

## Takeaways

Joint accuracy at 10% sensor noise unless stated.

**1. Five frames is nearly all the history worth having.** Going from 1 frame to 5 is worth +13.6 points for the filtered decoder (67.4 -> 81.0); the curve then flattens, reaching only 82.7 at its best (W=16), +1.7 over W=5. The transition model prunes the state space hard - from any state only about seven of the 120 are reachable in one step - so a couple of frames already collapse most of the ambiguity. A short buffer is not a meaningful handicap: 5 frames instead of 16 costs about a point.

**2. Bounded history only costs you at power-on.** A recursive filter carries a 120-number belief vector, not a buffer of readings, so in steady state its history is unbounded at constant memory and constant cost per sample. The W sweep is therefore the price of a *cold start*, and by W=5 that price is nearly paid off. Only methods that re-read a raw window (the GRU as run here) genuinely need the buffer.

**3. The dynamics are worth more than the network.** At W=5, handing per-frame MLP posteriors to the HMM filter gives 81.0 against 67.6 for the same network read frame by frame, and 79.8 for a GRU left to learn the dynamics itself. Telling a decoder the transition rules beats making it infer them, and costs nothing at inference.

**4. You may not need a network at all.** The physics emission - just the mean unit's map and a Gaussian - scores 80.9 through the identical filter, against 81.0 for the learned emission. The network buys convenience (no forward model at inference), not accuracy.

**5. Not knowing how the user moves costs more than not knowing the hardware.** Usage rates now vary per unit and per session and are never recorded. Handing the decoder the true rates is worth +1.1 points at W=5 (81.8 -> 82.9) on top of an already perfect map - a bigger prize than per-unit calibration of the map itself. It is also the easier of the two to recover: counting moves in a decoded start-up buffer gets +1.0 of it back with about ten lines of arithmetic.

**6. One start-up fit handles both unknowns.** Estimating the unit's map and the session's rates together from the opening frames scores 82.4 at W=5, against 80.9 assuming population averages for both and 82.9 for the full oracle - so +1.5 points recovered out of +2.0 available. The map fit is the fiddly half (a rank-10 subspace and a shrinkage prior); the rate fit is a few counters. Both are unsupervised and neither needs the device to be measured.

## Recommended decoder

Per-frame MLP (or the physics map, if you would rather not train anything) -> HMM forward filter carrying its belief vector across samples -> read the argmax each frame. Fit the tolerance coefficients once per unit over the first few seconds of use and fold them into the emission map. That is a few hundred lines, needs no sequence model, runs in constant time and memory per sample, and beats everything else here.

## On the other approaches

- **Markov chain / FSA.** The legality rules in `loader.filter_legal_tilt` and `filter_legal_rotation` are exactly a finite state automaton over the 120 states. Adding probabilities to its edges gives the Markov chain used here, so a hard FSA constraint is the special case with uniform weights on legal moves. The filter rows are the FSA approach, with unlikely-but-legal moves penalised rather than merely allowed.
- **Kalman filter.** Not applicable to the *state*, which is 120 discrete cells with categorical jumps rather than a continuous linear-Gaussian quantity; the HMM forward filter is its discrete counterpart, same predict/update recursion with a sum over states instead of a covariance update. It does apply to the *tolerance*, which is continuous and low-dimensional: running the coefficient fit recursively would give an online self-calibrator that sharpens with use.
- **Angle as a continuous quantity.** Angle is 24 classes rather than a regression, which throws away the fact that class 23 neighbours class 0. Angle accuracy is high enough that this costs little, but a circular (sin/cos) head would be the thing to try for finer resolution.

## Caveats

- Tolerances are whatever `parameter_factory` models: Gaussian, independent, 0.1 mm on positions and 0.1 deg on orientations. Real production spread may be correlated (a mis-set jig moves several magnets together) or have tails, either of which would hurt more.
- Magnetisation is fixed across units at the measured values, so magnet strength variation is not represented. Adding it would widen the across-unit spread and lower every number here.
- Noise is i.i.d. isotropic Gaussian. Real sensors drift and have cross-axis and temperature effects, which the filter's persistence assumption handles less gracefully than white noise.
- The transition model assumes it knows `p_rotate`/`p_tilt`. A user moving the stick much faster than the training prior would be over-smoothed. The matrix itself is not an oracle advantage: estimating it by counting training transitions changes accuracy by under 0.05 points.
- Windows are cut at stride 4 from one trajectory per unit, so neighbouring windows overlap in state but get independent noise draws.
- Accuracy is not perfectly monotone in W: the largest window scores a few tenths below the peak. Longer windows fit fewer times into the scoring region, so they yield fewer and differently-placed windows - a sampling artefact, not evidence that extra history hurts. Differences under about half a point should not be read as real.
