# Status

Where the joystick inversion work stands, as of 2026-08-06 (branch `rnn`).

## The problem

A magnetic joystick carries 4 cuboid magnets and reads the field at one 3-D Hall sensor.
From that field reading we want to recover the joystick's state:

- **tilt** — one of 5: south, north, east, west, ground (the un-tilted rest position)
- **rotation** — one of 24 discrete steps, 15° apart, wrapping at 360°

The catch is manufacturing tolerance. Every physical unit has slightly different magnet
positions, orientations, polarizations and travel, so a model fitted to one unit does not
transfer for free. Everything below is therefore scored on **units the model has never
seen**, not on held-out states of the training units.

## Repo layout

| file | what it holds |
|---|---|
| [parameters.py](parameters.py) | `Parameters` dataclass — one unit's geometry, magnetics and travel. `parameter_factory` builds the nominal design, or a tolerance-perturbed unit from a generator. |
| [joystick.py](joystick.py) | magpylib simulation. `make_dataset` sweeps one unit through all 120 states; `make_transitions` enumerates the legal state graph; `make_transitions_datasets` stacks units. |
| [train.py](train.py) | angle encoding helpers (`process_angle`/`unprocess_angle`/`angle_index`/`angular_error`), the `Predictions` container, and `make_dataloader`. |
| [run.py](run.py) | the model (`mlp`), training loop, `evaluate`/`metrics`/`report`, and `main`. |
| [plot.py](plot.py) | evaluation figures, plus the pre-existing `show_system`/`plot_loops` helpers. |
| [constants.py](constants.py) | tilt naming and the mT/T conversion. |
| [utils.py](utils.py) | `timed` decorator, and `load_measurement_data` for the real CSVs in `data/`. |

`uv run run.py` trains and evaluates end to end in ~9 s. `uv run joystick.py` and
`uv run train.py` run their self-checks.

## Strategy

**1. Simulate a population of units, not one joystick.**
`parameter_factory(generator)` perturbs every parameter of a unit by a named 1-sigma
tolerance, currently:

| tolerance | value |
|---|---|
| `POSITION_TOLERANCE` | 1e-4 m (0.1 mm) — magnet and sensor positions |
| `SIZE_TOLERANCE` | 1e-4 m on a 5 mm cube edge |
| `ANGLE_TOLERANCE` | 1.0 deg — magnet/sensor orientations and tilt travel |
| `POLARIZATION_TOLERANCE` | 15e-3 T |

Unit *i* uses `seed + i`, so a run is reproducible and train/test units cannot overlap.

**2. Work on the state graph, not isolated states.**
`make_transitions` enumerates every legal single-step move: rotate one step clockwise or
anticlockwise, tilt out of ground into one of 4 directions, return to ground from a tilt,
or hold still. That is **552 transitions per unit** (7 actions from each of the 24 ground
states, 4 from each of the 96 tilted states). The self-check in `joystick.py` validates the
per-state action counts.

**3. Predict the end state from a length-2 field trajectory.**
Model input is `(B_start, B_end)` shaped `(batch, 2, 3)` in mT — the reading before and
after one move. A single reading is genuinely ambiguous; one predecessor resolves most of
it. See "What we tried and rejected" for why it is 2 and not 16.

**4. Two separate heads, one shared input.**
- tilt: 5-way classifier, `CrossEntropyLoss`
- rotation: 2-vector `(sin θ, cos θ)` regression, `MSELoss` — avoids the discontinuity a
  raw-angle target would have at 0/360

Both are the same MLP: `6 → 128 → 128 → out`, Adam at 1e-3, 150 epochs. 40 epochs
underfits badly (~10° error, train error equal to test); past ~150 the train/test gap
opens and the extra capacity goes into memorising training units.

**5. Inject sensor noise at the loader.**
`make_dataloader(noise=...)` adds Gaussian noise in mT, drawn once per sample. Train and
test use the same level, so the sweep answers "how good is this sensor good enough to be".

Train units are seeded from 2 (8 units, 4416 transitions), test units from 100 (4 units,
2208 transitions).

## Results

Cross-unit test set, 4 unseen units. `angle acc` snaps the continuous prediction to the
nearest of the 24 rest positions; `within 1` allows a single-step miss. Errors in degrees.

| noise / mT | tilt acc | angle acc | within 1 | err mean | err med | err p95 |
|---|---|---|---|---|---|---|
| 0.00 | 0.942 | 0.904 | 0.970 | 5.098 | 2.434 | 12.249 |
| 0.01 | 0.947 | 0.901 | 0.969 | 4.944 | 2.311 | 12.823 |
| 0.05 | 0.934 | 0.894 | 0.966 | 5.101 | 2.425 | 12.670 |
| 0.10 | 0.926 | 0.891 | 0.968 | 5.084 | 2.327 | 13.910 |
| 0.50 | 0.852 | 0.845 | 0.957 | 6.035 | 2.792 | 18.564 |

Reading this:

- **Noise up to 0.1 mT is nearly free.** Fields at the sensor are a few mT peak to peak,
  so 0.1 mT is a few percent of signal and costs ~1.6 points of tilt accuracy. At 0.5 mT
  it starts to bite (−9 points). The 0.01 row scoring marginally above 0.00 is seed noise,
  not a real effect — run-to-run spread on this config is about ±0.02.
- **The mean angular error is misleading.** Median error is 2.3–2.8° while the mean is
  ~5°, and 97% of predictions land within one 15° step. The model is usually far better
  than 5°, with a thin tail of gross failures dragging the mean up.
- **That tail is a distinct failure mode, not imprecision.** The miss-distance histogram
  has a bump at **9–10 steps out** (135–150°), and the polar plot shows p95 spiking at
  rotation states 10 and 21 — nearly opposite each other on the dial — while the median
  stays flat everywhere. This looks like an occasional near-antipodal misread, i.e. a sign
  or symmetry ambiguity in the field rather than a resolution limit. **Not yet diagnosed.**
- **Tilt confusion is structured.** South and west over-predict (recall 0.97/0.98,
  precision 0.86/0.88); east and ground under-predict (recall 0.87/0.88, precision
  0.97/0.96). East and west should be symmetric, so this asymmetry is suspicious.
- **The 6 inputs are not 6 degrees of freedom.** `Bx_start`/`Bz_start` correlate at −0.86
  and `Bx_start`/`Bx_end` at +0.76, which is consistent with the second timestep adding
  little.

Figures are written to `plots/` (gitignored) by `uv run run.py`:
`error.png`, `confusion.png`, `rotation_error.png` (polar), `correlation.png`.

## What we tried and rejected

**An RNN over long trajectories.** We generated random walks over the legal transition
graph, each trajectory drawing its own Dirichlet action weights so the set spanned a range
of usage patterns (spinning, tilting, idling), and trained a GRU with per-timestep
supervision. Scored on identical targets, the one-step pair model won:

| model | context per prediction | tilt acc | angle err |
|---|---|---|---|
| one-step pairs | `B[t-1], B[t]` | 0.973 | 4.14° |
| trajectory GRU | `B[0..t]` | 0.922 | 4.10° |

Retraining the pair model on pairs drawn from the same trajectories — matching the GRU's
uneven state coverage — cost it only ~1.7 points, so the gap is the architecture, not the
data. The reason is that this inversion is **Markovian in the field reading**: one reading
nearly determines the state, one predecessor resolves the rest, and readings 3 onward carry
no new information. Per-step GRU accuracy was flat from step 1 (0.91, 0.91, 0.93, …) and
never approached the pair model's 0.97.

Both arms of that comparison ran on the same simulation, so the conclusion holds — but note
those absolute numbers predate the tilt-chain fix below and are **not** comparable to the
results table above. The trajectory code has been removed; see git history if it is wanted
back. It would only become worth revisiting if the problem stopped being memoryless — much
higher sensor noise, where averaging over many readings wins, or a target that needs
history such as cumulative revolutions rather than angle mod 360°.

**LightGBM.** The original `train_tilt`/`train_angle` gradient-boosting path has been
deleted along with the `lightgbm` dependency.

## Fixed along the way

**The tilt chain was silently wrong.** `make_sensor_readings` applies each tilt block as a
cumulative delta down the rotation path (`-n*2`, `+e`, `-w*2`), which only cancels correctly
when `s == n` and `e == w` exactly. Nominal parameters satisfy that by luck; per-angle
tolerances do not. With `ANGLE_TOLERANCE = 1.0` on a nominal 4° tilt the labels became
close to meaningless — the "ground" block was a randomly tilted state, differing per unit:

```
tilt acc   0.285 -> 0.926      (0.2 is chance for 5 classes)
angle err  82.8° -> 5.08°      (90° is chance)
```

Each block's delta is now derived from its intended absolute tilt. The self-check in
`joystick.py` reads each block's orientation relative to the ground block and asserts
south=`+s`, north=`−n`, east=`+e`, west=`−w`, ground=identity, over several seeds. It fails
on the old code.

**12 dead parameters.** Magnet y/x offsets, the theta orientations and the per-unit
polarizations were in the parameter vector but never read by the simulation, so their
tolerances did nothing. All are now wired in, along with the sensor's own phi/theta on top
of its −45° mounting. `setup_magnets` became a loop over 4 magnets rather than four
near-identical blocks.

**Mislabelled classification reports.** `DIRECTIONS` is ordered `north, south, …` but tilt
class 0 is *south*, so every `classification_report(target_names=DIRECTIONS)` swapped those
two labels. `TILT_NAMES`, derived from `DIRECTION_MAP` so it cannot drift, replaces it at
the call sites. `DIRECTIONS` itself is untouched — `load_measurement_data` uses it only for
name membership, where order is irrelevant.

## Known gaps

- **The antipodal tail is undiagnosed.** The single most useful next step: find out whether
  rotation states 10 and 21 produce near-mirror-image field readings, and whether the bad
  cases cluster by unit or by state.
- **The east/ground asymmetry in tilt confusion is unexplained.**
- **No validation against real hardware.** `data/sensor1` holds measured CSVs and
  `load_measurement_data` still reads them, but nothing calls it — the current `run.py`
  is simulation-only. The measurement rows are single states, so they would need pairing
  into transitions before this model could consume them. This is the biggest hole in the
  story: every number above is simulated.
- **Sensor 2 is dormant.** `Parameters.sensor_position` holds both, but only sensor 1 is
  instantiated; its design positions are not trusted, and using it would double the
  feature space.
- **`run.py` currently evaluates a single noise level** (0.1 mT) rather than looping
  `NOISE_LEVELS`. The table above came from a sweep run manually.
- **The right-handedness flip is not implemented.** `setup_sensor` notes that the real
  Infineon part is left-handed and would need `B[2]` negated; the simulation never does it.
  Harmless while everything is simulated, a real discrepancy once measured data is used.
