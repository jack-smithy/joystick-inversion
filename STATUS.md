# Status

Where the joystick inversion work stands, as of 2026-08-06 (branch `rnn`).

## The problem

A magnetic joystick carries 4 cuboid magnets and reads the field at two 3-D Hall sensors.
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
| [train.py](train.py) | `state_index` (the joint 120-state label), `bin_error`, the `Predictions` container, and `make_dataloader`. |
| [run.py](run.py) | the model (`mlp`), training loop, `evaluate`/`metrics`/`report`, and `main`. |
| [plot.py](plot.py) | evaluation figures, plus the pre-existing `show_system`/`plot_loops` helpers. |
| [constants.py](constants.py) | tilt naming, `FIELD_COLUMNS` (the two sensors' 6 field components), and the mT/T conversion. |
| [utils.py](utils.py) | `timed` decorator, and `load_measurement_data` for the real CSVs in `data/`. |
| [RESULTS.md](RESULTS.md) | every experiment run, grouped by regime. This file holds the conclusions; that one holds the measurements, including superseded ones. |

`uv run run.py` trains and evaluates end to end in ~2.5 min. `uv run joystick.py` and
`uv run train.py` run their self-checks.

## Strategy

**1. Simulate a population of units, not one joystick.**
`parameter_factory(generator)` perturbs every parameter of a unit by a named 1-sigma
tolerance, currently:

| tolerance | value |
|---|---|
| `POSITION_TOLERANCE` | 1e-4 m (0.1 mm) — magnet and sensor positions |
| `SIZE_TOLERANCE` | 1e-4 m on a 5 mm cube edge |
| `ANGLE_TOLERANCE` | 1.0 deg — magnet/sensor orientations |
| `TILT_TOLERANCE` | 1.0 deg — per-direction tilt travel |
| `POLARIZATION_TOLERANCE` | 15e-3 T |

`TILT_TOLERANCE` used to be folded into `ANGLE_TOLERANCE`. It is split out because it is a
different physical thing — a mechanical stop, not an assembly alignment — and because it
is now **the single parameter that limits accuracy**. See "The travel tolerance is the
whole error budget" below.

Unit *i* uses `seed + i`, so a run is reproducible and train/test units cannot overlap.

**2. Work on the state graph, not isolated states.**
`make_transitions` enumerates every legal single-step move: rotate one step clockwise or
anticlockwise, tilt out of ground into one of 4 directions, return to ground from a tilt,
or hold still. That is **552 transitions per unit** (7 actions from each of the 24 ground
states, 4 from each of the 96 tilted states). The self-check in `joystick.py` validates the
per-state action counts.

**3. Predict the end state from a length-2 field trajectory, at both sensors.**
Model input is `(B_start, B_end)` shaped `(batch, 2, 6)` in mT — the reading before and
after one move, each timestep carrying both 3-D sensors. A single reading is genuinely
ambiguous; one predecessor resolves most of it. See "What we tried and rejected" for why
it is 2 timesteps and not 16, and "Sensor 2 was not optional" for why it is 6 components
and not 3.

**4. One classifier over all 120 states.**
The joystick has 5 tilts × 24 rotations and nothing in between, so the target is a single
120-way softmax rather than two heads. The label is `state = tilt * 24 + angle_idx`,
decoded back with `// 24` and `% 24`, so every per-tilt and per-rotation metric survives.

This replaced a 5-way tilt classifier plus a `(sin θ, cos θ)` regression head. Two heads
could disagree, and the regression output had to be snapped to the nearest rest position
after the fact — a step that could only ever lose information, since the model was being
asked for a continuous angle it was then forbidden to return.

On sensor 1 with 8 training units, folding both into one classifier was worth **+4.5
points of tilt accuracy and +9 points of rotation accuracy**. That gain is real but it is
**specific to an information-starved regime**: once sensor 2 is added and the unit count
is raised, a head-to-head over four architectures finds no significant difference between
any of them (see "Four architectures, one regime"). The joint head is kept because it is
simpler — one model, one loss, no post-hoc snapping — not because it is measurably more
accurate under the current setup.

The network is a plain MLP: `12 → 256 → 256 → 120`, Adam at 1e-3, 200 epochs. 40 epochs
underfits badly; past ~200 the extra capacity goes into memorising training units. Three
layers instead of two measurably made it worse. One head instead of two halves training
time.

**6. Fix the sensor noise at 0.5 mT.**
`run.py` trains and tests at `NOISE = 0.5` and no longer sweeps. The signal std across
states is **22.5 mT**, so 0.5 mT is ~2% of signal. The previous 0.1 mT default was 0.44%,
low enough that nothing was ambiguous — which is why every architecture scored identically
under it, and why raising it separated them immediately. Quote any accuracy number
together with the noise it was measured at.

**5. Inject sensor noise at the loader.**
`make_dataloader(noise=...)` adds Gaussian noise in mT, drawn once per sample. Train and
test use the same level, so the sweep answers "how good is this sensor good enough to be".

Train units are seeded from 2 (128 units, 70656 transitions), test units from 100
(16 units, 8832 transitions). Both counts went up sharply, and both mattered:

- **Training units are the cheapest accuracy on offer.** 32 → 64 → 128 units took the
  miss count 263 → 152 → 82. It has not flattened; raise it further if the runtime is
  affordable. Capacity does not substitute — at fixed data, wider helped a little and
  deeper hurt.
- **4 test units was too few to trust.** It scored the current model at 5 misses when the
  honest figure over 16 units is 82. Most units are near-perfect and a couple are not, so
  a small test population mostly measures which units it happened to draw. Every number
  below is on 16 units.

`uv run run.py` now takes ~2.5 min rather than ~9 s, almost all of it in training.

## Results

Cross-unit test set, 16 unseen units, 8832 transitions, sensor noise 0.5 mT, tolerances at
the current (doubled) values. `state acc` is the headline: the fraction of transitions
where the full 120-way state is exactly right. `tilt acc` and `angle acc` decompose it and
are both upper bounds on it. `within 1` allows a single-step rotation miss. `err mean` is
the mean miss distance in degrees, quantized to multiples of 15° because the classifier can
only ever answer with a rest position.

| state acc | tilt acc | angle acc | within 1 | err mean | misses |
|---|---|---|---|---|---|
| 0.941 | 0.941 | 0.999 | 1.000 | 0.036 | ~521 / 8832 |

How it got here. Every row changed something about the problem as well as the model, so
read the "regime" column before comparing any two:

| change | state acc | misses | regime |
|---|---|---|---|
| two heads, sensor 1, 8 train units | n/a (0.926 tilt) | — | B, 4-unit test, 0.1 mT |
| joint 120-way classifier | 0.971 | 65 / 2208 | B, 4-unit test, 0.1 mT |
| + sensor 2 | 0.982 | 39 / 2208 | C, 4-unit test, 0.1 mT |
| + 128 train units, 256 wide | 0.991 | 82 / 8832 | C, 16-unit test, 0.1 mT |
| + doubled tolerances, 0.5 mT noise | **0.941** | ~521 / 8832 | D, 16-unit test |

Two rows need their caveats read with them. The joint-classifier gain was measured before
sensor 2 and does not survive into regime C at 0.1 mT (Strategy §4) — though it returns
under noise, see "Raise the noise". And the 4 → 16 unit test change makes rows 3 and 4 look
like a regression in raw miss count when the model in fact improved; on the old 4-unit set
that model scores 5 misses. That is why the test set was enlarged.

The last row is not a regression either: it is the same model on a deliberately harder
problem — doubled position, alignment and polarization spread, and 5× the sensor noise.

Reading the current row:

- **The rotation is still essentially solved.** `angle acc` 0.999, and `within 1` is 1.000,
  so every rotation miss that does occur is a single step. `state acc == tilt acc` still
  holds: every failure is a tilt failure.
- **The failure changed shape, not just size.** Per-class recall is now south 0.98, north
  0.94, east 0.92, west 0.96, **ground 0.92** — spread across every class, with ground the
  weakest. Under regime C the deficit was east alone (0.96, everything else 1.00) and was
  traceable to two low-travel units. Doubling the alignment and polarization spread
  produces a different, more evenly distributed failure mode; the low-travel story below
  explains regime C's errors, not these.
- **Worst-case misses grew slightly.** Worst rotation miss is now 1–7 steps depending on
  tilt state, against 0 steps for four of five classes under regime C.

Figures are written to `plots/` (gitignored) by `uv run run.py`:
`error.png`, `confusion.png`, `rotation_error.png` (polar), `correlation.png`.
`error.png` is a single miss-distance panel: the continuous error histogram was dropped,
since against a quantized error it drew the same figure on a rescaled x-axis. The polar
plot summarises each state by mean and p95 rather than median and p95, for the same
reason — the median is zero at every state.

## Sensor 2 was not optional

Sensor 2 was dormant on the grounds that it would double the feature space for unclear
gain. It was in fact the fix for the failure mode that had been open longest.

The old "near-antipodal tail" was misdiagnosed. It was not a 180° symmetry: measuring the
field-space distance between every pair of the 120 states on the nominal unit, and
comparing it against how far a single state's reading wanders across the unit population,
gives the real picture with sensor 1 alone:

```
state       nearest other state    gap      own spread   ratio
south/13    east/18                0.233 mT   0.359 mT    0.65
east/18     south/13               0.233 mT   0.484 mT    0.48
south/1     east/6                 0.300 mT   0.444 mT    0.68
```

A ratio below 1 means the two states sit closer to each other than one state sits to
itself across manufacturing tolerance — **no model can separate them**. 9 of the 120
states were in that position, and the model's actual confusions matched the list exactly
(`east/6 → south/1`, `east/18 → south/13`). It is a tilt/rotation coupling: tilting south
at one rotation produces almost the field of tilting east at another.

Adding sensor 2 takes the count of such states from **9 to 0**, and the worst gap/spread
ratio from 0.48 to 2.08. It also confirmed the mechanism: 34 of the 65 original misses
were `noop` transitions, which are 22% of the data but were 52% of the errors — a noop
gives two noisy reads of one state, so it has nothing extra to disambiguate with. With
sensor 2 the noop misses collapsed from 34 to 12 to 4 as training units were added.

Caveat, unchanged: **sensor 2's design positions are not trusted.** Everything above is
simulation, so this says the second sensor is worth having, not that these exact
coordinates are right. Both sensors also share the un-applied left-handedness flip noted
in `setup_sensors`.

## The travel tolerance is the whole error budget

The 82 remaining misses are not spread across the test population. Per unit:

- 12 of 16 units: **zero** misses.
- Units 6 and 13 alone: **90%** of all misses.
- State accuracy excluding those two units: **0.9981**.
- 127 of 152 misses (at the 64-unit config, where they are easier to count) are the single
  confusion `east → ground`.

Those two units are exactly the ones whose drawn east travel is small: **0.775°** on unit
13 and **1.926°** on unit 6, against a 4° nominal. `TILT_TOLERANCE` is 1.0°, which is 25%
of the travel, so a −3σ unit has essentially no travel at all. The model is not wrong to
call it ground; the joystick has barely moved.

Sweeping the tolerance, everything else held fixed:

| `TILT_TOLERANCE` | state acc | misses / 8832 | test units with <2° travel |
|---|---|---|---|
| 1.00° (current) | 0.9907 | 82 | 2 |
| 0.75° | 0.9980 | 18 | 1 |
| 0.50° | 0.9997 | 3 | 0 |
| 0.25° | 1.0000 | 0 | 0 |

**Holding the tilt travel to 0.5° buys more than any model change on the table.** The
default is deliberately left at 1.0° — tightening it in the repo would flatter the model
by quietly making the problem easier. It is a hardware question: if the mechanism can hold
0.5°, the inversion is effectively exact.

## Four architectures, one regime

The head structure and the context window had only ever been compared across regimes that
also differed in sensors, unit counts and tolerances, so none of the comparisons meant
anything on their own. Run head to head — same 64 train / 16 test units, same
trajectories, same noise draw, same width, same optimizer, both sensors, only the model
varying — over 3 model seeds:

| model | mean | seed spread | context |
|---|---|---|---|
| 1. separate tilt/rotation heads | 0.9878 | 0.0006 | `B_t` |
| 2. joint 120-way classifier | 0.9866 | **0.0058** | `B_t` |
| 3. joint causal GRU | 0.9893 | 0.0018 | `B_0..B_t` |
| 4. joint 120-way classifier | 0.9853 | 0.0039 | `B_t-1, B_t` |

**Nothing separates them.** The full range of model means is 0.0040, while model 2's own
seed-to-seed spread is 0.0058 — one model's noise is wider than the gap between best and
worst. The ranking reshuffles across seeds (model 2 places 3rd, 1st and 4th). Every model
reached a training loss under 0.02, so none of this is under-training.

Two prior claims do not survive this:

- BENCHMARKS.md (commit `3859109`) says separate tilt/angle heads are "worse than one
  joint head, because the two errors are correlated". Not reproducible here.
- The joint head's +4.5 tilt points over two heads, measured earlier in this repo, was a
  sensor-1 / 8-unit result. See Strategy §4.

**Why nothing separates them: they all fail on the same physical positions.** Miss overlap
between any two models is a Jaccard of 0.54–0.74. Two models each missing ~150 of 11904
positions independently would overlap at a Jaccard of about 0.006, so the observed
agreement is ~100x chance. Per unit, at seed 1:

```
unit      m1    m2    m4   GRU
   6      37    59    63    59
  13      58    54    60    54
   2      16    25    21     0
others    31    21    36    25
```

Units 6 and 13 — the two low-travel units from the previous section — carry 67-82% of every
model's misses, and 4 of the 16 units are perfect for all four models. The architectures
are all sitting on the floor set by `TILT_TOLERANCE`, which no head structure or context
window can move.

**History buys nothing *at this noise level, with this much data*.** The GRU's accuracy is
flat against how much context it has: 0.9917 at t=1-5, 0.9828 at t=11-15, 0.9844 at t=31.

Both qualifiers turned out to matter, and the null result survives neither:

- **Noise.** 0.1 mT is 0.44% of the 22.5 mT signal std — so little that nothing is
  ambiguous and there is nothing for extra context to resolve. Raise it and every model
  separates cleanly (see "Raise the noise").
- **Data.** At 64 units all four have saturated at 0.986-0.990, which is why they look
  identical. Cut the training set and they separate just as cleanly: at 4 units the GRU
  scores 0.9486 against 0.9048 for the next best. Reaching 0.97 takes the GRU ~8 units,
  the pair model ~16, and either single-frame model ~32 — **recurrence is worth about 2x
  the training data**. Full curve in [RESULTS.md](RESULTS.md) C9.

The ranking also inverts at small data: the pair model ties for best of the non-recurrent
three at 64 units but is *worst* at 2 units, because doubling the input width costs more
than the extra frame returns when there is almost nothing to learn from.

One loose end, deliberately not over-read: the GRU took unit 2 from ~20 misses to **0**
while matching the other models everywhere else. Unit 2's tilt travel is unremarkable, so
this may be a case where history genuinely helps, or it may be seed luck. One seed cannot
tell the difference and no claim is made either way.

The comparison needs trajectories rather than the transition table, since a pair gives an
RNN nothing to integrate. That generator lives in the scratchpad, not in the repo — the
walk semantics were recovered from `loader.make_trajectory` in commit `3859109`.

## Raise the noise and the architectures do separate

Repeating the same four-way comparison with doubled tolerances (0.2 mm, 2.0° alignment,
30 mT polarization; **tilt travel deliberately held at 1.0°**, since C3/C7 showed low-travel
units dominate every model's misses equally and would swamp the comparison) and noise swept
over 2%, 9% and 20% of signal std. Mean joint accuracy over 3 seeds:

| model | context | 0.5 mT | 2.0 mT | 4.5 mT |
|---|---|---|---|---|
| 1. separate tilt/rotation heads | `B_t` | 0.9129 | 0.7534 | 0.5216 |
| 2. joint 120-way | `B_t` | 0.9091 | 0.7505 | 0.5722 |
| 4. joint 120-way | `B_t-1, B_t` | 0.9196 | 0.7595 | 0.5938 |
| 3. joint causal GRU | `B_0..B_t` | **0.9341** | **0.7895** | **0.6445** |
| max seed spread | | 0.0079 | 0.0081 | 0.0094 |

Every gap now exceeds the seed spread, and the ranking is identical at all three levels:
**GRU > pair > single-frame joint**. Concretely:

- **Context helps, and more so as noise rises.** The GRU's margin over the best
  non-recurrent model grows +1.45 → +3.00 → +5.07 points. One extra frame is worth
  +1.05 → +0.90 → +2.16.
- **The joint head beats separate heads only when the problem is hard.** A tie at 0.5 and
  2.0 mT; **+5.06 points** at 4.5 mT. That vindicates the old BENCHMARKS.md claim,
  conditionally — and explains why it looked wrong at 0.1 mT.
- **The regression head degrades worst.** Model 1 goes from best single-frame model at
  0.5 mT to clearly worst at 4.5 mT: `(sin θ, cos θ)` cannot express uncertainty, so it
  fails ungracefully as readings get noisier. This is the strongest argument for the joint
  classifier, and it only shows up under noise.

**Caveat on the GRU's margin — treat it as an upper bound.** Its accuracy is flat to
*declining* with context (0.665 → 0.622 across a trajectory at 4.5 mT), which is not what
evidence accumulation looks like. Every trajectory in the generator starts at `ground`, so
the GRU can exploit "trajectories begin at ground", a cue the other three cannot see
because they do not know their position in the sequence. Regime A's benchmark avoided this
with a 32-frame warm-up. Isolating the real integration benefit needs random start states
or a warm-up before scoring; until then the GRU's lead mixes noise-averaging with
start-state knowledge. Full numbers in [RESULTS.md](RESULTS.md) D1/D2.

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
no new information. Per-step GRU accuracy was flat from step 1 (0.91, 0.91, 0.93, …).

**Half of that has since been superseded.** Re-run head to head under the current setup
(see "Four architectures, one regime"), the GRU is *not* beaten by the pair model — it
scores 0.9893 against 0.9853, and the two are inside seed noise of each other. The
"one-step pairs win" conclusion above was an artefact of the old regime; those numbers
predate the tilt-chain fix, sensor 2 and the joint classifier, and both arms were two-head
models.

What does survive, and is now confirmed twice on different simulations, is the **memoryless
finding**: the GRU's accuracy does not improve with context (0.9917 at t=1-5 against 0.9844
at t=31). The RNN is not worse, it is simply pointless here — it costs 4x the training
time to reach the same answer as a model that sees one frame.

The trajectory code is not in the repo; the walk semantics live in git history at
`3859109:loader.py` and were reused for the comparison above. Recurrence would only become
worth revisiting if the problem stopped being memoryless — much higher sensor noise, where
averaging over many readings wins, or a target that needs history such as cumulative
revolutions rather than angle mod 360°.

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

(Two-head numbers, measured when the bug was found. The joint classifier reaches 0.971
tilt on the fixed simulation.)

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

- **The travel tolerance, not the model, is what limits accuracy now.** Everything else in
  this list is second order next to it. It is a hardware question, not a modelling one.
- **Nothing is left to chase in the model at this tolerance.** 12 of 16 test units are
  perfect, rotation is exact, and the residual is two units that barely tilt. Further
  training units still help (the 32 → 128 curve has not flattened) but only by teaching
  the model to recognise low-travel units, which is treating the symptom.
- **No validation against real hardware, and the model now depends on sensor 2.** This is
  the biggest hole in the story: every number above is simulated, and the accuracy rests on
  a second sensor whose design coordinates are not trusted. `data/sensor1` holds measured
  CSVs and `load_measurement_data` still reads them, but nothing calls it — and it only
  covers sensor 1, so it can no longer feed this model at all. Real validation needs
  measurements from both sensors, paired into transitions (the CSV rows are single states).
- **The right-handedness flip is not implemented.** `setup_sensors` notes that the real
  Infineon part is left-handed and would need `B[2]` negated; the simulation never does it.
  Harmless while everything is simulated, a real discrepancy once measured data is used,
  and it now applies to both sensors.
- **`run.py` takes ~2.5 min.** Almost all of it is training on 128 units. Fine for a batch
  run, slow for iterating; drop `N_UNITS_TRAIN` while experimenting.
