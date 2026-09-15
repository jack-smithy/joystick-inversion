# Results

Every experiment run on this problem, with the regime each one belongs to. Numbers from
different regimes are **not** comparable — the regime table below is the first thing to
check before quoting anything.

[STATUS.md](STATUS.md) holds the conclusions and the current design. This file holds the
measurements they came from, including the ones that were superseded.

## Regimes

A "regime" is a fixed combination of simulation fidelity, sensors and tolerances. Changing
any of them invalidates comparison with earlier numbers.

| id | tilt chain | sensors | tolerances (pos / angle / tilt / pol) | test set | noise |
|---|---|---|---|---|---|
| **A** | buggy, dormant | 1 | 0.1 mm / 0.1° / none / none | 200 units | % of signal std |
| **B** | fixed | 1 | 0.1 mm / 1.0° / 1.0° / 15 mT | 4 units, 2208 | 0–0.5 mT |
| **C** | fixed | 2 | 0.1 mm / 1.0° / 1.0° / 15 mT | 16 units, 8832 | 0.1 mT |
| **D** | fixed | 2 | 0.2 mm / 2.0° / **1.0°** / 30 mT | 16 units | 0.5 / 2.0 / 4.5 mT |

Regime A is commit `3859109`. Its tilt-chain bug was dormant because tilt travel was never
perturbed there — which also makes it a much easier problem than B/C/D.
Signal std across states is **22.51 mT**, peak component 95.8 mT; use that to convert
absolute mT noise to a fraction of signal.

---

## Regime B — one sensor, fixed simulation

### B1. Two heads vs one joint classifier

Two-head model = 5-way tilt classifier + `(sin θ, cos θ)` regression. Joint = one 120-way
softmax over `tilt * 24 + angle_idx`. Same MLP (`6 → 128 → 128 → out`), same data,
8 train units, transition pairs `(B_start, B_end)`.

| model | tilt acc | angle acc |
|---|---|---|
| two heads | 0.926 | 0.891 |
| joint 120-way | **0.971** | **0.984** |

Superseded in scope, not in fact: see C6 — this gain does not survive into regimes C/D.

### B2. Noise sweep, joint classifier

| noise / mT | state acc | tilt acc | angle acc | within 1 | err mean |
|---|---|---|---|---|---|
| 0.00 | 0.977 | 0.977 | 0.986 | 0.986 | 1.841 |
| 0.01 | 0.975 | 0.975 | 0.988 | 0.988 | 1.535 |
| 0.05 | 0.975 | 0.975 | 0.987 | 0.988 | 1.610 |
| 0.10 | 0.971 | 0.971 | 0.984 | 0.984 | 2.235 |
| 0.50 | 0.901 | 0.902 | 0.977 | 0.977 | 2.812 |

`state acc == tilt acc` throughout. Samples with rotation wrong but tilt right: **0** at
0.00–0.1 mT, 2/2208 at 0.5 mT. Every failure is a tilt failure.

### B3. Field-space degeneracy, sensor 1 alone

For each of the 120 states, distance to the nearest *other* state on the nominal unit,
against how far that state's own reading wanders across 12 units. Ratio < 1 means the two
states are closer to each other than one state is to itself — unresolvable by any model.

| state | nearest | gap / mT | own spread / mT | ratio |
|---|---|---|---|---|
| east/18 | south/13 | 0.233 | 0.484 | **0.48** |
| east/10 | south/21 | 8.247 | 15.191 | 0.54 |
| east/6 | south/1 | 0.300 | 0.526 | 0.57 |
| west/10 | south/20 | 2.241 | 3.487 | 0.64 |
| south/13 | east/18 | 0.233 | 0.359 | 0.65 |
| south/1 | east/6 | 0.300 | 0.444 | 0.68 |

**9 of 120 states** had ratio < 1. Median nearest-neighbour gap 4.098 mT, median spread
1.420 mT. The model's actual confusions matched this list (`east/6 → south/1`,
`east/18 → south/13`), so the "near-antipodal tail" in the old notes was misdiagnosed: it
is a tilt/rotation coupling, not a 180° symmetry.

Corroborating: 34 of 65 misses were `noop` transitions, which are 22% of the data but were
52% of the errors — a noop gives two noisy reads of one state and cannot disambiguate.

### B4. Adding sensor 2 (geometry only, no model)

| feature set | states with gap < own spread | worst gap/spread ratio | median gap |
|---|---|---|---|
| sensor 1 only | 9 / 120 | 0.48 | 4.098 mT |
| both sensors | **0 / 120** | **2.08** | 8.548 mT |

### B5. Capacity and data ablation, sensor 1, 4-unit test set

| config | state acc | misses | noop | moved |
|---|---|---|---|---|
| 8 units / 128 wide / 150e | 0.9706 | 65 | 34 | 31 |
| 16 units | 0.9796 | 45 | 22 | 23 |
| 32 units | 0.9841 | 35 | 28 | 7 |
| 32 units / 300e | 0.9855 | 32 | 20 | 12 |
| 32 units / 256 wide | 0.9810 | 42 | 32 | 10 |
| 32 units / 3 layers | 0.9746 | 56 | 40 | 16 |
| 32 units / 256x3 / 300e | 0.9841 | 35 | 29 | 6 |

More units fixes the *moved* transitions (31 → 7) but not the noop ones (~28), which is
the signature of an information limit rather than a capacity limit.

---

## Regime C — both sensors, fixed simulation

### C1. Capacity and data ablation, both sensors, 4-unit test set

| config | state acc | misses | noop | moved |
|---|---|---|---|---|
| 8 units / 150e | 0.9823 | 39 | 12 | 27 |
| 8 units / 300e | 0.9814 | 41 | 9 | 32 |
| 16 units / 300e | 0.9855 | 32 | 8 | 24 |
| 32 units / 150e | 0.9851 | 33 | 12 | 21 |
| 32 units / 300e | 0.9864 | 30 | 10 | 20 |
| 32 units / 300e / 256 wide | 0.9891 | 24 | 6 | 18 |
| 64 units / 300e | 0.9941 | 13 | 4 | 9 |
| 64 units / 300e / 256 wide | 0.9977 | 5 | 1 | 4 |
| 128 units / 200e | 0.9973 | 6 | 2 | 4 |
| 128 units / 200e / 256 wide | **0.9982** | 4 | 1 | 3 |

Sensor 2 collapsed the noop misses (34 → 12 → 4), confirming B3/B4.

### C2. The same configs on a 16-unit test set

| config | state acc | misses / 8832 | noop | moved |
|---|---|---|---|---|
| 32 units / 256 / 300e | 0.9702 | 263 | 72 | 191 |
| 64 units / 256 / 300e | 0.9828 | 152 | 36 | 116 |
| 128 units / 256 / 200e | **0.9907** | 82 | 24 | 58 |

**The 4-unit test set was not trustworthy.** It scored the 64-unit model at 5 misses when
the honest figure over 16 units is 152. Most units are near-perfect and a couple are not,
so a small test population mostly measures which units it drew.

### C3. Where the misses actually are

Per test unit, 64-unit model, against that unit's drawn tilt travel:

| unit | misses | acc | south | north | east | west |
|---|---|---|---|---|---|---|
| 0 | 0 | 1.0000 | 4.38 | 4.24 | 4.62 | 3.18 |
| 2 | 3 | 0.9946 | 4.25 | 4.59 | 3.38 | 4.31 |
| 5 | 5 | 0.9909 | 3.90 | 3.93 | 3.83 | 5.47 |
| **6** | **40** | 0.9275 | 3.84 | 4.09 | **1.93** | 3.54 |
| 10 | 1 | 0.9982 | 3.68 | 2.27 | 4.73 | 5.71 |
| **13** | **97** | 0.8243 | 3.94 | 6.28 | **0.77** | 2.67 |
| (10 others) | 0–2 | ≥0.9964 | | | | |

- Units 6 and 13 are **12.5% of the test set and 90.1% of the misses**.
- State accuracy excluding them: **0.9981**.
- 127 of 152 misses are the single confusion `east → ground`.
- Rotation is wrong in only **3 of 152** misses.

Both units are exactly the ones whose east travel drew small (0.775° and 1.926° against a
4° nominal). A joystick that tilts 0.775° has barely left the ground state.

### C4. Tilt travel tolerance sweep

Everything else held fixed, 128 train units:

| `TILT_TOLERANCE` | state acc | misses / 8832 | test units with <2° travel |
|---|---|---|---|
| 1.00° | 0.9907 | 82 | 2 |
| 0.75° | 0.9980 | 18 | 1 |
| 0.50° | 0.9997 | 3 | 0 |
| 0.25° | **1.0000** | 0 | 0 |

**The travel tolerance is the entire remaining error budget.** Holding it to 0.5° buys more
than any model change tried.

### C5. Headline configuration

128 train units, 256 wide, 200 epochs, both sensors, 0.1 mT noise, 16-unit test set:

| state acc | tilt acc | angle acc | within 1 | err mean | misses |
|---|---|---|---|---|---|
| 0.991 | 0.991 | 1.000 | 1.000 | 0.008° | 82 / 8832 |

Per-class recall: south 1.00, north 1.00, east 0.96, west 1.00, ground 1.00.

Superseded as the repo default by D3, which is the same model on a harder problem.

---

## Regime C — four architectures head to head

Same 64 train / 16 test units, same trajectories, same noise draw, same width (256), same
optimizer, both sensors. Only the model varies. Scored on the newest frame at every
position t ≥ 1 (11904 positions), so model 4 always has a predecessor.

Needs trajectories rather than transition pairs — a pair gives an RNN nothing to
integrate. Random walks over the legal state graph, each drawing its own Dirichlet mix
over rotate/tilt/hold. The GRU is matched on **gradient steps**, not epochs (batch 32
gives it 48 steps/epoch against the MLPs' 192, so it runs 4× the epochs for the same
28800 updates).

### C6. Four architectures head to head, three model seeds

| model | context | seed 1 | seed 2 | seed 3 | mean | spread |
|---|---|---|---|---|---|---|
| 1. separate tilt/rotation heads | `B_t` | 0.9881 | 0.9875 | 0.9877 | 0.9878 | 0.0006 |
| 2. joint 120-way | `B_t` | 0.9866 | 0.9895 | 0.9837 | 0.9866 | **0.0058** |
| 3. joint causal GRU | `B_0..B_t` | 0.9884 | 0.9892 | 0.9902 | 0.9893 | 0.0018 |
| 4. joint 120-way | `B_t-1, B_t` | 0.9849 | 0.9835 | 0.9874 | 0.9853 | 0.0039 |

**Null result.** Range across model means is 0.0040; model 2's own seed-to-seed spread is
0.0058. One model's noise is wider than the gap between best and worst, and the ranking
reshuffles across seeds (model 2 places 3rd, 1st, 4th). All converged — training loss
< 0.02 for every model, so this is not under-training.

Two prior claims fail to reproduce here:

- Regime A's BENCHMARKS.md: separate heads "worse than one joint head, because the two
  errors are correlated". Not reproducible — they are nominally *ahead*.
- B1's +4.5 tilt points for the joint head. Real in regime B, gone in regime C.
- STATUS's older "one-step pairs beat the GRU" (0.973 vs 0.922) has the direction wrong
  under this setup; the GRU is nominally ahead.

### C7. Why nothing separated them

Miss overlap between models, Jaccard `|A ∩ B| / |A ∪ B|`, seed 1:

| | 1 | 2 | 4 | 3 |
|---|---|---|---|---|
| 1 | 1.00 | 0.56 | 0.54 | 0.62 |
| 2 | 0.56 | 1.00 | 0.57 | 0.74 |
| 4 | 0.54 | 0.57 | 1.00 | 0.63 |
| 3 | 0.62 | 0.74 | 0.63 | 1.00 |

Two models each missing ~150 of 11904 positions independently would overlap at a Jaccard
of about **0.006**. Observed is 0.54–0.74, roughly **100× chance**. They fail on the same
physical positions.

Misses by test unit, seed 1:

| unit | m1 | m2 | m4 | GRU |
|---|---|---|---|---|
| 6 | 37 | 59 | 63 | 59 |
| 13 | 58 | 54 | 60 | 54 |
| 2 | 16 | 25 | 21 | **0** |
| 10 | 14 | 9 | 7 | 10 |
| all others | 17 | 12 | 29 | 15 |

Units 6 and 13 carry **67–82%** of every model's misses; 4 of 16 units are perfect for all
four. Every architecture is pinned to the floor set by `TILT_TOLERANCE`.

Unresolved: the GRU took unit 2 from ~20 misses to **0** while matching elsewhere. Unit 2's
travel is unremarkable, so this may be genuine or seed luck. One seed cannot tell.

### C8. History buys nothing at this noise

GRU joint accuracy by position in the trajectory (context grows left to right):

| t | 1–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 | 31 |
|---|---|---|---|---|---|---|---|
| acc | 0.9917 | 0.9911 | 0.9828 | 0.9849 | 0.9875 | 0.9932 | 0.9844 |

Thirty frames score the same as one. Reproduces the older "Markovian in the field reading"
conclusion on the fixed simulation with both sensors. At 0.1 mT — 0.44% of the 22.5 mT
signal std — there is simply nothing for averaging to remove.

---

### C9. How little training data does each architecture need?

Regime C throughout (0.1 mm / 1.0° / 1.0° / 15 mT, 0.1 mT noise), training units swept
2 → 64, **validation set frozen** at the same 16 units, same trajectories, same noise draw
at every point. 3 seeds per point.

Two design choices shape how this reads:

- **Optimization budget fixed at 20000 gradient steps, not fixed epochs.** At a fixed
  epoch count the 2-unit run would get 1/32 the updates of the 64-unit run and the curve
  would measure optimizer budget as much as data. The cost is that small-data runs make
  far more passes over their data (3333 epochs at 2 units against 104 at 64) and overfit
  hard. That is real, and it hits all four equally.
- **Data is swept by unit count**, so units and frames scale together. The problem is
  cross-unit generalization, so "2 units" means two joysticks to generalize from.

| train units | frames | 1. sep heads | 2. joint `B_t` | 4. joint pair | 3. GRU |
|---|---|---|---|---|---|
| 2 | 1536 | 0.8673 | 0.8491 | 0.8282 | **0.8940** |
| 4 | 3072 | 0.8997 | 0.8950 | 0.9048 | **0.9486** |
| 8 | 6144 | 0.9239 | 0.9364 | 0.9497 | **0.9701** |
| 16 | 12288 | 0.9598 | 0.9616 | 0.9715 | **0.9794** |
| 32 | 24576 | 0.9702 | 0.9705 | 0.9749 | **0.9804** |
| 64 | 49152 | 0.9873 | 0.9858 | 0.9873 | **0.9897** |
| max seed spread | | 0.0254 | 0.0199 | 0.0162 | **0.0076** |

**The C6 null result was a saturation artefact.** At 64 units and 0.1 mT all four converge
to 0.986–0.990 and become indistinguishable — which is exactly the regime C6 measured. Cut
the data and they separate cleanly.

**The GRU wins at every data size, and by most at small data.** Its margin over the best
non-recurrent model: +0.027, +0.044, +0.020, +0.008, +0.006, +0.002 as units double. The
advantage decays monotonically from 4 units onward — the signature of a data-efficiency
benefit rather than a higher ceiling.

**Read as data required for a fixed target, the GRU is worth about 2x the training units:**

| target | 3. GRU | 4. pair | 2. joint `B_t` | 1. sep heads |
|---|---|---|---|---|
| 0.95 | ~4 units | ~8 units | ~12 units | ~13 units |
| 0.97 | ~8 units | ~16 units | ~32 units | ~32 units |

**The ranking inverts at small data.** At 64 units the pair model ties for best of the
non-recurrent three (0.9873); at 2 units it is **worst** (0.8282, below even separate
heads at 0.8673). Doubling the input width to 12 features costs more than the extra frame
returns when there are only two units to learn from. Separate heads are the opposite: best
of the three at 2 units, because 5 + 2 outputs are far easier to fit from 1536 frames than
120 classes are.

**The GRU is also the most stable.** Max seed spread 0.0076 against 0.016–0.025 for the
others — roughly a third the run-to-run variance.

**Nothing has saturated at 64 units.** All four curves are still climbing, and C2 confirms
it: 128 units reaches 0.9907. The "smallest set within 1 point of its own 64-unit score"
statistic the script printed is therefore uninformative for three of the four methods — it
returned "64 units" simply because they had not plateaued. The equal-accuracy table above
is the meaningful reading.

**Caveat, unchanged from D2:** the GRU's margin here may still be inflated by the
ground-start confound — every trajectory begins at `ground`, a cue the other three cannot
use. That confound is constant across data sizes, so it does not explain the *shape* of the
curve, but it may shift the GRU's whole line upward.

## Regime D — doubled tolerances, higher noise

Motivated by C6/C8: at 0.1 mT nothing was ambiguous enough to reward extra context, so no
architecture could distinguish itself. Noise is the ambiguity history *can* average away;
tolerance spread is a fixed per-unit bias it cannot. Doubling both separates those two
effects.

Tolerances doubled: 0.2 mm positions and size, 2.0° orientations, 30 mT polarization.

**`TILT_TOLERANCE` deliberately held at 1.0°.** Doubling it too was tried and abandoned
before scoring: at 2.0° on a 4° nominal travel, ~16% of units tilt under 2° and ~2% travel
backwards, a negative draw inverting the tilt so that a commanded east is physically a
west. C3/C7 already showed that low-travel units dominate every model's misses equally, so
raising that floor would swamp the very differences this run exists to resolve. Holding it
keeps the added difficulty on the axes that can actually discriminate between models.

Test population under this regime is unchanged from C: 2 of 16 units with any travel
< 2°, 0 inverted.

### D1. Four architectures across noise

Mean joint accuracy over 3 model seeds. Same 64 train / 16 test units, same trajectories,
same noise draw, both sensors; the clean fields are built once and only the noise draw
changes between columns.

| model | context | 0.5 mT (2.2%) | 2.0 mT (8.9%) | 4.5 mT (20%) |
|---|---|---|---|---|
| 1. separate tilt/rotation heads | `B_t` | 0.9129 | 0.7534 | 0.5216 |
| 2. joint 120-way | `B_t` | 0.9091 | 0.7505 | 0.5722 |
| 4. joint 120-way | `B_t-1, B_t` | 0.9196 | 0.7595 | 0.5938 |
| 3. joint causal GRU | `B_0..B_t` | **0.9341** | **0.7895** | **0.6445** |
| max seed spread | | 0.0079 | 0.0081 | 0.0094 |
| gap, best − worst | | 0.0250 | 0.0390 | 0.1228 |

**The null result in C6 was an artefact of the noise level, not a property of the
architectures.** At 0.1 mT (0.44% of signal) nothing was ambiguous enough to reward extra
context. Here every gap exceeds the seed spread, and the ranking is identical at all three
noise levels: **GRU > pair > single-frame joint**.

Three findings, all outside seed noise:

1. **Context helps, and the benefit grows with noise.** The GRU beats the best
   non-recurrent model by +1.45, +3.00 and +5.07 points as noise rises. One extra frame
   (model 4 over model 2) is worth +1.05, +0.90, +2.16.
2. **The joint head beats separate heads, but only when the problem is hard.** At 0.5 and
   2.0 mT they tie (0.9129 vs 0.9091, 0.7534 vs 0.7505 — inside spread). At 4.5 mT the
   joint head wins by **+5.06 points** (0.5722 vs 0.5216). Regime A's claim that separate
   heads are "worse than one joint head, because the two errors are correlated" is
   vindicated — conditionally on the noise being high enough to matter.
3. **Separate heads degrade worst.** Model 1 goes from best-of-the-single-frame models at
   0.5 mT to clearly worst at 4.5 mT. The `(sin θ, cos θ)` regression head has no way to
   express uncertainty, so it degrades ungracefully as the reading gets noisier.

### D2. But the GRU's advantage is not long-range integration

GRU joint accuracy by position in the trajectory (context grows left to right):

| noise | t=1–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 | 31 |
|---|---|---|---|---|---|---|---|
| 0.5 mT | 0.940 | 0.932 | 0.914 | 0.930 | 0.938 | 0.932 | 0.919 |
| 2.0 mT | 0.809 | 0.796 | 0.793 | 0.788 | 0.793 | 0.780 | 0.799 |
| 4.5 mT | 0.665 | 0.651 | 0.657 | 0.658 | 0.617 | 0.621 | 0.622 |

**Flat to declining at every noise level.** If the GRU were accumulating evidence about the
current state, later frames — with more history behind them — would score better. They do
not. Whatever the GRU is exploiting, it is available within the first few frames.

**Known confound, not yet controlled.** `walk()` starts every trajectory at `ground`, so
early frames are genuinely easier and the GRU can exploit "trajectories begin at ground" —
a cue models 1, 2 and 4 cannot see, since they do not know their position in the sequence.
The declining curve is consistent with a decaying start-state advantage masking any real
accumulation. Regime A's benchmark avoided this with a 32-frame warm-up before scoring.

A partial check from the data above: at late frames, where the start advantage has decayed,
the GRU still leads the pair model at high noise (t≥26: 0.62 against 0.5938 overall at
4.5 mT) but the lead nearly vanishes at low noise (0.92–0.93 against 0.9196 at 0.5 mT).
Per-position accuracy was not recorded for the non-recurrent models, so this cannot be
decomposed cleanly.

**What this means for the D1 ranking:** the GRU's win is real on these positions, but part
of it may be start-state knowledge rather than noise averaging. Isolating it needs either
random start states or a warm-up before scoring. Until that is run, treat the GRU's margin
as an upper bound.

### D3. Headline configuration under the harder regime

`run.py` as committed: 128 train units, 256 wide, 200 epochs, both sensors, 0.5 mT noise,
16-unit test set. This is the same model as C5 on a harder problem, not a worse model.

| state acc | tilt acc | angle acc | within 1 | err mean | misses |
|---|---|---|---|---|---|
| 0.941 | 0.941 | 0.999 | 1.000 | 0.036° | ~521 / 8832 |

Per-class recall: south 0.98, north 0.94, east 0.92, west 0.96, ground 0.92.
Per-class precision: south 0.95, north 0.97, east 0.97, west 0.94, ground 0.91.

**The failure changed shape, not just size.** Under C5 the deficit was east alone (0.96,
every other class 1.00) and traced to two low-travel units (C3). Here it is spread across
all five classes with **ground weakest**, and ground is also the least precise — it is being
over-predicted. Doubling position, alignment and polarization spread produces an evenly
distributed failure mode rather than a few pathological units, so C3's low-travel
explanation covers regime C's errors and not these.

Rotation survives the harder regime nearly intact: `angle acc` 0.999 and `within 1` 1.000,
so every rotation miss is a single step. `state acc == tilt acc` still holds — every failure
is a tilt failure. Worst rotation miss is 1–7 steps by tilt state, against 0 for four of
five classes under C5.

---

## Reference: regime A (commit `3859109`)

Kept for provenance. Not comparable to anything above: no tilt-travel tolerance, 0.1°
orientations, no polarization spread, sensor 1 only. Joint accuracy, noise as % of signal
std:

| method | 0% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| joint MLP, W=1 | 99.63 | 93.71 | 81.87 | 67.57 | 50.63 |
| joint GRU, W=1 | 99.61 | 93.60 | 81.93 | 67.45 | 50.73 |
| joint GRU, W=2 | 99.59 | 96.53 | 87.64 | 74.88 | 59.43 |
| joint GRU, W=5 | 99.91 | 97.31 | 90.24 | 79.77 | 65.72 |
| joint GRU, W=16 | 99.95 | 97.69 | 91.60 | 82.08 | 69.75 |

Note the GRU's gain over W=1 grows with noise — 0 points at 0%, 19 points at 20%. That is
the effect regime D is built to test for under the current simulation.

Separate tilt/angle heads were run in regime A but the numbers were **not recorded**; only
the claim that they were "worse than one joint head" survives, and C6 does not reproduce it.

---

## Where the code lives

`run.py` reproduces C5 and D-regime headline numbers. The ablations, degeneracy analysis,
tolerance sweep and four-model comparison were scratchpad scripts, not repo code; the
trajectory-walk semantics were recovered from `3859109:loader.py`.
