# Development log

How Claw Crew got from a single-file prototype to the module layout in
`README.md`, and — more usefully — which numbers forced which decisions. Every
table here is measured output from the run that motivated the change, not an
estimate.

The short version: the refactor itself was mechanical and reproduced the
prototype bit for bit. The real work was **M2**, where the prototype's tracking
was 3× outside the milestone's budget for two structural reasons that no amount
of test-tuning could hide.

---

## Stage 0 — environment

The default interpreter here is Python 3.14.7, which has no `mujoco` wheels.
Pinned the venv to Python 3.12 instead:

```
mujoco 3.12.0   pygame 2.6.1   numpy 2.5.2   pytest 9.1.1
```

Baseline check on the untouched prototype before changing anything, so there
was a known-good reference to diff against:

```
$ python arm_game.py --headless
cube spawn [ 0.261 -0.151  0.03 ]
lift cube_z 0.301 held: True
final cube [0.21  0.174 0.024] score 1 -> SCORED
```

---

## Stage 1 — the refactor (`scene` → `kin` → `game` → `input` → `render`)

Split in dependency order, verifying each module against the prototype as it
landed rather than at the end.

| step | check | result |
|---|---|---|
| `scene.py` | model compiles | `nq=15 nv=14 nu=8 ngeom=11` |
| `scene.py` | constraint 4 holds | finger gap 68 mm vs 48 mm cube |
| `kin.py` | HOME IK converges | **0.092 mm**, `q = [0, 0.181, 1.167, 1.569, 0, 0.224]` |
| `game.py` | matches the prototype | same spawn, `ncon = 0`, `score 1` |

That `kin.py` row is also the first independent confirmation of **constraint 5**:
`j2 + j3 + j4 + j6 = 0.181 + 1.167 + 1.569 + 0.224 = 3.141 = π`. The pitch joints
all turn about y, so summing them to π gives `R_y(π) = diag(-1, 1, -1)` — the
wrist-neutral tool frame the convention claims. `test_m1` now asserts this
against the model instead of trusting the derivation.

### One structural decision

The spec calls for readers producing a common `ControlInput`, but the obvious
wiring (`game.step(ci)`) would make `game.py` import `input.py` — which imports
pygame — and quietly break the hard rule.

Resolved by moving the camera-relative rotation **onto the input side**
(`ControlInput.world_xy(cam_az)`) and having `Game.step` take plain world-frame
floats. `game.py` now imports neither `input.py` nor `render.py`, so the rule
holds by construction rather than by discipline. `test_no_pygame.py` enforces it
in a subprocess where `import pygame` raises — including for `main.py --headless`.

---

## Stage 2 — M1, and why `ncon == 0` isn't enough on its own

M1 passed immediately (12 tests). But `d.ncon == 0` at rest is a weak guard: the
cube spawns 6 mm above the floor, so a *totally* broken model could still read
zero. Added two stronger checks alongside it:

- every geom that isn't finger/hand/cube/floor must have **both** `contype` and
  `conaffinity` at 0 — this fails loudly if someone "tidies up" constraint 1;
- after a second of settling, no contact may involve a structural link body.

---

## Stage 3 — M2, the one that didn't pass

The milestone asks for **< 10 mm** end-effector tracking on a swept target.
Measured the prototype first rather than picking a sweep speed that would pass:

| sweep speed | mean error | max error |
|---|---|---|
| 0.45 m/s (full stick) | 84.8 mm | 90.5 mm |
| 0.30 m/s | 65.0 mm | 70.5 mm |
| 0.15 m/s | 36.4 mm | 41.3 mm |
| 0.05 m/s | 13.6 mm | 17.4 mm |

Error scaling linearly with speed is the signature of a *lag*, not a tuning
wobble — and it never reached 10 mm even at a tenth of full stick. Two separate
sources, found by decomposing the error at 0.30 m/s:

```
IK command vs target :  8.36 mm     <- the commanded pose trails the goal
measured vs IK command: 21.36 mm    <- the joints trail the command
total                : 28.99 mm
static hold (parked) :  4.49 mm     <- gravity droop, speed-independent floor
```

### Cause 1 — a proportional chase always trails a moving goal

`IKController` drove the commanded twist at `kp · error`. Against a goal moving
at `v` that settles at a constant `e = v / kp` — 0.30 / 6.0 = 50 mm — forever.

Fix: **velocity feedforward**, `kp · error + goal_velocity`. This is safe
precisely *because of* constraint 6: the goal is rate-limited, so differencing it
is bounded, and the existing twist clamp is the backstop. It does **not** touch
constraint 2 — the Jacobian and error are still evaluated on the scratch
`MjData`, and nothing measured enters the loop.

Result: 65.0 → 28.99 mm at 0.30 m/s. Better, still not passing.

### Cause 2 — the position servo trails its own command

The remaining 21 mm was actuator following error, which for a position servo is
roughly `kv · q̇ / kp`. Swept the gains against **all three** M2 criteria at once,
so tracking couldn't be bought with stability:

| kp | kv | peak \|qvel\| | ramp err | sweep mean | sweep max | static |
|---|---|---|---|---|---|---|
| 800 | 60 | 1.000 | 7e-14 | 28.99 | 36.08 | 4.49 |
| 1500 | 40 | 1.000 | 7e-16 | 15.01 | 22.65 | 2.39 |
| **2500** | **40** | **1.019** | **2e-16** | **11.79** | **19.42** | **1.44** |
| 2500 | 25 | 1.151 | 0 | 10.18 | 17.83 | 1.44 |
| 4000 | 25 | 1.247 | 2e-16 | 9.06 | 16.61 | 0.90 |

Gain alone plateaus around 10 mm, because by then the 8 mm IK-command lag
dominates. So the last term had to come from the solver.

### Cause 3 — DLS damping *is* lag

`Arm.dls` damping is what makes the solver undershoot the twist it was asked
for, and that undershoot shows up directly as end-effector lag. At the chosen
gains:

| `lam` | mean | max | peak actuator force |
|---|---|---|---|
| 0.10 | 11.79 mm | 19.42 mm | 12.9 N / 200 N |
| 0.06 | 7.34 mm | 13.45 mm | 13.1 N / 200 N |
| **0.04** | **4.56 mm** | **8.22 mm** | **13.2 N / 200 N** |

Safe here because the Cartesian target is clamped well inside `REACH_MAX`, so
the tracking loop never sits on the singularity that damping exists to survive.
The force column was the reason to check: 13 N of a 200 N `forcerange` means the
stiffer gains buy tracking without saturating anything.

### Where it landed

**65 mm → 8.2 mm peak.** Recorded as constraints 7–9 in `CLAUDE.md`, because
each one looks like gratuitous tuning to a future reader and all three are
load-bearing.

Deliberately **not** taken further: `ik_kp = 30` reaches 5.6 mm, but it passes
with margin at the settled value of 6.0 and a stiffer teleop loop is a real
feel regression. Tuning stopped at "passes with headroom", not "minimised".

### Regression check

Retuning the actuators risked the validated grasp, so M1 and the grasp were
re-run immediately: **M1 12/12**, grasp **8/8 seeds** before writing any more
tests.

---

## Stage 4 — M3–M7

| milestone | notes | tests |
|---|---|---|
| M3 grasp | 12 seeds, all must pass; plus a carry test and a joint-limit check | 19 |
| M4 rules | end-to-end score, and each scoring condition isolated | 17 |
| M5 input | fake pad, no hardware; deadzone continuity, rising-edge latch | 33 |
| M6 render | thin by design — asserts it renders and *doesn't mutate the sim* | 5 |
| M7 record | LeRobot v2.1 parquet + PNG + meta sidecars | 7 |

Two bugs found, both in tests rather than in the code under test — worth
recording because they are the kind that silently pass:

- **M4** asserted a partial reset preserves the clock, but never stepped, so
  `time_left` was still exactly `ROUND_SECONDS`. The assertion was vacuous;
  fixed by burning ticks first.
- **M5** used `np.cross` on 2-D vectors, which **numpy 2 removed**. Replaced with
  the explicit z-component.

M7 was verified end to end, not just at unit level: `--headless --record` wrote
**1680 ticks and 1680 PNGs** in full LeRobot layout. `pyarrow`/`pillow` are
imported lazily so the core keeps three dependencies.

---

## Final state

```
$ python -m pytest -q
103 passed in 8.25s
```

| milestone | file | tests |
|---|---|---|
| M1 scene + kinematics | `test_m1_scene_kin.py` | 12 |
| M2 stable tracking | `test_m2_tracking.py` | 6 |
| M3 grasping | `test_m3_grasp.py` | 19 |
| M4 game rules | `test_m4_rules.py` | 17 |
| M5 input layer | `test_m5_input.py` | 33 |
| M6 render + HUD | `test_m6_render.py` | 5 |
| M7 recording | `test_m7_record.py` | 7 |
| hard rule | `test_no_pygame.py` | 4 |

## Known limits

- Tracking degrades with speed by design; the 10 mm budget is verified at
  0.05/0.15/0.30 m/s. Full-stick 0.45 m/s is separately asserted only to stay
  **bounded** (< 20 mm), not under 10 mm.
- The ~1.4 mm static floor is gravity droop against a position servo. Removing
  it needs gravity compensation, not more gain.
- `--record` writes one PNG per tick, so a 90 s round is ~9000 files. Fine for
  collecting demos, but video encoding would be the real answer for volume.
- M6 is the one layer without meaningful headless coverage. It is kept logic-free
  so that matters as little as possible.
