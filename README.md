# Skillcrane

> **Teleoperated manipulation in sim, and the skills it collects.**
>
> Gamepad teleoperation for MuJoCo manipulation that records LeRobot
> demonstrations — natively, or on robosuite, Meta-World, Fetch and LIBERO.

A 6-DoF robot arm you teleoperate with an 8BitDo gamepad, and the rig that
records what you did so a policy can learn it.

A *skill crane* is the trade name for an arcade claw machine, which is what this
started as: grab the cube, drop it in the zone, beat the clock. Building that
well meant solving the hard part of a teleop rig anyway — an arm that tracks
your hand smoothly enough to do fine manipulation with. Once that worked, every
round played was already a demonstration, so the same rig now logs each control
tick as an imitation-learning episode. The game is the interface; the dataset is
the output.

Runs under plain `python` on macOS/Apple Silicon — MuJoCo renders offscreen and
pygame owns the window, so nothing fights over the main thread.

It also drives **existing benchmarks** — robosuite, Meta-World, Fetch and
LIBERO — with the same gamepad and the same recorder, so demonstrations are
collected against standard tasks rather than only our own.

Project page: [`docs/index.html`](docs/index.html) · benchmarks:
[`BENCHMARKS.md`](BENCHMARKS.md) · development history: [`DEVLOG.md`](DEVLOG.md)
· physics constraints: [`CLAUDE.md`](CLAUDE.md)

## Install

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:

```sh
CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
conda activate skillcrane
```

The `CONDA_SUBDIR` prefix matters only if `conda info` reports `platform :
osx-64` — an Anaconda from before Apple Silicon builds x86_64 environments, and
mujoco ships no x86_64 macOS wheel, so the install fails building it from
source. `environment.yml` explains it and how to make the setting stick.

## Run

```sh
python main.py                  # play
python main.py --list-envs      # benchmark suites, their tasks, and what's installed
python main.py --env robosuite:Lift          # a suite and one of its tasks
python main.py --env libero                  # a suite, on its first task
python main.py --headless       # scripted pick-and-place, no window, exits 0 on a score
python main.py --seed 7         # fixed cube spawns
python main.py --record         # collect the session into runs/ (off unless asked)
python main.py --record DIR     # ... into DIR instead
python main.py --record --record-views all   # ... logging every camera, not just one
python main.py --window 1600x1000            # open a bigger window (drag to resize too)
python main.py --record --record-size 640x480   # bigger frames in the dataset
python main.py --eval 12        # score a policy over 12 seeds, no window
python main.py --eval 50 --record runs/         # ... and collect them as a dataset
python main.py --eval 1 --policy replay:runs/:0 # replay a recorded episode
python gamepad_probe.py         # see what your pad reports, and how it is read
```

## Controls

| | gamepad | keyboard |
|---|---|---|
| move gripper (camera-relative) | left stick | `WASD` |
| raise / lower | right stick Y | `Q` / `E` |
| rotate wrist | right stick X | `Z` / `C` |
| gripper | `A` (toggle) | `SPACE` (hold) |
| orbit camera | `LB` / `RB` | `←` / `→` |
| zoom camera in / out | `RT` / `LT` | `=` / `-` |
| camera follow on / off | `B` | `F` |
| cycle view layout | `X` | `V` |
| switch **task** (inside the suite) | d-pad ← → | `,` / `.` |
| switch **suite** (the benchmark) | `Back` / `Start` | `[` / `]` |
| reset round | `Y` | `R` |
| quit | — | `ESC` |

The pad is read through **SDL's game-controller layer**, which recognises the
device from SDL's own database and reports a standard layout — A is A, LB is LB
— whatever pairing mode it is in. That is what makes an 8BitDo work without
hand-mapping; the two numberings really do disagree, SDL calling the shoulders
9 and 10 where a raw HID pad reports 4 and 5.

A pad SDL does not know falls back to raw indices, and *those* are the ones that
differ per pairing mode (Apple = hold Start+A on power-on, D-input = Start+B;
some models, the Ultimate 2C included, use a switch on the back). Run
`gamepad_probe.py`: it says which path is live, shows what `input.py` makes of
every control next to the raw numbers, and if the layout is raw, prints the
config block to edit at the top of `input.py`.

**Only the native arm has a camera you can move.** The benchmark suites render
from fixed cameras their own model defines, so orbit, zoom and follow do nothing
there — by design, not because the buttons are unmapped.

## Views

`X` on the pad (or `V`) cycles three layouts:

| layout | what it shows |
|---|---|
| `single` | the operator's own view, full width |
| `inset` | that view, with the other cameras stacked down the right edge |
| `grid` | up to four cameras tiled 2×2, all the same size |

The native arm has four: `scene` (the free camera you drive), `wrist`
(eye-in-hand, mounted beside the gripper), `front`, and `top`. A benchmark
offers whatever cameras its own model has — robosuite adds `robot0_eye_in_hand`,
`frontview` and `birdview` where the task defines them, LIBERO offers the
cameras it rendered into the observation, and Meta-World and Fetch have exactly
one each. An environment with one camera always draws it full width, so cycling
the layout there does nothing.

Each view is rendered at exactly the panel size it will be drawn into, so
`inset` costs one big frame plus three small ones rather than four big ones. The
cameras are declared in the MJCF (`scene.py`), not built in code, so anything
else that loads the model sees the same views.

### Window size

The window opens at **1280×860** and is **resizable** — drag a corner, or start
with `--window 1600x1000`. Nothing is upscaled when you do: the layout recomputes
its panel rects, the environment is asked for frames at exactly those sizes, and
the HUD's type, band height and margins scale with the window, so a 2560-wide
window is a 2560-wide render rather than a stretched 1280 one.

The one ceiling is MuJoCo's offscreen framebuffer, declared in the MJCF at
2048×1152 (`scene.OFF_W/OFF_H`). MuJoCo *raises* rather than downscaling when
asked for more, so panels are clamped to it and scaled up beyond that point —
a window dragged across a 4K display degrades in sharpness instead of ending the
round.

Recorded frames are separate and do not follow the window: `--record-size WxH`
(default 320×240) sets them, because one dataset column has one image shape.

### Driving the operator's camera

All three views are framed for *control*, which means close:

- **`scene`** sits 0.9 m out and **follows the work** — the midpoint between the
  gripper and whatever it is reaching for (the cube, or the drop zone once it is
  holding one). `RT`/`LT` (or `=`/`-`) dolly between 0.45 m and 2.2 m, `LB`/`RB`
  orbit, and `B` (or `F`) freezes the follow where it is if you would rather aim
  the camera yourself.
- **`front`** and **`top`** are aimed at the middle of the spawn arc rather than
  at the horizon, and sit as close as they can while still showing every cube
  spawn and the whole drop zone — `test_m1` checks that against the model's own
  camera matrices, so pulling them in further fails a test rather than quietly
  cropping the cube.
- **`top`** is turned 90°, screen-right being −y. The workspace is wide in y and
  a view panel is wide too, so putting them on the same axis is worth 0.33 m of
  camera height for free.

## Layout

| file | what it holds |
|---|---|
| `scene.py` | MJCF as a string + model constants |
| `kin.py` | `Arm` (FK/Jacobian/pose error), `IKController`, `down_R` |
| `game.py` | physics, scoring, spawn logic, the demo waypoints — **no pygame** |
| `input.py` | `GamepadReader`/`KeyboardReader` → `ControlInput` |
| `render.py` | offscreen render + HUD |
| `recorder.py` | LeRobot-format episode logging |
| `policy.py` | scripted / replayed / learned policies + rollout and eval — **no pygame** |
| `benchmarks/` | adapters: robosuite, RoboCasa, Meta-World, Fetch, LIBERO |
| `main.py` | CLI entry point |

`game.py` never imports `input.py` or `render.py`, so the whole sim/rules layer
runs headless. `policy.py` is on that side of the line too: it is what a data
pipeline calls, so it must run where there is no display. See `CLAUDE.md` for the physics constraints that must not be
"simplified" away.

## Tests

```sh
python -m pytest -q          # 230 tests, ~15 s, no display and no gamepad needed
```

Benchmark tests skip cleanly when the benchmark isn't installed, so a bare
checkout stays green (230 passed / 17 skipped).

Physics tests assert on numbers — contact count, joint velocity, tracking error,
cube height, score — and the grasp tests are parametrised over 12 random spawns,
all of which must pass.

## Data collection

Data collection is **off by default**. `--record` turns it on (bare, it collects
into `runs/`; `--record DIR` picks the directory) and writes one row per 100 Hz
control tick — `observation.state` (6 joints + gripper + cube xyz), `action`
(the operator's stick input), what the rules made of it (`next.reward` is 1.0 on
the tick that scored, `next.done` marks the last tick) and a rendered frame — in
LeRobot v2.1 layout:

```
DIR/data/chunk-000/episode_000000.parquet
DIR/images/observation.images.scene/episode_000000/frame_%06d.png
DIR/meta/{info,episodes,tasks,episodes_stats}.{json,jsonl}
```

which is the shape `lerobot` expects for training an ACT policy.

`--record-views` chooses the cameras: `main` (the default — the operator's own
view), `all`, or a comma-separated list such as `scene,wrist`. Each view becomes
its own `observation.images.<view>` column with its own PNG per tick, which is
how LeRobot describes a multi-camera rig.

Recorded frames are **not** the ones on screen. They are rendered separately at
a fixed size — `--record-size`, 320×240 by default — because a dataset column has
one image shape for the whole episode, while the operator can change the layout
mid-round and resize the window. Bigger frames cost disk in a straight line: the
7.4 s headless episode is 29 MB with one camera at 320×240 and 94 MB with two at
640×480.

One thing to know before training on them: `scene` is the **operator's** camera,
so it orbits, zooms and follows the gripper during the episode, and none of that
is in `observation.state`. That is fine for a human watching a replay and it is
a moving viewpoint for a policy. `wrist`, `front` and `top` are fixed to the
robot and the world respectively — record those (`--record-views wrist,front`)
when you want a stationary camera.

### Example: collect an episode and read it back

Recording needs the two optional dependencies, then it is just a flag:

```sh
pip install pyarrow pillow
python main.py --record                  # play a round; ESC ends the episode
# recording to runs/
# ...on exit: recorded <n> ticks -> runs/data/chunk-000/episode_000000.parquet
```

`--headless` records the scripted pick-and-place instead, which is the quickest
way to get a well-formed episode without touching a gamepad:

```sh
python main.py --headless --seed 2 --record runs/
# recorded    736 ticks -> runs/data/chunk-000/episode_000000.parquet (success=True)
```

Read it back with plain pyarrow — no `lerobot` install needed to check the file:

```python
import pyarrow.parquet as pq
t = pq.read_table("runs/data/chunk-000/episode_000000.parquet")
print(t.num_rows, t.column_names)
# 736 ['observation.state', 'action', 'timestamp', 'frame_index',
#      'episode_index', 'index', 'task_index', 'next.reward', 'next.done',
#      'observation.images.scene']
t.slice(300, 1).to_pylist()[0]
# {'observation.state': [-0.024, 0.246, 1.04, 1.758, 0.071, 0.102,   # j1..j6
#                        0.016, 0.296, -0.006, 0.297],               # grip, cube xyz
#  'action': [0.08, 0.311, 0.751, 0.0, 1.0],                         # dx dy dz dyaw grip
#  'timestamp': 3.0, 'frame_index': 300, 'episode_index': 0,
#  'next.reward': 0.0, 'next.done': False,
#  'observation.images.scene': 'images/.../frame_000300.png'}
```

Read the `action` row next to the `observation.state` row: those four numbers are
the stick deflection that moved the arm to that state. They have to be, or the
episode is untrainable — see [Policies, replay and eval](#policies-replay-and-eval)
for the check that they are.

The same run with every camera gives one image column per view:

```sh
python main.py --headless --seed 2 --record runs/ --record-views all
# recording views: scene, wrist, front, top
# recorded    736 ticks -> runs/data/chunk-000/episode_000000.parquet
ls runs/images
# observation.images.front  observation.images.scene
# observation.images.top    observation.images.wrist
```

`meta/` carries what a `LeRobotDataset` reads at load time — `info.json` (fps
100, the feature schema, `codebase_version` v2.1), `tasks.jsonl` (the language
instruction), `episodes.jsonl` (one row per episode with its length **and whether
it worked**), and `episodes_stats.jsonl` (per-field mean/std/min/max, which is
what a policy's input normalisation uses):

```sh
cat runs/meta/episodes.jsonl
# {"episode_index": 0, "tasks": ["Pick up the cube and place it in the target zone."], "length": 736, "success": true, "score": 1}
```

`success` and `score` are the ones to read before training. The usual ACT recipe
learns from *successful* demonstrations only, and a round here can score more
than once — the cube respawns, so a good operator scores several times in 90 s —
which is why the count is there beside the boolean. Filtering is a read of one
small file rather than a scan of every frame of every round.

Two things to know before collecting in bulk:

- **Episodes accumulate; they do not overwrite.** The episode index comes from
  reading the directory, not from a counter, so recording into the same `DIR`
  across separate runs of the program appends `episode_000001`,
  `episode_000002`, … and `meta/` stays consistent with them.
- **Frames dominate the size.** One 320×240 PNG per 100 Hz tick is ~40 KB, so
  the 7.4 s episode above is 29 MB for one camera and 84 MB for all four; at
  640×480 two cameras cost 94 MB for the same 7.4 s, and a 90 s round is ~360 MB
  per camera at 320×240. `EpisodeRecorder` accepts frames at a lower rate (pass
  `None` on the ticks you skip; each image column repeats its last frame to stay
  1:1 with the ticks), and `--record-views` is the other dial.

## Policies, replay and eval

`--record` fills a dataset; `--eval` is the other end of the same pipe. It runs a
policy over N seeds with no window and prints a per-seed table:

```sh
python main.py --eval 6 --seed 0
# eval       native  policy scripted  seeds 0..5
#   seed   ticks  score   t_score  result
#      0     717      1     5.30s  ok
#      1     714      1     5.27s  ok
#      2     736      1     5.49s  ok
#      3     786      1     5.99s  ok
#      4     791      1     6.03s  ok
#      5     758      1     5.70s  ok
# success 6/6 = 100%
```

Seeds are consecutive from `--seed`, so two policies can be compared on the same
worlds — an average over different spawns is not a comparison. The command exits
non-zero unless every episode succeeded.

`--policy` chooses what drives it:

| `--policy` | what it is |
|---|---|
| `scripted` (default) | the built-in demonstrator: approach → grasp → carry → release |
| `replay:DIR[:EP]` | play back the `action` column of a recorded episode |
| `act:PATH` | a LeRobot checkpoint — **untested here**, `lerobot` is not installed on this machine |

Every one of them emits the same five numbers the sticks do — `dx, dy, dz, dyaw,
grip` — and goes through the same `TeleopEnv.step`, so a demonstration, a replay
and a trained network are scored by identical rules.

Add `--record` to collect the rollouts, one episode per seed. This is how you get
a few hundred demonstrations without touching a gamepad:

```sh
python main.py --eval 50 --record runs/ --record-views wrist,front
```

### Replay is the dataset integrity check

This is the reason `ReplayPolicy` exists. A recording can be perfectly
well-formed — every column present, every shape right, every PNG on disk — and
still be untrainable, because the numbers in `action` are not the numbers that
produced the motion in `observation.state` beside them. Nothing else in the
project can see that. Replay can:

```sh
python main.py --eval 2 --seed 0 --record /tmp/d      # collect seeds 0 and 1
python main.py --eval 1 --seed 0 --policy replay:/tmp/d:0
# success 1/1 = 100%     <- same ticks, same score, same instant it scored
python main.py --eval 1 --seed 5 --policy replay:/tmp/d:0
# success 0/1 = 0%       <- and it can fail, so the check above means something
```

Replaying an episode into the seed it was collected at reproduces it to within
float32 rounding (max state divergence 1.6e-7 over ~700 ticks; the cube lands in
exactly the same place). Replaying it into a different world misses, which is
what stops the check from being vacuous.

It was worth building: `--headless --record` used to log `[0, 0, 0, 0, grip]` on
every tick of a completed pick-and-place. The scripted demo moved the Cartesian
target directly — correct for a physics test, silently useless as a
demonstration — so every headless dataset was a constant action column beside a
moving arm. The file looked fine. `main.py --headless` now drives through
`policy.ScriptedPickPlace`, which walks the same waypoints through the sticks.

## Benchmarks

Two words, kept apart everywhere in this project:

- a **suite** is a benchmark — `native`, `robosuite`, `robocasa`, `metaworld`,
  `fetch`, `libero`. A different robot, a different wrapper, a different
  observation width.
- a **task** is one setting inside a suite — `Lift`, `Stack`, `PickPlaceCan`.
  Same robot, same adapter, different scene.

`--env` takes both: `--env robosuite:Lift` is a suite and a task, `--env libero`
is a suite on its first task. `--list-envs` prints every suite, its tasks, and
whether it is installed.

Verified working headless on Apple Silicon: **robosuite** (Lift, Stack,
PickPlaceCan, Door), **Meta-World** (50 tasks), **Fetch**, and **LIBERO** (130
tasks, in its own environment). **RoboCasa** is registered and goes through the
robosuite adapter — its kitchens are robosuite environments once `robocasa` is
imported — but it is not installed here, so that path is wiring, not a measured
result.

```sh
pip install -r requirements-benchmarks.txt      # robosuite, Meta-World, Fetch
CONDA_SUBDIR=osx-arm64 conda env create -f environment-libero.yml   # LIBERO (robosuite 1.4)
pip install git+https://github.com/robocasa/robocasa.git \
  && python -m robocasa.scripts.download_kitchen_assets                # RoboCasa
```

They need `mujoco<3.12` — see [`BENCHMARKS.md`](BENCHMARKS.md) for why, plus the
gripper-sign table and two other install landmines.

### Switching without restarting

Two rings, because they are two different moves.

`[` / `]` on the keyboard, or `Back` / `Start` on the pad, cycles the **suite**
— the benchmark itself. Only suites that are actually installed are in the ring,
which is why a bare checkout says *"native is the only benchmark suite
installed"* when you press it: install one and it joins the ring. `--list-envs`
shows what exists and what installs it. The task resets to the new suite's
default, because a task id from one suite means nothing in the next.

`,` / `.` on the keyboard, or the **d-pad**, cycles the **task** inside the
suite you are on: `Lift` → `Stack` → `PickPlaceCan` → … The ring is that suite's
task list in `--list-envs`. The native arm has one scene, so it says so instead
of rebuilding the same environment.

The HUD names both, next to *both* controls that change each, with the ring
position beside it:

```
suite native 1/1 <[ ] Back/Start>   task default 1/1 <, . d-pad>   views single <V X>: scene
```

The `1/1` is the part worth reading when a key seems dead: it means the ring has
one entry, so cycling it correctly does nothing. On a bare checkout both rings
are one long — nothing but the native arm is installed, and the native arm has
one scene. Press the key anyway and the HUD says why, in gold, for five seconds:
*"native is the only suite installed -- python main.py --list-envs"*. That line
goes to the terminal too, but the HUD is where you are looking.

A switch tears down the old environment and builds the new one, so it is not
instant; the window title and the HUD both name the environment you are on. If
a backend fails to build (a bad install, an asset download that hasn't run), the
session says so and stays on the environment you were already driving.

While recording, either switch **closes the current episode and opens the next
one** — the two environments have different `observation.state` widths and
different objects in them, so they must not share a parquet. The new episode
also picks up the new environment's cameras.

## Note

`arm_game.py` is the original single-file prototype, kept for reference. It is
**superseded** by the modules above: its actuator tuning predates the tracking
work (constraints 7–9 in `CLAUDE.md`), and it predates the rename, so it still
says "Claw Crew" inside. Read the modules, not it.
