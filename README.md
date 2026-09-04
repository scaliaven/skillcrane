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
python main.py --list-envs      # benchmark environments, and what's installed
python main.py --env robosuite:Lift          # start on a benchmark ([ / ] switches live)
python main.py --headless       # scripted pick-and-place, no window, exits 0 on a score
python main.py --seed 7         # fixed cube spawns
python main.py --record         # collect the session into runs/ (off unless asked)
python main.py --record DIR     # ... into DIR instead
python gamepad_probe.py         # see what your pad reports, and how it is read
```

## Controls

| | gamepad | keyboard |
|---|---|---|
| move gripper (camera-relative) | left stick | `WASD` |
| raise / lower | right stick Y | `Q` / `E` |
| rotate wrist | right stick X | `Z` / `C` |
| gripper | `A` (toggle) | `SPACE` (hold) |
| orbit camera | `LB` / `RB`, or d-pad ← → | `←` / `→` |
| switch environment | `Back` / `Start` | `[` / `]` |
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

**Only the native arm has a camera you can orbit.** The benchmark families
render from a fixed camera their own adapter picks, so `LB`/`RB` and the arrow
keys do nothing there — by design, not because the buttons are unmapped.

## Layout

| file | what it holds |
|---|---|
| `scene.py` | MJCF as a string + model constants |
| `kin.py` | `Arm` (FK/Jacobian/pose error), `IKController`, `down_R` |
| `game.py` | physics, scoring, spawn logic — **no pygame** |
| `input.py` | `GamepadReader`/`KeyboardReader` → `ControlInput` |
| `render.py` | offscreen render + HUD |
| `recorder.py` | LeRobot-format episode logging |
| `benchmarks/` | adapters: robosuite, Meta-World, Fetch, LIBERO |
| `main.py` | CLI entry point |

`game.py` never imports `input.py` or `render.py`, so the whole sim/rules layer
runs headless. See `CLAUDE.md` for the physics constraints that must not be
"simplified" away.

## Tests

```sh
python -m pytest -q          # 134 tests, ~10 s, no display and no gamepad needed
```

Benchmark tests skip cleanly when the benchmark isn't installed, so a bare
checkout stays green (134 passed / 10 skipped).

Physics tests assert on numbers — contact count, joint velocity, tracking error,
cube height, score — and the grasp tests are parametrised over 12 random spawns,
all of which must pass.

## Data collection

Data collection is **off by default**. `--record` turns it on (bare, it collects
into `runs/`; `--record DIR` picks the directory) and writes one row per 100 Hz
control tick — `observation.state` (6 joints + gripper + cube xyz), `action`
(the operator's stick input), and a rendered frame — in LeRobot v2.1 layout:

```
DIR/data/chunk-000/episode_000000.parquet
DIR/images/observation.images.cam/episode_000000/frame_%06d.png
DIR/meta/{info,episodes,tasks,episodes_stats}.{json,jsonl}
```

which is the shape `lerobot` expects for training an ACT policy.

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
# recorded    1680 ticks -> runs/data/chunk-000/episode_000000.parquet
```

Read it back with plain pyarrow — no `lerobot` install needed to check the file:

```python
import pyarrow.parquet as pq
t = pq.read_table("runs/data/chunk-000/episode_000000.parquet")
print(t.num_rows, t.column_names)
# 1680 ['observation.state', 'action', 'timestamp', 'frame_index',
#       'episode_index', 'index', 'task_index', 'observation.images.cam']
t.slice(900, 1).to_pylist()[0]
# {'observation.state': [0.777, 0.016, 1.289, 1.842, 0.201, -0.001,   # j1..j6
#                        0.016, 0.16, 0.158, 0.293],                  # grip, cube xyz
#  'action': [0.0, 0.0, 0.0, 0.0, 1.0],                               # dx dy dz dyaw grip
#  'timestamp': 9.0, 'frame_index': 900, 'episode_index': 0,
#  'observation.images.cam': 'images/.../frame_000900.png'}
```

`meta/` carries what a `LeRobotDataset` reads at load time — `info.json` (fps
100, the feature schema, `codebase_version` v2.1), `tasks.jsonl` (the language
instruction), `episodes.jsonl` (one row per episode with its length), and
`episodes_stats.jsonl` (per-field mean/std/min/max, which is what a policy's
input normalisation uses):

```sh
cat runs/meta/episodes.jsonl
# {"episode_index": 0, "tasks": ["Pick up the cube and place it in the target zone."], "length": 1680}
```

Two things to know before collecting in bulk:

- **Episodes accumulate; they do not overwrite.** The episode index comes from
  reading the directory, not from a counter, so recording into the same `DIR`
  across separate runs of the program appends `episode_000001`,
  `episode_000002`, … and `meta/` stays consistent with them.
- **Frames dominate the size.** One PNG per 100 Hz tick is ~37 KB, so the 16.8 s
  episode above is 63 MB and a 90 s round is ~340 MB. `EpisodeRecorder` accepts
  frames at a lower rate (pass `None` on the ticks you skip; the image column
  repeats the last frame to stay 1:1 with the ticks).

## Benchmarks

`python main.py --env <family>:<task>` drives an existing benchmark with the same
rig. Verified working headless on Apple Silicon: **robosuite** (Lift, Stack,
PickPlaceCan, Door), **Meta-World** (50 tasks), **Fetch**, and **LIBERO** (130
tasks, in its own environment).

```sh
pip install -r requirements-benchmarks.txt      # robosuite, Meta-World, Fetch
CONDA_SUBDIR=osx-arm64 conda env create -f environment-libero.yml   # LIBERO (robosuite 1.4)
```

They need `mujoco<3.12` — see [`BENCHMARKS.md`](BENCHMARKS.md) for why, plus the
gripper-sign table and two other install landmines.

### Switching without restarting

`[` / `]` on the keyboard, or `Back` / `Start` on the pad, cycles through the
families that are actually installed — `--list-envs` shows the ring, and a
family you have not installed is never in it. The task resets to that family's
default, because a task id from one family means nothing in the next.

A switch tears down the old environment and builds the new one, so it is not
instant; the window title and the HUD both name the environment you are on. If
a backend fails to build (a bad install, an asset download that hasn't run), the
session says so and stays on the environment you were already driving.

While recording, a switch **closes the current episode and opens the next one**
— the two environments have different `observation.state` widths, so they must
not share a parquet.

## Note

`arm_game.py` is the original single-file prototype, kept for reference. It is
**superseded** by the modules above: its actuator tuning predates the tracking
work (constraints 7–9 in `CLAUDE.md`), and it predates the rename, so it still
says "Claw Crew" inside. Read the modules, not it.
