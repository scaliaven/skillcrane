# Claw Crew

**Manipulation in sim + demonstration data collection.** A 6-DoF robot arm you
teleoperate in MuJoCo with an 8BitDo gamepad — and the rig that records what you
did, so a policy can learn it.

It started as a manipulation *game*: grab the cube, drop it in the zone, beat the
clock. Building that well meant solving the hard part of a teleop rig anyway — an
arm that tracks your hand smoothly enough to do fine manipulation with. Once that
worked, every round played was already a demonstration, so the same rig now logs
each control tick as an imitation-learning episode. The game is the interface;
the dataset is the output.

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

## Run

```sh
python main.py                  # play
python main.py --list-envs      # benchmark environments, and what's installed
python main.py --env robosuite:Lift          # teleop a benchmark instead
python main.py --headless       # scripted pick-and-place, no window, exits 0 on a score
python main.py --seed 7         # fixed cube spawns
python main.py --record runs/   # log the session as a LeRobot dataset
python gamepad_probe.py         # discover your pad's axis/button indices
```

## Controls

| | gamepad | keyboard |
|---|---|---|
| move gripper (camera-relative) | left stick | `WASD` |
| raise / lower | right stick Y | `Q` / `E` |
| rotate wrist | right stick X | `Z` / `C` |
| gripper | `A` (toggle) | `SPACE` (hold) |
| orbit camera | `LB` / `RB` | `←` / `→` |
| reset round | `Y` | `R` |
| quit | — | `ESC` |

8BitDo pads enumerate differently per pairing mode. On macOS use Apple mode
(hold Start+A on power-on) or D-input (Start+B). If the arm moves on the wrong
stick, run `gamepad_probe.py` and edit the config block at the top of `input.py`.

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
python -m pytest -q          # 116 tests, ~8 s, no display and no gamepad needed
```

Benchmark tests skip cleanly when the benchmark isn't installed, so a bare
checkout stays green (116 passed / 10 skipped; 125 passed with the benchmark
families installed).

Physics tests assert on numbers — contact count, joint velocity, tracking error,
cube height, score — and the grasp tests are parametrised over 12 random spawns,
all of which must pass.

## Data collection

`--record DIR` writes one row per 100 Hz control tick — `observation.state`
(6 joints + gripper + cube xyz), `action` (the operator's stick input), and a
rendered frame — in LeRobot v2.1 layout:

```
DIR/data/chunk-000/episode_000000.parquet
DIR/images/observation.images.cam/episode_000000/frame_%06d.png
DIR/meta/{info,episodes,tasks,episodes_stats}.{json,jsonl}
```

which is the shape `lerobot` expects for training an ACT policy.

## Benchmarks

`python main.py --env <family>:<task>` drives an existing benchmark with the same
rig. Verified working headless on Apple Silicon: **robosuite** (Lift, Stack,
PickPlaceCan, Door), **Meta-World** (50 tasks), **Fetch**, and **LIBERO** (130
tasks, in its own environment).

```sh
pip install -r requirements-benchmarks.txt      # robosuite, Meta-World, Fetch
conda env create -f environment-libero.yml      # LIBERO (pins robosuite 1.4)
```

They need `mujoco<3.12` — see [`BENCHMARKS.md`](BENCHMARKS.md) for why, plus the
gripper-sign table and two other install landmines.

## Note

`arm_game.py` is the original single-file prototype, kept for reference. It is
**superseded** by the modules above and its actuator tuning predates the
tracking work (constraints 7–9 in `CLAUDE.md`) — read the modules, not it.
