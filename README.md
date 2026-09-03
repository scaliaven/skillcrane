# Claw Crew

Teleoperate a 6-DoF robot arm in MuJoCo with an 8BitDo gamepad and score
pick-and-place. Runs under plain `python` on macOS/Apple Silicon — MuJoCo renders
offscreen and pygame owns the window, so nothing fights over the main thread.

## Install

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
python main.py                  # play
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
| `main.py` | CLI entry point |

`game.py` never imports `input.py` or `render.py`, so the whole sim/rules layer
runs headless. See `CLAUDE.md` for the physics constraints that must not be
"simplified" away.

## Tests

```sh
python -m pytest -q          # 98 tests, ~7 s, no display and no gamepad needed
```

Physics tests assert on numbers — contact count, joint velocity, tracking error,
cube height, score — and the grasp tests are parametrised over 12 random spawns,
all of which must pass.

## Recording (stretch)

`--record DIR` writes one row per 100 Hz control tick — `observation.state`
(6 joints + gripper + cube xyz), `action` (the operator's stick input), and a
rendered frame — in LeRobot v2.1 layout:

```
DIR/data/chunk-000/episode_000000.parquet
DIR/images/observation.images.cam/episode_000000/frame_%06d.png
DIR/meta/{info,episodes,tasks,episodes_stats}.{json,jsonl}
```

which is the shape `lerobot` expects for training an ACT policy.

## Note

`arm_game.py` is the original single-file prototype, kept for reference. It is
**superseded** by the modules above and its actuator tuning predates the
tracking work (constraints 7–9 in `CLAUDE.md`) — read the modules, not it.
