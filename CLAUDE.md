# Claw Crew — manipulation in sim + data collection

A 6-DoF arm in MuJoCo, driven with an 8BitDo gamepad, scored as pick-and-place.
Target platform is **macOS on Apple Silicon**.

It began as a manipulation game and became a demonstration-collection rig: the
scoring loop is the teleop interface, and every round played is an
imitation-learning episode (`--record`, LeRobot format, trains an ACT policy).
That is why the sim/rules layer stays free of UI code — `game.py` is the part a
data pipeline consumes, and it must never need a window.

## Stack

- Python 3.11+, `mujoco`, `pygame`, `numpy`. `pytest` for tests.
  (`pyarrow` + `pillow` are optional, needed only for `--record`.)
- Render MuJoCo **offscreen** via `mujoco.Renderer`; pygame owns the window and
  draws the HUD. Do not use `mujoco.viewer`.
- Must run under plain `python`, not `mjpython`.

## Architecture

```
clawcrew/
  scene.py        MJCF as a string + model constants
  kin.py          Arm (FK/Jacobian/pose error), IKController, down_R
  game.py         Game: physics, scoring, spawn logic. NO pygame import.
  input.py        GamepadReader + KeyboardReader -> a common ControlInput dataclass
  render.py       offscreen render + HUD drawing
  recorder.py     LeRobot-format episode logging (optional deps)
  main.py         CLI entry: --headless, --seed, --record
  tests/
```

Hard rule: `game.py` must be importable and fully runnable without pygame or a
display. `tests/test_no_pygame.py` enforces this in a subprocess where
`import pygame` raises — do not weaken it. Every acceptance test runs headless.

Note the dependency direction: `game.py` never imports `input.py`. `Game.step`
takes plain world-frame floats, and the camera-relative rotation lives in
`ControlInput.world_xy` on the input side.

## Non-negotiable constraints

Settled findings. They look arbitrary and will be "cleaned up" otherwise.

1. **Structural arm links must have `contype="0" conaffinity="0"`.** Collision
   stays only on fingers, hand, cube, floor. MuJoCo skips its parent–child
   contact filter for bodies welded to the world, and a fixed base with no joint
   *is* world-welded — so the base geom collides with link 1 and destabilises the
   yaw joint. Symptom: base joint oscillates at >15 rad/s and never tracks its
   command. `test_m1` asserts `d.ncon == 0` at rest and that every non-colliding
   geom really has both flags off.

2. **The IK controller integrates on a scratch `MjData`, never on measured
   state.** Evaluate the Jacobian and pose error at the *commanded* joint
   configuration. Feeding measured pose error back into the command while
   position actuators chase that same command closes a loop through the actuator
   dynamics and oscillates hard.

3. **Declare `<visual><global offwidth=... offheight=.../></visual>`** sized to
   the window. MuJoCo's offscreen framebuffer defaults to 640×480 and *raises*
   rather than downscaling. `render.py` asserts its window fits.

4. **Gripper joint 0 = fully open**, increasing closes. Fingers must open wider
   than the cube (cube 48 mm across, finger inner faces 68 mm apart when open).

5. **Tool orientation convention.** The wrist-neutral tool frame is
   `diag(-1, 1, -1)` — the pitch joints j2/j3/j4/j6 all turn about y and sum to
   π to point the tool down, and `R_y(π) = diag(-1, 1, -1)`. So `down_R(yaw)` has
   tool x at −world x. Pass `yaw = atan2(y, x) + user_yaw` so the wrist joint
   stays mid-range instead of jamming against a limit. `test_m1` verifies this
   against the model rather than trusting the derivation.

6. **Rate-limit the Cartesian target.** It moves by `stick * speed * dt`; never
   let it jump, or the arm flings the payload. `drive_to` obeys the same rule.

### Tuning that is load-bearing (added on top of the original prototype)

7. **IK velocity feedforward.** `IKController.update` commands
   `kp * error + goal_velocity`. A pure proportional chase trails a moving goal
   by `v / kp` forever — 65 mm at a 0.3 m/s sweep. This is safe precisely
   because of constraint 6: the goal is rate-limited, so differencing it is
   bounded, and the twist clamp is the backstop. It does **not** violate
   constraint 2 — nothing measured enters the loop.

8. **Actuator gains `kp=2500 kv=40` (not 800/60).** A position servo trails its
   moving command by roughly `kv * qdot / kp`, and that following error was the
   single biggest term in end-effector tracking (~21 mm at 0.3 m/s, over the
   10 mm budget). Peaks at 13 N of the 200 N `forcerange`, so nothing saturates.

9. **`TRACK_LAM = 0.04`** for the per-tick DLS, lower than the batch solver's.
   Damping is what makes the solver undershoot the twist it was asked for, and
   that undershoot is end-effector lag. Safe because the target is clamped well
   inside `REACH_MAX`, so the tracking loop never sits on a singularity.

   Together 7–9 take a 0.3 m/s sweep from 65 mm of error to 8.2 mm peak.

## Milestones — all complete, all tests passing

| # | Milestone | Test | Status |
|---|-----------|------|--------|
| M1 | scene + kinematics | `test_m1_scene_kin.py` | 12 passed |
| M2 | stable tracking | `test_m2_tracking.py` | 6 passed |
| M3 | grasping (12 seeds) | `test_m3_grasp.py` | 19 passed |
| M4 | game rules | `test_m4_rules.py` | 17 passed |
| M5 | input layer | `test_m5_input.py` | 33 passed |
| M6 | render + HUD | `test_m6_render.py` | 5 passed |
| M7 | recording (stretch) | `test_m7_record.py` | 7 passed |
| — | hard rule | `test_no_pygame.py` | 4 passed |

## Gamepad notes

8BitDo pads report different layouts per pairing mode. On macOS use Apple mode
(hold Start+A on power-on) or D-input (Start+B); XInput mode is a Windows API and
is useless here. Axis and button indices live in **one config block at the top of
`input.py`** — never scattered. `gamepad_probe.py` prints live axis and button
indices so the mapping can be discovered rather than guessed.

## Testing rules

- Every test runs headless with no display and no gamepad.
- Physics tests assert on numbers (`ncon`, `|qvel|`, tracking error, cube z,
  score), never on "it looked fine."
- Grasp tests are parametrised over random seeds. A single lucky seed is not a
  pass.
- Do not mark a milestone done without running its test and showing the output.

## Style

Small modules, plain functions, no framework. Comment *why* for anything
non-obvious — especially the constraints above. No premature abstraction over
robots or input devices; there is one arm and one pad.
