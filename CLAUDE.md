# Skillcrane — manipulation in sim + data collection

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
skillcrane/
  scene.py        MJCF as a string + model constants
  kin.py          Arm (FK/Jacobian/pose error), IKController, down_R
  game.py         Game: physics, scoring, spawn logic. NO pygame import.
  input.py        GamepadReader + KeyboardReader -> a common ControlInput dataclass
                  (SDL standard layout when available, raw indices otherwise)
  render.py       offscreen render + HUD drawing + multi-view layouts
  recorder.py     LeRobot-format episode logging (optional deps)
  benchmarks/     adapters: native, robosuite, Meta-World, Fetch, LIBERO
  main.py         CLI entry: --headless, --seed, --record, --record-views,
                  --env, --list-envs
  tests/
```

`benchmarks/` is an adapter layer, not an abstraction over robots: every backend
implements one small `TeleopEnv` contract (Cartesian delta + gripper bit), and
the native arm goes through the same path so the adapters stay honest. Importing
`benchmarks` costs nothing with none installed -- every backend imports lazily
inside its factory.

Hard rule: `game.py` must be importable and fully runnable without pygame or a
display. `tests/test_no_pygame.py` enforces this in a subprocess where
`import pygame` raises — do not weaken it. Every acceptance test runs headless.

Note the dependency direction: `game.py` never imports `input.py`. `Game.step`
takes plain world-frame floats, and the camera-relative rotation lives in
`ControlInput.world_xy` on the input side.

### Cameras and views

An env declares `view_names` (operator's view first) and answers
`frames({view: (w, h)})`. `render.py` decides the layout, computes the panel
rects, and asks for exactly those sizes -- nothing is rendered large and then
thrown away, and a one-camera env simply gets one full-width panel.

Two settled details behind the native cameras:

- **They live in the MJCF, not in code.** `scene.CAMERAS` names them; anything
  that loads the model gets the same views. Only the orbiting `scene` camera is
  a code-side `MjvCamera`, because it is view *state*, not model data.
- **The TCP marker is site group 3, and MuJoCo hides groups 3+ by default.**
  `NativeEnv` re-enables it for the operator's view only. In the wrist camera,
  5 cm away, that 8 mm dot covers exactly the object being grasped -- it would
  be in the middle of every recorded eye-in-hand frame.

Recording is deliberately decoupled from the window: `--record-views` renders
its own frames at a fixed 320x240, because a dataset column has one image shape
for the whole episode and the operator can change the layout mid-round. Each
view becomes its own `observation.images.<view>` column.

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
| M5 | input layer | `test_m5_input.py` | 48 passed |
| M6 | render + HUD | `test_m6_render.py` | 14 passed |
| M7 | recording (stretch) | `test_m7_record.py` | 17 passed |
| M8 | benchmark adapters | `test_benchmarks.py` | 32 passed, 13 skipped |
| — | hard rule | `test_no_pygame.py` | 4 passed |

Totals: **169 passed / 13 skipped** with no benchmarks installed. The counts
with the benchmark families installed have not been re-measured since the
switching work.

## Benchmark constraints

10. **`mujoco<3.12` for anything robosuite-based** (robosuite, LIBERO). Its
    mujoco-py shim asserts `joint_type in (mjJNT_HINGE, mjJNT_SLIDE)`, and in
    3.12 the enum's *reflected* comparison with `numpy.int32` began returning
    False, so the membership test fails and every env dies in
    `_setup_references`. Bisected: fine through 3.11.0, broken at 3.12.0.

11. **Never install LIBERO beside the robosuite backend.** LIBERO pins
    `robosuite==1.4.0`, whose controller API is not the 1.5 one
    `benchmarks/robosuite_env.py` uses. `registry.installed()` version-checks
    for this so it reports unavailable instead of failing at `make()`.

12. **Clear `MUJOCO_GL` when it holds a backend gymnasium can't use.**
    Importing robosuite sets `MUJOCO_GL=cgl` on macOS; gymnasium only knows
    glfw/egl/osmesa and dies with `KeyError: 'cgl'`. Only shows up when two
    families share a process -- which the test suite does.

13. **Gripper close signs differ per family and were measured**: robosuite +1,
    Meta-World +1, Fetch **-1**, LIBERO +1. A wrong sign makes the task quietly
    unsolvable while looking like bad teleop. See BENCHMARKS.md for evidence.

## Gamepad notes

Read the pad through **SDL's game-controller layer** (`pygame._sdl2.controller`)
whenever SDL recognises it, and only fall back to raw joystick indices when it
does not. SDL's controller database reports a standard layout — A is A, LB is
LB — regardless of pairing mode, which is the actual fix for "8BitDo pads
enumerate differently per mode". The two numberings are not compatible: SDL
calls the shoulders 9 and 10, a raw HID pad usually 4 and 5, so reading a
controller through the raw table silently breaks the camera buttons.

Raw indices still exist for unknown pads, and both tables live in **one config
block at the top of `input.py`** — never scattered. `GamepadReader` reads by
role (`"grip"`, `"cam_l"`), so neither table leaks into the read path.
`gamepad_probe.py` reports which path is live, what `input.py` makes of each
control next to the raw numbers, and the block to edit if the layout is raw.

Roles as bound: A grip, X view layout, Y reset, LB/RB orbit, d-pad task,
Back/Start environment family. The d-pad used to double as orbit; it steps the
task now, and the bumpers keep the camera. The d-pad is two buttons under SDL
and a *hat* on a raw pad, which is why `_dpad_x()` reads it separately.

Pairing modes still matter for the fallback: Apple (hold Start+A on power-on),
D-input (Start+B); XInput is a Windows API and is useless here. Some models,
including the Ultimate 2C, use a switch on the back instead.

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
