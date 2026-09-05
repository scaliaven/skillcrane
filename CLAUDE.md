# Skillcrane — manipulation in sim + data collection

A 6-DoF arm in MuJoCo, driven with an 8BitDo gamepad, scored as pick-and-place.
Target platform is **macOS on Apple Silicon**.

It began as a manipulation game and became a demonstration-collection rig: the
scoring loop is the teleop interface, and every round played is an
imitation-learning episode (`--record`, LeRobot format, trains an ACT policy) --
and `--eval` runs one back through the same loop and scores it.
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
  policy.py       Policy protocol + ScriptedPickPlace / ReplayPolicy /
                  LeRobotPolicy, rollout(), evaluate(). NO pygame.
  benchmarks/     adapters: native, robosuite, RoboCasa, Meta-World, Fetch,
                  LIBERO. A *suite* is a benchmark, a *task* is a setting in
                  one; registry.py cycles them separately.
  main.py         CLI entry: --headless, --seed, --record, --record-views,
                  --env, --list-envs, --eval, --policy
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

`policy.py` sits on `game.py`'s side of the pygame line, not `render.py`'s: it
is what a data pipeline calls, so `test_no_pygame.py` covers it too.

Note the dependency direction: `game.py` never imports `input.py`. `Game.step`
takes plain world-frame floats, and the camera-relative rotation lives in
`ControlInput.world_xy` on the input side.

### Suite vs task -- keep them apart

Two different switches, and conflating them is how the rig gets confusing:

- **suite** = which benchmark (`native`, `robosuite`, `robocasa`, `metaworld`,
  `fetch`, `libero`). Different robot, different adapter, different observation
  width. `registry.cycle_suite`, bound to `Back`/`Start` and `[` / `]`. The ring
  holds only *installed* suites, so on a bare checkout it is one entry long and
  says so rather than looking broken.
- **task** = one setting inside a suite (`Lift` -> `Stack`). Same robot, same
  adapter. `registry.cycle_task`, bound to the d-pad and `,` / `.`.

An *environment spec* names both: `robosuite:Lift`. `--env` takes one, `parse()`
splits it, and the HUD prints each next to the control that changes it.

Both rings can legitimately be one entry long -- one suite installed, one task
in it -- and then the key correctly does nothing. That is indistinguishable from
a broken binding unless the rig says so, so two things make it visible: the HUD
carries the ring position (`suite native 1/1`), and every switch outcome goes
through `run_game`'s `say()`, which prints *and* calls `Display.notify()` for a
five-second gold line in the HUD band. Reporting a no-op only on stdout is not
reporting it -- the operator is watching the window. The HUD also names the
keyboard binding next to the pad one for the same reason: it used to say only
`<Back/Start>`, so a keyboard operator had no way to learn `[` / `]` exists. The word
"family" used to do duty for both and was removed; `ControlInput.suite`,
`SUITES`, `suites()` and `cycle_suite()` are the current names.

RoboCasa is registered but **untested here** -- it is not installed on this
machine. It reuses `robosuite_env.py` (importing `robocasa` registers its
kitchens in robosuite's registry); the robot name and default task are read off
its docs, not measured, and the factory tries `PandaOmron` then `PandaMobile`
because the name changed between releases.

### Policies -- and why replay is the only dataset test that matters

`recorder.py` fills a dataset and `policy.py` reads one back. Everything here
emits the same five numbers the sticks do -- `(dx, dy, dz, dyaw, grip)` -- and
goes through the same `TeleopEnv.step`, so a demonstration, a replay and a
trained checkpoint are scored by identical rules and recorded through identical
columns. `--eval N` runs one over N consecutive seeds and prints a per-seed
table; `--policy` picks between `scripted`, `replay:DIR[:EP]` and `act:PATH`.

Two findings behind it:

- **A well-formed dataset can still be untrainable, and nothing else here can
  see that.** `--headless --record` logged `[0, 0, 0, 0, grip]` on every tick of
  a completed pick-and-place: `game.scripted_pick_and_place` moves `game.tgt`
  directly (right for a physics test, it isolates the arm from the input layer)
  so the action beside the motion was not the action that caused it. Every
  column was present, every shape correct, every PNG on disk. `ReplayPolicy` is
  the check -- same seed, same actions, same outcome -- and `main.py --headless`
  now drives through `ScriptedPickPlace`. Two drivers is deliberate; two copies
  of the script was not, so the waypoint table is declared once as
  `game.WAYPOINTS` and both walk it -- one by moving `game.tgt`, one through the
  sticks. The rate limit of constraint 6 is likewise one function,
  `game.step_toward`. Replay reproduces an episode to 1.6e-7 of state over ~700
  ticks; replayed into a *different* seed it misses, which is what stops the
  check from being vacuous, and `test_m9_policy.py` asserts both directions.
- **Arriving is not stopping.** The commanded target is pure integration, so it
  reaches a waypoint while the arm is still crossing the tolerance ball at
  0.3 m/s -- position tolerance cannot express "settled" at any radius. The two
  legs that end in a gripper action carry `settle=SETTLE_QVEL` (0.05 rad/s) as
  well; without it the fingers closed at |qvel| ~0.7 and opened at ~0.9, so
  every demonstration grasped and released from a moving gripper. Costs ~15
  ticks of a ~700-tick round. `settle` is the one field `policy.Waypoint` adds
  to `game.WAYPOINTS`; the tick budgets stay the old script's fixed counts, so a
  leg that never settles falls back to exactly the behaviour the grasp tests
  were written against.

`LeRobotPolicy` is **untested here** -- `lerobot` is not installed on this
machine -- in the same way RoboCasa is: written against the documented
`from_pretrained` / `select_action` interface, not measured.

Every recorded tick carries `next.reward` (1.0 on the tick that scored) and
`next.done` (the last tick), and every episode's row in `meta/episodes.jsonl`
carries `success` and `score`. Without them a directory of rounds cannot be
filtered into the successful demonstrations the ACT recipe trains on, and that
filtering happens long after the collecting process has exited. `score` is a
count, not a bool, because a native round is not one-shot -- the cube respawns
and a good operator scores several times in 90 s.

### Cameras and views

An env declares `view_names` (operator's view first) and answers
`frames({view: (w, h)})`. `render.py` decides the layout, computes the panel
rects, and asks for exactly those sizes -- nothing is rendered large and then
thrown away, and a one-camera env simply gets one full-width panel.

Settled details behind the native cameras:

- **They live in the MJCF, not in code.** `scene.CAMERAS` names them; anything
  that loads the model gets the same views. Only the orbiting `scene` camera is
  a code-side `MjvCamera`, because it is view *state*, not model data.
- **The TCP marker is site group 3, and MuJoCo hides groups 3+ by default.**
  `NativeEnv` re-enables it for the operator's view only. In the wrist camera,
  5 cm away, that 8 mm dot covers exactly the object being grasped -- it would
  be in the middle of every recorded eye-in-hand frame.
- **They are framed for control, and "how close" has a test.** The fixed cameras
  aim at the middle of the spawn arc, not at the horizon, and sit as close as
  they can while still seeing every spawn position and the whole drop zone.
  `test_m1` checks that against the model's own camera matrices at a 16:9 panel
  (narrower than any layout actually uses), so pulling them in further fails a
  test instead of quietly cropping the cube.
- **The `top` camera is turned 90 degrees** (`xyaxes="0 -1 0  1 0 0"`), so world
  y runs across the panel. The workspace is wide in y (+/-0.36 m of spawn arc)
  and shallow in x, and a view panel is wide too; matching those axes is worth
  0.33 m of camera height -- 0.82 m instead of 1.15 m for the same coverage.
- **The operator's `scene` camera follows the work and zooms.** It sits 0.90 m
  out (was 1.35 m, where a 48 mm cube was 30 px of a 900 px panel) and eases its
  lookat onto the midpoint of gripper-and-goal with a 0.3 s time constant: the
  thing being judged is a *gap*, so centring the midpoint keeps both ends of it
  on screen. `zoom()` is multiplicative and clamped to 0.45-2.2 m; `B` / `F`
  freezes the follow. `track()` is keyed on **sim time**, not on render calls --
  recording renders the same tick twice (window panels, then the dataset's own
  fixed-size views) and a per-render lerp would move at whatever rate the layout
  happened to demand.

### Render settings that were tuned by looking at the output

The scene is lit and sized for an operator reading depth off a 2-D panel, not
for a screenshot. Four of these are findings, not preferences:

- **`offsamples="8"`** in `<quality>` (default 4). Every frame this project
  shows or records comes off the offscreen buffer, so it is the one anti-alias
  knob that reaches all of them, including the 320×240 recorded ones.
- **`shadowsize` stays 2048.** 4096 is not free quality: the finer shadow texel
  put visible acne -- a green stipple -- across the flat drop-zone site, which
  sits 2 mm off the floor.
- **The key light stays positional.** A directional one lights the scene just as
  well and casts no usable shadow here, and the contact shadow under the gripper
  is the only cue on a 2-D screen for how far above the cube the fingers are.
- **`<map znear="0.004">`.** znear is a *fraction of the model extent* (~1.2 m),
  so the default put the near plane at ~1 cm short of where the wrist camera
  needs it; the eye-in-hand view sits 10 cm from jaws it has to see.
- Fingers are light grey (0.52), not the 0.20 they started at: in the wrist
  camera they were a black silhouette against a dark floor, in every recorded
  eye-in-hand frame.

The window is **resizable** (default 1280×860, `--window WxH`). Nothing may
assume a size: `panels()` takes the viewport, `Display._measure()` re-derives
the type sizes, HUD band and margins from the window height, and the environment
is asked for frames at exactly the new panel sizes rather than being upscaled.
`NativeEnv` caches renderers per size and **bounds that cache** (MAX_RENDERERS)
-- a window drag is one resize per mouse move, and each size is a GL context.

Measured on this machine (M-series, with the physics in the loop): 6.8 ms/frame
for one 1280x638 panel, 11.7 ms for a four-panel grid at 1600x1000, 10.2 ms for
a single 2048x1000. All inside the 16.7 ms the 60 Hz loop has, which is why
offsamples went to 8 and the default window got bigger rather than either being
traded against frame rate.

Recording is deliberately decoupled from the window: `--record-views` renders
its own frames at a fixed size (`--record-size`, 320x240 by default), because a
dataset column has one image shape for the whole episode while the operator can
change the layout and resize the window mid-round. Each view becomes its own
`observation.images.<view>` column.

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

3. **Declare `<visual><global offwidth=... offheight=.../></visual>`** big
   enough for the window. MuJoCo's offscreen framebuffer defaults to 640×480 and
   *raises* rather than downscaling. It is 2048×1152 here; `render.py` asserts
   its **default** window fits and `Display.view_sizes` clamps panel requests to
   it, because the window is resizable and can be dragged past the cap. Past it,
   frames are rendered at the cap and scaled up — losing sharpness is acceptable
   mid-round, raising is not.

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
| M1 | scene + kinematics | `test_m1_scene_kin.py` | 15 passed |
| M2 | stable tracking | `test_m2_tracking.py` | 6 passed |
| M3 | grasping (12 seeds) | `test_m3_grasp.py` | 19 passed |
| M4 | game rules | `test_m4_rules.py` | 17 passed |
| M5 | input layer | `test_m5_input.py` | 54 passed |
| M6 | render + HUD | `test_m6_render.py` | 25 passed |
| M7 | recording (stretch) | `test_m7_record.py` | 23 passed |
| M8 | benchmark adapters | `test_benchmarks.py` | 39 passed, 17 skipped |
| M9 | policies + replay + eval | `test_m9_policy.py` | 26 passed |
| — | hard rule | `test_no_pygame.py` | 6 passed |
| — | docs match the suite | `test_docs.py` | 12 passed |

Totals: **242 passed / 17 skipped** with no benchmarks installed. The counts
with the benchmark suites installed have not been re-measured since the
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
    suites share a process -- which the test suite does.

13. **Gripper close signs differ per suite and were measured**: robosuite +1,
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

Roles as bound: A grip, B camera follow, X view layout, Y reset, LB/RB orbit,
LT/RT zoom, d-pad **task**, Back/Start **suite**. The d-pad used to double as
orbit; it steps the task now, and the bumpers keep the camera. The d-pad is two
buttons under SDL and a *hat* on a raw pad, which is why `_dpad_x()` reads it
separately.

**Triggers are read one-sided** (`max(0, value)`, `_trigger()`). SDL reports a
released trigger as 0, but a raw pad usually rests at -1 and runs to +1; flooring
costs a raw pad the bottom half of its travel and buys the thing that matters --
a pad resting at -1 does not zoom the camera out for the whole session.

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
- **The docs are tested too.** Six files quote test counts -- README, this file,
  DEVLOG, BENCHMARKS, `environment.yml` and the project page -- and all six had
  drifted, one by four commits and 114 tests, because nothing was looking.
  `test_docs.py` compares every one of them against what pytest actually
  collects, so a new test fails the suite until the tables are updated.

  Ground truth is *collection*, not a second full run, so the rule is that the
  numbers a line states must **sum** to what is collected: `242 passed, 17
  skipped` is 259, and `39 (+17 skipped)` is 56 on the M8 row. That also makes
  the numbers machine-independent -- installing robosuite changes which tests
  pass, not how many exist. Adding a test module fails the suite too, until it
  has a row.

  A number that is deliberately frozen -- the benchmark-installed worlds, last
  measured when the adapters landed -- is exempt only if its own line says so, in
  words a reader sees: `re-measured`, `measured at`, or a `†`. The exemption is
  the disclosure, and it is per line, because a paragraph-level one let a single
  disclaimer cover a whole table and the live row in it went unchecked.

## Style

Small modules, plain functions, no framework. Comment *why* for anything
non-obvious — especially the constraints above. No premature abstraction over
robots or input devices; there is one arm and one pad.
