"""Skillcrane -- teleoperate a robot arm in MuJoCo, and record what you did.

    python main.py                        play the native Skillcrane arm
    python main.py --list-envs            show benchmark environments
    python main.py --env robosuite:Lift   teleop a benchmark instead
    python main.py --headless             scripted pick-and-place, no window
    python main.py --record               collect the session into runs/
    python main.py --record DIR           ... into DIR instead (LeRobot format)

Runs under plain `python`, not `mjpython`: environments render offscreen and
pygame owns the window, so nothing fights over the macOS main thread.

Controls
    left stick    move gripper horizontally (camera-relative)
    right stick Y raise / lower          right stick X  rotate wrist
    A             toggle gripper          LB / RB       orbit camera
    Y             reset the round
    keyboard      WASD move, QE up/down, ZC wrist, SPACE grip, R reset,
                  arrow keys orbit, ESC quit
"""
import argparse
import math
import sys

import numpy as np

import scene
from game import Game, drive_to, scripted_grasp

MAX_STEPS_PER_FRAME = 8      # keeps a slow env from spiralling on catch-up
DEFAULT_RECORD_DIR = "runs/"  # where a bare --record collects to


def run_headless(seed: int = 2, record=None, env_spec: str = "native") -> int:
    """Scripted acceptance run for the native arm; a smoke test for benchmarks."""
    if env_spec != "native":
        return _headless_benchmark(env_spec, seed, record)

    g = Game(seed=seed)
    print("cube spawn ", np.round(g.cube_pos(), 3), " ncon at rest:", g.d.ncon)

    rec = ren = None
    if record is not None:
        import mujoco
        from recorder import EpisodeRecorder
        rec = EpisodeRecorder(record)
        ren = mujoco.Renderer(g.m, height=240, width=320)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.10, 0.10, 0.15]
        cam.distance, cam.elevation, cam.azimuth = 1.35, -22.0, 130.0

    def on_tick(game):
        if rec is None:
            return
        ren.update_scene(game.d, cam)
        rec.add(game.observation(), [0, 0, 0, 0, float(game.closed)], ren.render())

    scripted_grasp(g, on_tick=on_tick)
    print("lift       cube_z", round(float(g.cube_pos()[2]), 3), " held:", g.held())
    drive_to(g, [*g.target, 0.30], True, 400, on_tick=on_tick)
    drive_to(g, [*g.target, 0.09], True, 250, on_tick=on_tick)
    print("transit    |qvel|", round(float(np.linalg.norm(g.d.qvel[g.arm.dof])), 3))
    scored = False
    for _ in range(200):
        scored |= bool(g.step(0, 0, 0, 0, False))
        on_tick(g)
    print("final      ", np.round(g.cube_pos(), 3), " score:", g.score,
          "->", "SCORED" if scored or g.score else "MISS")
    if rec is not None:
        print("recorded   ", len(rec), "ticks ->", rec.save())
    return 0 if g.score else 1


def _headless_benchmark(env_spec: str, seed: int, record=None, ticks: int = 60) -> int:
    """Smoke test for a benchmark backend: build it, drive it, report.

    Deliberately not an acceptance test -- these environments have their own
    success criteria and we are only proving the adapter wiring is live.
    """
    import benchmarks
    from recorder import EpisodeRecorder

    env = benchmarks.make(env_spec, seed=seed)
    print(f"env        {env_spec}  ({env.task})")
    print(f"control_dt {env.control_dt:.4f}s   state dim {env.observation().size}")
    rec = EpisodeRecorder(record, state_names=env.state_names,
                          action_names=env.action_names, task=env.task) \
        if record is not None else None

    scored = 0
    for i in range(ticks):
        action = (0.0, 0.0, -0.4, 0.0, i > ticks // 2)     # descend, then close
        scored += bool(env.step(*action, env.control_dt))
        if rec is not None:
            rec.add(env.observation(),
                    [*action[:4], float(action[4])], env.frame(320, 240))
    hud = env.hud()
    print(f"after {ticks} ticks: ee {np.round(hud.ee, 3)}  score {hud.score}")
    if rec is not None:
        print("recorded   ", len(rec), "ticks ->", rec.save())
    env.close()
    return 0


def run_game(seed=None, record=None, env_spec: str = "native") -> int:
    import pygame

    import benchmarks
    from input import GamepadReader, KeyboardReader, merge
    from render import CAM_SPEED, RENDER_H, RENDER_W, Display

    if seed is None:
        seed = int(np.random.randint(1 << 30))
    env = benchmarks.make(env_spec, seed=seed)
    display = Display(caption=f"Skillcrane - {env_spec}")

    pad = GamepadReader.open()
    if pad:
        print(f"gamepad: {pad.name}  axes={pad.pad.get_numaxes()} "
              f"buttons={pad.pad.get_numbuttons()}")
    else:
        print("no gamepad found - keyboard only (WASD/QE/ZC/SPACE). "
              "Run gamepad_probe.py if yours is connected but idle.")
    kb = KeyboardReader()

    rec = None
    if record is not None:
        from recorder import EpisodeRecorder
        rec = EpisodeRecorder(record, state_names=env.state_names,
                              action_names=env.action_names, task=env.task)
        print(f"recording to {record}")

    frame_dt = 1 / 60
    accum = 0.0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        ci = kb.read(pygame.key.get_pressed())
        if pad:
            ci = merge(ci, pad.read())
        if ci.reset:
            env.reset(full=True)
        if ci.cam:
            env.orbit(ci.cam * CAM_SPEED * frame_dt)

        dx, dy = ci.world_xy(math.radians(env.azimuth))
        frame = env.frame(RENDER_W, RENDER_H)

        # Each environment runs at its own control rate (100 Hz here, 20 Hz for
        # most benchmarks), so drive it from a time accumulator rather than
        # assuming a fixed number of ticks per drawn frame.
        accum += frame_dt
        scored = False
        steps = 0
        while accum >= env.control_dt and steps < MAX_STEPS_PER_FRAME:
            scored |= bool(env.step(dx, dy, ci.mz, ci.dyaw, ci.grip, env.control_dt))
            accum -= env.control_dt
            steps += 1
            if rec is not None:
                rec.add(env.observation(),
                        [dx, dy, ci.mz, ci.dyaw, float(ci.grip)],
                        frame if steps == 1 else None)

        display.draw(env.hud(), frame, scored, frame_dt)
        display.tick(60)

    if rec is not None and len(rec):
        print(f"recorded {len(rec)} ticks -> {rec.save()}")
    env.close()
    display.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Skillcrane teleop + data collection")
    ap.add_argument("--env", default="native", metavar="FAMILY[:TASK]",
                    help="environment to drive, e.g. robosuite:Lift (see --list-envs)")
    ap.add_argument("--list-envs", action="store_true",
                    help="show benchmark environments and whether they are installed")
    ap.add_argument("--headless", action="store_true",
                    help="run a scripted pick-and-place with no window")
    ap.add_argument("--seed", type=int, default=None, help="spawn / task seed")
    # Off unless asked for: bare --record collects into DEFAULT_RECORD_DIR,
    # --record DIR picks the directory.
    ap.add_argument("--record", metavar="DIR", nargs="?",
                    default=None, const=DEFAULT_RECORD_DIR,
                    help="collect this session as a LeRobot dataset in DIR "
                         f"(default {DEFAULT_RECORD_DIR}); off when omitted")
    a = ap.parse_args(argv)

    if a.list_envs:
        import benchmarks
        print(benchmarks.describe())
        return 0
    if a.headless:
        return run_headless(seed=2 if a.seed is None else a.seed,
                            record=a.record, env_spec=a.env)
    return run_game(seed=a.seed, record=a.record, env_spec=a.env)


if __name__ == "__main__":
    sys.exit(main())
