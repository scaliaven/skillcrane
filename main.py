"""Claw Crew -- teleoperate a 6-DoF arm in MuJoCo with an 8BitDo gamepad.

    python main.py                 play (gamepad if present, else keyboard)
    python main.py --headless      scripted pick-and-place, no window
    python main.py --seed 7        fixed cube spawns
    python main.py --record runs/  log the session in LeRobot dataset format

Runs under plain `python`, not `mjpython`: MuJoCo renders offscreen and pygame
owns the window, so nothing fights over the macOS main thread.

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
from game import Game, scripted_grasp, drive_to


def run_headless(seed: int = 2, record=None) -> int:
    """Scripted smoke test. Exit 0 if the round scored."""
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
        # The scripted run has no stick input; the action is the target motion
        # the script asked for, recovered from the commanded gripper state.
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


def run_game(seed=None, record=None) -> int:
    import pygame
    from render import Display
    from input import GamepadReader, KeyboardReader, merge

    if seed is None:
        seed = int(np.random.randint(1 << 30))
    g = Game(seed=seed)
    display = Display(g.m)

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
        rec = EpisodeRecorder(record)
        print(f"recording to {record}")

    frame_dt = 1 / 60
    ticks_per_frame = max(1, int(round(frame_dt / scene.CTRL_DT)))
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
            g.reset(full=True)

        display.orbit(ci.cam, frame_dt)
        dx, dy = ci.world_xy(math.radians(display.cam.azimuth))

        frame = display.frame(g.d) if rec is not None else None
        scored = False
        for i in range(ticks_per_frame):
            scored |= bool(g.step(dx, dy, ci.mz, ci.dyaw, ci.grip))
            if rec is not None:
                # One row per control tick; only the first carries a new image.
                rec.add(g.observation(), [dx, dy, ci.mz, ci.dyaw, float(ci.grip)],
                        frame if i == 0 else None)

        display.draw(g, scored, frame_dt)
        display.tick(60)

    if rec is not None and len(rec):
        print(f"recorded {len(rec)} ticks -> {rec.save()}")
    display.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Claw Crew robot arm teleop game")
    ap.add_argument("--headless", action="store_true",
                    help="run a scripted pick-and-place with no window")
    ap.add_argument("--seed", type=int, default=None, help="cube spawn seed")
    ap.add_argument("--record", metavar="DIR", default=None,
                    help="log the episode to DIR in LeRobot dataset format")
    a = ap.parse_args(argv)
    if a.headless:
        return run_headless(seed=2 if a.seed is None else a.seed, record=a.record)
    return run_game(seed=a.seed, record=a.record)


if __name__ == "__main__":
    sys.exit(main())
