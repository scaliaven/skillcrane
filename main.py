"""Skillcrane -- teleoperate a robot arm in MuJoCo, and record what you did.

    python main.py                        play the native Skillcrane arm
    python main.py --list-envs            show benchmark environments
    python main.py --env robosuite:Lift   teleop a benchmark instead
    python main.py --headless             scripted pick-and-place, no window
    python main.py --record               collect the session into runs/
    python main.py --record DIR           ... into DIR instead (LeRobot format)
    python main.py --record --record-views all      log every camera, not just one

Runs under plain `python`, not `mjpython`: environments render offscreen and
pygame owns the window, so nothing fights over the macOS main thread.

Controls
    left stick    move gripper horizontally (camera-relative)
    right stick Y raise / lower          right stick X  rotate wrist
    A             toggle gripper          LB / RB       orbit camera
    X             cycle view layout       d-pad L/R     previous / next task
    Back / Start  previous / next environment family
    Y             reset the round
    keyboard      WASD move, QE up/down, ZC wrist, SPACE grip, R reset,
                  arrow keys orbit, V views, , / . task, [ / ] family, ESC quit
"""
import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

import scene
from game import drive_to, scripted_grasp

MAX_STEPS_PER_FRAME = 8      # keeps a slow env from spiralling on catch-up
DEFAULT_RECORD_DIR = "runs/"  # where a bare --record collects to
# Recorded frames are deliberately not the ones on screen: a dataset column has
# one image shape for the whole episode, and the operator can change the layout
# mid-round. So recording renders its own fixed-size views.
RECORD_W, RECORD_H = 320, 240
DEFAULT_RECORD_VIEWS = "main"


@dataclass
class Session:
    """What a switch replaces as a unit: the environment, its name, its logging.

    Switching either the family or the task rebuilds the environment, and both
    change the observation schema -- so the recorder and the set of cameras
    being recorded have to travel with it rather than be tracked separately.
    """
    env: object
    spec: str
    rec: object = None
    rec_sizes: dict = field(default_factory=dict)


def _new_recorder(root, env=None):
    """Recorder for `env`'s schema, writing into the next free episode slot.

    Episode index comes from the directory, not a counter, so a second run into
    the same DIR adds an episode instead of overwriting the first.
    """
    from recorder import EpisodeRecorder, next_episode_index
    kw = {} if env is None else {"state_names": env.state_names,
                                 "action_names": env.action_names, "task": env.task}
    return EpisodeRecorder(root, episode_index=next_episode_index(root), **kw)


def _record_sizes(env, views: str) -> dict:
    """{view: (w, h)} to record, from --record-views.

    "main" is the operator's own view only, "all" is every camera the env has,
    and anything else is a comma-separated list of view names. A name the env
    does not have is reported rather than silently dropped -- a typo would
    otherwise look like a camera that quietly recorded nothing.
    """
    names = list(env.view_names)
    if views == "all":
        chosen = names
    elif views == "main":
        chosen = names[:1]
    else:
        wanted = [v.strip() for v in views.split(",") if v.strip()]
        chosen = [v for v in wanted if v in names]
        missing = [v for v in wanted if v not in names]
        if missing:
            print(f"no camera named {', '.join(missing)} here; "
                  f"this env has: {', '.join(names)}")
    return {v: (RECORD_W, RECORD_H) for v in chosen}


def run_headless(seed: int = 2, record=None, env_spec: str = "native",
                 record_views: str = DEFAULT_RECORD_VIEWS) -> int:
    """Scripted acceptance run for the native arm; a smoke test for benchmarks."""
    if env_spec != "native":
        return _headless_benchmark(env_spec, seed, record, record_views=record_views)

    import benchmarks

    # Goes through the adapter rather than Game directly, so the headless run
    # records through exactly the same camera path as a played round.
    env = benchmarks.make("native", seed=seed)
    g = env.game
    print("cube spawn ", np.round(g.cube_pos(), 3), " ncon at rest:", g.d.ncon)

    rec = None
    sizes = {}
    if record is not None:
        rec = _new_recorder(record, env)
        sizes = _record_sizes(env, record_views)
        print("recording views:", ", ".join(sizes) or "(none)")

    def on_tick(game):
        if rec is None:
            return
        rec.add(game.observation(), [0, 0, 0, 0, float(game.closed)],
                env.frames(sizes))

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
    env.close()
    return 0 if g.score else 1


def _headless_benchmark(env_spec: str, seed: int, record=None, ticks: int = 60,
                        record_views: str = DEFAULT_RECORD_VIEWS) -> int:
    """Smoke test for a benchmark backend: build it, drive it, report.

    Deliberately not an acceptance test -- these environments have their own
    success criteria and we are only proving the adapter wiring is live.
    """
    import benchmarks

    env = benchmarks.make(env_spec, seed=seed)
    print(f"env        {env_spec}  ({env.task})")
    print(f"control_dt {env.control_dt:.4f}s   state dim {env.observation().size}")
    print(f"views      {', '.join(env.view_names)}")
    rec = _new_recorder(record, env) if record is not None else None
    sizes = _record_sizes(env, record_views) if rec is not None else {}

    scored = 0
    for i in range(ticks):
        action = (0.0, 0.0, -0.4, 0.0, i > ticks // 2)     # descend, then close
        scored += bool(env.step(*action, env.control_dt))
        if rec is not None:
            rec.add(env.observation(),
                    [*action[:4], float(action[4])], env.frames(sizes))
    hud = env.hud()
    print(f"after {ticks} ticks: ee {np.round(hud.ee, 3)}  score {hud.score}")
    if rec is not None:
        print("recorded   ", len(rec), "ticks ->", rec.save())
    env.close()
    return 0


def _open(spec: str, seed: int, record, views: str) -> Session:
    """Build `spec` and everything that has to be replaced along with it."""
    import benchmarks

    return _session(benchmarks.make(spec, seed=seed), spec, record, views)


def _session(env, spec: str, record, views: str) -> Session:
    if record is None:
        return Session(env, spec)
    rec = _new_recorder(record, env)
    sizes = _record_sizes(env, views)
    print(f"recording to {record}  (episode {rec.episode_index}, "
          f"views: {', '.join(sizes) or 'none'})")
    return Session(env, spec, rec, sizes)


def _switch(s: Session, new_spec: str, seed, record, views, nothing: str) -> Session:
    """Rebuild the session on `new_spec` without leaving the round.

    A switch changes the observation and action schema -- a different family has
    a different state width, a different task has different objects in it -- so
    a recording in progress is closed out here and the next episode opens under
    the new environment's columns. Otherwise one parquet would carry two
    different meanings of `observation.state`.

    Returns the old session unchanged if there is nowhere to go or the new
    environment fails to build: a bad install must not end the session.
    """
    import benchmarks

    if new_spec == s.spec:
        print(nothing)
        return s
    try:
        env = benchmarks.make(new_spec, seed=seed)
    except (SystemExit, Exception) as exc:
        # Broad on purpose: a backend that fails to build (a bad install, a
        # missing asset download) must not take the live session down with it.
        print(f"could not switch to {new_spec}: {exc}")
        return s

    if s.rec is not None and len(s.rec):
        print(f"recorded {len(s.rec)} ticks -> {s.rec.save()}")
    s.env.close()
    print(f"env -> {new_spec}  ({env.task})   views: {', '.join(env.view_names)}")
    return _session(env, new_spec, record, views)


def run_game(seed=None, record=None, env_spec: str = "native",
             record_views: str = DEFAULT_RECORD_VIEWS) -> int:
    import pygame

    import benchmarks
    from input import GamepadReader, KeyboardReader, merge
    from render import CAM_SPEED, Display

    if seed is None:
        seed = int(np.random.randint(1 << 30))
    s = _open(env_spec, seed, record, record_views)
    display = Display(caption=f"Skillcrane - {s.spec}")
    print("environments:", "  ".join(benchmarks.switchable()),
          "   ([ / ] or Back/Start)")
    print("tasks:", ", ".join(benchmarks.tasks(s.spec)) or "(one)",
          "   (, / . or d-pad)")
    print("views:", ", ".join(s.env.view_names), "   (V or X cycles the layout)")

    pad = GamepadReader.open()
    if pad:
        print(f"gamepad: {pad.name}  axes={pad.pad.get_numaxes()} "
              f"buttons={pad.pad.get_numbuttons()}")
    else:
        print("no gamepad found - keyboard only (WASD/QE/ZC/SPACE). "
              "Run gamepad_probe.py if yours is connected but idle.")
    kb = KeyboardReader()

    frame_dt = 1 / 60
    accum = 0.0
    running = True
    # Switching and cycling step once per press, not once per frame.
    env_latch = task_latch = view_latch = False
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
            s.env.reset(full=True)
        if ci.env and not env_latch:
            s = _switch(s, benchmarks.cycle(s.spec, ci.env), seed, record,
                        record_views,
                        nothing=f"only {s.spec} is installed -- nothing to switch "
                                f"to; see python main.py --list-envs")
            display.set_caption(f"Skillcrane - {s.spec}")
            accum = 0.0         # the new env has its own control rate
        env_latch = bool(ci.env)
        if ci.task and not task_latch:
            s = _switch(s, benchmarks.cycle_task(s.spec, ci.task), seed, record,
                        record_views,
                        nothing=f"{benchmarks.parse(s.spec)[0]} has one task setting")
            display.set_caption(f"Skillcrane - {s.spec}")
            accum = 0.0
        task_latch = bool(ci.task)
        if ci.view and not view_latch:
            layout = display.cycle_layout()
            print(f"views -> {layout}: "
                  f"{', '.join(display.view_sizes(s.env.view_names))}")
        view_latch = bool(ci.view)
        if ci.cam:
            s.env.orbit(ci.cam * CAM_SPEED * frame_dt)

        dx, dy = ci.world_xy(math.radians(s.env.azimuth))
        # Rendered at exactly the panel sizes the layout will blit into.
        views = s.env.frames(display.view_sizes(s.env.view_names))
        logged = s.env.frames(s.rec_sizes) if s.rec is not None else None

        # Each environment runs at its own control rate (100 Hz here, 20 Hz for
        # most benchmarks), so drive it from a time accumulator rather than
        # assuming a fixed number of ticks per drawn frame.
        accum += frame_dt
        scored = False
        steps = 0
        while accum >= s.env.control_dt and steps < MAX_STEPS_PER_FRAME:
            scored |= bool(s.env.step(dx, dy, ci.mz, ci.dyaw, ci.grip,
                                      s.env.control_dt))
            accum -= s.env.control_dt
            steps += 1
            if s.rec is not None:
                s.rec.add(s.env.observation(),
                          [dx, dy, ci.mz, ci.dyaw, float(ci.grip)],
                          logged if steps == 1 else None)

        display.draw(s.env.hud(), views, scored, frame_dt,
                     # The layout is obvious from the panels; the environment
                     # name is not, so that is what the line spends room on.
                     controls=f"[{s.spec}]  L-stick move  R-stick lift/rot  "
                              f"A grip  LB/RB cam  X view  d-pad task  "
                              f"Back/Start env  Y reset")
        display.tick(60)

    if s.rec is not None and len(s.rec):
        print(f"recorded {len(s.rec)} ticks -> {s.rec.save()}")
    s.env.close()
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
    ap.add_argument("--record-views", metavar="VIEWS", default=DEFAULT_RECORD_VIEWS,
                    help="cameras to record: main (default), all, or a "
                         "comma-separated list of view names. Every view is its "
                         "own image column, and its own PNG per tick")
    a = ap.parse_args(argv)

    if a.list_envs:
        import benchmarks
        print(benchmarks.describe())
        return 0
    if a.headless:
        return run_headless(seed=2 if a.seed is None else a.seed,
                            record=a.record, env_spec=a.env,
                            record_views=a.record_views)
    return run_game(seed=a.seed, record=a.record, env_spec=a.env,
                    record_views=a.record_views)


if __name__ == "__main__":
    sys.exit(main())
