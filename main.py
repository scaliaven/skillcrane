"""Skillcrane -- teleoperate a robot arm in MuJoCo, and record what you did.

    python main.py                        play the native Skillcrane arm
    python main.py --list-envs            show the benchmark suites and tasks
    python main.py --env robosuite:Lift   teleop a benchmark suite instead
    python main.py --env libero           ... its first task
    python main.py --headless             scripted pick-and-place, no window
    python main.py --record               collect the session into runs/
    python main.py --record DIR           ... into DIR instead (LeRobot format)
    python main.py --record --record-views all      log every camera, not just one
    python main.py --window 1600x1000     open a bigger window (it also resizes)
    python main.py --eval 12              score a policy over 12 seeds, no window
    python main.py --eval 12 --record D   ... and collect them as a dataset
    python main.py --eval 1 --policy replay:runs/   replay a recorded episode

Runs under plain `python`, not `mjpython`: environments render offscreen and
pygame owns the window, so nothing fights over the macOS main thread.

An environment is a *suite* and a *task*: "robosuite:Lift" is the robosuite
suite on its Lift task. The two are switched separately while playing, because
they are different moves -- a suite change swaps the robot and the simulator, a
task change keeps both and changes the scene.

Controls
    left stick    move gripper horizontally (camera-relative)
    right stick Y raise / lower          right stick X  rotate wrist
    A             toggle gripper          LB / RB       orbit camera
    LT / RT       zoom out / in           B             camera follow on/off
    X             cycle view layout       d-pad L/R     previous / next TASK
    Back / Start  previous / next SUITE (benchmark)
    Y             reset the round
    keyboard      WASD move, QE up/down, ZC wrist, SPACE grip, R reset,
                  arrows orbit, -/= zoom, F follow, V views,
                  , / . task, [ / ] suite, ESC quit
"""
import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

import scene

MAX_STEPS_PER_FRAME = 8      # keeps a slow env from spiralling on catch-up
DEFAULT_RECORD_DIR = "runs/"  # where a bare --record collects to
# Recorded frames are deliberately not the ones on screen: a dataset column has
# one image shape for the whole episode, and the operator can change the layout
# mid-round -- and now the window itself resizes. So recording renders its own
# fixed-size views, at --record-size, independent of the window.
RECORD_SIZE = (320, 240)
DEFAULT_RECORD_VIEWS = "main"
DEFAULT_POLICY = "scripted"


@dataclass
class Session:
    """What a switch replaces as a unit: the environment, its name, its logging.

    Switching either the suite or the task rebuilds the environment, and both
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


def _record_sizes(env, views: str, size=RECORD_SIZE) -> dict:
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
    return {v: tuple(size) for v in chosen}


def run_headless(seed: int = 2, record=None, env_spec: str = "native",
                 record_views: str = DEFAULT_RECORD_VIEWS,
                 record_size=RECORD_SIZE) -> int:
    """Scripted acceptance run for the native arm; a smoke test for benchmarks."""
    if env_spec != "native":
        return _headless_benchmark(env_spec, seed, record, record_views=record_views,
                                   record_size=record_size)

    import benchmarks
    from policy import ScriptedPickPlace, rollout

    # Goes through the adapter rather than Game directly, so the headless run
    # records through exactly the same camera path as a played round.
    env = benchmarks.make("native", seed=seed)
    g = env.game
    print("cube spawn ", np.round(g.cube_pos(), 3), " ncon at rest:", g.d.ncon)

    rec = None
    sizes = {}
    if record is not None:
        rec = _new_recorder(record, env)
        sizes = _record_sizes(env, record_views, record_size)
        print("recording views:", ", ".join(sizes) or "(none)",
              f"at {record_size[0]}x{record_size[1]}")

    # The policy, not game.scripted_pick_and_place. Both walk game.WAYPOINTS;
    # the difference is that the script moves the Cartesian target directly, so a
    # round recorded from it logged [0, 0, 0, 0, grip] on every tick while the
    # arm crossed the table. The dataset was unlearnable and looked fine.
    pol = ScriptedPickPlace()
    reported = set()

    def on_tick(env, action, scored):
        # The phase has already advanced when a leg finishes, so seeing a new
        # name here means the previous one just completed -- which is where the
        # two progress lines belong.
        if pol.phase not in reported:
            reported.add(pol.phase)
            if pol.phase == "carry":
                print("lift       cube_z", round(float(g.cube_pos()[2]), 3),
                      " held:", g.held())
            elif pol.phase == "release":
                print("transit    |qvel|",
                      round(float(np.linalg.norm(g.d.qvel[g.arm.dof])), 3))
        if rec is not None:
            rec.add(env.observation(), action, env.frames(sizes),
                    reward=float(scored))

    r = rollout(env, pol, on_tick=on_tick, seed=seed)
    print("final      ", np.round(g.cube_pos(), 3), " score:", g.score,
          "->", "SCORED" if r.success else "MISS")
    if rec is not None:
        print("recorded   ", len(rec), "ticks ->", rec.save(),
              f"(success={rec.success()})")
    env.close()
    return 0 if r.success else 1


def _headless_benchmark(env_spec: str, seed: int, record=None, ticks: int = 60,
                        record_views: str = DEFAULT_RECORD_VIEWS,
                        record_size=RECORD_SIZE) -> int:
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
    sizes = _record_sizes(env, record_views, record_size) if rec is not None else {}

    for i in range(ticks):
        action = (0.0, 0.0, -0.4, 0.0, i > ticks // 2)     # descend, then close
        hit = bool(env.step(*action, env.control_dt))
        if rec is not None:
            rec.add(env.observation(),
                    [*action[:4], float(action[4])], env.frames(sizes),
                    reward=float(hit))
    hud = env.hud()
    print(f"after {ticks} ticks: ee {np.round(hud.ee, 3)}  score {hud.score}")
    if rec is not None:
        print("recorded   ", len(rec), "ticks ->", rec.save())
    env.close()
    return 0


def _make_policy(spec: str, size=RECORD_SIZE):
    """Turn a --policy string into a Policy.

    Same shape as --env: a kind, a colon, and what the kind needs. One instance
    serves every seed -- `rollout` rewinds it through `reset` -- so a replayed
    episode is read once here and a checkpoint is loaded once.
    """
    import policy as pol

    kind, _, rest = str(spec).partition(":")
    if kind == "scripted":
        return pol.ScriptedPickPlace()
    if kind == "replay":
        root, ep = rest, 0
        head, sep, tail = rest.rpartition(":")
        # "replay:runs" is episode 0; "replay:runs:3" is episode 3. Split from
        # the right and only when the tail is digits, so a Windows-ish or
        # colon-bearing path is not mistaken for an episode number.
        if sep and tail.isdigit():
            root, ep = head, int(tail)
        if not root:
            raise SystemExit("--policy replay:DIR wants the dataset directory")
        loaded = pol.ReplayPolicy.from_dataset(root, ep)
        print(f"replaying episode {ep} of {root}: {len(loaded)} ticks")
        return loaded
    if kind in ("act", "lerobot"):
        if not rest:
            raise SystemExit("--policy act:PATH wants a checkpoint directory")
        return pol.LeRobotPolicy(rest, size=size)
    raise SystemExit(f"unknown --policy {spec!r}; want scripted, "
                     f"replay:DIR[:EP] or act:PATH")


def run_eval(episodes: int, seed=None, policy_spec: str = DEFAULT_POLICY,
             record=None, env_spec: str = "native",
             record_views: str = DEFAULT_RECORD_VIEWS, record_size=RECORD_SIZE) -> int:
    """Score a policy over `episodes` seeds, headless, and report per seed.

    The seeds are consecutive from --seed so a run is reproducible and two
    policies can be compared on the same worlds -- an average over different
    spawns is not a comparison. Exits non-zero unless every episode succeeded:
    the number to read is the table, but a rig where the scripted policy stops
    scoring is broken, and CI should hear about it.
    """
    import benchmarks
    from policy import evaluate, summarise

    the_policy = _make_policy(policy_spec, record_size)
    suite = benchmarks.parse(env_spec)[0]
    # Asked of the policy and the registry rather than compared against the
    # string "native": `--env native:default` is a spec the registry itself
    # produces, and a literal here rejects it.
    if the_policy.suites and suite not in the_policy.suites:
        # Caught before an env is built, rather than as a TypeError out of the
        # first waypoint several hundred ticks in.
        raise SystemExit(
            f"--policy {policy_spec} is {'/'.join(the_policy.suites)}-only; "
            f"{suite} has no cube of ours. Use --policy replay:DIR or act:PATH")
    seed = 0 if seed is None else seed
    seeds = list(range(seed, seed + max(1, int(episodes))))
    print(f"eval       {env_spec}  policy {policy_spec}  seeds {seeds[0]}..{seeds[-1]}")

    # One episode per seed, so the recorder is opened and closed around each
    # rollout rather than spanning the whole run: a dataset wants an episode
    # boundary wherever the world was rebuilt.
    rec, sizes = None, {}

    def on_start(s, env):
        nonlocal rec, sizes
        if record is None:
            return
        rec = _new_recorder(record, env)
        sizes = _record_sizes(env, record_views, record_size)

    def on_tick(env, action, scored):
        if rec is not None:
            rec.add(env.observation(), action, env.frames(sizes),
                    reward=float(scored))

    def on_episode(r):
        nonlocal rec
        if rec is not None and len(rec):
            print(f"  seed {r.seed}: {len(rec)} ticks -> {rec.save()} "
                  f"(success={rec.success()})")
        rec = None

    results = evaluate(lambda s: benchmarks.make(env_spec, seed=s), the_policy,
                       seeds, on_start=on_start, on_tick=on_tick,
                       on_episode=on_episode)
    print(summarise(results))
    return 0 if all(r.success for r in results) else 1


def _ring(items, current) -> str:
    """"2/6" -- where `current` sits in the ring it can be cycled around.

    On screen next to the suite and the task, because the commonest report about
    the switch keys is that they do nothing, and the commonest reason is a ring
    one entry long: nothing installed beside the native arm, or a suite with a
    single task. "1/1" answers that without the operator having to go and read
    the terminal.
    """
    items = list(items)
    if not items:
        return ""
    i = items.index(current) if current in items else 0
    return f"{i + 1}/{len(items)}"


def _open(spec: str, seed: int, record, views: str, size=RECORD_SIZE) -> Session:
    """Build `spec` and everything that has to be replaced along with it."""
    import benchmarks

    return _session(benchmarks.make(spec, seed=seed), spec, record, views, size)


def _session(env, spec: str, record, views: str, size=RECORD_SIZE) -> Session:
    if record is None:
        return Session(env, spec)
    rec = _new_recorder(record, env)
    sizes = _record_sizes(env, views, size)
    print(f"recording to {record}  (episode {rec.episode_index}, "
          f"views: {', '.join(sizes) or 'none'})")
    return Session(env, spec, rec, sizes)


def _switch(s: Session, new_spec: str, seed, record, views, nothing: str,
            what: str = "env", size=RECORD_SIZE, say=print) -> Session:
    """Rebuild the session on `new_spec` without leaving the round.

    `what` is the noun to report it as -- "suite" or "task" -- because from here
    the two switches are the same operation and only the operator can tell them
    apart, which they cannot do if both print the same word.

    `say` reports the outcome. It defaults to print, but the live loop passes
    one that also puts the line on the HUD: every branch below can end in
    "nothing changed", and an operator watching the window would otherwise read
    a correct no-op as a dead key.

    A switch changes the observation and action schema -- a different suite has
    a different state width, a different task has different objects in it -- so
    a recording in progress is closed out here and the next episode opens under
    the new environment's columns. Otherwise one parquet would carry two
    different meanings of `observation.state`.

    Returns the old session unchanged if there is nowhere to go or the new
    environment fails to build: a bad install must not end the session.
    """
    import benchmarks

    if new_spec == s.spec:
        say(nothing)
        return s
    try:
        env = benchmarks.make(new_spec, seed=seed)
    except (SystemExit, Exception) as exc:
        # Broad on purpose: a backend that fails to build (a bad install, a
        # missing asset download) must not take the live session down with it.
        say(f"could not switch to {new_spec}: {exc}")
        return s

    if s.rec is not None and len(s.rec):
        print(f"recorded {len(s.rec)} ticks -> {s.rec.save()} "
              f"(success={s.rec.success()})")
    s.env.close()
    suite, task = benchmarks.parse(new_spec)
    say(f"{what} -> {new_spec}   suite {suite}  task {task or '(default)'}   "
        f"views: {', '.join(env.view_names)}")
    return _session(env, new_spec, record, views, size)


def run_game(seed=None, record=None, env_spec: str = "native",
             record_views: str = DEFAULT_RECORD_VIEWS, record_size=RECORD_SIZE,
             window=None) -> int:
    import pygame

    import benchmarks
    from input import GamepadReader, KeyboardReader, merge
    from render import CAM_SPEED, Display

    if seed is None:
        seed = int(np.random.randint(1 << 30))
    s = _open(env_spec, seed, record, record_views, record_size)
    display = Display(caption=f"Skillcrane - {s.spec}", size=window)
    # Spelled out at startup because the two rings are the thing operators get
    # wrong: the suite ring holds only what is installed, the task ring is
    # whatever the suite you are on offers.
    print("suites installed:", "  ".join(benchmarks.suites()),
          "   ([ / ] or Back/Start switches suite;",
          "python main.py --list-envs shows the rest)")
    print("tasks in", benchmarks.parse(s.spec)[0] + ":",
          ", ".join(benchmarks.tasks(s.spec)) or "(one)",
          "   (, / . or d-pad switches task)")
    print("views:", ", ".join(s.env.view_names), "   (V or X cycles the layout,",
          "-/= or the triggers zoom, F or B toggles camera follow)")
    print(f"window: {display.screen.get_size()[0]}x{display.screen.get_size()[1]}"
          f"  -- drag the corner to resize, or start with --window WxH")

    pad = GamepadReader.open()
    if pad:
        print(f"gamepad: {pad.name}  axes={pad.pad.get_numaxes()} "
              f"buttons={pad.pad.get_numbuttons()}")
    else:
        print("no gamepad found - keyboard only (WASD/QE/ZC/SPACE). "
              "Run gamepad_probe.py if yours is connected but idle.")
    kb = KeyboardReader()

    def say(text: str) -> None:
        """Report a switch to both places the operator might be looking.

        The terminal keeps the full line for later; the HUD gets it too, because
        "nothing happened" is a legitimate outcome of pressing [ or ] and the
        window is where the operator's eyes are.
        """
        print(text)
        display.notify(text)

    frame_dt = 1 / 60
    accum = 0.0
    running = True
    # Switching and cycling step once per press, not once per frame.
    suite_latch = task_latch = view_latch = follow_latch = False
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                # Every panel is re-rendered at the new size next frame; the
                # environment is asked for those sizes, so nothing is upscaled
                # from the old window.
                display.resize(ev.w, ev.h)

        ci = kb.read(pygame.key.get_pressed())
        if pad:
            ci = merge(ci, pad.read())
        if ci.reset:
            s.env.reset(full=True)
        # Two switches, deliberately kept apart: `suite` swaps the benchmark,
        # `task` swaps the setting inside the one already running.
        if ci.suite and not suite_latch:
            s = _switch(s, benchmarks.cycle_suite(s.spec, ci.suite), seed, record,
                        record_views, what="suite", size=record_size, say=say,
                        # Short enough to survive the HUD's trim at the
                        # default window: this line is the answer to "why did
                        # nothing happen", so its actionable half must not be
                        # the half that gets cut.
                        nothing=f"{s.spec} is the only suite installed -- "
                                f"python main.py --list-envs")
            display.set_caption(f"Skillcrane - {s.spec}")
            accum = 0.0         # the new env has its own control rate
        suite_latch = bool(ci.suite)
        if ci.task and not task_latch:
            suite = benchmarks.parse(s.spec)[0]
            s = _switch(s, benchmarks.cycle_task(s.spec, ci.task), seed, record,
                        record_views, what="task", size=record_size, say=say,
                        nothing=f"{suite} has one task -- [ ] or Back/Start "
                                f"switches suite instead")
            display.set_caption(f"Skillcrane - {s.spec}")
            accum = 0.0
        task_latch = bool(ci.task)
        if ci.view and not view_latch:
            layout = display.cycle_layout()
            say(f"views -> {layout}: "
                f"{', '.join(display.view_sizes(s.env.view_names))}")
        view_latch = bool(ci.view)
        if ci.follow and not follow_latch:
            say(f"camera follow {'on' if s.env.toggle_follow() else 'off'}")
        follow_latch = bool(ci.follow)
        if ci.cam:
            s.env.orbit(ci.cam * CAM_SPEED * frame_dt)
        if ci.zoom:
            s.env.zoom(ci.zoom * frame_dt)

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
            hit = bool(s.env.step(dx, dy, ci.mz, ci.dyaw, ci.grip,
                                  s.env.control_dt))
            scored |= hit
            accum -= s.env.control_dt
            steps += 1
            if s.rec is not None:
                # Per tick, not per frame: `scored` is the flash on the HUD and
                # covers the whole frame, but the reward column has to name the
                # one tick the cube actually landed.
                s.rec.add(s.env.observation(),
                          [dx, dy, ci.mz, ci.dyaw, float(ci.grip)],
                          logged if steps == 1 else None, reward=float(hit))

        suite, task = benchmarks.parse(s.spec)
        display.draw(s.env.hud(), views, scored, frame_dt,
                     # Each switchable thing next to *both* ways to switch it,
                     # keyboard first -- the HUD used to name only the pad, so a
                     # keyboard operator had no way to learn [ ] and , . exist.
                     #
                     # The counts are the other half of that: "1/1" says there
                     # is nowhere to go, so a key that does nothing reads as an
                     # empty ring rather than as a dead binding.
                     status=f"suite {suite} {_ring(benchmarks.suites(), suite)} "
                            f"<[ ] Back/Start>   "
                            f"task {task or 'default'} "
                            f"{_ring(benchmarks.tasks(s.spec), task)} "
                            f"<, . d-pad>   "
                            f"views {display.layout} <V X>: "
                            f"{', '.join(display.view_sizes(s.env.view_names))}",
                     # Both devices on one line, key first, sized to fit the
                     # default window without being trimmed. The suite, task
                     # and view bindings are not repeated here -- they are on
                     # the status line, next to what they change.
                     controls="move WASD/stick  lift QE  wrist ZC  grip SPC/A  "
                              "cam arrows/LB-RB  zoom -=/LT-RT  follow F/B  "
                              "reset R/Y")
        display.tick(60)

    if s.rec is not None and len(s.rec):
        print(f"recorded {len(s.rec)} ticks -> {s.rec.save()} "
              f"(success={s.rec.success()})")
    s.env.close()
    display.close()
    return 0


def _size(text: str, what: str) -> tuple:
    """Parse a WxH argument, or exit saying what was wrong with it."""
    try:
        w, h = (int(v) for v in str(text).lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--{what} wants WxH, e.g. 1600x1000 (got {text!r})")
    return (w, h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Skillcrane teleop + data collection")
    ap.add_argument("--env", default="native", metavar="SUITE[:TASK]",
                    help="environment to drive: a benchmark suite and one of its "
                         "tasks, e.g. robosuite:Lift. A bare suite name takes its "
                         "first task (see --list-envs)")
    ap.add_argument("--list-envs", "--list-suites", action="store_true",
                    dest="list_envs",
                    help="show the benchmark suites, their tasks, and whether "
                         "each suite is installed")
    ap.add_argument("--headless", action="store_true",
                    help="run a scripted pick-and-place with no window")
    ap.add_argument("--eval", metavar="N", type=int, default=None,
                    help="score a policy over N seeds (from --seed) with no "
                         "window, and print a per-seed table. Exits non-zero "
                         "unless every episode succeeded")
    ap.add_argument("--policy", metavar="SPEC", default=DEFAULT_POLICY,
                    help="what drives --eval: scripted (default), "
                         "replay:DIR[:EP] to play back a recorded episode, or "
                         "act:PATH for a LeRobot checkpoint")
    ap.add_argument("--seed", type=int, default=None, help="spawn / task seed")
    # Off unless asked for: bare --record collects into DEFAULT_RECORD_DIR,
    # --record DIR picks the directory.
    ap.add_argument("--record", metavar="DIR", nargs="?",
                    default=None, const=DEFAULT_RECORD_DIR,
                    help="collect this session as a LeRobot dataset in DIR "
                         f"(default {DEFAULT_RECORD_DIR}); off when omitted")
    ap.add_argument("--window", metavar="WxH", default=None,
                    help="window size to open at, e.g. 1600x1000. The window is "
                         "resizable either way -- drag its corner")
    ap.add_argument("--record-size", metavar="WxH",
                    default=f"{RECORD_SIZE[0]}x{RECORD_SIZE[1]}",
                    help="size of recorded frames (default "
                         f"{RECORD_SIZE[0]}x{RECORD_SIZE[1]}). Independent of the "
                         "window: one dataset column has one image shape")
    ap.add_argument("--record-views", metavar="VIEWS", default=DEFAULT_RECORD_VIEWS,
                    help="cameras to record: main (default), all, or a "
                         "comma-separated list of view names. Every view is its "
                         "own image column, and its own PNG per tick")
    a = ap.parse_args(argv)

    if a.list_envs:
        import benchmarks
        print(benchmarks.describe())
        return 0
    record_size = _size(a.record_size, "record-size")
    if a.eval is not None:
        return run_eval(a.eval, seed=a.seed, policy_spec=a.policy,
                        record=a.record, env_spec=a.env,
                        record_views=a.record_views, record_size=record_size)
    if a.headless:
        return run_headless(seed=2 if a.seed is None else a.seed,
                            record=a.record, env_spec=a.env,
                            record_views=a.record_views, record_size=record_size)
    return run_game(seed=a.seed, record=a.record, env_spec=a.env,
                    record_views=a.record_views, record_size=record_size,
                    window=_size(a.window, "window") if a.window else None)


if __name__ == "__main__":
    sys.exit(main())
