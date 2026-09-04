"""Which benchmark environments this rig can drive, and whether they're installed.

Two things get switched at runtime and they are not the same thing:

    suite   which benchmark is running -- native, robosuite, LIBERO, RoboCasa.
            A different robot, a different simulator wrapper, a different
            observation width. Cycled with `cycle_suite`.
    task    which setting inside that suite -- Lift, Stack, PickPlaceCan.
            Same robot, same adapter, different scene. Cycled with `cycle_task`.

An *environment spec* names both: "robosuite:Lift" is a suite and a task, and a
bare "robosuite" means that suite's first task.

Everything here is probed, not assumed: nothing is imported until you ask, and a
suite that is not installed reports the command that would fix it rather than
raising.

Gripper conventions differ between suites and were each measured against the
simulator rather than taken from documentation:

    robosuite    action[-1] = +1  closes   (Panda finger joints 0.042 -> 0.001)
    Meta-World   action[-1] = +1  closes   (gripper obs 1.00 -> 0.30)
    Fetch        action[-1] = -1  closes   (finger joints stay 0 on +1)

Get one backwards and the task is quietly unsolvable while looking like bad
teleop, so they live in one table with the evidence attached.
"""
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from typing import Callable

NATIVE = "native"


@dataclass(frozen=True)
class Suite:
    name: str
    module: str                 # import name used to probe installation
    install: str                # pip command that provides it
    summary: str
    tasks: tuple                # a few known-good task ids
    factory: Callable = None
    note: str = ""
    supported: bool = True
    #: Minimum distribution version the adapter's API needs, as a tuple.
    min_version: tuple = ()
    dist: str = ""              # distribution name, if it differs from `module`


def _native(task, seed):
    from .native import NativeEnv
    return NativeEnv(seed=seed)


def _robosuite(task, seed):
    from .robosuite_env import RobosuiteEnv
    return RobosuiteEnv(task=task or "Lift", seed=seed)


# RoboCasa's mobile Panda was renamed between releases (v0.1 PandaMobile,
# v0.2 PandaOmron) and we cannot probe which one is present without importing
# it, so the factory tries them in order rather than pinning a guess.
ROBOCASA_ROBOTS = ("PandaOmron", "PandaMobile")


def _robocasa(task, seed):
    """RoboCasa kitchens, which are robosuite envs once robocasa is imported.

    Importing robocasa is what registers its environments in robosuite's
    registry, so the adapter underneath is the robosuite one -- same OSC
    controller, same Cartesian delta, same gripper sign. Untested here: the
    package is not installed on this machine, so treat the robot names and the
    default task as the documented starting point, not as measured facts.
    """
    import robocasa  # noqa: F401  (registers the kitchen envs with robosuite)
    from .robosuite_env import RobosuiteEnv

    last = None
    for robot in ROBOCASA_ROBOTS:
        try:
            return RobosuiteEnv(task=task or "PnPCounterToCab", robot=robot,
                                seed=seed, camera=None)
        except Exception as exc:        # wrong robot name for this release
            last = exc
    raise SystemExit(f"robocasa built no environment with any of "
                     f"{', '.join(ROBOCASA_ROBOTS)}: {last}")


def _metaworld(task, seed):
    import metaworld  # noqa: F401  (registers the Meta-World gym namespace)
    from .gym_env import METAWORLD, GymEnv
    name = task or "pick-place-v3"
    env = GymEnv("Meta-World/MT1", METAWORLD, seed=seed,
                 task=f"Meta-World {name}.", make_kwargs={"env_name": name})
    env.task_name = name          # MT1 is the wrapper id; show the real task
    return env


def _libero(task, seed):
    from .libero_env import LiberoEnv
    return LiberoEnv(task=task or "libero_spatial", seed=seed)


def _fetch(task, seed):
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    from .gym_env import FETCH, GymEnv
    return GymEnv(task or "FetchPickAndPlace-v4", FETCH, seed=seed)


SUITES = {
    NATIVE: Suite(
        NATIVE, "game", "(built in)",
        "Skillcrane's own 6-DoF arm and cube.",
        ("default",), _native),
    "robosuite": Suite(
        "robosuite", "robosuite", "pip install -r requirements-benchmarks.txt",
        "ARISE robosuite manipulation suite (Panda, OSC control).",
        ("Lift", "Stack", "PickPlaceCan", "Door", "NutAssemblyRound"), _robosuite,
        note="Needs mujoco<3.12 and robosuite>=1.5 -- see BENCHMARKS.md.",
        min_version=(1, 5)),
    "robocasa": Suite(
        "robocasa", "robocasa",
        "pip install git+https://github.com/robocasa/robocasa.git "
        "&& python -m robocasa.scripts.download_kitchen_assets",
        "RoboCasa kitchen tasks (robosuite envs, mobile Panda).",
        ("PnPCounterToCab", "PnPCabToCounter", "OpenSingleDoor", "CloseDrawer",
         "TurnOnStove"), _robocasa,
        note="Built on robosuite, so the same mujoco<3.12 rule applies, and it "
             "needs its kitchen assets downloaded once. Not installed here: "
             "the adapter path is untested.",
    ),
    "metaworld": Suite(
        "metaworld", "metaworld", "pip install -r requirements-benchmarks.txt",
        "Meta-World 50-task manipulation benchmark (Sawyer).",
        ("pick-place-v3", "reach-v3", "push-v3", "door-open-v3", "drawer-close-v3"),
        _metaworld, note="No wrist joint: dyaw is recorded but does nothing."),
    "fetch": Suite(
        "fetch", "gymnasium_robotics", "pip install -r requirements-benchmarks.txt",
        "Gymnasium-Robotics Fetch tasks (7-DoF mobile manipulator).",
        ("FetchPickAndPlace-v4", "FetchReach-v4", "FetchPush-v4", "FetchSlide-v4"),
        _fetch, note="No wrist joint: dyaw is recorded but does nothing."),
    # Known, verified to install and run, but not teleoperable through this
    # rig -- recorded here so the next person does not re-derive it.
    "aloha": Suite(
        "aloha", "gym_aloha", "pip install gym-aloha",
        "LeRobot's bimanual ALOHA sim.",
        ("AlohaTransferCube-v0", "AlohaInsertion-v0"), None,
        note="14-D bimanual joint-position control, not a Cartesian delta -- "
             "one gamepad cannot drive it without an IK layer.",
        supported=False),
    "libero": Suite(
        "libero", "libero", "conda env create -f environment-libero.yml",
        "LIBERO lifelong-learning benchmark (130 tasks, HF LeRobot build).",
        ("libero_spatial", "libero_object", "libero_goal", "libero_90",
         "libero_10", "libero_100"), _libero,
        note="Pins robosuite==1.4.0 -- run it in its own env, never beside "
             "the robosuite backend. Task ids take an index: libero_spatial/3."),
}


def _version(dist: str) -> tuple:
    try:
        raw = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ()
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def installed(suite: str) -> bool:
    """True only if the suite is importable *and* new enough for the adapter.

    The version check is not pedantry: LIBERO pins robosuite==1.4.0, whose
    controller API is not the 1.5 one this project's adapter uses. Inside a
    LIBERO environment robosuite imports fine and then fails at make() time, so
    presence alone is the wrong question.
    """
    fam = SUITES.get(suite)
    if fam is None:
        return False
    if suite == NATIVE:
        return True
    if importlib.util.find_spec(fam.module) is None:
        return False
    if fam.min_version:
        return _version(fam.dist or fam.module) >= fam.min_version
    return True


def parse(spec: str):
    """Split an environment spec into its two halves.

    "robosuite:Lift" -> ("robosuite", "Lift"); a bare suite name -> (name, None),
    meaning "that suite's default task".
    """
    suite, _, task = (spec or NATIVE).partition(":")
    return suite, (task or None)


def make(spec: str = NATIVE, seed: int = 0):
    """Build a TeleopEnv from a "suite:task" spec, or explain what is missing."""
    suite, task = parse(spec)
    fam = SUITES.get(suite)
    if fam is None:
        raise SystemExit(f"unknown environment {spec!r}. Try --list-envs.")
    if not fam.supported:
        raise SystemExit(f"{suite} is not teleoperable here: {fam.note}")
    if not installed(suite):
        have = _version(fam.dist or fam.module)
        why = (f"{suite} {'.'.join(map(str, have))} is too old for this adapter"
               if have else f"{suite} is not installed")
        raise SystemExit(f"{why}.\n  {fam.install}\n  {fam.note}".rstrip())
    return fam.factory(task, seed)


def suites() -> tuple:
    """Benchmark suites that can be cycled through live, in declaration order.

    Installed, teleoperable, and actually constructible -- `aloha` is registered
    but has no factory, so it can never be switched to. A suite you have not
    installed is simply not in the ring; `describe()` is where you find out it
    exists and what would install it.
    """
    return tuple(name for name, fam in SUITES.items()
                 if fam.supported and fam.factory is not None and installed(name))


def cycle_suite(spec: str, step: int = 1) -> str:
    """The next installed *suite* around the ring from `spec`.

    This is the big switch: a different benchmark, a different robot, a
    different observation width. Returns a bare suite name, because the task
    resets to that suite's default -- a task id from one suite means nothing in
    the next. An unknown or uninstalled current suite starts the ring from the
    beginning, so this can never strand the operator.
    """
    ring = suites()
    if not ring:                    # pragma: no cover - native is always in
        return spec
    suite, _ = parse(spec)
    if suite in ring:
        return ring[(ring.index(suite) + step) % len(ring)]
    return ring[0]


def tasks(spec: str) -> tuple:
    """The task ids registered for `spec`'s suite, or () if it has none."""
    fam = SUITES.get(parse(spec)[0])
    return fam.tasks if fam else ()


def cycle_task(spec: str, step: int = 1) -> str:
    """The next task *within* the current suite, e.g. Lift -> Stack.

    Switching suite is the other move (`cycle_suite`): this one keeps the robot
    and the adapter and changes the setting, which is what "try the same rig on
    the next task" means. Suites list a few known-good task ids each, and that
    list is the ring. A suite with fewer than two -- the native arm has one
    scene -- returns `spec` unchanged, so the caller can say so instead of
    silently rebuilding the identical environment.

    A task id that is not on the list (a hand-typed `--env libero:libero_90/17`)
    is treated as position 0, so stepping forward lands on the second entry
    rather than stranding the operator outside the ring.
    """
    suite, task = parse(spec)
    fam = SUITES.get(suite)
    if fam is None or len(fam.tasks) < 2:
        return spec
    ids = list(fam.tasks)
    i = ids.index(task) if task in ids else 0
    return f"{suite}:{ids[(i + step) % len(ids)]}"


def describe() -> str:
    """Human-readable table for --list-envs, in the suite / task vocabulary."""
    rows = []
    for fam in SUITES.values():
        if not fam.supported:
            mark = "n/a "
        else:
            mark = " ok " if installed(fam.name) else "  - "
        rows.append(f"[{mark}] {fam.name:<11} {fam.summary}")
        rows.append(f"         tasks: {', '.join(fam.tasks)}")
        if fam.note:
            rows.append(f"         note : {fam.note}")
        if fam.supported and not installed(fam.name):
            rows.append(f"         get  : {fam.install}")
    ring = ", ".join(suites())
    return "\n".join(
        ["Benchmark suites  ([ ok ] installed, [  - ] available, "
         "[n/a ] not teleoperable)",
         "A suite is a benchmark; a task is one setting inside it. "
         "--env takes SUITE[:TASK].", ""]
        + rows
        + ["",
           "Use:  python main.py --env robosuite:Lift      one suite, one task",
           "      python main.py --env libero               that suite's first task",
           "",
           "Live:  [ / ]  or Back/Start   next SUITE  (only installed ones: "
           f"{ring})",
           "       , / .  or the d-pad    next TASK   (inside the suite you are on)"])
