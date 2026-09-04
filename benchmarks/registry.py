"""Which benchmark environments this rig can drive, and whether they're installed.

Everything here is probed, not assumed: `available()` imports nothing until you
ask, and a family that is not installed reports the pip command that would fix
it instead of raising.

Gripper conventions differ between families and were each measured against the
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
class Family:
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


FAMILIES = {
    NATIVE: Family(
        NATIVE, "game", "(built in)",
        "Skillcrane's own 6-DoF arm and cube.",
        ("default",), _native),
    "robosuite": Family(
        "robosuite", "robosuite", "pip install -r requirements-benchmarks.txt",
        "ARISE robosuite manipulation suite (Panda, OSC control).",
        ("Lift", "Stack", "PickPlaceCan", "Door", "NutAssemblyRound"), _robosuite,
        note="Needs mujoco<3.12 and robosuite>=1.5 -- see BENCHMARKS.md.",
        min_version=(1, 5)),
    "metaworld": Family(
        "metaworld", "metaworld", "pip install -r requirements-benchmarks.txt",
        "Meta-World 50-task manipulation benchmark (Sawyer).",
        ("pick-place-v3", "reach-v3", "push-v3", "door-open-v3", "drawer-close-v3"),
        _metaworld, note="No wrist joint: dyaw is recorded but does nothing."),
    "fetch": Family(
        "fetch", "gymnasium_robotics", "pip install -r requirements-benchmarks.txt",
        "Gymnasium-Robotics Fetch tasks (7-DoF mobile manipulator).",
        ("FetchPickAndPlace-v4", "FetchReach-v4", "FetchPush-v4", "FetchSlide-v4"),
        _fetch, note="No wrist joint: dyaw is recorded but does nothing."),
    # Known, verified to install and run, but not teleoperable through this
    # rig -- recorded here so the next person does not re-derive it.
    "aloha": Family(
        "aloha", "gym_aloha", "pip install gym-aloha",
        "LeRobot's bimanual ALOHA sim.",
        ("AlohaTransferCube-v0", "AlohaInsertion-v0"), None,
        note="14-D bimanual joint-position control, not a Cartesian delta -- "
             "one gamepad cannot drive it without an IK layer.",
        supported=False),
    "libero": Family(
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


def installed(family: str) -> bool:
    """True only if the family is importable *and* new enough for the adapter.

    The version check is not pedantry: LIBERO pins robosuite==1.4.0, whose
    controller API is not the 1.5 one this project's adapter uses. Inside a
    LIBERO environment robosuite imports fine and then fails at make() time, so
    presence alone is the wrong question.
    """
    fam = FAMILIES.get(family)
    if fam is None:
        return False
    if family == NATIVE:
        return True
    if importlib.util.find_spec(fam.module) is None:
        return False
    if fam.min_version:
        return _version(fam.dist or fam.module) >= fam.min_version
    return True


def parse(spec: str):
    """"robosuite:Lift" -> ("robosuite", "Lift"). Bare name -> (name, None)."""
    family, _, task = (spec or NATIVE).partition(":")
    return family, (task or None)


def make(spec: str = NATIVE, seed: int = 0):
    """Build a TeleopEnv, with a useful message when the family is missing."""
    family, task = parse(spec)
    fam = FAMILIES.get(family)
    if fam is None:
        raise SystemExit(f"unknown environment {spec!r}. Try --list-envs.")
    if not fam.supported:
        raise SystemExit(f"{family} is not teleoperable here: {fam.note}")
    if not installed(family):
        have = _version(fam.dist or fam.module)
        why = (f"{family} {'.'.join(map(str, have))} is too old for this adapter"
               if have else f"{family} is not installed")
        raise SystemExit(f"{why}.\n  {fam.install}\n  {fam.note}".rstrip())
    return fam.factory(task, seed)


def switchable() -> tuple:
    """Families that can be cycled through at runtime, in declaration order.

    Installed, teleoperable, and actually constructible -- `aloha` is registered
    but has no factory, so it can never be switched to.
    """
    return tuple(name for name, fam in FAMILIES.items()
                 if fam.supported and fam.factory is not None and installed(name))


def cycle(spec: str, step: int = 1) -> str:
    """The next installed family around the ring from `spec`.

    Returns a bare family name: the task resets to that family's default,
    because a task id from one family means nothing in the next. An unknown or
    uninstalled current family starts the ring from the beginning, so this can
    never strand the operator.
    """
    ring = switchable()
    if not ring:                    # pragma: no cover - native is always in
        return spec
    family, _ = parse(spec)
    if family in ring:
        return ring[(ring.index(family) + step) % len(ring)]
    return ring[0]


def describe() -> str:
    """Human-readable table for --list-envs."""
    rows = []
    for fam in FAMILIES.values():
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
    return "\n".join(
        ["Environments  ([ ok ] installed, [  - ] available, [n/a ] not teleoperable)", ""]
        + rows
        + ["", "Use:  python main.py --env robosuite:Lift"])
