"""Policies that drive a TeleopEnv -- scripted, replayed, or learned.

recorder.py is one end of the data pipe; this is the other. A policy produces
exactly what the sticks produce -- `(dx, dy, dz, dyaw, grip)`, each in [-1, 1] --
and it goes through the same `TeleopEnv.step`, so a scripted demonstration, a
replayed episode and a trained network are all comparable, all recordable, and
all scored by the same rules.

HARD RULE, same as game.py: no pygame, no display. This has to run wherever the
sim runs, because it is what a data pipeline calls.

Why replay exists at all: it is the only check that a recorded dataset means
what it claims. Same seed, same actions, same simulator -- if the cube does not
end up where it did when the episode was collected, then the action column does
not explain the motion beside it, and nothing trained on that column can work.
That is not hypothetical. `--headless --record` used to log [0, 0, 0, 0, grip]
for every tick while the arm crossed the table, because the scripted demo moved
the Cartesian target directly (see `game.drive_to`) instead of through the
sticks -- correct for a physics test, silently useless as a demonstration.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import scene
from game import SCRIPT_SPEED, WAYPOINTS, step_toward

# Arrival is measured on the *commanded* target, not the measured end effector.
# The target is pure integration, so it always converges and the phase machine
# can never stall waiting on an arm that is fighting contact. 3 mm is well
# inside the tolerance the waypoints themselves are chosen at.
REACH_TOL = 0.003
# "Arrived" is not "stopped": the target gets to a waypoint first, with the arm
# still crossing the tolerance ball at SCRIPT_SPEED, so no position tolerance can
# express settling at any radius. The two legs that end in a gripper action wait
# for the joints as well. 0.05 rad/s is the same order as the cube-settled test
# in Game.check_score, and costs ~15 ticks of a ~700-tick round.
SETTLE_QVEL = 0.05
# One pick-and-place is ~750 ticks. The cap is a backstop against a policy that
# never finishes, not a budget anyone should hit.
MAX_TICKS = 2500


def _game(env):
    """The underlying Game, whether handed an env or a Game.

    The scripted policy is native-only -- it reads the cube and the drop zone,
    which no other suite exposes -- so it takes the short way in rather than
    pretending the TeleopEnv contract carries them.
    """
    return getattr(env, "game", env)


class Policy(ABC):
    """Something that produces one action per control tick.

    Takes the env rather than a bare observation: a scripted policy needs the
    world, a learned one needs to ask for camera frames at its own size, and
    both are cheaper to write than a lowest-common-denominator observation dict
    that neither wants. Everything they *emit* is the same five numbers.

    One instance is reused across episodes; `reset` is what rewinds it.
    """

    name = "policy"
    #: Suites this policy can drive, or None for any. Only the scripted one is
    #: restricted, and it says so itself rather than having callers test for it.
    suites = None

    def reset(self, env) -> None:
        """Start a fresh episode. The env is already reset by the caller."""

    @abstractmethod
    def act(self, env) -> tuple:
        """(dx, dy, dz, dyaw, grip) for this tick. Each of the four in [-1, 1]."""

    def done(self) -> bool:
        """True when the policy has nothing left to do (replay ran out, etc)."""
        return False


# --- the demo script, as a policy -------------------------------------------

@dataclass(frozen=True)
class Waypoint:
    """One leg of game.WAYPOINTS plus the one thing only a policy needs."""
    name: str
    where: object           # callable(game) -> xyz, latched on entry; None = hold
    closed: bool
    budget: int             # ticks: the whole leg if holding, else a backstop
    settle: float = 0.0     # also wait for |qvel| under this before advancing


# game.py owns the script; this adds the settle condition to the two legs that
# end with the fingers moving. The budgets stay the old fixed tick counts, so a
# leg that never settles falls back to exactly the behaviour the grasp tests
# were written against.
SETTLE_LEGS = {"descend": SETTLE_QVEL, "lower": SETTLE_QVEL}
SCRIPT = tuple(Waypoint(name, where, closed, budget, SETTLE_LEGS.get(name, 0.0))
               for name, where, closed, budget in WAYPOINTS)


class ScriptedPickPlace(Policy):
    """Approach -> grasp -> carry -> release, emitted as stick values.

    `game.scripted_pick_and_place` walks the same `game.WAYPOINTS` by driving
    `game.tgt` directly. That is right for a physics test -- it isolates the arm
    from the input layer -- and wrong for a *demonstration*, because the action
    recorded next to the motion is then not the action that caused it. This
    converts each leg into the stick deflection that moves the target there,
    which is what an operator's thumb would have had to do.

    Native-only, on purpose: it reads the cube position and the drop zone.
    """

    name = "scripted"
    suites = ("native",)

    def __init__(self, speed: float = SCRIPT_SPEED, cycles: int = 1, script=SCRIPT):
        self.speed = float(speed)
        self.cycles = int(cycles)       # rounds to play before done(); 0 = forever
        self.script = tuple(script)
        self._rewind()

    def _rewind(self) -> None:
        self._i = 0
        self._ticks = 0
        self._goal = None
        self.completed = 0

    @property
    def phase(self) -> str:
        """Which leg is running. Derived, so it cannot drift from `_i`."""
        return self.script[self._i].name

    def reset(self, env) -> None:
        # Checked here rather than left to fail deep in a waypoint lambda: on a
        # benchmark suite the first symptom would be AttributeError: cube_pos,
        # several hundred ticks into what looked like a working eval.
        if not hasattr(_game(env), "cube_pos"):
            raise TypeError(
                "the scripted policy is native-only -- it reads the cube and the "
                "drop zone, which no benchmark suite exposes. Use "
                "--policy replay:DIR or act:PATH on other suites.")
        self._rewind()

    def done(self) -> bool:
        return bool(self.cycles) and self.completed >= self.cycles

    def act(self, env) -> tuple:
        g = _game(env)
        wp = self.script[self._i]
        # Latched on entry, not recomputed: once the fingers have the cube,
        # "13 cm above the cube" moves with the gripper and the leg never ends.
        if self._goal is None and wp.where is not None:
            self._goal = np.asarray(wp.where(g), dtype=float)

        action = self._stick(env, g, wp.closed)
        self._ticks += 1
        if self._arrived(g, wp):
            self._advance()
        return action

    def _stick(self, env, g, closed) -> tuple:
        """The stick that moves the commanded target toward the goal this tick.

        `game.step_toward` decides how far to travel -- the same rate limit
        `drive_to` obeys -- and this divides it by what a full stick is worth on
        each axis. The axes have different speeds (MOVE_SPEED horizontally,
        LIFT_SPEED vertically), so the division is per axis, not one factor.
        """
        if self._goal is None:
            return (0.0, 0.0, 0.0, 0.0, bool(closed))
        dt = env.control_dt
        d = step_toward(g.tgt, self._goal, self.speed, dt)
        full = np.array([scene.MOVE_SPEED * dt, scene.MOVE_SPEED * dt,
                         scene.LIFT_SPEED * dt])
        dx, dy, dz = np.clip(d / full, -1.0, 1.0)
        return (float(dx), float(dy), float(dz), 0.0, bool(closed))

    def _arrived(self, g, wp) -> bool:
        # A hold has no goal, so its budget *is* the leg. Otherwise the budget is
        # the backstop: the target is clamped to the reachable shell, so a goal
        # outside it is never reached however long we wait.
        there = (self._goal is not None
                 and float(np.linalg.norm(self._goal - g.tgt)) < REACH_TOL)
        if there and wp.settle:
            there = float(np.linalg.norm(g.d.qvel[g.arm.dof])) < wp.settle
        return there or self._ticks >= wp.budget

    def _advance(self) -> None:
        self._i = (self._i + 1) % len(self.script)
        self._ticks = 0
        self._goal = None
        if self._i == 0:                        # wrapped: one full round done
            self.completed += 1


# --- replaying a recorded episode -------------------------------------------

class ReplayPolicy(Policy):
    """Feed back the action column of a recorded episode, tick for tick.

    The dataset integrity check: run this on a fresh env with the seed the
    episode was collected at, and the round should come out the same way. If it
    does not, the recording is lying about one of the things it is easy to get
    wrong and impossible to notice later -- the action convention, the control
    rate, or the frame the deltas are expressed in.
    """

    name = "replay"

    def __init__(self, actions, loop: bool = False):
        self.actions = np.asarray(actions, dtype=float).reshape(-1, 5)
        self.loop = loop
        self._i = 0

    @classmethod
    def from_dataset(cls, root, episode: int = 0, **kw) -> "ReplayPolicy":
        """Load one episode's actions from a LeRobot directory written here."""
        from recorder import episode_path     # the writer owns the layout

        # Checked before pyarrow is imported, so a wrong path reports the wrong
        # path. The other order answered "--policy replay:typo/" with
        # ModuleNotFoundError: pyarrow on any machine that had not installed the
        # optional recording deps, which named neither the real problem nor the
        # file the caller got wrong.
        path = episode_path(root, episode)
        if not path.exists():
            raise FileNotFoundError(f"no episode {episode} in {root} ({path})")

        import pyarrow.parquet as pq          # optional dep, same as recorder.py
        return cls(pq.read_table(path, columns=["action"]).column("action").to_pylist(),
                   **kw)

    def __len__(self) -> int:
        return len(self.actions)

    def reset(self, env) -> None:
        self._i = 0

    def done(self) -> bool:
        return not self.loop and self._i >= len(self.actions)

    def act(self, env) -> tuple:
        if len(self.actions) == 0:
            return (0.0, 0.0, 0.0, 0.0, False)
        a = self.actions[self._i % len(self.actions)]
        self._i += 1
        return (float(a[0]), float(a[1]), float(a[2]), float(a[3]), bool(a[4] > 0.5))


# --- a policy trained by LeRobot --------------------------------------------

class LeRobotPolicy(Policy):
    """A checkpoint from `lerobot/scripts/train.py --policy.type=act`.

    UNTESTED HERE -- lerobot is not installed on this machine, so this is
    written against its documented interface rather than measured, in the same
    way benchmarks/robosuite_env.py carries RoboCasa. Two things are assumed:
    the checkpoint loads with `ACTPolicy.from_pretrained(path)`, and the loaded
    object exposes `select_action(batch) -> tensor` over a batch keyed by the
    same column names recorder.py writes. Both have been stable across releases;
    neither has been run here.

    The module path moved between lerobot versions, so both are tried -- the
    same fallback benchmarks/robosuite_env.py does for the Panda name.
    """

    name = "lerobot"

    def __init__(self, path, views=None, size=(320, 240), device: str = "cpu"):
        self.path = str(path)
        self.views = views                  # None -> the env's operator view
        self.size = tuple(size)
        self.device = device
        self._policy = None

    def _load(self):
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError:                 # pre-0.2 layout
            from lerobot.common.policies.act.modeling_act import ACTPolicy
        p = ACTPolicy.from_pretrained(self.path).to(self.device)
        p.eval()
        return p

    def reset(self, env) -> None:
        if self._policy is None:
            self._policy = self._load()
        if hasattr(self._policy, "reset"):
            self._policy.reset()            # ACT keeps an action chunk queue

    def act(self, env) -> tuple:
        import torch

        from recorder import EpisodeRecorder

        names = self.views or list(env.view_names[:1])
        frames = env.frames({v: self.size for v in names})
        batch = {"observation.state":
                 torch.as_tensor(np.asarray(env.observation(), dtype=np.float32))[None]}
        for view, f in frames.items():
            if f is None:
                continue
            # CHW float in [0, 1], batched -- LeRobot's image convention.
            img = torch.as_tensor(np.asarray(f, dtype=np.float32) / 255.0)
            batch[EpisodeRecorder.image_column(view)] = img.permute(2, 0, 1)[None]
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            a = self._policy.select_action(batch)
        a = np.asarray(a.squeeze(0).cpu(), dtype=float).reshape(-1)
        return (float(a[0]), float(a[1]), float(a[2]), float(a[3]), bool(a[4] > 0.5))


# --- running one -------------------------------------------------------------

@dataclass
class Rollout:
    """What one episode did. Everything --eval reports is derived from these."""
    seed: int
    ticks: int
    score: int                      # times the round scored
    dt: float                       # the env's control period
    first_score_tick: int = None    # None if it never scored

    @property
    def success(self) -> bool:
        return self.score > 0

    @property
    def seconds(self) -> float:
        """Simulated, not wall clock."""
        return self.ticks * self.dt

    @property
    def time_to_score(self):
        if self.first_score_tick is None:
            return None
        return self.first_score_tick * self.dt


def rollout(env, policy, max_ticks: int = MAX_TICKS, on_tick=None,
            seed: int = 0) -> Rollout:
    """Drive `env` with `policy` until it is done or `max_ticks` runs out.

    `on_tick(env, action, scored)` fires after every step -- that is where the
    recorder hangs, so a rollout and a played round log identically.
    """
    policy.reset(env)
    dt = env.control_dt
    ticks = score = 0
    first = None
    while ticks < max_ticks and not policy.done():
        a = policy.act(env)
        scored = bool(env.step(a[0], a[1], a[2], a[3], bool(a[4]), dt))
        ticks += 1
        if scored:
            score += 1
            if first is None:
                first = ticks
        if on_tick is not None:
            on_tick(env, a, scored)
    return Rollout(seed=seed, ticks=ticks, score=score, dt=dt, first_score_tick=first)


def evaluate(make_env, policy, seeds, max_ticks: int = MAX_TICKS,
             on_start=None, on_episode=None, on_tick=None) -> list:
    """One rollout per seed, on a freshly built env each time.

    `make_env` is a factory because the whole point is that each seed gets a
    clean world -- reusing one env would carry the previous round's score, clock
    and cube into the next and make the numbers meaningless. `policy` is a single
    instance because `rollout` rewinds it through `reset` anyway.

    `on_start(seed, env)` fires before each episode and `on_episode(rollout)`
    after it, which is where recording hangs: one episode per seed, opened and
    closed here rather than in a loop main.py would have to keep in step.
    """
    out = []
    for seed in seeds:
        env = make_env(seed)
        try:
            if on_start is not None:
                on_start(seed, env)
            r = rollout(env, policy, max_ticks=max_ticks, seed=seed, on_tick=on_tick)
        finally:
            env.close()
        out.append(r)
        if on_episode is not None:
            on_episode(r)
    return out


def success_rate(results) -> float:
    return sum(r.success for r in results) / len(results) if results else 0.0


def summarise(results) -> str:
    """The table --eval prints. Per seed, because an average hides which ones."""
    if not results:
        return "no episodes"
    lines = [f"{'seed':>6}  {'ticks':>6}  {'score':>5}  {'t_score':>8}  result"]
    for r in results:
        tts = "-" if r.time_to_score is None else f"{r.time_to_score:7.2f}s"
        lines.append(f"{r.seed:>6}  {r.ticks:>6}  {r.score:>5}  {tts:>8}  "
                     f"{'ok' if r.success else 'MISS'}")
    ok = sum(r.success for r in results)
    lines.append(f"success {ok}/{len(results)} = {100 * success_rate(results):.0f}%")
    return "\n".join(lines)
