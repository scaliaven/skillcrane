"""Policies that drive a TeleopEnv -- scripted, replayed, or learned.

recorder.py is one end of the data pipe; this is the other. A policy produces
exactly what the sticks produce -- `(dx, dy, dz, dyaw, grip)`, each in
[-1, 1] -- and it goes through the same `TeleopEnv.step`, so a scripted
demonstration, a replayed episode and a trained network are all comparable, all
recordable, and all scored by the same rules.

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

# Arrival is measured on the *commanded* target, not the measured end effector.
# The target is pure integration, so it always converges and the phase machine
# can never stall waiting on an arm that is fighting contact. 3 mm is well
# inside the tolerance the waypoints themselves are chosen at.
REACH_TOL = 0.003
# Waypoint travel speed. Deliberately under MOVE_SPEED (0.45) and LIFT_SPEED
# (0.35) so the emitted stick never saturates: a clipped action is one that no
# longer describes the motion it produced, which is the exact defect this module
# exists to catch.
SCRIPT_SPEED = 0.30
# "Arrived" on the commanded target is not "stopped": the target is pure
# integration and gets there first, with the arm still crossing the tolerance
# ball at 0.3 m/s. Legs that end in a gripper action wait for the joints as well,
# or the fingers close on a moving cube and open on a moving one. 0.05 rad/s is
# the same order as the cube-settled test in Game.check_score, and costs ~15
# ticks of a ~700-tick round.
SETTLE_QVEL = 0.05
# One pick-and-place is ~1430 ticks of the script below. The cap is a backstop
# against a policy that never finishes, not a budget anyone should hit.
MAX_TICKS = 2500


def _game(env):
    """The underlying Game, whether handed an env or a Game.

    The scripted policy is native-only -- it reads the cube and the drop zone,
    which no other suite exposes -- so it takes the short way in rather than
    pretending the TeleopEnv contract carries them.
    """
    return getattr(env, "game", env)


def _dt(env) -> float:
    return float(getattr(env, "control_dt", scene.CTRL_DT))


class Policy(ABC):
    """Something that produces one action per control tick.

    Takes the env rather than a bare observation: a scripted policy needs the
    world, a learned one needs to ask for camera frames at its own size, and
    both are cheaper to write than a lowest-common-denominator observation dict
    that neither wants. Everything they *emit* is the same five numbers.
    """

    name = "policy"

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
    """One leg of the script. `where` is None for 'hold still and wait'."""
    name: str
    where: object           # callable(game) -> xyz, latched on entry
    closed: bool
    budget: int             # ticks: the whole phase if holding, else a backstop
    settle: float = 0.0     # also wait for |qvel| under this before advancing


# The same waypoints and the same tick budgets as game.scripted_pick_and_place,
# because that one is the version 19 grasp tests have already vouched for. The
# only difference is that this one goes through the sticks.
# The budgets double as the old script's fixed tick counts, so a leg that never
# settles falls back to exactly the behaviour the grasp tests were written
# against rather than to something new.
SCRIPT = (
    Waypoint("above",   lambda g: g.cube_pos() + [0, 0, 0.13],  False, 250),
    Waypoint("descend", lambda g: g.cube_pos() + [0, 0, 0.012], False, 250,
             settle=SETTLE_QVEL),                       # then the fingers close
    Waypoint("close",   None,                                   True,   80),
    Waypoint("lift",    lambda g: np.array([0.30, 0.0, 0.32]),  True,  250),
    Waypoint("carry",   lambda g: np.array([*g.target, 0.30]),  True,  400),
    Waypoint("lower",   lambda g: np.array([*g.target, 0.09]),  True,  250,
             settle=SETTLE_QVEL),                       # then the fingers open
    Waypoint("release", None,                                   False, 200),
)


class ScriptedPickPlace(Policy):
    """Approach -> grasp -> carry -> release, emitted as stick values.

    `game.scripted_pick_and_place` drives `game.tgt` directly. That is right
    for a physics test -- it isolates the arm from the input layer -- and wrong
    for a *demonstration*, because the action recorded next to the motion is
    then not the action that caused it. This walks the identical waypoints and
    converts each one into the stick deflection that moves the target there,
    which is what an operator's thumb would have had to do.

    Native-only, on purpose: it reads the cube position and the drop zone.
    """

    name = "scripted"

    def __init__(self, speed: float = SCRIPT_SPEED, cycles: int = 1, script=SCRIPT):
        self.speed = float(speed)
        self.cycles = int(cycles)       # rounds to play before done(); 0 = forever
        self.script = tuple(script)
        self.reset(None)

    def reset(self, env) -> None:
        # Checked here rather than left to fail deep in a waypoint lambda: on a
        # benchmark suite the first symptom would be AttributeError: cube_pos,
        # several hundred ticks into what looked like a working eval.
        if env is not None and not hasattr(_game(env), "cube_pos"):
            raise TypeError(
                "the scripted policy is native-only -- it reads the cube and the "
                "drop zone, which no benchmark suite exposes. Use "
                "--policy replay:DIR or act:PATH on other suites.")
        self.phase = self.script[0].name
        self._i = 0
        self._ticks = 0
        self._goal = None
        self.completed = 0

    def done(self) -> bool:
        return bool(self.cycles) and self.completed >= self.cycles

    def act(self, env) -> tuple:
        g = _game(env)
        wp = self.script[self._i]
        # Latched on entry, not recomputed: once the fingers have the cube,
        # "13 cm above the cube" moves with the gripper and the leg never ends.
        if self._goal is None and wp.where is not None:
            self._goal = np.asarray(wp.where(g), dtype=float)

        action = self._stick(env, g, self._goal, wp.closed)
        self._ticks += 1
        if self._arrived(g, wp):
            self._advance()
        return action

    def _stick(self, env, g, goal, closed) -> tuple:
        """The stick that moves the commanded target toward `goal` this tick.

        Mirrors game.drive_to exactly -- travel at most `speed * dt`, and never
        past the goal -- then divides by what a full stick is worth on each axis.
        The axes have different speeds (MOVE_SPEED horizontally, LIFT_SPEED
        vertically), so the division is per axis, not one scale factor.
        """
        if goal is None:
            return (0.0, 0.0, 0.0, 0.0, bool(closed))
        dt = _dt(env)
        d = goal - g.tgt
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            d = d / n * min(self.speed * dt, n)
        else:
            d = np.zeros(3)
        full = np.array([scene.MOVE_SPEED * dt, scene.MOVE_SPEED * dt,
                         scene.LIFT_SPEED * dt])
        dx, dy, dz = np.clip(d / full, -1.0, 1.0)
        return (float(dx), float(dy), float(dz), 0.0, bool(closed))

    def _arrived(self, g, wp) -> bool:
        if wp.where is None:                    # a hold: the budget *is* the leg
            return self._ticks >= wp.budget
        there = float(np.linalg.norm(self._goal - g.tgt)) < REACH_TOL
        if there and wp.settle:
            there = float(np.linalg.norm(g.d.qvel[g.arm.dof])) < wp.settle
        # The target is clamped to the reachable shell, so a goal outside it is
        # never reached however long we wait -- hence the budget as well.
        return there or self._ticks >= wp.budget

    def _advance(self) -> None:
        self._i = (self._i + 1) % len(self.script)
        self._ticks = 0
        self._goal = None
        self.phase = self.script[self._i].name
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
        import pyarrow.parquet as pq          # optional dep, same as recorder.py
        from pathlib import Path

        path = Path(root) / "data" / "chunk-000" / f"episode_{int(episode):06d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no episode {episode} in {root} ({path})")
        return cls(pq.read_table(path).column("action").to_pylist(), **kw)

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
    seconds: float                  # simulated, not wall clock
    first_score_tick: int = None    # None if it never scored

    @property
    def success(self) -> bool:
        return self.score > 0

    @property
    def time_to_score(self):
        dt = self.seconds / self.ticks if self.ticks else 0.0
        return None if self.first_score_tick is None else self.first_score_tick * dt


def rollout(env, policy, max_ticks: int = MAX_TICKS, on_tick=None,
            seed: int = 0) -> Rollout:
    """Drive `env` with `policy` until it is done or `max_ticks` runs out.

    `on_tick(env, action, scored)` fires after every step -- that is where the
    recorder hangs, so a rollout and a played round log identically.
    """
    policy.reset(env)
    dt = _dt(env)
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
    return Rollout(seed=seed, ticks=ticks, score=score, seconds=ticks * dt,
                   first_score_tick=first)


def evaluate(make_env, make_policy, seeds, max_ticks: int = MAX_TICKS,
             on_start=None, on_episode=None, on_tick=None) -> list:
    """One rollout per seed, on a freshly built env each time.

    Factories rather than instances because the whole point is that each seed
    gets a clean world -- reusing one env would carry the previous round's
    score, clock and cube into the next and make the numbers meaningless.

    `on_start(seed, env)` fires before each episode and `on_episode(rollout)`
    after it, which is where recording hangs: one episode per seed, opened and
    closed here rather than in a loop main.py would otherwise have to keep in
    step with this one.
    """
    out = []
    for seed in seeds:
        env = make_env(seed)
        try:
            if on_start is not None:
                on_start(seed, env)
            r = rollout(env, make_policy(), max_ticks=max_ticks, seed=seed,
                        on_tick=on_tick)
        finally:
            # A TeleopEnv owns renderers and has to be closed; a bare Game does
            # not have the method at all, and the rest of this module accepts
            # either -- so ask rather than assume.
            close = getattr(env, "close", None)
            if callable(close):
                close()
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
