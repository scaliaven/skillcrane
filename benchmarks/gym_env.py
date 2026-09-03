"""Gymnasium manipulation benchmarks (Meta-World, Gymnasium-Robotics Fetch).

Both families expose a 4-D action that is already this project's control
interface:

    [dx, dy, dz, grip]

so the adapter is a remapping. They have no wrist joint to drive, so `dyaw` is
ignored -- the sticks that rotate the Claw Crew wrist do nothing here, which is
a property of the benchmark, not a bug.

The gripper sign is NOT consistent across families and was measured rather than
assumed (see benchmarks/registry.py):

    Meta-World  action[-1] = +1  closes
    Fetch       action[-1] = -1  closes

Getting it backwards makes the task quietly unsolvable, which is exactly the
kind of thing that looks like bad teleop rather than a wrong constant.
"""
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Hud, TeleopEnv

ROUND_SECONDS = 90.0
MOVE_GAIN = 1.0

# gymnasium's MuJoCo renderer only knows these backends. robosuite sets
# MUJOCO_GL=cgl on macOS when it is imported, and gymnasium then dies with
# KeyError: 'cgl' -- so importing one benchmark can break another's rendering
# inside the same process. Clearing the variable lets gymnasium pick its own
# default, which is what it does when nothing is set.
GYM_BACKENDS = ("glfw", "egl", "osmesa")


def _clear_foreign_gl_backend():
    backend = os.environ.get("MUJOCO_GL")
    if backend is not None and backend.lower() not in GYM_BACKENDS:
        os.environ.pop("MUJOCO_GL")


@dataclass(frozen=True)
class GymSpec:
    """Everything that differs between one gym benchmark family and another."""
    grip_close: float                     # sign of action[-1] that shuts the gripper
    flatten: Callable                     # obs -> flat float32 vector
    ee: Callable                          # obs -> end-effector xyz
    obj: Callable                         # obs -> manipulated object xyz
    success_keys: tuple = ("success", "is_success")


def _fetch_obs(o):
    return np.concatenate([o["observation"], o["achieved_goal"], o["desired_goal"]])


FETCH = GymSpec(
    grip_close=-1.0,                                  # measured
    flatten=lambda o: _fetch_obs(o).astype(np.float32),
    ee=lambda o: np.asarray(o["observation"][:3]),
    obj=lambda o: np.asarray(o["observation"][3:6]),
)

METAWORLD = GymSpec(
    grip_close=+1.0,                                  # measured
    flatten=lambda o: np.asarray(o, dtype=np.float32).ravel(),
    ee=lambda o: np.asarray(o[:3]),
    obj=lambda o: np.asarray(o[4:7]),
)


class GymEnv(TeleopEnv):
    action_names = ("dx", "dy", "dz", "dyaw", "grip")   # dyaw recorded but unused

    def __init__(self, env_id, spec: GymSpec, seed=0, task=None, make_kwargs=None):
        _clear_foreign_gl_backend()
        import gymnasium as gym
        self.spec = spec
        self.seed = seed
        self.env = gym.make(env_id, render_mode="rgb_array", **(make_kwargs or {}))
        self.control_dt = 1.0 / float(self.env.metadata.get("render_fps", 25))
        self.task = task or f"Gymnasium benchmark {env_id}."
        self.task_name = env_id
        self.closed = False
        self._obs = None
        self.reset(full=True)
        self.state_names = tuple(f"obs{i:02d}" for i in range(self.observation().size))

    def reset(self, full: bool = False) -> None:
        self._obs, _ = self.env.reset(seed=self.seed if full else None)
        self.closed = False
        if full or not hasattr(self, "score"):
            self.score = 0
            self.streak = self.best_streak = 0
            self.time_left = ROUND_SECONDS

    def step(self, dx, dy, dz, dyaw, want_closed, dt=None) -> bool:
        self.closed = bool(want_closed)
        a = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        a[0], a[1], a[2] = dx * MOVE_GAIN, dy * MOVE_GAIN, dz * MOVE_GAIN
        a[-1] = self.spec.grip_close if self.closed else -self.spec.grip_close
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        self._obs, _, terminated, truncated, info = self.env.step(a)

        self.time_left = max(0.0, self.time_left - self.control_dt)
        won = any(bool(info.get(k)) for k in self.spec.success_keys)
        if won:
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        if won or terminated or truncated:
            self.reset(full=False)
        return won

    def observation(self) -> np.ndarray:
        return self.spec.flatten(self._obs)

    def hud(self) -> Hud:
        return Hud(score=self.score, time_left=self.time_left, streak=self.streak,
                   best_streak=self.best_streak,
                   grip="CLOSED" if self.closed else "OPEN",
                   ee=self.spec.ee(self._obs), obj=self.spec.obj(self._obs),
                   task=self.task_name)

    def frame(self, width: int, height: int):
        # Gym envs fix their frame size at make() time; render.py scales to fit.
        img = self.env.render()
        return None if img is None else np.asarray(img, dtype=np.uint8)

    def close(self) -> None:
        # gymnasium's OffScreenViewer.__del__ can raise during teardown; a
        # broken teardown must not take down a teleop session.
        try:
            self.env.close()
        except Exception:
            pass
