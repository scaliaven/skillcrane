"""Skillcrane's own arm, wrapped in the TeleopEnv contract.

This is the reference implementation: `Game` already had every method in the
right shape, so the adapter is mostly forwarding. It exists so that the native
game and the benchmark environments go through exactly one code path in
main.py, which is what keeps them honest against each other.

The mujoco.Renderer lives *here*, not in game.py -- creating one needs a GL
context, and game.py must stay runnable with no display.
"""
import numpy as np

import scene
from game import Game

from .base import Hud, TeleopEnv

STATE_NAMES = tuple([f"j{i}" for i in range(1, 7)] + ["grip", "cube_x", "cube_y", "cube_z"])
TASK = "Pick up the cube and place it in the target zone."


class NativeEnv(TeleopEnv):
    state_names = STATE_NAMES
    task = TASK

    def __init__(self, seed: int = 0):
        self.game = Game(seed=seed)
        self._renderer = None
        self._size = None

    def reset(self, full: bool = False) -> None:
        self.game.reset(full=full)

    def step(self, dx, dy, dz, dyaw, want_closed, dt=scene.CTRL_DT) -> bool:
        return self.game.step(dx, dy, dz, dyaw, want_closed, dt)

    def observation(self) -> np.ndarray:
        return self.game.observation()

    def hud(self) -> Hud:
        g = self.game
        return Hud(score=g.score, time_left=g.time_left, streak=g.streak,
                   best_streak=g.best_streak,
                   grip="HOLDING" if g.held() else ("CLOSED" if g.closed else "OPEN"),
                   ee=g.arm.ee(), obj=g.cube_pos(), task=self.task)

    def frame(self, width: int, height: int):
        import mujoco
        if self._renderer is None or self._size != (width, height):
            self._renderer = mujoco.Renderer(self.game.m, height=height, width=width)
            self._size = (width, height)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.10, 0.10, 0.15]
            self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.35, -22.0, 130.0
        self._renderer.update_scene(self.game.d, self.cam)
        return self._renderer.render()

    # The native env is the only one with an orbitable camera of our own.
    def orbit(self, degrees: float) -> None:
        if self._renderer is not None:
            self.cam.azimuth += degrees

    @property
    def azimuth(self) -> float:
        return getattr(self, "cam", None).azimuth if self._renderer is not None else 130.0
