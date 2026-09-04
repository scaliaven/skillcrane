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

MAIN_VIEW = "scene"                     # the free camera the operator orbits
HOME_CAM = (1.35, -22.0, 130.0)         # distance, elevation, azimuth


class NativeEnv(TeleopEnv):
    state_names = STATE_NAMES
    task = TASK
    #: The orbit camera first, then the fixed cameras baked into the MJCF.
    view_names = (MAIN_VIEW,) + scene.CAMERAS

    def __init__(self, seed: int = 0):
        self.game = Game(seed=seed)
        self._renderers = {}            # (w, h) -> mujoco.Renderer
        self.cam = None                 # free camera, built on the first render
        self._opt = None                # scene options for the main view

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

    # -- rendering -----------------------------------------------------------
    def _renderer(self, width: int, height: int):
        """One renderer per distinct panel size, kept for the session.

        A renderer's size is fixed at construction, and a multi-view layout asks
        for two or three different ones every frame -- rebuilding them per frame
        would rebuild a GL context per frame.
        """
        import mujoco
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = self._renderers[key] = mujoco.Renderer(self.game.m, height=height,
                                                       width=width)
            if self.cam is None:
                self.cam = mujoco.MjvCamera()
                self.cam.lookat[:] = [0.10, 0.10, 0.15]
                (self.cam.distance, self.cam.elevation,
                 self.cam.azimuth) = HOME_CAM
                # The TCP marker is site group 3, which MuJoCo hides by default.
                # Show it in the operator's view only -- see scene.py.
                self._opt = mujoco.MjvOption()
                self._opt.sitegroup[3] = 1
        return r

    def frame(self, width: int, height: int, view: str = MAIN_VIEW):
        r = self._renderer(width, height)
        if view == MAIN_VIEW:
            r.update_scene(self.game.d, self.cam, self._opt)
        else:
            r.update_scene(self.game.d, view)   # a camera name from the MJCF
        return r.render()

    def frames(self, sizes: dict) -> dict:
        return {view: self.frame(*sizes[view], view=view) for view in sizes}

    # The native env is the only one with an orbitable camera of our own.
    def orbit(self, degrees: float) -> None:
        if self.cam is not None:
            self.cam.azimuth += degrees

    @property
    def azimuth(self) -> float:
        return self.cam.azimuth if self.cam is not None else HOME_CAM[2]

    def close(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:           # pragma: no cover - driver-dependent
                pass
        self._renderers.clear()
