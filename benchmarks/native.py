"""Skillcrane's own arm, wrapped in the TeleopEnv contract.

This is the reference implementation: `Game` already had every method in the
right shape, so the adapter is mostly forwarding. It exists so that the native
game and the benchmark environments go through exactly one code path in
main.py, which is what keeps them honest against each other.

The mujoco.Renderer lives *here*, not in game.py -- creating one needs a GL
context, and game.py must stay runnable with no display.
"""
import math

import numpy as np

import scene
from game import Game

from .base import Hud, TeleopEnv

STATE_NAMES = tuple([f"j{i}" for i in range(1, 7)] + ["grip", "cube_x", "cube_y", "cube_z"])
TASK = "Pick up the cube and place it in the target zone."

MAIN_VIEW = "scene"                     # the free camera the operator orbits
# Distance, elevation, azimuth. 0.90 m rather than the 1.35 m this started at:
# the whole point of the operator's view is the 48 mm cube and the 68 mm gripper
# around it, and at 1.35 m the cube was 30 px of a 900 px panel. The camera
# follows the work (see `_focus`), so it can sit this close without the arm
# driving out of frame.
HOME_CAM = (0.90, -30.0, 130.0)
ZOOM_RANGE = (0.45, 2.20)               # metres of camera distance, hard limits
ZOOM_RATE = 1.2                         # e-folds per second at full trigger
                                        # (~0.6 s to halve the distance)
# Seconds for the camera to close most of the way onto a moved focus point. Long
# enough that a flicked stick does not whip the view, short enough that the
# gripper never leaves the middle of the panel during a reach.
FOLLOW_TAU = 0.30
# Renderers are cached per panel size and each one owns a GL context. A window
# drag emits a resize per mouse move, so the cache has to be bounded or a few
# seconds of resizing leaks dozens of contexts. Four covers any one layout
# (a main view plus three insets) with a little slack for the frame after a
# resize and for the recorder's own fixed size.
MAX_RENDERERS = 6


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
        self.following = True           # main view tracks the work by default
        self._cam_t = None              # sim time the camera was last advanced

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
        """One renderer per distinct panel size, most recently used last.

        A renderer's size is fixed at construction, and a multi-view layout asks
        for two or three different ones every frame -- rebuilding them per frame
        would rebuild a GL context per frame. The cache is bounded (see
        MAX_RENDERERS) because a resizable window makes the set of sizes
        unbounded, not the two or three it used to be.
        """
        import mujoco
        key = (width, height)
        r = self._renderers.pop(key, None)
        if r is None:
            self._ensure_cam()
            r = mujoco.Renderer(self.game.m, height=height, width=width)
        self._renderers[key] = r                # move to the fresh end
        while len(self._renderers) > MAX_RENDERERS:
            self._close(self._renderers.pop(next(iter(self._renderers))))
        return r

    @staticmethod
    def _close(renderer) -> None:
        try:
            renderer.close()
        except Exception:               # pragma: no cover - driver-dependent
            pass

    def _ensure_cam(self):
        """Build the free camera and its scene options, once.

        Separate from `_renderer` because a camera is not a GL object: zooming,
        orbiting and following all work on a machine that cannot open a
        framebuffer, and so do their tests.
        """
        import mujoco
        if self.cam is None:
            self.cam = mujoco.MjvCamera()
            # Opens already pointed at the work rather than easing onto it over
            # the first half second of the round.
            self.cam.lookat[:] = self._focus()
            (self.cam.distance, self.cam.elevation, self.cam.azimuth) = HOME_CAM
            # The TCP marker is site group 3, which MuJoCo hides by default.
            # Show it in the operator's view only -- see scene.py.
            self._opt = mujoco.MjvOption()
            self._opt.sitegroup[3] = 1
        return self.cam

    def frame(self, width: int, height: int, view: str = MAIN_VIEW):
        r = self._renderer(width, height)
        if view == MAIN_VIEW:
            self.track()
            r.update_scene(self.game.d, self.cam, self._opt)
        else:
            r.update_scene(self.game.d, view)   # a camera name from the MJCF
        return r.render()

    def frames(self, sizes: dict) -> dict:
        return {view: self.frame(*sizes[view], view=view) for view in sizes}

    # -- the operator's camera -----------------------------------------------
    def _focus(self) -> np.ndarray:
        """Where the main view should be pointed: halfway into the next move.

        Not the gripper alone. What the operator is judging is a *gap* -- to the
        cube while reaching for it, to the drop zone while carrying it -- and
        centring the midpoint keeps both ends of that gap on screen at a
        distance where either one alone would fill the panel.
        """
        ee = np.asarray(self.game.arm.ee(), dtype=float)
        if self.game.held():
            other = np.array([*self.game.target, scene.CUBE_HALF * 2])
        else:
            other = np.asarray(self.game.cube_pos(), dtype=float)
        mid = 0.5 * (ee + other)
        mid[2] = max(mid[2], 0.06)      # never look at a point under the floor
        return mid

    def track(self) -> None:
        """Ease the camera's lookat onto `_focus`, once per simulated instant.

        Rendering the main view calls this; it is public so that a test can
        advance the camera without needing a GL context.

        Keyed on sim time because a frame is rendered more than once per tick
        when recording (the window's panels, then the dataset's own fixed-size
        views), and a lerp that ran per *render* would move at whatever rate the
        layout happened to demand.
        """
        self._ensure_cam()
        now = float(self.game.d.time)
        was, self._cam_t = self._cam_t, now
        if not self.following or was is None or now <= was:
            return
        dt = min(now - was, 0.1)        # a switch or a reset must not teleport it
        self.cam.lookat[:] += (self._focus() - self.cam.lookat) * \
            (1.0 - math.exp(-dt / FOLLOW_TAU))

    # The native env is the only one with an orbitable camera of our own.
    def orbit(self, degrees: float) -> None:
        self._ensure_cam().azimuth += degrees

    def zoom(self, amount: float) -> None:
        """Dolly in (+) / out (-). Multiplicative, and clamped to ZOOM_RANGE.

        The clamp is not decoration: inside 0.45 m the near plane starts eating
        the gripper, and past 2.2 m the cube is the handful of pixels this
        camera was moved in to avoid.
        """
        cam, (lo, hi) = self._ensure_cam(), ZOOM_RANGE
        cam.distance = float(np.clip(cam.distance * math.exp(-amount * ZOOM_RATE),
                                     lo, hi))

    def toggle_follow(self) -> bool:
        """Follow the work, or freeze the lookat where it currently is."""
        self.following = not self.following
        return self.following

    @property
    def azimuth(self) -> float:
        return self.cam.azimuth if self.cam is not None else HOME_CAM[2]

    def close(self) -> None:
        for r in self._renderers.values():
            self._close(r)
        self._renderers.clear()
