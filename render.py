"""Offscreen frame + pygame HUD.

pygame owns the window; the environment renders itself offscreen and hands us a
frame. That is what lets this run under plain `python` instead of `mjpython`:
mujoco.viewer wants the macOS main thread and so does SDL.

This module holds no logic and knows nothing about any particular simulator --
it draws a `benchmarks.Hud` and blits whatever frames it is given, so the native
arm and every benchmark environment share one HUD.

It does own the *layout* of those frames. An environment says which cameras it
has; `panels()` decides where each one goes and how big it is, and the caller
asks the environment for exactly those sizes. That is the whole multi-view
mechanism: no camera is rendered at a size it will not be drawn at.
"""
import numpy as np
import pygame

import scene

# Default window. The window is resizable and `--window WxH` overrides this, so
# nothing downstream may assume a size: the viewport, the panel rects, the HUD
# type sizes and the frames asked of the environment are all derived from
# whatever the window currently is.
WINDOW_W, WINDOW_H = 1280, 860
MIN_W, MIN_H = 720, 520                # below this the HUD stops fitting
HUD_H = 160                            # HUD band height at BASE_H, scaled with it
BASE_H = 620                           # window height the HUD type sizes were drawn at
MAX_SCALE = 1.8                        # past this the HUD eats the viewport
CAM_SPEED = 90.0                       # degrees/second of camera orbit at full input

# Multi-view layouts. An environment declares its cameras (TeleopEnv.view_names)
# and the layout decides how many of them fit and how big each one is; the env
# is then asked to render exactly those sizes, so nothing is rendered large and
# thrown away. A one-camera environment always gets `single`, whatever the
# operator last picked -- there is no second view to show.
LAYOUTS = ("single", "inset", "grid")
INSET_FRAC = 0.24                      # inset width as a fraction of the viewport
INSET_PAD_FRAC = 0.011

# MuJoCo's offscreen framebuffer is fixed at model-compile time and it raises
# rather than downscaling. The default window has to fit inside it; a window
# dragged past it is not an error, but its panels are rendered at the cap and
# scaled up (see `Display.view_sizes`), because the alternative is a crash
# mid-session.
assert WINDOW_W <= scene.OFF_W and WINDOW_H <= scene.OFF_H, \
    f"default window {WINDOW_W}x{WINDOW_H} exceeds MJCF offscreen " \
    f"{scene.OFF_W}x{scene.OFF_H}"

MARGIN = 24                            # HUD text inset, at scale 1

BG = (14, 15, 18)
HUD_BG = (22, 24, 28)
WHITE = (235, 235, 240)
GREY = (150, 155, 165)
DIM = (110, 115, 125)
FAINT = (90, 95, 105)
GOLD = (250, 190, 60)
GREEN = (90, 230, 140)
RED = (240, 120, 110)


def panels(layout: str, n: int, size) -> list:
    """(x, y, w, h) per view under `layout` inside a `size` viewport, main first.

    Returns at most `n` rects and never more than the layout holds -- 4 in the
    grid, 1 + 3 insets -- so an environment with six cameras simply shows the
    first few rather than shrinking everything into unreadable tiles.

    `size` is passed in rather than read from a constant because the window is
    resizable: the panels have to be recomputed for whatever it is now.
    """
    vw, vh = int(size[0]), int(size[1])
    n = max(1, int(n))
    full = (0, 0, vw, vh)
    if n == 1 or layout == "single":
        return [full]
    if layout == "grid":
        if n == 2:                     # two halves read better than two tiles
            w = vw // 2
            return [(0, 0, w, vh), (w, 0, vw - w, vh)]
        w, h = vw // 2, vh // 2
        return [(x, y, w, h) for x, y in ((0, 0), (w, 0), (0, h), (w, h))][:min(n, 4)]
    iw = max(48, round(vw * INSET_FRAC))
    ih = max(32, round(iw * vh / vw))
    pad = max(4, round(vw * INSET_PAD_FRAC))
    return [full] + [(vw - iw - pad, pad + i * (ih + pad), iw, ih)
                     for i in range(min(n - 1, 3))]


class Display:
    """Window, HUD, and a place to blit whatever the environment rendered."""

    def __init__(self, caption="Skillcrane", size=None):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode(self._clamp(size or (WINDOW_W, WINDOW_H)),
                                              pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.flash = 0.0
        self.layout = LAYOUTS[0]
        self._measure()

    @staticmethod
    def _clamp(size) -> tuple:
        """A window no smaller than the HUD needs. No upper bound on purpose --
        the operator gets to make it as big as their screen; only the *rendered*
        panels are capped, by `view_sizes`."""
        return (max(MIN_W, int(size[0])), max(MIN_H, int(size[1])))

    def _measure(self) -> None:
        """Recompute everything that depends on the window size.

        Called on open and on every resize. The HUD was laid out at BASE_H, so
        one scale factor drives the type sizes, the band height and the text
        insets together; the viewport is whatever is left above it.
        """
        w, h = self.screen.get_size()
        self.scale = min(max(h / BASE_H, 1.0), MAX_SCALE)
        self.hud_h = round(HUD_H * self.scale)
        self.render_w, self.render_h = w, max(120, h - self.hud_h)
        font = "menlo,dejavusansmono,monospace"
        self.f_big = pygame.font.SysFont(font, round(30 * self.scale), bold=True)
        self.f_sm = pygame.font.SysFont(font, round(17 * self.scale))
        self.f_tiny = pygame.font.SysFont(font, round(13 * self.scale))
        self.margin = round(MARGIN * self.scale)
        # Hint lines are monospace, so how much fits is just a character count.
        self.hint_chars = max(8, (w - 2 * self.margin) // self.f_tiny.size("M")[0])

    @property
    def viewport(self) -> tuple:
        """(w, h) of the 3D area above the HUD."""
        return (self.render_w, self.render_h)

    def resize(self, w: int, h: int) -> None:
        """Follow a window drag. `main.py` calls this on pygame.VIDEORESIZE."""
        self.screen = pygame.display.set_mode(self._clamp((w, h)), pygame.RESIZABLE)
        self._measure()

    def set_caption(self, caption: str) -> None:
        """Window title, so switching environments is visible outside the HUD."""
        pygame.display.set_caption(caption)

    def cycle_layout(self) -> str:
        """Next view layout, and the name of it."""
        self.layout = LAYOUTS[(LAYOUTS.index(self.layout) + 1) % len(LAYOUTS)]
        return self.layout

    def panels(self, n: int) -> list:
        """Panel rects for `n` views in this window's current viewport."""
        return panels(self.layout, n, self.viewport)

    def view_sizes(self, view_names) -> dict:
        """{view: (w, h)} to ask the environment for, under the current layout.

        Only the views the layout has room for appear, so `env.frames()` never
        renders a camera that would not be drawn.

        Sizes are capped at the MJCF's offscreen buffer: MuJoCo raises rather
        than downscaling, and a window dragged wider than that must degrade to a
        scaled-up frame, not an exception in the middle of a round.
        """
        return {name: (min(r[2], scene.OFF_W), min(r[3], scene.OFF_H))
                for name, r in zip(view_names, self.panels(len(view_names)))}

    def blit_frame(self, rgb, rect=None) -> None:
        """Blit an RGB frame into `rect`, scaled if it is a different size.

        Benchmark environments fix their frame size at construction, so scaling
        here is what lets one window serve all of them.
        """
        x, y, w, h = rect or (0, 0, self.render_w, self.render_h)
        if rgb is None:
            pygame.draw.rect(self.screen, (30, 32, 38), (x, y, w, h))
            return
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surf.get_size() != (w, h):
            surf = pygame.transform.smoothscale(surf, (w, h))
        self.screen.blit(surf, (x, y))

    def blit_views(self, views) -> None:
        """One frame, or a {view: RGB} dict laid out by the current layout.

        Dict order is the environment's `view_names` order, i.e. the operator's
        main view first, which is what the layout puts in the big panel.
        """
        if views is None or isinstance(views, np.ndarray):
            self.blit_frame(views)
            return
        rects = self.panels(len(views))
        for (name, rgb), rect in zip(views.items(), rects):
            self.blit_frame(rgb, rect)
            if len(rects) > 1:
                self._label(name, rect)

    def _label(self, name: str, rect) -> None:
        """Name a panel, on a dark strip so it reads over any frame."""
        x, y, w, h = rect
        strip = self.f_tiny.get_height() + round(5 * self.scale)
        pygame.draw.rect(self.screen, BG, (x, y, w, strip))
        self.screen.blit(self.f_tiny.render(name, True, GREY),
                         (x + round(6 * self.scale), y + round(2 * self.scale)))
        pygame.draw.rect(self.screen, (60, 64, 72), rect, 1)

    def draw(self, hud, frame=None, scored: bool = False, dt: float = 1 / 60,
             controls: str = "", status: str = "") -> None:
        """`frame` is one RGB array, or {view: RGB} for a multi-camera layout.

        `status` says *what* is running (which suite, which task, which layout)
        and `controls` says how to change it; they are separate lines because
        they answer separate questions and either one can outgrow the window.
        """
        if scored:
            self.flash = 0.7

        self.screen.fill(BG)
        self.blit_views(frame)
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, HUD_BG, (0, self.render_h, w, h - self.render_h))

        # Everything below is placed relative to the HUD band and the scale
        # factor, so the same layout holds in a 720-wide window and a 2560-wide
        # one. The numbers are the pixel offsets it was drawn at, at scale 1.
        def at(x, y):
            return (round(x * self.scale), self.render_h + round(y * self.scale))

        acc = GREEN if self.flash > 0 else GOLD
        blit = self.screen.blit
        blit(self.f_big.render(f"SCORE {hud.score}", True, acc), at(MARGIN, 20))
        blit(self.f_big.render(f"{hud.time_left:5.1f}s", True, WHITE), at(250, 20))
        blit(self.f_sm.render(f"streak {hud.streak}  best {hud.best_streak}",
                              True, GREY), at(420, 28))
        blit(self.f_sm.render(f"gripper {hud.grip}", True, GREY), at(420, 50))
        # Three hint lines, in the order an operator needs them: what is
        # running, how to change it, and what the environment says it wants.
        # All are trimmed rather than allowed to run off the edge -- the
        # controls line names ten bindings and "robosuite:NutAssemblyRound" is
        # a legitimate environment name.
        self._hint(status, 82, GREY)
        self._hint(controls or "L-stick move  R-stick lift/rotate  A grip  "
                               "LB/RB camera  Y reset", 106, DIM)
        self._hint(f"{hud.task}   ee {np.round(hud.ee, 2)}   "
                   f"obj {np.round(hud.obj, 2)}", 130, FAINT)
        if hud.time_left <= 0:
            blit(self.f_big.render("TIME  -  press R", True, RED),
                 (w - round(280 * self.scale), self.render_h + round(20 * self.scale)))

        self.flash = max(0.0, self.flash - dt)
        pygame.display.flip()

    def _hint(self, text: str, y: int, colour) -> None:
        """One HUD hint line, at `y` inside the HUD band, trimmed to fit."""
        if len(text) > self.hint_chars:
            text = text[:self.hint_chars - 3] + "..."
        self.screen.blit(self.f_tiny.render(text, True, colour),
                         (self.margin, self.render_h + round(y * self.scale)))

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
