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

WINDOW_W, WINDOW_H = 900, 620
RENDER_W, RENDER_H = 900, 460          # 3D viewport; the rest is HUD
HUD_TOP = RENDER_H
CAM_SPEED = 90.0                       # degrees/second of camera orbit at full input

# Multi-view layouts. An environment declares its cameras (TeleopEnv.view_names)
# and the layout decides how many of them fit and how big each one is; the env
# is then asked to render exactly those sizes, so nothing is rendered large and
# thrown away. A one-camera environment always gets `single`, whatever the
# operator last picked -- there is no second view to show.
LAYOUTS = ("single", "inset", "grid")
INSET_W = 216                          # a quarter of the viewport, near enough
INSET_PAD = 10

# MuJoCo's offscreen framebuffer is fixed at model-compile time and it raises
# rather than downscaling, so a window bigger than it is a hard error.
assert RENDER_W <= scene.OFF_W and RENDER_H <= scene.OFF_H, \
    f"render {RENDER_W}x{RENDER_H} exceeds MJCF offscreen {scene.OFF_W}x{scene.OFF_H}"

MARGIN = 24                            # HUD text inset from the window edge

BG = (14, 15, 18)
HUD_BG = (22, 24, 28)
WHITE = (235, 235, 240)
GREY = (150, 155, 165)
DIM = (110, 115, 125)
FAINT = (90, 95, 105)
GOLD = (250, 190, 60)
GREEN = (90, 230, 140)
RED = (240, 120, 110)


def panels(layout: str, n: int) -> list:
    """(x, y, w, h) per view under `layout`, main view first.

    Returns at most `n` rects and never more than the layout holds -- 4 in the
    grid, 1 + 3 insets -- so an environment with six cameras simply shows the
    first few rather than shrinking everything into unreadable tiles.
    """
    n = max(1, int(n))
    full = (0, 0, RENDER_W, RENDER_H)
    if n == 1 or layout == "single":
        return [full]
    if layout == "grid":
        if n == 2:                     # two halves read better than two tiles
            w = RENDER_W // 2
            return [(0, 0, w, RENDER_H), (w, 0, RENDER_W - w, RENDER_H)]
        w, h = RENDER_W // 2, RENDER_H // 2
        return [(x, y, w, h) for x, y in ((0, 0), (w, 0), (0, h), (w, h))][:min(n, 4)]
    h = round(INSET_W * RENDER_H / RENDER_W)
    return [full] + [(RENDER_W - INSET_W - INSET_PAD, INSET_PAD + i * (h + INSET_PAD),
                      INSET_W, h) for i in range(min(n - 1, 3))]


class Display:
    """Window, HUD, and a place to blit whatever the environment rendered."""

    def __init__(self, caption="Skillcrane"):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 30, bold=True)
        self.f_sm = pygame.font.SysFont("menlo,dejavusansmono,monospace", 17)
        self.f_tiny = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.flash = 0.0
        self.layout = LAYOUTS[0]
        # Hint lines are monospace, so how much fits is just a character count.
        self.hint_chars = max(8, (WINDOW_W - 2 * MARGIN) // self.f_tiny.size("M")[0])

    def set_caption(self, caption: str) -> None:
        """Window title, so switching environments is visible outside the HUD."""
        pygame.display.set_caption(caption)

    def cycle_layout(self) -> str:
        """Next view layout, and the name of it."""
        self.layout = LAYOUTS[(LAYOUTS.index(self.layout) + 1) % len(LAYOUTS)]
        return self.layout

    def view_sizes(self, view_names) -> dict:
        """{view: (w, h)} to ask the environment for, under the current layout.

        Only the views the layout has room for appear, so `env.frames()` never
        renders a camera that would not be drawn.
        """
        rects = panels(self.layout, len(view_names))
        return {name: (r[2], r[3]) for name, r in zip(view_names, rects)}

    def blit_frame(self, rgb, rect=None) -> None:
        """Blit an RGB frame into `rect`, scaled if it is a different size.

        Benchmark environments fix their frame size at construction, so scaling
        here is what lets one window serve all of them.
        """
        x, y, w, h = rect or (0, 0, RENDER_W, RENDER_H)
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
        rects = panels(self.layout, len(views))
        for (name, rgb), rect in zip(views.items(), rects):
            self.blit_frame(rgb, rect)
            if len(rects) > 1:
                self._label(name, rect)

    def _label(self, name: str, rect) -> None:
        """Name a panel, on a dark strip so it reads over any frame."""
        x, y, w, h = rect
        pygame.draw.rect(self.screen, BG, (x, y, w, 18))
        self.screen.blit(self.f_tiny.render(name, True, GREY), (x + 6, y + 2))
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
        pygame.draw.rect(self.screen, HUD_BG, (0, HUD_TOP, WINDOW_W, WINDOW_H - HUD_TOP))

        acc = GREEN if self.flash > 0 else GOLD
        blit = self.screen.blit
        blit(self.f_big.render(f"SCORE {hud.score}", True, acc), (MARGIN, 480))
        blit(self.f_big.render(f"{hud.time_left:5.1f}s", True, WHITE), (250, 480))
        blit(self.f_sm.render(f"streak {hud.streak}  best {hud.best_streak}",
                              True, GREY), (420, 488))
        blit(self.f_sm.render(f"gripper {hud.grip}", True, GREY), (420, 510))
        # Three hint lines, in the order an operator needs them: what is
        # running, how to change it, and what the environment says it wants.
        # All are trimmed rather than allowed to run off the edge -- the
        # controls line names ten bindings and "robosuite:NutAssemblyRound" is
        # a legitimate environment name.
        self._hint(status, 542, GREY)
        self._hint(controls or "L-stick move  R-stick lift/rotate  A grip  "
                               "LB/RB camera  Y reset", 566, DIM)
        self._hint(f"{hud.task}   ee {np.round(hud.ee, 2)}   "
                   f"obj {np.round(hud.obj, 2)}", 590, FAINT)
        if hud.time_left <= 0:
            blit(self.f_big.render("TIME  -  press R", True, RED), (620, 480))

        self.flash = max(0.0, self.flash - dt)
        pygame.display.flip()

    def _hint(self, text: str, y: int, colour) -> None:
        """One HUD hint line, trimmed rather than allowed to run off the edge."""
        if len(text) > self.hint_chars:
            text = text[:self.hint_chars - 3] + "..."
        self.screen.blit(self.f_tiny.render(text, True, colour), (MARGIN, y))

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
