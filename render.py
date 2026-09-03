"""Offscreen frame + pygame HUD.

pygame owns the window; the environment renders itself offscreen and hands us a
frame. That is what lets this run under plain `python` instead of `mjpython`:
mujoco.viewer wants the macOS main thread and so does SDL.

This module holds no logic and knows nothing about any particular simulator --
it draws a `benchmarks.Hud` and blits whatever frame it is given, so the native
arm and every benchmark environment share one HUD.
"""
import numpy as np
import pygame

import scene

WINDOW_W, WINDOW_H = 900, 620
RENDER_W, RENDER_H = 900, 460          # 3D viewport; the rest is HUD
HUD_TOP = RENDER_H
CAM_SPEED = 90.0                       # degrees/second of camera orbit at full input

# MuJoCo's offscreen framebuffer is fixed at model-compile time and it raises
# rather than downscaling, so a window bigger than it is a hard error.
assert RENDER_W <= scene.OFF_W and RENDER_H <= scene.OFF_H, \
    f"render {RENDER_W}x{RENDER_H} exceeds MJCF offscreen {scene.OFF_W}x{scene.OFF_H}"

BG = (14, 15, 18)
HUD_BG = (22, 24, 28)
WHITE = (235, 235, 240)
GREY = (150, 155, 165)
DIM = (110, 115, 125)
FAINT = (90, 95, 105)
GOLD = (250, 190, 60)
GREEN = (90, 230, 140)
RED = (240, 120, 110)


class Display:
    """Window, HUD, and a place to blit whatever the environment rendered."""

    def __init__(self, caption="Claw Crew"):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 30, bold=True)
        self.f_sm = pygame.font.SysFont("menlo,dejavusansmono,monospace", 17)
        self.flash = 0.0

    def blit_frame(self, rgb) -> None:
        """Blit an RGB frame, scaled to the viewport if it is a different size.

        Benchmark environments fix their frame size at construction, so scaling
        here is what lets one window serve all of them.
        """
        if rgb is None:
            pygame.draw.rect(self.screen, (30, 32, 38), (0, 0, RENDER_W, RENDER_H))
            return
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surf.get_size() != (RENDER_W, RENDER_H):
            surf = pygame.transform.smoothscale(surf, (RENDER_W, RENDER_H))
        self.screen.blit(surf, (0, 0))

    def draw(self, hud, frame=None, scored: bool = False, dt: float = 1 / 60,
             controls: str = "") -> None:
        if scored:
            self.flash = 0.7

        self.screen.fill(BG)
        self.blit_frame(frame)
        pygame.draw.rect(self.screen, HUD_BG, (0, HUD_TOP, WINDOW_W, WINDOW_H - HUD_TOP))

        acc = GREEN if self.flash > 0 else GOLD
        blit = self.screen.blit
        blit(self.f_big.render(f"SCORE {hud.score}", True, acc), (24, 480))
        blit(self.f_big.render(f"{hud.time_left:5.1f}s", True, WHITE), (250, 480))
        blit(self.f_sm.render(f"streak {hud.streak}  best {hud.best_streak}",
                              True, GREY), (420, 488))
        blit(self.f_sm.render(f"gripper {hud.grip}", True, GREY), (420, 510))
        blit(self.f_sm.render(
            controls or "L-stick move  R-stick lift/rotate  A grip  LB/RB camera  Y reset",
            True, DIM), (24, 545))
        blit(self.f_sm.render(
            f"{hud.task}   ee {np.round(hud.ee, 2)}   obj {np.round(hud.obj, 2)}",
            True, FAINT), (24, 572))
        if hud.time_left <= 0:
            blit(self.f_big.render("TIME  -  press R", True, RED), (620, 480))

        self.flash = max(0.0, self.flash - dt)
        pygame.display.flip()

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
