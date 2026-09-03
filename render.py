"""Offscreen MuJoCo render + pygame HUD.

pygame owns the window; MuJoCo renders into a texture we blit. That is what
lets this run under plain `python` instead of `mjpython`: mujoco.viewer wants
the main thread on macOS and so does SDL, and only one of them can have it.

This is the one module that cannot be tested headless, so it holds no logic --
every number it draws is read straight off the Game.
"""
import numpy as np
import pygame
import mujoco

import scene

WINDOW_W, WINDOW_H = 900, 620
RENDER_W, RENDER_H = 900, 460          # 3D viewport; the rest is HUD
HUD_TOP = RENDER_H
CAM_SPEED = 90.0                       # degrees/second of camera orbit at full input

# The offscreen framebuffer is fixed at model-compile time and MuJoCo raises
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
    """Window, offscreen renderer, orbit camera and HUD."""

    def __init__(self, model, caption="Claw Crew"):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 30, bold=True)
        self.f_sm = pygame.font.SysFont("menlo,dejavusansmono,monospace", 17)

        self.renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.10, 0.10, 0.15]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.35, -22.0, 130.0
        self.flash = 0.0

    def orbit(self, amount: float, dt: float) -> None:
        self.cam.azimuth += amount * CAM_SPEED * dt

    def frame(self, data) -> np.ndarray:
        self.renderer.update_scene(data, self.cam)
        return self.renderer.render()

    def draw(self, game, scored: bool = False, dt: float = 1 / 60) -> None:
        if scored:
            self.flash = 0.7

        rgb = self.frame(game.d)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self.screen.fill(BG)
        self.screen.blit(surf, (0, 0))
        pygame.draw.rect(self.screen, HUD_BG, (0, HUD_TOP, WINDOW_W, WINDOW_H - HUD_TOP))

        acc = GREEN if self.flash > 0 else GOLD
        blit = self.screen.blit
        blit(self.f_big.render(f"SCORE {game.score}", True, acc), (24, 480))
        blit(self.f_big.render(f"{game.time_left:5.1f}s", True, WHITE), (250, 480))
        blit(self.f_sm.render(f"streak {game.streak}  best {game.best_streak}",
                              True, GREY), (420, 488))
        state = "HOLDING" if game.held() else ("CLOSED" if game.closed else "OPEN")
        blit(self.f_sm.render(f"gripper {state}", True, GREY), (420, 510))
        blit(self.f_sm.render(
            "L-stick move  R-stick lift/rotate  A grip  LB/RB camera  Y reset",
            True, DIM), (24, 545))
        blit(self.f_sm.render(
            f"ee {np.round(game.arm.ee(), 2)}   cube {np.round(game.cube_pos(), 2)}",
            True, FAINT), (24, 572))
        if game.time_left <= 0:
            blit(self.f_big.render("TIME  -  press R", True, RED), (620, 480))

        self.flash = max(0.0, self.flash - dt)
        pygame.display.flip()

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()

