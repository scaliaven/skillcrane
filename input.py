"""Gamepad and keyboard readers, both producing a common ControlInput.

The stick -> world mapping math is plain functions so it can be unit tested with
a fake pad: no hardware, no display, no pygame needed for the maths.
"""
from dataclasses import dataclass

import math

import numpy as np

import scene

try:                            # pygame is only needed to talk to real devices
    import pygame
except ImportError:             # pragma: no cover - exercised by the CI env
    pygame = None

# ---------------------------------------------------------------------------
# PAD MAPPING -- the single place axis/button indices are allowed to live.
# 8BitDo pads enumerate differently per pairing mode (Start+A = Apple,
# Start+B = D-input; XInput is a Windows API and is useless on macOS). If the
# arm moves on the wrong stick, run `python gamepad_probe.py`, note the live
# indices, and edit these four lines -- nowhere else.
# ---------------------------------------------------------------------------
AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3
BTN_GRIP, BTN_RESET, BTN_CAM_L, BTN_CAM_R = 0, 3, 4, 5
DEADZONE = 0.15

CAM_ORBIT_SPEED = 90.0          # degrees/second while a bumper is held


@dataclass
class ControlInput:
    """One tick of operator intent, device independent.

    mx/my are raw stick values in the *camera* frame; call world_xy() to get the
    world-frame direction the Cartesian target should move in.
    """
    mx: float = 0.0             # camera-right
    my: float = 0.0             # camera-forward (away from the camera)
    mz: float = 0.0             # up
    dyaw: float = 0.0           # wrist rotation
    grip: bool = False          # True = closed
    reset: bool = False
    cam: float = 0.0            # camera orbit, -1..1

    def world_xy(self, cam_az: float) -> tuple:
        """Rotate the planar stick input into world x/y.

        cam_az is the MuJoCo camera azimuth in *radians*. The camera sits at
        (cos az, sin az) from its lookat point, so "away from the camera" -- the
        direction the operator reads as forward -- is the negative of that.
        Right is forward rotated -90 degrees.
        """
        fwd = np.array([-math.cos(cam_az), -math.sin(cam_az)])
        rgt = np.array([-math.sin(cam_az), math.cos(cam_az)])
        v = self.my * fwd + self.mx * rgt
        return float(v[0]), float(v[1])

    def clipped(self) -> "ControlInput":
        """Axes clamped to [-1, 1] after any merging."""
        return ControlInput(
            mx=float(np.clip(self.mx, -1, 1)), my=float(np.clip(self.my, -1, 1)),
            mz=float(np.clip(self.mz, -1, 1)), dyaw=float(np.clip(self.dyaw, -1, 1)),
            grip=self.grip, reset=self.reset,
            cam=float(np.clip(self.cam, -1, 1)))


def deadzone(v: float, d: float = DEADZONE) -> float:
    """Deadzone with the live range rescaled, so output is continuous at +/-d."""
    if abs(v) < d:
        return 0.0
    return (abs(v) - d) / (1.0 - d) * (1.0 if v > 0 else -1.0)


def merge(a: ControlInput, b: ControlInput) -> ControlInput:
    """Sum two devices' axes and OR their buttons, then clip.

    Means a held key wins over an idle stick, and hold-to-close beats the
    gamepad's toggle without either device needing to know about the other.
    """
    return ControlInput(a.mx + b.mx, a.my + b.my, a.mz + b.mz, a.dyaw + b.dyaw,
                        a.grip or b.grip, a.reset or b.reset, a.cam + b.cam).clipped()


class GamepadReader:
    """Reads an 8BitDo-style pad. `pad` is any object with the pygame joystick
    API (get_numaxes/get_numbuttons/get_axis/get_button), so tests inject a fake.
    """

    def __init__(self, pad):
        self.pad = pad
        self.grip_latch = False     # previous frame's button state
        self.closed = False         # A is a toggle, so the reader holds the state

    @classmethod
    def open(cls, index: int = 0):
        """First connected pad, or None. Requires pygame."""
        if pygame is None:
            return None
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            return None
        pad = pygame.joystick.Joystick(index)
        pad.init()
        return cls(pad)

    @property
    def name(self) -> str:
        return self.pad.get_name() if hasattr(self.pad, "get_name") else "pad"

    def _axis(self, i: int) -> float:
        return self.pad.get_axis(i) if i < self.pad.get_numaxes() else 0.0

    def _button(self, i: int) -> bool:
        return bool(self.pad.get_button(i)) if i < self.pad.get_numbuttons() else False

    def read(self) -> ControlInput:
        grip_btn = self._button(BTN_GRIP)
        if grip_btn and not self.grip_latch:     # rising edge only
            self.closed = not self.closed
        self.grip_latch = grip_btn
        return ControlInput(
            mx=deadzone(self._axis(AX_LX)),
            my=-deadzone(self._axis(AX_LY)),     # pads report stick-up as negative
            mz=-deadzone(self._axis(AX_RY)),
            dyaw=deadzone(self._axis(AX_RX)),
            grip=self.closed,
            reset=self._button(BTN_RESET),
            cam=float(self._button(BTN_CAM_R)) - float(self._button(BTN_CAM_L)),
        )


class KeyboardReader:
    """Fallback when no pad is connected. `pressed` is pygame.key.get_pressed()
    (or any sequence indexable by key code), so tests can pass a plain list.
    """

    def read(self, pressed) -> ControlInput:
        if pygame is None:      # pragma: no cover
            return ControlInput()
        k = pygame
        return ControlInput(
            mx=float(pressed[k.K_d]) - float(pressed[k.K_a]),
            my=float(pressed[k.K_w]) - float(pressed[k.K_s]),
            mz=float(pressed[k.K_e]) - float(pressed[k.K_q]),
            dyaw=float(pressed[k.K_c]) - float(pressed[k.K_z]),
            grip=bool(pressed[k.K_SPACE]),       # hold to close
            reset=bool(pressed[k.K_r]),
            cam=float(pressed[k.K_RIGHT]) - float(pressed[k.K_LEFT]),
        )
