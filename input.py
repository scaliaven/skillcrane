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
#
# Two paths, and the first is why an 8BitDo now works without hand-mapping:
#
# 1. SDL's game-controller layer. SDL ships a controller database, so a pad it
#    recognises reports a *standard* layout -- A is A, LB is LB -- whatever
#    pairing mode the pad is in. GamepadReader.open() prefers this.
# 2. Raw joystick indices, for a pad SDL has never heard of. These are the
#    numbers gamepad_probe.py prints; edit them here and nowhere else.
#
# The two numberings genuinely disagree: SDL calls the shoulders 9 and 10,
# while a raw HID pad usually reports them at 4 and 5. Reading a controller
# through the raw table is how "the camera buttons do nothing" happens.
# ---------------------------------------------------------------------------

# -- 1. standard layout, by role. Values are filled in from SDL's own ids.
if pygame is not None:
    ROLE_AXIS = {
        "lx": pygame.CONTROLLER_AXIS_LEFTX, "ly": pygame.CONTROLLER_AXIS_LEFTY,
        "rx": pygame.CONTROLLER_AXIS_RIGHTX, "ry": pygame.CONTROLLER_AXIS_RIGHTY,
    }
    ROLE_BUTTON = {
        "grip": pygame.CONTROLLER_BUTTON_A,
        "reset": pygame.CONTROLLER_BUTTON_Y,
        "view": pygame.CONTROLLER_BUTTON_X,
        "cam_l": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
        "cam_r": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
        "dpad_l": pygame.CONTROLLER_BUTTON_DPAD_LEFT,
        "dpad_r": pygame.CONTROLLER_BUTTON_DPAD_RIGHT,
        "env_prev": pygame.CONTROLLER_BUTTON_BACK,
        "env_next": pygame.CONTROLLER_BUTTON_START,
    }
else:                           # pragma: no cover - no pygame, no real device
    ROLE_AXIS, ROLE_BUTTON = {}, {}

# -- 2. raw joystick fallback. 8BitDo pads enumerate differently per pairing
#       mode (Start+A = Apple, Start+B = D-input; XInput is a Windows API and
#       is useless on macOS), so these are a starting point, not a promise.
AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3
BTN_GRIP, BTN_VIEW, BTN_RESET, BTN_CAM_L, BTN_CAM_R = 0, 2, 3, 4, 5
BTN_ENV_PREV, BTN_ENV_NEXT = 6, 7          # usually Select/Back and Start

RAW_AXIS = {"lx": AX_LX, "ly": AX_LY, "rx": AX_RX, "ry": AX_RY}
RAW_BUTTON = {"grip": BTN_GRIP, "reset": BTN_RESET, "view": BTN_VIEW,
              "cam_l": BTN_CAM_L, "cam_r": BTN_CAM_R,
              "env_prev": BTN_ENV_PREV, "env_next": BTN_ENV_NEXT}
# The d-pad is a hat on a raw pad, not two buttons, so it is read separately.

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
    env: int = 0                # -1 previous environment family, +1 next, 0 stay
    task: int = 0               # -1 previous task in this family, +1 next, 0 stay
    view: bool = False          # cycle the view layout (single / inset / grid)

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
            cam=float(np.clip(self.cam, -1, 1)),
            # env and task are directions, not magnitudes: two devices asking
            # at once must still step exactly one environment, one task.
            env=int(np.sign(self.env)), task=int(np.sign(self.task)),
            view=bool(self.view))


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
                        a.grip or b.grip, a.reset or b.reset, a.cam + b.cam,
                        a.env or b.env, a.task or b.task,
                        a.view or b.view).clipped()


class GamepadReader:
    """Reads a pad by role -- "grip", "cam_l" -- not by index.

    `pad` is any object with the pygame joystick API (get_numaxes/get_numbuttons/
    get_axis/get_button), so tests inject a fake. `ctl` is the optional SDL
    game-controller view of the same physical device; when present it is used
    instead, because its layout is standard across pairing modes.
    """

    def __init__(self, pad, ctl=None):
        self.pad = pad
        self.ctl = ctl
        self.grip_latch = False     # previous frame's button state
        self.closed = False         # A is a toggle, so the reader holds the state

    @classmethod
    def open(cls, index: int = 0):
        """First connected pad, or None. Requires pygame.

        Prefers SDL's game-controller view. SDL recognises the pad from its own
        database and reports a standard layout, which is what makes the face
        and shoulder buttons land where their labels say on an 8BitDo. A pad
        SDL does not know falls back to the raw indices above.
        """
        if pygame is None:
            return None
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            return None
        pad = pygame.joystick.Joystick(index)
        pad.init()
        return cls(pad, ctl=_open_controller(index))

    @property
    def name(self) -> str:
        return self.pad.get_name() if hasattr(self.pad, "get_name") else "pad"

    @property
    def standard(self) -> bool:
        """True when reading through SDL's standard layout rather than raw ids."""
        return self.ctl is not None

    def _axis(self, role: str) -> float:
        if self.ctl is not None:
            # SDL reports axes as signed 16-bit, the joystick API as -1..1.
            return self.ctl.get_axis(ROLE_AXIS[role]) / 32767.0
        i = RAW_AXIS[role]
        return self.pad.get_axis(i) if i < self.pad.get_numaxes() else 0.0

    def _button(self, role: str) -> bool:
        if self.ctl is not None:
            return bool(self.ctl.get_button(ROLE_BUTTON[role]))
        i = RAW_BUTTON.get(role)
        if i is None:               # d-pad: a hat on a raw pad, handled below
            return False
        return bool(self.pad.get_button(i)) if i < self.pad.get_numbuttons() else False

    def _dpad_x(self) -> float:
        """D-pad left/right, which steps the task. Buttons under SDL, a hat raw."""
        if self.ctl is not None:
            return float(self._button("dpad_r")) - float(self._button("dpad_l"))
        if not hasattr(self.pad, "get_numhats") or self.pad.get_numhats() < 1:
            return 0.0
        return float(self.pad.get_hat(0)[0])

    def read(self) -> ControlInput:
        grip_btn = self._button("grip")
        if grip_btn and not self.grip_latch:     # rising edge only
            self.closed = not self.closed
        self.grip_latch = grip_btn
        cam = float(self._button("cam_r")) - float(self._button("cam_l"))
        return ControlInput(
            mx=deadzone(self._axis("lx")),
            my=-deadzone(self._axis("ly")),      # pads report stick-up as negative
            mz=-deadzone(self._axis("ry")),
            dyaw=deadzone(self._axis("rx")),
            grip=self.closed,
            reset=self._button("reset"),
            cam=cam,
            env=int(self._button("env_next")) - int(self._button("env_prev")),
            task=int(self._dpad_x()),
            view=self._button("view"),
        )


def _open_controller(index: int):
    """SDL game-controller handle for joystick `index`, or None.

    None is the normal outcome for a pad missing from SDL's database, not an
    error -- the caller falls back to raw indices.
    """
    try:
        from pygame._sdl2 import controller
    except ImportError:             # pragma: no cover - very old pygame
        return None
    try:
        controller.init()
        if not controller.is_controller(index):
            return None
        ctl = controller.Controller(index)
        ctl.init()
        return ctl
    except Exception:               # pragma: no cover - driver-dependent
        return None


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
            env=int(bool(pressed[k.K_RIGHTBRACKET])) - int(bool(pressed[k.K_LEFTBRACKET])),
            task=int(bool(pressed[k.K_PERIOD])) - int(bool(pressed[k.K_COMMA])),
            view=bool(pressed[k.K_v]),
        )
