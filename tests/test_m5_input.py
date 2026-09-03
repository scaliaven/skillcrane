"""M5 -- deadzone, camera-relative mapping and the readers, with a fake pad.

No hardware and no display: GamepadReader takes any object with the pygame
joystick API, and KeyboardReader takes anything indexable by key constant.
"""
import math
from collections import defaultdict

import numpy as np
import pygame
import pytest

import input as inp
from input import ControlInput, GamepadReader, KeyboardReader, deadzone, merge


class FakePad:
    """Minimal stand-in for pygame.joystick.Joystick."""

    def __init__(self, axes=(0.0, 0.0, 0.0, 0.0), buttons=(0,) * 8):
        self.axes = list(axes)
        self.buttons = list(buttons)

    def get_numaxes(self):
        return len(self.axes)

    def get_numbuttons(self):
        return len(self.buttons)

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return self.buttons[i]

    def get_name(self):
        return "FakePad"


def keys(*pressed):
    d = defaultdict(bool)
    for k in pressed:
        d[k] = True
    return d


# --- deadzone ---------------------------------------------------------------

@pytest.mark.parametrize("v", [0.0, 0.05, 0.1499, -0.1499, -0.05])
def test_deadzone_suppresses_small_input(v):
    assert deadzone(v) == 0.0


def test_deadzone_is_continuous_at_the_threshold():
    just_outside = deadzone(inp.DEADZONE + 1e-9)
    assert just_outside == pytest.approx(0.0, abs=1e-6), \
        "output must not jump when the stick crosses the deadzone"


def test_deadzone_rescales_to_full_range():
    assert deadzone(1.0) == pytest.approx(1.0)
    assert deadzone(-1.0) == pytest.approx(-1.0)
    mid = deadzone((1.0 + inp.DEADZONE) / 2)
    assert mid == pytest.approx(0.5, abs=1e-6)


def test_deadzone_preserves_sign():
    assert deadzone(0.5) > 0 and deadzone(-0.5) < 0
    assert deadzone(0.5) == pytest.approx(-deadzone(-0.5))


# --- camera-relative mapping ------------------------------------------------

@pytest.mark.parametrize("az_deg,expected", [
    (0.0, (-1.0, 0.0)),      # camera on +x, so "away" is -x
    (90.0, (0.0, -1.0)),     # camera on +y, so "away" is -y
    (180.0, (1.0, 0.0)),
    (270.0, (0.0, 1.0)),
])
def test_stick_forward_points_away_from_the_camera(az_deg, expected):
    dx, dy = ControlInput(my=1.0).world_xy(math.radians(az_deg))
    assert (dx, dy) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("az_deg", [0.0, 37.0, 130.0, 245.0])
def test_stick_right_is_perpendicular_and_right_handed(az_deg):
    az = math.radians(az_deg)
    f = np.array(ControlInput(my=1.0).world_xy(az))
    r = np.array(ControlInput(mx=1.0).world_xy(az))
    assert f @ r == pytest.approx(0.0, abs=1e-9), "forward and right must be perpendicular"
    # right = forward rotated -90 degrees, i.e. cross(forward, up) with up = +z.
    # numpy 2 dropped the 2-D cross product, so take the z component directly.
    cross_z = f[0] * r[1] - f[1] * r[0]
    assert cross_z < 0


@pytest.mark.parametrize("az_deg", [0.0, 37.0, 130.0, 245.0])
def test_mapping_is_a_rotation_and_preserves_magnitude(az_deg):
    ci = ControlInput(mx=0.6, my=-0.8)
    v = np.array(ci.world_xy(math.radians(az_deg)))
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)


def test_camera_azimuth_actually_changes_the_world_direction():
    a = ControlInput(my=1.0).world_xy(0.0)
    b = ControlInput(my=1.0).world_xy(math.radians(90.0))
    assert not np.allclose(a, b)


# --- gamepad ----------------------------------------------------------------

def test_gamepad_axis_mapping_and_stick_up_is_positive():
    pad = FakePad(axes=(0.0, 0.0, 0.0, 0.0))
    r = GamepadReader(pad)
    pad.axes[inp.AX_LY] = -1.0          # pads report stick-up as negative
    assert r.read().my == pytest.approx(1.0), "pushing the stick up must move forward"
    pad.axes[inp.AX_LY] = 0.0
    pad.axes[inp.AX_LX] = 1.0
    assert r.read().mx == pytest.approx(1.0)
    pad.axes[inp.AX_LX] = 0.0
    pad.axes[inp.AX_RY] = -1.0
    assert r.read().mz == pytest.approx(1.0), "right stick up must raise the gripper"
    pad.axes[inp.AX_RY] = 0.0
    pad.axes[inp.AX_RX] = 1.0
    assert r.read().dyaw == pytest.approx(1.0)


def test_gamepad_applies_the_deadzone():
    pad = FakePad(axes=(0.1, -0.1, 0.05, 0.0))
    ci = GamepadReader(pad).read()
    assert (ci.mx, ci.my, ci.mz, ci.dyaw) == (0.0, 0.0, 0.0, 0.0)


def test_grip_button_toggles_on_the_rising_edge_only():
    pad = FakePad()
    r = GamepadReader(pad)
    assert r.read().grip is False

    pad.buttons[inp.BTN_GRIP] = 1
    assert r.read().grip is True        # pressed -> closed
    assert r.read().grip is True        # still held -> stays closed
    pad.buttons[inp.BTN_GRIP] = 0
    assert r.read().grip is True        # released -> stays closed
    pad.buttons[inp.BTN_GRIP] = 1
    assert r.read().grip is False       # pressed again -> opens


def test_reset_and_camera_buttons():
    pad = FakePad()
    r = GamepadReader(pad)
    pad.buttons[inp.BTN_RESET] = 1
    assert r.read().reset is True
    pad.buttons[inp.BTN_RESET] = 0
    pad.buttons[inp.BTN_CAM_R] = 1
    assert r.read().cam == pytest.approx(1.0)
    pad.buttons[inp.BTN_CAM_R] = 0
    pad.buttons[inp.BTN_CAM_L] = 1
    assert r.read().cam == pytest.approx(-1.0)


def test_a_pad_with_fewer_axes_than_the_mapping_does_not_crash():
    """8BitDo modes enumerate differently; a short pad must degrade, not raise."""
    ci = GamepadReader(FakePad(axes=(0.0,), buttons=(0,))).read()
    assert (ci.mx, ci.my, ci.mz, ci.dyaw) == (0.0, 0.0, 0.0, 0.0)
    assert ci.grip is False and ci.reset is False and ci.cam == 0.0


# --- keyboard fallback ------------------------------------------------------

def test_keyboard_movement_keys():
    kb = KeyboardReader()
    assert kb.read(keys(pygame.K_w)).my == pytest.approx(1.0)
    assert kb.read(keys(pygame.K_s)).my == pytest.approx(-1.0)
    assert kb.read(keys(pygame.K_d)).mx == pytest.approx(1.0)
    assert kb.read(keys(pygame.K_a)).mx == pytest.approx(-1.0)
    assert kb.read(keys(pygame.K_e)).mz == pytest.approx(1.0)
    assert kb.read(keys(pygame.K_q)).mz == pytest.approx(-1.0)
    assert kb.read(keys(pygame.K_c)).dyaw == pytest.approx(1.0)
    assert kb.read(keys(pygame.K_z)).dyaw == pytest.approx(-1.0)


def test_opposed_keys_cancel():
    ci = KeyboardReader().read(keys(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d))
    assert ci.mx == 0.0 and ci.my == 0.0


def test_keyboard_grip_is_hold_to_close():
    kb = KeyboardReader()
    assert kb.read(keys(pygame.K_SPACE)).grip is True
    assert kb.read(keys()).grip is False, "releasing SPACE must open the gripper"


def test_keyboard_reset_and_orbit():
    kb = KeyboardReader()
    assert kb.read(keys(pygame.K_r)).reset is True
    assert kb.read(keys(pygame.K_RIGHT)).cam == pytest.approx(1.0)
    assert kb.read(keys(pygame.K_LEFT)).cam == pytest.approx(-1.0)


# --- merging both devices ---------------------------------------------------

def test_merge_ors_buttons_so_hold_to_close_beats_an_idle_toggle():
    kb_ci = ControlInput(grip=True)
    pad_ci = ControlInput(grip=False)
    assert merge(kb_ci, pad_ci).grip is True
    assert merge(ControlInput(reset=True), ControlInput()).reset is True


def test_merge_sums_and_clips_axes():
    m = merge(ControlInput(mx=0.8, my=0.5), ControlInput(mx=0.8, my=-0.5))
    assert m.mx == pytest.approx(1.0), "summed axes must clip to the unit range"
    assert m.my == pytest.approx(0.0)


def test_idle_input_is_all_zero():
    ci = merge(KeyboardReader().read(keys()), GamepadReader(FakePad()).read())
    assert (ci.mx, ci.my, ci.mz, ci.dyaw, ci.cam) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert ci.grip is False and ci.reset is False
