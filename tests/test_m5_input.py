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


# --- switching the benchmark suite ------------------------------------------
# A *suite* is the benchmark (native, robosuite, LIBERO). A *task* is one
# setting inside it. They are separate controls and separate fields on purpose.

def test_suite_buttons_step_one_suite_each_way():
    pad = FakePad()
    r = GamepadReader(pad)
    assert r.read().suite == 0
    pad.buttons[inp.BTN_SUITE_NEXT] = 1
    assert r.read().suite == 1
    pad.buttons[inp.BTN_SUITE_NEXT] = 0
    pad.buttons[inp.BTN_SUITE_PREV] = 1
    assert r.read().suite == -1


def test_keyboard_brackets_switch_suites():
    kb = KeyboardReader()
    assert kb.read(keys(pygame.K_RIGHTBRACKET)).suite == 1
    assert kb.read(keys(pygame.K_LEFTBRACKET)).suite == -1
    assert kb.read(keys()).suite == 0


def test_suite_and_task_are_different_controls():
    """The whole point of the split: neither key touches the other field."""
    kb = KeyboardReader()
    assert kb.read(keys(pygame.K_RIGHTBRACKET)).task == 0
    assert kb.read(keys(pygame.K_PERIOD)).suite == 0
    pad = FakePad()
    pad.buttons[inp.BTN_SUITE_NEXT] = 1
    assert GamepadReader(pad).read().task == 0


def test_merge_never_steps_more_than_one_suite():
    """Both devices asking at once must still move exactly one place."""
    m = merge(ControlInput(suite=1), ControlInput(suite=1))
    assert m.suite == 1


# --- SDL's standard layout --------------------------------------------------

class FakeController:
    """Stand-in for pygame._sdl2.controller.Controller.

    Indexed by SDL's own ids, and axes are signed 16-bit, which is what makes
    it different from the raw joystick API.
    """

    def __init__(self):
        self.axes = defaultdict(int)
        self.buttons = defaultdict(bool)

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return self.buttons[i]


def test_controller_path_scales_sdl_int16_axes_to_unit_range():
    ctl = FakeController()
    ctl.axes[pygame.CONTROLLER_AXIS_LEFTY] = -32767
    ci = GamepadReader(FakePad(), ctl=ctl).read()
    assert ci.my == pytest.approx(1.0), "full stick up must read as +1, not 32767"


def test_controller_shoulders_orbit_where_the_raw_indices_would_not():
    """The regression this path exists for.

    SDL numbers the shoulders 9 and 10; a raw HID pad usually reports them at
    4 and 5. Reading a controller through the raw table is why the camera
    buttons did nothing on an 8BitDo.
    """
    assert pygame.CONTROLLER_BUTTON_LEFTSHOULDER != inp.BTN_CAM_L
    ctl = FakeController()
    ctl.buttons[pygame.CONTROLLER_BUTTON_RIGHTSHOULDER] = True
    assert GamepadReader(FakePad(), ctl=ctl).read().cam == pytest.approx(1.0)
    ctl.buttons[pygame.CONTROLLER_BUTTON_RIGHTSHOULDER] = False
    ctl.buttons[pygame.CONTROLLER_BUTTON_LEFTSHOULDER] = True
    assert GamepadReader(FakePad(), ctl=ctl).read().cam == pytest.approx(-1.0)


def test_controller_ignores_the_raw_indices_entirely():
    pad = FakePad()
    pad.buttons[inp.BTN_CAM_L] = 1        # would orbit on the raw path
    pad.buttons[inp.BTN_GRIP] = 1         # would close the gripper
    ci = GamepadReader(pad, ctl=FakeController()).read()
    assert ci.cam == 0.0 and ci.grip is False


def test_controller_dpad_steps_the_task():
    """The d-pad walks the task ring; the bumpers keep the camera."""
    ctl = FakeController()
    ctl.buttons[pygame.CONTROLLER_BUTTON_DPAD_RIGHT] = True
    ci = GamepadReader(FakePad(), ctl=ctl).read()
    assert ci.task == 1 and ci.cam == 0.0
    ctl.buttons[pygame.CONTROLLER_BUTTON_DPAD_RIGHT] = False
    ctl.buttons[pygame.CONTROLLER_BUTTON_DPAD_LEFT] = True
    assert GamepadReader(FakePad(), ctl=ctl).read().task == -1


def test_controller_x_cycles_the_view_layout():
    ctl = FakeController()
    assert GamepadReader(FakePad(), ctl=ctl).read().view is False
    ctl.buttons[pygame.CONTROLLER_BUTTON_X] = True
    assert GamepadReader(FakePad(), ctl=ctl).read().view is True


def test_controller_triggers_zoom_and_b_toggles_follow():
    ctl = FakeController()
    ctl.axes[pygame.CONTROLLER_AXIS_TRIGGERRIGHT] = 32767
    assert GamepadReader(FakePad(), ctl=ctl).read().zoom == pytest.approx(1.0)
    ctl.axes[pygame.CONTROLLER_AXIS_TRIGGERRIGHT] = 0
    ctl.axes[pygame.CONTROLLER_AXIS_TRIGGERLEFT] = 32767
    assert GamepadReader(FakePad(), ctl=ctl).read().zoom == pytest.approx(-1.0)
    ctl.axes[pygame.CONTROLLER_AXIS_TRIGGERLEFT] = 0
    ctl.buttons[pygame.CONTROLLER_BUTTON_B] = True
    ci = GamepadReader(FakePad(), ctl=ctl).read()
    assert ci.follow is True and ci.grip is False, "B must not touch the gripper"


def test_standard_flag_says_which_path_is_live():
    assert GamepadReader(FakePad()).standard is False
    assert GamepadReader(FakePad(), ctl=FakeController()).standard is True


# --- raw d-pad --------------------------------------------------------------

class FakeHatPad(FakePad):
    """A raw pad whose d-pad is a hat, which is how pygame reports one."""

    def __init__(self, hat=(0, 0), **kw):
        super().__init__(**kw)
        self.hats = [hat]

    def get_numhats(self):
        return len(self.hats)

    def get_hat(self, i):
        return self.hats[i]


def test_raw_dpad_hat_steps_the_task():
    assert GamepadReader(FakeHatPad(hat=(1, 0))).read().task == 1
    assert GamepadReader(FakeHatPad(hat=(-1, 0))).read().task == -1
    assert GamepadReader(FakeHatPad(hat=(0, 1))).read().task == 0, \
        "up/down on the hat is not a task step"


def test_a_pad_with_no_hat_does_not_crash():
    """FakePad has no get_numhats at all, like a minimal joystick."""
    ci = GamepadReader(FakePad()).read()
    assert ci.cam == 0.0 and ci.task == 0


# --- switching environment, task and view -----------------------------------

def test_task_and_view_default_to_doing_nothing():
    ci = GamepadReader(FakePad()).read()
    assert (ci.suite, ci.task, ci.view) == (0, 0, False)
    assert KeyboardReader().read(keys()).task == 0


def test_keyboard_steps_the_task_and_cycles_views():
    r = KeyboardReader()
    assert r.read(keys(pygame.K_PERIOD)).task == 1
    assert r.read(keys(pygame.K_COMMA)).task == -1
    assert r.read(keys(pygame.K_v)).view is True
    assert r.read(keys(pygame.K_w)).view is False


def test_merging_two_devices_still_steps_exactly_one_task():
    """Both devices asking at once must not skip a task."""
    both = merge(ControlInput(task=1), ControlInput(task=1))
    assert both.task == 1
    assert merge(ControlInput(task=-1), ControlInput()).task == -1
    assert merge(ControlInput(view=True), ControlInput()).view is True


def test_the_view_button_is_not_the_grip_button():
    """A layout cycle that also opened the gripper would drop the payload."""
    assert inp.BTN_VIEW != inp.BTN_GRIP
    assert inp.ROLE_BUTTON["view"] not in (inp.ROLE_BUTTON["grip"],
                                           inp.ROLE_BUTTON["reset"])


# --- camera zoom and follow -------------------------------------------------

def test_keyboard_zooms_and_toggles_follow():
    r = KeyboardReader()
    assert r.read(keys(pygame.K_EQUALS)).zoom == pytest.approx(1.0)
    assert r.read(keys(pygame.K_MINUS)).zoom == pytest.approx(-1.0)
    assert r.read(keys()).zoom == 0.0
    assert r.read(keys(pygame.K_f)).follow is True
    assert r.read(keys()).follow is False


def test_raw_triggers_zoom_and_a_pad_resting_at_minus_one_does_not():
    """The reason triggers are floored at 0 rather than rescaled.

    Plenty of raw pads report a released trigger as -1. Rescaling that into the
    zoom would dolly the camera out for the whole session with nothing pressed.
    """
    def with_triggers(lt, rt):
        axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        axes[inp.AX_LT], axes[inp.AX_RT] = lt, rt
        return GamepadReader(FakePad(axes=axes)).read()

    assert with_triggers(0.0, 1.0).zoom == pytest.approx(1.0)
    assert with_triggers(1.0, 0.0).zoom == pytest.approx(-1.0)
    assert with_triggers(-1.0, -1.0).zoom == 0.0, "a resting trigger is not a zoom"


def test_a_four_axis_pad_has_no_triggers_and_does_not_crash():
    """The fallback table names axes 4 and 5; plenty of pads have neither."""
    assert GamepadReader(FakePad()).read().zoom == 0.0


def test_zoom_merges_and_clips_like_any_other_axis():
    assert merge(ControlInput(zoom=0.8), ControlInput(zoom=0.8)).zoom == \
        pytest.approx(1.0)
    assert merge(ControlInput(follow=True), ControlInput()).follow is True
