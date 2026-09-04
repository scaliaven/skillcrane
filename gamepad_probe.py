"""
See what your pad reports, and whether SDL already knows it.

    python gamepad_probe.py

Two things are printed. The first matters most:

  standard : SDL recognised the pad from its controller database and reports a
             standard layout (A is A, LB is LB) whatever pairing mode it is in.
             input.py uses this path automatically -- nothing to configure.
  raw      : SDL has never seen this pad, so input.py falls back to the raw
             indices in its config block. Move one control at a time, note the
             numbers, and edit that block.

8BitDo pads enumerate differently per pairing mode. Hold a face button while
powering on with Start:
    Start + A  ->  Apple  (macOS/iOS)   <- try this first on a Mac
    Start + B  ->  D-input             <- and this if Apple mode looks odd
    Start + X  ->  XInput (Windows; does nothing useful on macOS)
    Start + Y  ->  Nintendo Switch
Some models, including the Ultimate 2C, use a switch on the back instead.
"""
import pygame

import input as inp

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise SystemExit(
        "No gamepad detected.\n"
        "  - Pair it in System Settings > Bluetooth first.\n"
        "  - On macOS prefer Apple mode (Start+A) or D-input (Start+B).\n"
        "  - The 2.4GHz USB dongle has noticeably lower latency than Bluetooth."
    )

pad = pygame.joystick.Joystick(0)
pad.init()

ctl = inp._open_controller(0)
print(f"name    : {pad.get_name()}")
print(f"axes    : {pad.get_numaxes()}   buttons: {pad.get_numbuttons()}   "
      f"hats: {pad.get_numhats()}")
print(f"layout  : {'standard (SDL controller database)' if ctl else 'raw indices'}")
if ctl is None:
    print("          SDL does not know this pad. If the labels below do not match\n"
          "          the buttons you press, edit the config block in input.py.")

# Roles, in the order the HUD lists them, so a mismatch is obvious at a glance.
ROLES = [("grip", "A"), ("follow", "B"), ("view", "X"), ("reset", "Y"),
         ("cam_l", "LB"), ("cam_r", "RB"),
         ("suite_prev", "Back/Select"), ("suite_next", "Start")]

reader = inp.GamepadReader(pad, ctl=ctl)

screen = pygame.display.set_mode((520, 90))
pygame.display.set_caption("gamepad probe - keep this window focused")
clock = pygame.time.Clock()

print("\nMove sticks / press buttons. Ctrl-C to quit.")
print("Top line: what input.py makes of it. Bottom: the raw numbers.\n")

try:
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt

        ci = reader.read()
        named = [f"{label}" for role, label in ROLES if reader._button(role)]
        if abs(ci.mx) > 0 or abs(ci.my) > 0:
            named.append(f"move({ci.mx:+.2f},{ci.my:+.2f})")
        if abs(ci.mz) > 0 or abs(ci.dyaw) > 0:
            named.append(f"lift{ci.mz:+.2f} wrist{ci.dyaw:+.2f}")
        if ci.cam:
            named.append(f"camera{ci.cam:+.0f}")
        if ci.zoom:
            named.append(f"zoom{ci.zoom:+.2f}")
        if ci.suite:
            named.append(f"suite{ci.suite:+d}")
        if ci.task:
            named.append(f"task{ci.task:+d}")
        if ci.view:
            named.append("views")

        axes = [(i, round(pad.get_axis(i), 2)) for i in range(pad.get_numaxes())]
        raw = [f"ax{i}={v:+.2f}" for i, v in axes if abs(v) > inp.DEADZONE]
        raw += [f"btn{i}" for i in range(pad.get_numbuttons()) if pad.get_button(i)]
        raw += [f"hat{i}={pad.get_hat(i)}" for i in range(pad.get_numhats())
                if pad.get_hat(i) != (0, 0)]

        if named or raw:
            print(f"  {'  '.join(named):<46} | {'  '.join(raw)}".ljust(110),
                  end="\r", flush=True)

        clock.tick(30)
except KeyboardInterrupt:
    print("\n")
    if ctl is not None:
        print("SDL maps this pad already -- there is nothing to edit.")
        print("If a control still lands wrong, its SDL mapping is:")
        print(" ", ctl.get_mapping())
    else:
        print("Map what you saw onto the config block at the top of input.py:")
        print("  AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3")
        print("  AX_LT, AX_RT = 4, 5")
        print("  BTN_GRIP, BTN_FOLLOW, BTN_VIEW, BTN_RESET, BTN_CAM_L, BTN_CAM_R "
              "= 0, 1, 2, 3, 4, 5")
        print("  BTN_SUITE_PREV, BTN_SUITE_NEXT = 6, 7")
        print("The d-pad steps the task and is read as a hat, not as buttons.")
        print("Triggers are read one-sided: a pad that rests at -1 only gets")
        print("the top half of the travel, which beats zooming out on its own.")
finally:
    pygame.quit()
