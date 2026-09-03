"""
Find your 8BitDo's axis and button indices, then paste them into input.py.

    pip install pygame
    python gamepad_probe.py

8BitDo pads report a different layout per pairing mode. Hold a face button
while powering on with Start:
    Start + A  ->  Apple  (macOS/iOS)   <- try this first on a Mac
    Start + B  ->  D-input             <- try this if A gives odd axes
    Start + X  ->  XInput (Windows; does nothing useful on macOS)
    Start + Y  ->  Nintendo Switch
Some models use a physical switch on the back (A / D / X / S) instead.

Move one stick at a time and note which index changes.
"""
import pygame

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
print(f"name    : {pad.get_name()}")
print(f"axes    : {pad.get_numaxes()}")
print(f"buttons : {pad.get_numbuttons()}")
print(f"hats    : {pad.get_numhats()}")
print("\nMove sticks / press buttons. Ctrl-C to quit.")
print("Anything past a deadzone of 0.15 is shown.\n")

screen = pygame.display.set_mode((360, 90))
pygame.display.set_caption("gamepad probe - keep this window focused")
clock = pygame.time.Clock()

try:
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt

        axes = [(i, round(pad.get_axis(i), 2)) for i in range(pad.get_numaxes())]
        live = [f"ax{i}={v:+.2f}" for i, v in axes if abs(v) > 0.15]
        btns = [f"btn{i}" for i in range(pad.get_numbuttons()) if pad.get_button(i)]
        hats = [f"hat{i}={pad.get_hat(i)}" for i in range(pad.get_numhats())
                if pad.get_hat(i) != (0, 0)]

        line = "  ".join(live + btns + hats)
        if line:
            print(line.ljust(78), end="\r", flush=True)

        clock.tick(30)
except KeyboardInterrupt:
    print("\n\nMap what you saw onto the config block at the top of input.py:")
    print("  AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3")
    print("  BTN_GRIP, BTN_RESET, BTN_CAM_L, BTN_CAM_R = 0, 3, 4, 5")
finally:
    pygame.quit()
