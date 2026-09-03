"""M2 -- the arm tracks its command without oscillating.

The base-joint ramp is the canary for the collision bug M1 guards against: with
collision left on the structural links the world-welded base grinds on link 1
and j1 oscillates at >15 rad/s, never settling on its command. Here it must
settle to 1e-3 with peak |qvel| under 2.0.
"""
import numpy as np
import mujoco
import pytest

import scene
from game import Game
from kin import Arm, down_R

RAMP_TO = 1.4           # rad
RAMP_RATE = 1.0         # rad/s
SWEEP_SPEED = 0.30      # m/s of Cartesian target travel
SWEEP_R = 0.10          # m, radius of the swept circle
SWEEP_CENTRE = np.array([0.30, 0.0, 0.30])


def test_base_joint_ramp_is_stable():
    """Drive j1 through 1.4 rad on the raw actuator, with no IK in the loop."""
    m = mujoco.MjModel.from_xml_string(scene.XML)
    d = mujoco.MjData(m)
    arm = Arm(m, d)
    q0 = arm.solve(scene.HOME, down_R(0.0), scene.HOME_SEED)

    mujoco.mj_resetData(m, d)
    d.qpos[arm.qadr] = q0
    d.ctrl[:scene.NARM] = q0
    d.ctrl[scene.NARM:] = scene.GRIP_OPEN
    mujoco.mj_forward(m, d)

    peak = 0.0
    ramp_steps = int(RAMP_TO / RAMP_RATE / scene.TIMESTEP)
    for i in range(ramp_steps + 1500):          # ramp, then hold to settle
        d.ctrl[0] = min(RAMP_TO, i * scene.TIMESTEP * RAMP_RATE)
        mujoco.mj_step(m, d)
        peak = max(peak, float(np.max(np.abs(d.qvel[arm.dof]))))

    final_err = abs(float(d.qpos[arm.qadr][0]) - RAMP_TO)
    assert final_err < 1e-3, f"j1 settled {final_err:.2e} rad from its command"
    assert peak < 2.0, f"peak |qvel| {peak:.2f} rad/s -- the arm is oscillating"


def _sweep(game, speed, ticks=1200, warmup=300):
    """Move the target around a vertical circle at `speed`, collecting error."""
    omega = speed / SWEEP_R
    errs = []
    for i in range(ticks):
        t = i * scene.CTRL_DT
        game.tgt[:] = SWEEP_CENTRE + [0.0,
                                      SWEEP_R * np.sin(omega * t),
                                      SWEEP_R * (np.cos(omega * t) - 1.0)]
        game.clamp_target()
        game.step(0, 0, 0, 0, False)
        if i >= warmup:                          # skip the start-up transient
            errs.append(float(np.linalg.norm(game.arm.ee() - game.tgt)))
    return np.array(errs)


def test_cartesian_sweep_tracking_under_10mm():
    errs = _sweep(Game(seed=1), SWEEP_SPEED)
    assert errs.max() < 0.010, (
        f"tracking error peaked at {errs.max() * 1000:.1f} mm "
        f"(mean {errs.mean() * 1000:.1f} mm) on a {SWEEP_SPEED} m/s sweep")


def test_full_stick_sweep_stays_bounded():
    """Full-stick teleop is faster than the swept test and is allowed more lag,
    but it must stay bounded and finite -- no windup, no divergence."""
    g = Game(seed=1)
    errs = []
    for i in range(900):
        # hard reversals: worst case for a rate-limited target
        dx = 1.0 if (i // 150) % 2 == 0 else -1.0
        g.step(dx, 0.0, 0.0, 0.0, False)
        if i > 100:
            errs.append(float(np.linalg.norm(g.arm.ee() - g.tgt)))
    errs = np.array(errs)
    assert np.isfinite(errs).all()
    assert errs.max() < 0.020, f"full-stick error peaked at {errs.max() * 1000:.1f} mm"


@pytest.mark.parametrize("speed", [0.05, 0.15, 0.30])
def test_tracking_degrades_gracefully_with_speed(speed):
    """Error grows with sweep speed but stays inside budget across the range."""
    errs = _sweep(Game(seed=1), speed)
    assert errs.max() < 0.010, f"{speed} m/s -> {errs.max() * 1000:.1f} mm"
