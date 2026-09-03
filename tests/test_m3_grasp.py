"""M3 -- a scripted approach/descend/close/lift picks the cube up.

Parametrised over spawns, and all of them must pass: one lucky seed proves
nothing about a grasp. Each seed puts the cube somewhere different in the
annulus, so this also exercises the yaw convention (down_R keeps j5 mid-range
regardless of which way the arm is reaching).
"""
import numpy as np
import pytest

import scene
from game import Game, scripted_grasp

SEEDS = list(range(12))
LIFT_Z = 0.15


@pytest.mark.parametrize("seed", SEEDS)
def test_scripted_grasp_lifts_cube(seed):
    g = Game(seed=seed)
    spawn = g.cube_pos().copy()
    assert spawn[2] < 0.05, "cube should start on the floor"

    scripted_grasp(g)

    z = float(g.cube_pos()[2])
    assert z > LIFT_Z, (
        f"seed {seed}: cube spawned at {np.round(spawn[:2], 3)} and only "
        f"reached z={z:.3f}")
    assert g.held(), f"seed {seed}: gripper is not holding the cube at z={z:.3f}"


@pytest.mark.parametrize("seed", SEEDS[:6])
def test_grasp_survives_a_carry(seed):
    """Lifting is not enough -- the cube must stay in the fingers while moving."""
    g = Game(seed=seed)
    scripted_grasp(g)
    from game import drive_to
    drive_to(g, [*g.target, 0.30], True, 400)
    z = float(g.cube_pos()[2])
    assert z > LIFT_Z, f"seed {seed}: cube dropped during the carry (z={z:.3f})"
    xy = float(np.linalg.norm(g.cube_pos()[:2] - g.target))
    assert xy < 0.10, f"seed {seed}: cube ended {xy:.3f} m from the drop zone"


def test_joints_stay_in_limits_through_a_grasp():
    g = Game(seed=3)
    lo, hi = g.arm.lo, g.arm.hi
    worst = 0.0

    def check(game):
        nonlocal worst
        q = game.d.qpos[game.arm.qadr]
        worst = max(worst, float(np.max(np.maximum(lo - q, q - hi))))

    scripted_grasp(g, on_tick=check)
    # Position actuators can overshoot a commanded limit slightly; the command
    # itself is clamped, so anything beyond a few mrad means the IK is pushing.
    assert worst < 0.02, f"joint exceeded its limit by {worst:.4f} rad"
