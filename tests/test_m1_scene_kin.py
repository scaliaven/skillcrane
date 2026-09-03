"""M1 -- the scene loads, nothing collides at rest, and IK converges.

The ncon check is the one that matters: MuJoCo skips its parent-child contact
filter for bodies welded to the world, and the fixed base has no joint, so it
*is* world-welded. Without contype/conaffinity 0 on the structural links the
base geom grinds against link 1 and the yaw joint never tracks its command.
"""
import math

import numpy as np
import mujoco
import pytest

import scene
from game import Game
from kin import Arm, down_R

# Only these geoms are allowed to collide. Everything structural is disabled.
COLLIDING = {"floor", "cube_g", "fingerL", "fingerR"}

WORKSPACE = [
    (0.30, 0.00, 0.30),
    (0.25, 0.15, 0.20),
    (0.20, -0.20, 0.15),
    (0.00, 0.34, 0.12),
    (0.35, 0.10, 0.25),
    (0.18, -0.10, 0.35),
]


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(scene.XML)


def test_model_loads(model):
    assert model.nu == 8                       # 6 arm joints + 2 fingers
    assert model.njnt >= len(scene.ARM_JOINTS) + len(scene.GRIP_JOINTS)


def test_structural_links_have_collision_disabled(model):
    """Every geom except the fingers, hand, cube and floor must be non-colliding."""
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        collidable = model.geom_contype[g] or model.geom_conaffinity[g]
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if name in COLLIDING or body == "hand":
            assert collidable, f"{name} should collide but does not"
        else:
            assert not collidable, (
                f"structural geom {name!r} (body {body!r}) has collision enabled; "
                "it will fight the world-welded base and destabilise the yaw joint")


def test_no_contacts_at_rest():
    """d.ncon == 0 with the arm homed and the cube clear of the floor."""
    g = Game(seed=0)
    mujoco.mj_forward(g.m, g.d)
    assert g.d.ncon == 0, f"{g.d.ncon} contacts at rest"


def test_no_arm_contacts_after_settling():
    """Hold the home pose for a second; the arm still touches nothing."""
    g = Game(seed=0)
    for _ in range(100):
        g.step(0, 0, 0, 0, False)
    link_bodies = {"base", "l1", "l2", "l3", "l4", "l5"}
    for i in range(g.d.ncon):
        c = g.d.contact[i]
        for geom in (c.geom1, c.geom2):
            body = mujoco.mj_id2name(g.m, mujoco.mjtObj.mjOBJ_BODY, g.m.geom_bodyid[geom])
            assert body not in link_bodies, f"structural link {body} is colliding"


@pytest.mark.parametrize("goal", WORKSPACE)
def test_ik_converges_within_1mm(model, goal):
    d = mujoco.MjData(model)
    arm = Arm(model, d)
    goal = np.array(goal)
    yaw = math.atan2(goal[1], goal[0])
    q = arm.solve(goal, down_R(yaw), scene.HOME_SEED)
    arm.fk(q)
    err = float(np.linalg.norm(arm.ee() - goal))
    assert err < 1e-3, f"IK error {err * 1000:.2f} mm at {goal}"
    # Strictly inside, not clamped against a stop -- a solution pinned at a
    # limit has no room left to track the operator.
    assert np.all(q > arm.lo + 1e-6) and np.all(q < arm.hi - 1e-6), \
        f"joint jammed at a limit: {q}"


def test_tool_frame_convention(model):
    """down_R matches the model, rather than us trusting the derivation.

    The pitch joints j2/j3/j4/j6 all turn about y, so summing them to pi points
    the tool down and gives R_y(pi) = diag(-1, 1, -1). j1 then yaws that about z.
    """
    d = mujoco.MjData(model)
    arm = Arm(model, d)
    for yaw in (0.0, 0.7, -1.2):
        # j1 = yaw, j5 = 0, pitch joints summing to pi
        arm.fk([yaw, 0.5, 1.3, 1.0, 0.0, math.pi - 2.8])
        assert np.allclose(arm.ee_mat(), down_R(yaw), atol=1e-6), \
            f"tool frame at yaw={yaw} is\n{arm.ee_mat()}\nexpected\n{down_R(yaw)}"


def test_fingers_open_wider_than_cube(model):
    """Gripper joint 0 = fully open, and open must clear the 48 mm cube."""
    gl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gL")
    assert model.jnt_range[gl][0] == 0.0, "joint 0 must be the fully open end"
    assert model.jnt_range[gl][1] > 0.0, "increasing the joint value must close"

    d = mujoco.MjData(model)
    d.qpos[model.jnt_qposadr[gl]] = scene.GRIP_OPEN
    mujoco.mj_forward(model, d)
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingerL")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingerR")
    gap = abs(d.geom_xpos[right][0] - d.geom_xpos[left][0]) - 2 * scene.FINGER_HALF_X
    assert gap == pytest.approx(scene.FINGER_GAP_OPEN, abs=1e-6)
    assert gap > 2 * scene.CUBE_HALF, \
        f"fingers open to {gap * 1000:.0f} mm, cube is {2 * scene.CUBE_HALF * 1000:.0f} mm"
