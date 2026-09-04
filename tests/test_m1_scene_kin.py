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


# --- the fixed cameras ------------------------------------------------------
# They were moved in close for teleoperation, and "close" has a limit: a camera
# that crops the spawn arc hides the cube the operator is being asked to fetch.
# Checked against the model's own camera matrices rather than by eye.

# Narrower than any panel the layout actually uses (all of them are about 2:1),
# so passing here means passing on screen.
PANEL_ASPECT = 16 / 9

def _sees(model, data, cam: str, point, aspect: float = PANEL_ASPECT) -> bool:
    """Is `point` inside camera `cam`'s frustum? MuJoCo cameras look down -z."""
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    rel = np.asarray(point, float) - data.cam_xpos[i]
    x, y, z = data.cam_xmat[i].reshape(3, 3).T @ rel
    depth = -z
    if depth <= 0:
        return False
    half_y = math.tan(math.radians(model.cam_fovy[i]) / 2)
    return abs(y / depth) <= half_y and abs(x / depth) <= half_y * aspect


def _workspace_points():
    """The corners of what the operator has to be able to see."""
    pts = [(r * math.cos(a), r * math.sin(a), 2 * scene.CUBE_HALF)
           for r in scene.SPAWN_R
           for a in (scene.SPAWN_ANGLE[0], 0.0, scene.SPAWN_ANGLE[1])]
    tx, ty = scene.TARGET_XY                    # the drop zone, rim included
    pts += [(tx + dx * scene.TARGET_R, ty + dy * scene.TARGET_R, 0.0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    return pts


@pytest.mark.parametrize("cam", ["front", "top"])
def test_the_fixed_cameras_see_the_whole_workspace(model, cam):
    """Every spawn position and the whole drop zone, in frame, from both."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    missed = [p for p in _workspace_points() if not _sees(model, d, cam, p)]
    assert not missed, f"{cam} cannot see {np.round(missed, 3).tolist()}"


def test_the_top_camera_is_turned_to_match_the_panel(model):
    """Why it is only 0.82 m up: the arc is wide in y, and so is a view panel.

    Screen-right is -y, so the +/-0.36 m spawn arc runs along the *long* axis of
    the panel. Point it the obvious way instead and the same arc needs the
    camera a third of a metre further back.
    """
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    right = model.cam_mat0[i].reshape(3, 3).T[0]
    assert np.allclose(right, [0, -1, 0], atol=1e-6), \
        "the top view must put world y across the screen, not up it"

    # fovy is the *vertical* angle, so the short axis is what sets the height:
    # turned the obvious way, the arc would have to fit up the screen instead.
    half_y = math.tan(math.radians(model.cam_fovy[i]) / 2)
    widest = scene.SPAWN_R[1] * math.sin(scene.SPAWN_ANGLE[1]) + scene.CUBE_HALF
    height = model.cam_pos[i][2]
    assert widest / half_y > height, \
        (f"turned the other way the arc needs {widest / half_y:.2f} m of height; "
         f"the camera is at {height:.2f} m")

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    assert _sees(model, d, "top", (0.0, widest, 0.0)), "and across, it fits"
