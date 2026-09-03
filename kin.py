"""Kinematics: forward pose, Jacobian, damped-least-squares IK.

Two solvers live here and they are not interchangeable:

  Arm.solve()      batch IK, iterates to convergence. Used to find a seed pose
                   (e.g. HOME) before the sim starts.
  IKController     per-tick resolved-rate IK for teleop. Integrates on its own
                   scratch MjData -- see the class docstring for why that
                   matters.
"""
import math

import numpy as np
import mujoco

import scene

# DLS damping for the per-tick tracking loop. Lower than Arm.dls's default
# because damping is what makes the solver undershoot the twist it was asked
# for, and that undershoot shows up directly as end-effector lag. The target is
# clamped well inside the arm's stretched-out reach (scene.REACH_MAX), so the
# solver never sits on the singularity this damping exists to survive.
TRACK_LAM = 0.04


def down_R(yaw: float) -> np.ndarray:
    """Tool z pointing down (-world z), rotated by `yaw` about vertical.

    The wrist-neutral tool frame is diag(-1, 1, -1): the pitch joints (j2+j3+j4,
    all about y) sum to pi to point the tool down, and R_y(pi) = diag(-1, 1, -1).
    So tool x lands on -world x, not +x. Composing the base yaw on top gives
    R_z(yaw) @ diag(-1, 1, -1), which is exactly the matrix below.

    Consequence for callers: pass `yaw = atan2(y, x) + user_yaw`. The arm already
    reaches a point at atan2(y, x) by rotating j1 there, so this leaves j5 near
    the middle of its range instead of jammed against a limit.

    tests/test_m1_scene_kin.py checks this against the model rather than
    trusting the derivation.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-c, -s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, -1.0]])


class Arm:
    """6-DoF kinematics bound to a particular (model, data) pair.

    Every method reads whatever is currently in `d`, so the caller decides
    whether that is measured state or a scratch buffer.
    """

    def __init__(self, m, d):
        self.m, self.d = m, d
        jn = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in scene.ARM_JOINTS]
        self.dof = [m.jnt_dofadr[j] for j in jn]
        self.qadr = [m.jnt_qposadr[j] for j in jn]
        self.lo = np.array([m.jnt_range[j][0] for j in jn])
        self.hi = np.array([m.jnt_range[j][1] for j in jn])
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))

    def ee(self) -> np.ndarray:
        """World position of the grip site."""
        return self.d.site_xpos[self.site].copy()

    def ee_mat(self) -> np.ndarray:
        """World rotation of the grip site."""
        return self.d.site_xmat[self.site].reshape(3, 3).copy()

    def pose_error(self, gp, gR) -> np.ndarray:
        """6-vector [translation, rotation] from the current pose to the goal."""
        ep = gp - self.ee()
        Re = gR @ self.d.site_xmat[self.site].reshape(3, 3).T
        q = np.zeros(4)
        aa = np.zeros(3)
        mujoco.mju_mat2Quat(q, Re.flatten())
        mujoco.mju_quat2Vel(aa, q, 1.0)
        return np.concatenate([ep, aa])

    def dls(self, twist, lam: float = 0.10) -> np.ndarray:
        """Damped least squares joint step for a desired end-effector twist.

        Damping keeps the step finite through singularities, which the arm hits
        whenever it is stretched straight out.
        """
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        J = np.vstack([self._jp, self._jr])[:, self.dof]
        return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), twist)

    def fk(self, q) -> None:
        """Write joint angles into `d` and refresh the derived frames."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def solve(self, gp, gR, q0, iters: int = 400) -> np.ndarray:
        """Batch IK. Returns joint angles, always clamped inside the limits."""
        self.d.qpos[self.qadr] = q0
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            e = self.pose_error(gp, gR)
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            dq = self.dls(np.clip(e, -0.3, 0.3), lam=0.06)
            self.d.qpos[self.qadr] = np.clip(self.d.qpos[self.qadr] + dq * 0.6, self.lo, self.hi)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.qpos[self.qadr].copy()


class IKController:
    """Resolved-rate IK integrated on a *virtual* arm.

    The Jacobian and the pose error are evaluated at the **commanded** joint
    configuration, held on a scratch MjData that no physics ever touches --
    never at the measured state.

    Why: the position actuators are already chasing this command. If the
    measured pose error were fed back into the command, the command would chase
    the state that is chasing the command, closing a loop through the actuator
    dynamics. It oscillates hard. The scratch buffer keeps the command an
    open-loop function of the operator's target.
    """

    def __init__(self, m, q_init):
        self.m = m
        self.dk = mujoco.MjData(m)      # scratch: kinematics only, never stepped
        self.arm = Arm(m, self.dk)
        self.q = np.array(q_init, dtype=float)
        self._prev_gp = None

    def update(self, gp, gR, dt: float, kp: float = 6.0) -> np.ndarray:
        """Advance the commanded configuration one control tick toward the goal.

        The linear twist is `kp * error + goal velocity`. Without that
        feedforward term a pure proportional chase trails a moving goal by
        v / kp forever -- 65 mm at a 0.3 m/s sweep, which is a visible lag and
        blows the 10 mm tracking budget. The goal velocity is differenced from
        the goal itself, which is safe because the caller rate-limits it
        (see Game.step); the twist clamp below is the backstop either way.
        """
        self.arm.fk(self.q)
        e = self.arm.pose_error(gp, gR)
        v_ff = np.zeros(3) if self._prev_gp is None else (gp - self._prev_gp) / dt
        self._prev_gp = np.array(gp, dtype=float)
        # Clamp the commanded twist so a far-away goal ramps in instead of
        # demanding an impossible joint rate.
        tw = np.concatenate([np.clip(e[:3] * kp + v_ff, -0.6, 0.6),
                             np.clip(e[3:] * kp, -2.5, 2.5)])
        self.q = np.clip(self.q + self.arm.dls(tw, lam=TRACK_LAM) * dt,
                         self.arm.lo, self.arm.hi)
        return self.q
