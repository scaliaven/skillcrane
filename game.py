"""Simulation and game rules. No UI.

HARD RULE: this module must stay importable and fully runnable with no pygame
and no display. Everything the tests assert on lives here; render.py and
input.py are thin layers on top. tests/test_no_pygame.py enforces it by
importing this module in a subprocess where `import pygame` raises.
"""
import math

import numpy as np
import mujoco

import scene
from kin import Arm, IKController, down_R


class Game:
    """Arm + cube + scoring, advanced one control tick at a time."""

    def __init__(self, seed: int = 0):
        self.m = mujoco.MjModel.from_xml_string(scene.XML)
        self.d = mujoco.MjData(self.m)
        self.arm = Arm(self.m, self.d)
        self.rng = np.random.default_rng(seed)

        self.cube_b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_q = self.m.jnt_qposadr[cj]
        self.cube_v = self.m.jnt_dofadr[cj]

        self.target = scene.TARGET_XY.copy()
        self.reset(full=True)

    # -- state ---------------------------------------------------------------
    def reset(self, full: bool = False) -> None:
        """Re-home the arm and respawn the cube. `full` also resets the round."""
        q0 = self.arm.solve(scene.HOME, down_R(0.0), scene.HOME_SEED)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm.qadr] = q0
        self.d.ctrl[:scene.NARM] = q0
        self.d.ctrl[scene.NARM:] = scene.GRIP_OPEN
        self.ik = IKController(self.m, q0)
        self.tgt = scene.HOME.copy()
        self.yaw = 0.0
        self.closed = False
        self.spawn_cube()
        mujoco.mj_forward(self.m, self.d)
        if full:
            self.score = 0
            self.time_left = scene.ROUND_SECONDS
            self.best_streak = self.streak = 0

    def spawn_cube(self) -> np.ndarray:
        """Drop the cube somewhere reachable but clear of the drop zone."""
        while True:
            a = self.rng.uniform(*scene.SPAWN_ANGLE)
            r = self.rng.uniform(*scene.SPAWN_R)
            p = np.array([r * math.cos(a), r * math.sin(a)])
            if np.linalg.norm(p - self.target) > scene.SPAWN_CLEAR:
                break
        self.d.qpos[self.cube_q:self.cube_q + 3] = [p[0], p[1], 0.03]
        self.d.qpos[self.cube_q + 3:self.cube_q + 7] = [1, 0, 0, 0]
        self.d.qvel[self.cube_v:self.cube_v + 6] = 0
        return p

    def cube_pos(self) -> np.ndarray:
        return self.d.xpos[self.cube_b].copy()

    def held(self) -> bool:
        """Gripper closed and the cube off the floor -- good enough for the HUD."""
        return self.closed and self.cube_pos()[2] > 0.10

    def observation(self) -> np.ndarray:
        """Flat state vector: 6 arm joints, gripper opening, cube xyz."""
        return np.concatenate([self.d.qpos[self.arm.qadr],
                               [self.d.ctrl[scene.NARM]],
                               self.cube_pos()])

    # -- control -------------------------------------------------------------
    def step(self, dx, dy, dz, dyaw, want_closed, dt=scene.CTRL_DT) -> bool:
        """Advance one control tick. Returns True if this tick scored.

        dx/dy/dz/dyaw are deadzoned stick values in [-1, 1], already rotated
        into the world frame by the input layer (game.py knows nothing about
        the camera). The Cartesian target is *rate limited*: it moves by
        stick * speed * dt and can never jump, because a jump makes the IK
        demand a huge joint step and the arm flings whatever it is holding.
        """
        self.tgt[0] += dx * scene.MOVE_SPEED * dt
        self.tgt[1] += dy * scene.MOVE_SPEED * dt
        self.tgt[2] += dz * scene.LIFT_SPEED * dt
        self.yaw = float(np.clip(self.yaw + dyaw * scene.YAW_SPEED * dt,
                                 -scene.YAW_LIMIT, scene.YAW_LIMIT))
        self.clamp_target()

        self.closed = bool(want_closed)
        # Yaw the tool with the reach direction so j5 stays mid-range (see down_R).
        gR = down_R(math.atan2(self.tgt[1], self.tgt[0]) + self.yaw)
        self.d.ctrl[:scene.NARM] = self.ik.update(self.tgt, gR, dt)
        self.d.ctrl[scene.NARM:] = scene.GRIP_SHUT if self.closed else scene.GRIP_OPEN

        for _ in range(scene.SUBSTEPS):
            mujoco.mj_step(self.m, self.d)

        self.time_left = max(0.0, self.time_left - dt)
        return self.check_score()

    def clamp_target(self) -> None:
        """Keep the Cartesian target inside the reachable shell."""
        r = float(np.linalg.norm(self.tgt[:2]))
        if r > 1e-6:
            self.tgt[:2] *= np.clip(r, scene.REACH_MIN, scene.REACH_MAX) / r
        self.tgt[2] = float(np.clip(self.tgt[2], scene.Z_MIN, scene.Z_MAX))

    # -- rules ---------------------------------------------------------------
    def check_score(self) -> bool:
        """Score when the cube is settled, inside the zone, and released."""
        c = self.cube_pos()
        settled = c[2] < 0.05 and np.linalg.norm(self.d.cvel[self.cube_b]) < 0.05
        in_zone = np.linalg.norm(c[:2] - self.target) < scene.TARGET_R
        if settled and in_zone and not self.closed:
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.spawn_cube()
            mujoco.mj_forward(self.m, self.d)
            return True
        return False


def drive_to(game: Game, goal, closed: bool, ticks: int, speed: float = 0.30,
             on_tick=None) -> bool:
    """Scripted rate-limited move of the Cartesian target toward `goal`.

    Shared by the headless demo and the physics tests so both exercise the same
    path the sticks drive. Advances the target by at most `speed * dt` per tick,
    honouring the same no-jump rule as Game.step.
    """
    scored = False
    goal = np.asarray(goal, dtype=float)
    for _ in range(ticks):
        delta = goal - game.tgt
        nrm = float(np.linalg.norm(delta))
        if nrm > 1e-9:
            game.tgt += delta / nrm * min(speed * scene.CTRL_DT, nrm)
            game.clamp_target()
        scored |= bool(game.step(0, 0, 0, 0, closed))
        if on_tick is not None:
            on_tick(game)
    return scored


def scripted_grasp(game: Game, lift_to=(0.30, 0.0, 0.32), on_tick=None) -> None:
    """Approach above the cube -> descend onto it -> close -> lift.

    Split out from the full cycle so the grasp milestone can be asserted on its
    own, before any carrying happens.
    """
    c = game.cube_pos()
    drive_to(game, c + [0, 0, 0.13], False, 250, on_tick=on_tick)   # above it
    drive_to(game, c + [0, 0, 0.012], False, 250, on_tick=on_tick)  # down around it
    for _ in range(80):                       # hold still while the fingers close
        game.step(0, 0, 0, 0, True)
        if on_tick is not None:
            on_tick(game)
    drive_to(game, lift_to, True, 250, on_tick=on_tick)


def scripted_pick_and_place(game: Game, on_tick=None) -> bool:
    """Full grasp -> carry to the drop zone -> release cycle. True if it scored."""
    scripted_grasp(game, on_tick=on_tick)
    drive_to(game, [*game.target, 0.30], True, 400, on_tick=on_tick)
    drive_to(game, [*game.target, 0.09], True, 250, on_tick=on_tick)
    scored = False
    for _ in range(200):                      # release and let it settle
        scored |= bool(game.step(0, 0, 0, 0, False))
        if on_tick is not None:
            on_tick(game)
    return scored
