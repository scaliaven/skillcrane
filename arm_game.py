"""
Claw Crew - teleoperate a robot arm in MuJoCo with an 8BitDo gamepad.

macOS notes
-----------
MuJoCo's interactive viewer must own the main thread on macOS (that's why other
projects tell you to run `mjpython`). SDL wants the main thread too. To avoid
the fight entirely, this renders MuJoCo *offscreen* and lets pygame own the
window -- so you run it with plain `python`, and you get a HUD for free.

    pip install mujoco pygame numpy
    python arm_game.py

Controls (defaults; run gamepad_probe.py if your pad differs)
    left stick    move gripper horizontally (camera-relative)
    right stick Y raise / lower
    right stick X rotate wrist
    A             toggle gripper open/closed
    L / R bumper  orbit camera
    Y             reset the round
    keyboard      WASD move, QE up/down, ZC wrist, SPACE grip, R reset
"""
import argparse
import math
import sys

import numpy as np
import mujoco

# ----------------------------------------------------------------------------
# Gamepad mapping. 8BitDo pads enumerate differently per mode (Start+A = Apple,
# Start+B = D-input). If the arm moves on the wrong stick, run gamepad_probe.py
# and edit these indices.
# ----------------------------------------------------------------------------
AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3
BTN_GRIP, BTN_RESET, BTN_CAM_L, BTN_CAM_R = 0, 3, 4, 5
DEADZONE = 0.15

# Tuning
CTRL_DT = 0.01           # 100 Hz control tick
SUBSTEPS = 5             # physics steps per control tick (dt = 0.002)
MOVE_SPEED = 0.45        # m/s at full stick
LIFT_SPEED = 0.35
YAW_SPEED = 2.0          # rad/s
ROUND_SECONDS = 90.0
GRIP_OPEN, GRIP_SHUT = 0.0, 0.016
TARGET_R = 0.07

REACH_MIN, REACH_MAX = 0.16, 0.46
Z_MIN, Z_MAX = 0.035, 0.45

XML = """
<mujoco model="clawcrew">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>

  <default>
    <joint damping="2" armature="0.05"/>
    <geom solref="0.005 1" solimp="0.95 0.99 0.001"/>
    <!-- Structural links carry no collision. MuJoCo skips its parent-child
         contact filter for bodies welded to the world, so the fixed base would
         otherwise grind against link 1 and destabilise the yaw joint. -->
    <default class="link">
      <geom type="capsule" size="0.035" rgba="0.75 0.78 0.82 1" mass="0.8"
            contype="0" conaffinity="0"/>
    </default>
    <default class="finger">
      <geom type="box" friction="2.0 0.1 0.002" condim="4" rgba="0.20 0.22 0.25 1" mass="0.05"/>
    </default>
    <default class="arm_act">
      <position kp="800" kv="60" forcerange="-200 200"/>
    </default>
  </default>

  <!-- MuJoCo's offscreen framebuffer defaults to 640x480; the render call
       fails outright if the requested image is larger, so declare it here. -->
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.10 0.11 0.14"
             rgb2="0.02 0.02 0.03" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.17 0.20"
             rgb2="0.20 0.21 0.24" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="0.6 -0.4 1.6" dir="-0.3 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="-0.6 0.6 1.2" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.40"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="grid" friction="1 0.05 0.001"/>

    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.09 0.05" rgba="0.30 0.32 0.38 1" mass="5"
            contype="0" conaffinity="0"/>
      <body name="l1" pos="0 0 0.05">
        <joint name="j1" axis="0 0 1" range="-3.0 3.0"/>
        <geom class="link" fromto="0 0 0 0 0 0.09"/>
        <body name="l2" pos="0 0 0.09">
          <joint name="j2" axis="0 1 0" range="-2.0 2.0"/>
          <geom class="link" fromto="0 0 0 0 0 0.26"/>
          <body name="l3" pos="0 0 0.26">
            <joint name="j3" axis="0 1 0" range="-2.6 2.6"/>
            <geom class="link" size="0.030" fromto="0 0 0 0 0 0.23"/>
            <body name="l4" pos="0 0 0.23">
              <joint name="j4" axis="0 1 0" range="-2.6 2.6"/>
              <geom class="link" size="0.026" fromto="0 0 0 0 0 0.09"/>
              <body name="l5" pos="0 0 0.09">
                <joint name="j5" axis="0 0 1" range="-3.0 3.0"/>
                <geom class="link" size="0.026" fromto="0 0 0 0 0 0.04"/>
                <body name="hand" pos="0 0 0.04">
                  <joint name="j6" axis="0 1 0" range="-2.0 2.0"/>
                  <geom type="box" size="0.045 0.022 0.020" pos="0 0 0.020"
                        rgba="0.35 0.38 0.44 1" mass="0.3"/>
                  <site name="grip" pos="0 0 0.070" size="0.008" rgba="1 0.4 0.2 1"/>
                  <!-- joint value 0 = fully OPEN; increasing closes -->
                  <body name="fL" pos="0 0 0.040">
                    <joint name="gL" type="slide" axis="1 0 0" range="0 0.032"
                           damping="8" armature="0.01"/>
                    <geom class="finger" size="0.008 0.018 0.030" pos="-0.042 0 0.030"/>
                  </body>
                  <body name="fR" pos="0 0 0.040">
                    <joint name="gR" type="slide" axis="-1 0 0" range="0 0.032"
                           damping="8" armature="0.01"/>
                    <geom class="finger" size="0.008 0.018 0.030" pos="0.042 0 0.030"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="cube" pos="0.34 0.0 0.03">
      <freejoint name="cube_free"/>
      <geom name="cube_g" type="box" size="0.024 0.024 0.024" rgba="0.95 0.55 0.20 1"
            friction="2.0 0.1 0.002" condim="4" mass="0.06"/>
    </body>

    <site name="target" pos="0.0 0.34 0.002" size="0.07 0.001" type="cylinder"
          rgba="0.30 0.85 0.55 0.35"/>
  </worldbody>

  <actuator>
    <position class="arm_act" name="a1" joint="j1"/>
    <position class="arm_act" name="a2" joint="j2"/>
    <position class="arm_act" name="a3" joint="j3"/>
    <position class="arm_act" name="a4" joint="j4"/>
    <position class="arm_act" name="a5" joint="j5"/>
    <position class="arm_act" name="a6" joint="j6"/>
    <position name="gripL" joint="gL" kp="900" kv="25" forcerange="-120 120" ctrlrange="0 0.032"/>
    <position name="gripR" joint="gR" kp="900" kv="25" forcerange="-120 120" ctrlrange="0 0.032"/>
  </actuator>
</mujoco>
"""


def down_R(yaw):
    """Tool z-axis pointing down, rotated by `yaw` about vertical.

    Matches the arm's wrist-neutral frame (tool x = -world x at yaw 0), so
    passing yaw = atan2(y, x) + user_yaw keeps j5 near the middle of its range
    instead of jammed against a limit.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-c, -s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, -1.0]])


class Arm:
    """6-DoF kinematics on a given MjData."""

    def __init__(self, m, d):
        self.m, self.d = m, d
        jn = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}") for i in range(1, 7)]
        self.dof = [m.jnt_dofadr[j] for j in jn]
        self.qadr = [m.jnt_qposadr[j] for j in jn]
        self.lo = np.array([m.jnt_range[j][0] for j in jn])
        self.hi = np.array([m.jnt_range[j][1] for j in jn])
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))

    def ee(self):
        return self.d.site_xpos[self.site].copy()

    def pose_error(self, gp, gR):
        ep = gp - self.ee()
        Re = gR @ self.d.site_xmat[self.site].reshape(3, 3).T
        q = np.zeros(4)
        aa = np.zeros(3)
        mujoco.mju_mat2Quat(q, Re.flatten())
        mujoco.mju_quat2Vel(aa, q, 1.0)
        return np.concatenate([ep, aa])

    def dls(self, twist, lam=0.10):
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        J = np.vstack([self._jp, self._jr])[:, self.dof]
        return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), twist)

    def solve(self, gp, gR, q0, iters=400):
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

    The Jacobian and error are evaluated at the commanded configuration on a
    scratch MjData, never at the measured state. Feeding measured error back
    into the command while position actuators chase that same command closes a
    loop through the actuator dynamics and oscillates hard.
    """

    def __init__(self, m, q_init):
        self.m = m
        self.dk = mujoco.MjData(m)
        self.arm = Arm(m, self.dk)
        self.q = np.array(q_init, dtype=float)

    def update(self, gp, gR, dt, kp=6.0):
        self.dk.qpos[self.arm.qadr] = self.q
        mujoco.mj_kinematics(self.m, self.dk)
        mujoco.mj_comPos(self.m, self.dk)
        e = self.arm.pose_error(gp, gR)
        tw = np.concatenate([np.clip(e[:3] * kp, -0.6, 0.6),
                             np.clip(e[3:] * kp, -2.5, 2.5)])
        self.q = np.clip(self.q + self.arm.dls(tw) * dt, self.arm.lo, self.arm.hi)
        return self.q


class Game:
    """Sim + rules. No pygame in here, so it can run headless in tests."""

    def __init__(self, seed=0):
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        self.arm = Arm(self.m, self.d)
        self.rng = np.random.default_rng(seed)
        self.cube_b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        _cj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_q = self.m.jnt_qposadr[_cj]
        self.cube_v = self.m.jnt_dofadr[_cj]
        self.target = np.array([0.0, 0.34])
        self.home = np.array([0.30, 0.0, 0.30])
        self.reset(full=True)

    # -- state ------------------------------------------------------------
    def reset(self, full=False):
        q0 = self.arm.solve(self.home, down_R(0.0), np.array([0, 0.5, 1.3, 1.34, 0, 0]))
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm.qadr] = q0
        self.d.ctrl[:6] = q0
        self.d.ctrl[6:8] = GRIP_OPEN
        self.ik = IKController(self.m, q0)
        self.tgt = self.home.copy()
        self.yaw = 0.0
        self.closed = False
        self.spawn_cube()
        mujoco.mj_forward(self.m, self.d)
        if full:
            self.score = 0
            self.time_left = ROUND_SECONDS
            self.best_streak = self.streak = 0

    def spawn_cube(self):
        while True:
            a = self.rng.uniform(-1.1, 1.1)
            r = self.rng.uniform(0.26, 0.40)
            p = np.array([r * math.cos(a), r * math.sin(a)])
            if np.linalg.norm(p - self.target) > 0.16:
                break
        self.d.qpos[self.cube_q:self.cube_q + 3] = [p[0], p[1], 0.03]
        self.d.qpos[self.cube_q + 3:self.cube_q + 7] = [1, 0, 0, 0]
        self.d.qvel[self.cube_v:self.cube_v + 6] = 0

    def cube_pos(self):
        return self.d.xpos[self.cube_b].copy()

    def held(self):
        return self.closed and self.cube_pos()[2] > 0.10

    # -- control ----------------------------------------------------------
    def step(self, mx, my, mz, dyaw, want_closed, cam_az, dt=CTRL_DT):
        """mx/my/mz/dyaw are already deadzoned in [-1, 1]. cam_az in radians."""
        fwd = np.array([-math.cos(cam_az), -math.sin(cam_az)])
        rgt = np.array([-math.sin(cam_az), math.cos(cam_az)])
        step_xy = (my * fwd + mx * rgt) * MOVE_SPEED * dt
        self.tgt[:2] += step_xy
        self.tgt[2] += mz * LIFT_SPEED * dt
        self.yaw += dyaw * YAW_SPEED * dt
        self.yaw = float(np.clip(self.yaw, -2.2, 2.2))

        r = np.linalg.norm(self.tgt[:2])
        if r > 1e-6:
            self.tgt[:2] *= np.clip(r, REACH_MIN, REACH_MAX) / r
        self.tgt[2] = float(np.clip(self.tgt[2], Z_MIN, Z_MAX))

        self.closed = want_closed
        gR = down_R(math.atan2(self.tgt[1], self.tgt[0]) + self.yaw)
        self.d.ctrl[:6] = self.ik.update(self.tgt, gR, dt)
        self.d.ctrl[6:8] = GRIP_SHUT if want_closed else GRIP_OPEN
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)

        self.time_left = max(0.0, self.time_left - dt)
        return self.check_score()

    def check_score(self):
        c = self.cube_pos()
        settled = c[2] < 0.05 and np.linalg.norm(self.d.cvel[self.cube_b]) < 0.05
        if settled and np.linalg.norm(c[:2] - self.target) < TARGET_R and not self.closed:
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.spawn_cube()
            mujoco.mj_forward(self.m, self.d)
            return True
        return False


def dz(v, d=DEADZONE):
    return 0.0 if abs(v) < d else (abs(v) - d) / (1 - d) * (1 if v > 0 else -1)


def run_headless(ticks=1500):
    """Smoke test: drive the arm through a scripted pick-and-place."""
    g = Game(seed=2)
    c = g.cube_pos()
    print("cube spawn", np.round(c, 3))

    def drive(goal, closed, n, speed=0.30):
        for _ in range(n):
            delta = goal - g.tgt
            nrm = np.linalg.norm(delta)
            unit = delta / nrm if nrm > 1e-9 else np.zeros(3)
            g.step(0, 0, 0, 0, closed, 0.0)
            g.tgt += unit * min(speed * CTRL_DT, nrm)

    drive(c + [0, 0, 0.13], False, 250)
    drive(c + [0, 0, 0.012], False, 250)
    print("descend ee", np.round(g.arm.ee(), 3))
    for _ in range(80):
        g.step(0, 0, 0, 0, True, 0.0)
    drive(np.array([0.30, 0.0, 0.32]), True, 250)
    print("lift cube_z", round(float(g.cube_pos()[2]), 3), "held:", g.held())
    drive(np.array([g.target[0], g.target[1], 0.30]), True, 400)
    drive(np.array([g.target[0], g.target[1], 0.09]), True, 250)
    print("transit |qvel|", round(float(np.linalg.norm(g.d.qvel[g.arm.dof])), 3))
    scored = False
    for _ in range(200):
        scored |= bool(g.step(0, 0, 0, 0, False, 0.0))
    cc = g.cube_pos()
    print("final cube", np.round(cc, 3), "score", g.score, "->",
          "SCORED" if scored or g.score else "MISS")
    return 0 if g.score else 1


def run_game():
    import pygame

    W, H = 900, 620
    pygame.init()
    pygame.display.set_caption("Claw Crew")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 30, bold=True)
    f_sm = pygame.font.SysFont("menlo,dejavusansmono,monospace", 17)

    pygame.joystick.init()
    pad = None
    if pygame.joystick.get_count():
        pad = pygame.joystick.Joystick(0)
        pad.init()
        print(f"gamepad: {pad.get_name()}  axes={pad.get_numaxes()} buttons={pad.get_numbuttons()}")
    else:
        print("no gamepad found - keyboard only (WASD/QE/ZC/SPACE)")

    g = Game(seed=np.random.randint(1 << 30))
    ren = mujoco.Renderer(g.m, height=460, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.10, 0.10, 0.15]
    cam.distance, cam.elevation, cam.azimuth = 1.35, -22.0, 130.0

    grip_latch = False
    flash = 0.0
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                g.reset(full=True)

        mx = my = mz = dyaw = 0.0
        want = g.closed
        k = pygame.key.get_pressed()
        mx += (k[pygame.K_d] - k[pygame.K_a])
        my += (k[pygame.K_w] - k[pygame.K_s])
        mz += (k[pygame.K_e] - k[pygame.K_q])
        dyaw += (k[pygame.K_c] - k[pygame.K_z])

        if pad:
            na, nb = pad.get_numaxes(), pad.get_numbuttons()
            ax = lambda i: pad.get_axis(i) if i < na else 0.0
            bt = lambda i: bool(pad.get_button(i)) if i < nb else False
            mx += dz(ax(AX_LX))
            my += -dz(ax(AX_LY))
            mz += -dz(ax(AX_RY))
            dyaw += dz(ax(AX_RX))
            if bt(BTN_GRIP) and not grip_latch:
                want = not g.closed
            grip_latch = bt(BTN_GRIP)
            if bt(BTN_RESET):
                g.reset(full=True)
            cam.azimuth += (bt(BTN_CAM_R) - bt(BTN_CAM_L)) * 90.0 * CTRL_DT

        # SPACE is hold-to-close and always wins over the gamepad toggle.
        if k[pygame.K_SPACE]:
            want = True
        elif not pad:
            want = False

        az = math.radians(cam.azimuth)
        scored = False
        for _ in range(2):  # 2 control ticks per 60 Hz frame
            scored |= bool(g.step(np.clip(mx, -1, 1), np.clip(my, -1, 1),
                                  np.clip(mz, -1, 1), np.clip(dyaw, -1, 1),
                                  want, az))
        if scored:
            flash = 0.7

        ren.update_scene(g.d, cam)
        frame = ren.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.fill((14, 15, 18))
        screen.blit(surf, (0, 0))

        pygame.draw.rect(screen, (22, 24, 28), (0, 460, W, H - 460))
        acc = (250, 190, 60) if flash <= 0 else (90, 230, 140)
        screen.blit(f_big.render(f"SCORE {g.score}", True, acc), (24, 480))
        screen.blit(f_big.render(f"{g.time_left:5.1f}s", True, (235, 235, 240)), (250, 480))
        screen.blit(f_sm.render(f"streak {g.streak}  best {g.best_streak}", True, (150, 155, 165)),
                    (420, 488))
        state = "HOLDING" if g.held() else ("CLOSED" if g.closed else "OPEN")
        screen.blit(f_sm.render(f"gripper {state}", True, (150, 155, 165)), (420, 510))
        screen.blit(f_sm.render(
            "L-stick move  R-stick lift/rotate  A grip  LB/RB camera  Y reset",
            True, (110, 115, 125)), (24, 545))
        screen.blit(f_sm.render(
            f"ee {np.round(g.arm.ee(), 2)}   cube {np.round(g.cube_pos(), 2)}",
            True, (90, 95, 105)), (24, 572))
        if g.time_left <= 0:
            screen.blit(f_big.render("TIME  -  press R", True, (240, 120, 110)), (620, 480))

        flash = max(0.0, flash - 1 / 60)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run a scripted smoke test")
    a = ap.parse_args()
    sys.exit(run_headless() if a.headless else run_game())
