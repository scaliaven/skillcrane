"""MJCF for the Skillcrane arm, plus the model constants everything else reads.

Nothing in here imports pygame or touches a display: the scene is pure model
data so that `game.py` and the tests can use it headless.
"""
import numpy as np

# --- Offscreen framebuffer ---------------------------------------------------
# MuJoCo's offscreen buffer defaults to 640x480 and *raises* if you ask
# mujoco.Renderer for anything larger -- it does not silently downscale. So the
# size is declared in the MJCF and render.py asserts its window fits inside it.
OFF_W, OFF_H = 1280, 720

# --- Physics / control rates -------------------------------------------------
TIMESTEP = 0.002          # MuJoCo integrator step
SUBSTEPS = 5              # physics steps per control tick
CTRL_DT = TIMESTEP * SUBSTEPS   # 0.01 s -> 100 Hz control

# --- Teleop tuning -----------------------------------------------------------
MOVE_SPEED = 0.45         # m/s of Cartesian target travel at full stick
LIFT_SPEED = 0.35         # m/s
YAW_SPEED = 2.0           # rad/s
YAW_LIMIT = 2.2           # rad, user wrist offset clamp

# --- Gripper -----------------------------------------------------------------
# Joint value 0 = fully OPEN; increasing closes. Both finger joints slide along
# opposed axes, so one ctrl value drives both symmetrically.
GRIP_OPEN, GRIP_SHUT = 0.0, 0.016
GRIP_RANGE = 0.032

# --- Objects and workspace ---------------------------------------------------
CUBE_HALF = 0.024                       # cube is 48 mm across
FINGER_HALF_X = 0.008                   # finger box half-thickness in x
FINGER_OFFSET_X = 0.042                 # finger centre offset when open
# Inner faces sit 68 mm apart when open -- wider than the 48 mm cube, or the
# gripper can never be lowered around it.
FINGER_GAP_OPEN = 2 * (FINGER_OFFSET_X - FINGER_HALF_X)

TARGET_XY = np.array([0.0, 0.34])       # drop zone centre
TARGET_R = 0.07                         # drop zone radius
SPAWN_R = (0.26, 0.40)                  # cube spawn annulus
SPAWN_ANGLE = (-1.1, 1.1)               # cube spawn arc, radians
SPAWN_CLEAR = 0.16                      # min distance from spawn to drop zone

REACH_MIN, REACH_MAX = 0.16, 0.46       # planar reach clamp on the target
Z_MIN, Z_MAX = 0.035, 0.45              # vertical clamp on the target
HOME = np.array([0.30, 0.0, 0.30])      # rest pose of the Cartesian target
HOME_SEED = np.array([0.0, 0.5, 1.3, 1.34, 0.0, 0.0])   # IK seed for HOME

ROUND_SECONDS = 90.0

# --- Cameras -----------------------------------------------------------------
# Fixed cameras declared in the MJCF, in the order the multi-view layouts show
# them. The orbiting free camera is not one of these -- it lives in
# benchmarks/native.py, because it is a view-state thing, not model data.
CAMERAS = ("wrist", "front", "top")

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIP_JOINTS = ("gL", "gR")
NARM = len(ARM_JOINTS)


def build_xml(offwidth: int = OFF_W, offheight: int = OFF_H) -> str:
    """MJCF as a string. Offscreen buffer size is baked in (see OFF_W above)."""
    return f"""
<mujoco model="skillcrane">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic"/>

  <default>
    <joint damping="2" armature="0.05"/>
    <geom solref="0.005 1" solimp="0.95 0.99 0.001"/>
    <!-- Structural links carry NO collision (contype/conaffinity 0). MuJoCo
         skips its parent-child contact filter for bodies welded to the world,
         and a fixed base with no joint *is* world-welded -- so the base geom
         would collide with link 1 and destabilise the yaw joint (base joint
         oscillating >15 rad/s, never tracking its command). Collision lives
         only on the fingers, hand, cube and floor. tests/test_m1 asserts
         ncon == 0 at rest; do not "tidy up" these flags. -->
    <default class="link">
      <geom type="capsule" size="0.035" rgba="0.75 0.78 0.82 1" mass="0.8"
            contype="0" conaffinity="0"/>
    </default>
    <default class="finger">
      <geom type="box" friction="2.0 0.1 0.002" condim="4" rgba="0.20 0.22 0.25 1" mass="0.05"/>
    </default>
    <!-- Position actuators. A position servo trails a moving command by
         roughly kv*qdot/kp, and that following error is the biggest single
         term in end-effector tracking: at kp=800/kv=60 a 0.3 m/s sweep lags
         by ~21 mm, well over the 10 mm budget. These gains cut it to ~4 mm
         and peak at 13 N of the 200 N forcerange, so there is no saturation.
         tests/test_m2_tracking.py pins both the tracking and the stability. -->
    <default class="arm_act">
      <position kp="2500" kv="40" forcerange="-200 200"/>
    </default>
  </default>

  <visual>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
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
                  <!-- TCP marker, in site group 3. MuJoCo hides groups 3+ by
                       default, so it appears only where a renderer opts in:
                       the operator's orbit view does, the eye-in-hand camera
                       does not -- at 5 cm this dot covers exactly the thing
                       the gripper is reaching for. The drop-zone site stays in
                       group 0 and is visible everywhere. -->
                  <site name="grip" pos="0 0 0.070" size="0.008" rgba="1 0.4 0.2 1"
                        group="3"/>
                  <!-- Eye-in-hand view. A MuJoCo camera looks down its own -z
                       with +y up, so xyaxes aims this one from beside the
                       wrist at the grip site: right = hand +x (the finger
                       separation axis), up tilted to match. -->
                  <camera name="wrist" pos="0 -0.10 0.015" fovy="70"
                          xyaxes="1 0 0  0 -0.65 0.76"/>
                  <!-- joint value 0 = fully OPEN; increasing closes -->
                  <body name="fL" pos="0 0 0.040">
                    <joint name="gL" type="slide" axis="1 0 0" range="0 {GRIP_RANGE}"
                           damping="8" armature="0.01"/>
                    <geom class="finger" name="fingerL" size="{FINGER_HALF_X} 0.018 0.030"
                          pos="-{FINGER_OFFSET_X} 0 0.030"/>
                  </body>
                  <body name="fR" pos="0 0 0.040">
                    <joint name="gR" type="slide" axis="-1 0 0" range="0 {GRIP_RANGE}"
                           damping="8" armature="0.01"/>
                    <geom class="finger" name="fingerR" size="{FINGER_HALF_X} 0.018 0.030"
                          pos="{FINGER_OFFSET_X} 0 0.030"/>
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
      <geom name="cube_g" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            rgba="0.95 0.55 0.20 1" friction="2.0 0.1 0.002" condim="4" mass="0.06"/>
    </body>

    <site name="target" pos="{TARGET_XY[0]} {TARGET_XY[1]} 0.002" size="{TARGET_R} 0.001"
          type="cylinder" rgba="0.30 0.85 0.55 0.35"/>

    <!-- Named here rather than built in code, so every renderer that loads this
         MJCF -- ours and anyone else's -- sees the same views.

         Both are framed on the *workspace*, not on the room: aimed at
         (0.13, 0.04, 0.07), roughly the middle of the spawn arc, and no further
         back than it takes to keep the whole arc in frame. The old framing sat
         a third of a metre further out and pointed at the horizon, which spent
         half of every panel on empty floor. test_m1 checks the arc still fits.
    -->
    <camera name="front" pos="0.86 -0.14 0.46" fovy="45"
            xyaxes="0.239 0.971 0  -0.447 0.110 0.888"/>
    <!-- The top view is turned 90 degrees on purpose: the workspace is wide in
         y (the spawn arc spans +/-0.36 m) and shallow in x, and a view panel is
         wide too, so putting y across the screen fits the arc in the long axis.
         That alone buys the height: 0.82 m instead of the 1.15 m the same arc
         needs when y runs up the short axis. Screen up is +x (away from the
         operator), screen right is -y, i.e. the view from behind the arm. -->
    <camera name="top" pos="0.17 0 0.82" fovy="45" xyaxes="0 -1 0  1 0 0"/>
  </worldbody>

  <actuator>
    <position class="arm_act" name="a1" joint="j1"/>
    <position class="arm_act" name="a2" joint="j2"/>
    <position class="arm_act" name="a3" joint="j3"/>
    <position class="arm_act" name="a4" joint="j4"/>
    <position class="arm_act" name="a5" joint="j5"/>
    <position class="arm_act" name="a6" joint="j6"/>
    <position name="gripL" joint="gL" kp="900" kv="25" forcerange="-120 120"
              ctrlrange="0 {GRIP_RANGE}"/>
    <position name="gripR" joint="gR" kp="900" kv="25" forcerange="-120 120"
              ctrlrange="0 {GRIP_RANGE}"/>
  </actuator>
</mujoco>
"""


XML = build_xml()
