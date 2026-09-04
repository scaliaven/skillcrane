"""robosuite environments driven by the Skillcrane teleop loop.

robosuite's BASIC/OSC controller takes exactly the action this project already
produces: a Cartesian delta plus a gripper bit. So the adapter is a remapping,
not a re-implementation --

    robosuite action = [dx, dy, dz, drx, dry, drz, grip]      (7-D)
    Skillcrane stick  = [dx, dy, dz,           dyaw,  grip]

with our yaw driving drz. The controller clamps its own output, which is what
rate-limits the target here (constraint 6) -- we scale the sticks and let OSC
do the limiting rather than integrating a target ourselves.

Gripper sign is +1 = close, verified against the Panda's finger joints rather
than assumed; see benchmarks/registry.py for the per-benchmark table.
"""
import numpy as np

from .base import Hud, TeleopEnv

GRIP_CLOSE = +1.0          # measured: action[-1]=+1 drives the fingers shut
MOVE_GAIN = 0.5            # stick -> normalised OSC delta
ROUND_SECONDS = 90.0

# Extra cameras to offer beside the main one. Which of these exist depends on
# the task XML and the robot, so they are probed at construction rather than
# assumed -- `robot0_eye_in_hand` comes from the robot model, the rest from the
# arena, and a task that lacks one must not leave a dead panel on screen.
CANDIDATE_VIEWS = ("robot0_eye_in_hand", "frontview", "birdview")


class RobosuiteEnv(TeleopEnv):
    action_names = ("dx", "dy", "dz", "dyaw", "grip")

    def __init__(self, task="Lift", robot="Panda", seed=0, control_freq=20,
                 camera="agentview"):
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        self.task_name = task
        self.camera = camera
        self.control_dt = 1.0 / control_freq
        cfg = load_composite_controller_config(controller="BASIC", robot=robot)
        self.env = suite.make(
            env_name=task, robots=robot, controller_configs=cfg,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
            camera_names=[camera, *CANDIDATE_VIEWS],
            control_freq=control_freq, horizon=10 ** 6,
            ignore_done=True, reward_shaping=True,
        )
        self.rng = np.random.default_rng(seed)
        self.task = f"robosuite {task} with a {robot}."
        self._obs = None
        self.closed = False
        self.reset(full=True)
        self.state_names = tuple(f"obs{i:02d}" for i in range(self.observation().size))
        self.view_names = (camera,) + tuple(v for v in CANDIDATE_VIEWS
                                            if self._renderable(v))

    def _renderable(self, view: str) -> bool:
        """Can this model render `view`? Asked once, with a tiny frame."""
        try:
            return self.frame(32, 32, view=view) is not None
        except Exception:
            return False

    # -- state ---------------------------------------------------------------
    def reset(self, full: bool = False) -> None:
        self._obs = self.env.reset()
        self.closed = False
        if full or not hasattr(self, "score"):
            self.score = 0
            self.streak = self.best_streak = 0
            self.time_left = ROUND_SECONDS

    def step(self, dx, dy, dz, dyaw, want_closed, dt=None) -> bool:
        self.closed = bool(want_closed)
        a = np.zeros(self.env.action_dim)
        a[0], a[1], a[2] = dx * MOVE_GAIN, dy * MOVE_GAIN, dz * MOVE_GAIN
        if self.env.action_dim >= 6:
            a[5] = dyaw * MOVE_GAIN            # drz -- wrist yaw
        a[-1] = GRIP_CLOSE if self.closed else -GRIP_CLOSE
        self._obs, _, _, _ = self.env.step(a)

        self.time_left = max(0.0, self.time_left - self.control_dt)
        if self.env._check_success():
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.reset(full=False)             # next attempt, keep the round
            return True
        return False

    def observation(self) -> np.ndarray:
        flat = [np.atleast_1d(np.asarray(v, dtype=np.float32)).ravel()
                for k, v in sorted(self._obs.items()) if not k.endswith("image")]
        return np.concatenate(flat).astype(np.float32)

    def hud(self) -> Hud:
        o = self._obs
        ee = np.asarray(o.get("robot0_eef_pos", np.zeros(3)))
        obj = next((np.asarray(v)[:3] for k, v in o.items() if k.endswith("_pos")
                    and not k.startswith("robot")), np.zeros(3))
        return Hud(score=self.score, time_left=self.time_left, streak=self.streak,
                   best_streak=self.best_streak,
                   grip="CLOSED" if self.closed else "OPEN",
                   ee=ee, obj=obj, task=self.task_name)

    def frame(self, width: int, height: int, view: str = None):
        img = self.env.sim.render(width=width, height=height,
                                  camera_name=view or self.camera)
        return np.flipud(img).copy()          # MuJoCo renders bottom-up

    def frames(self, sizes: dict) -> dict:
        return {v: self.frame(*sizes[v], view=v) for v in sizes}

    def close(self) -> None:
        self.env.close()
