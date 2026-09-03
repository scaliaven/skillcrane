"""LIBERO tasks driven by the Claw Crew teleop loop.

LIBERO is a robosuite environment underneath, so the action is the same 7-D
Cartesian delta the robosuite adapter uses. What is different is packaging:
LIBERO pins robosuite==1.4.0, whose controller API is not the 1.5 one, so this
backend only works inside the dedicated LIBERO environment
(`environment-libero.yml`) -- never alongside benchmarks/robosuite_env.py.

Two setup gotchas, both verified on macOS/Apple Silicon:

  * robosuite's mujoco-py compatibility shim breaks on mujoco >= 3.12, where
    `mjtJoint.mjJNT_HINGE == np.int32(3)` became False in the reflected
    direction, so its `joint_type in (HINGE, SLIDE)` check fails. Pin
    mujoco<3.12.
  * `import libero` prompts on first run and writes ~/.libero/config.yaml.
    Answer it once (see BENCHMARKS.md) or this raises with instructions.
"""
import os

import numpy as np

from .base import Hud, TeleopEnv

GRIP_CLOSE = +1.0          # robosuite Panda convention, same as 1.5
MOVE_GAIN = 0.5
ROUND_SECONDS = 90.0
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_90",
          "libero_10", "libero_100")


class LiberoEnv(TeleopEnv):
    action_names = ("dx", "dy", "dz", "dyaw", "grip")

    def __init__(self, task="libero_spatial", seed=0, task_index=0, image=84):
        cfg = os.path.expanduser("~/.libero/config.yaml")
        if not os.path.exists(cfg):
            raise SystemExit(
                "LIBERO is not configured yet -- its first import asks where to keep\n"
                "datasets and writes ~/.libero/config.yaml. Run this once:\n"
                "    printf 'N\\n' | python -c 'import libero.libero'")

        suite_name, _, idx = str(task or "libero_spatial").partition("/")
        if idx:
            task_index = int(idx)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[suite_name]()
        spec = suite.get_task(task_index)
        bddl = os.path.join(get_libero_path("bddl_files"), spec.problem_folder,
                            spec.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=image,
                                      camera_widths=image)
        self.env.seed(seed)
        self.control_dt = 1.0 / 20.0
        self.task = spec.language
        self.task_name = f"{suite_name}/{task_index}"
        self.closed = False
        self._obs = None
        self.reset(full=True)
        self.state_names = tuple(f"obs{i:02d}" for i in range(self.observation().size))

    def reset(self, full: bool = False) -> None:
        self._obs = self.env.reset()
        self.closed = False
        if full or not hasattr(self, "score"):
            self.score = 0
            self.streak = self.best_streak = 0
            self.time_left = ROUND_SECONDS

    def step(self, dx, dy, dz, dyaw, want_closed, dt=None) -> bool:
        self.closed = bool(want_closed)
        a = np.zeros(7, dtype=np.float32)
        a[0], a[1], a[2] = dx * MOVE_GAIN, dy * MOVE_GAIN, dz * MOVE_GAIN
        a[5] = dyaw * MOVE_GAIN
        a[6] = GRIP_CLOSE if self.closed else -GRIP_CLOSE
        self._obs, _, done, _ = self.env.step(a)

        self.time_left = max(0.0, self.time_left - self.control_dt)
        if done:
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.reset(full=False)
            return True
        return False

    def observation(self) -> np.ndarray:
        flat = [np.atleast_1d(np.asarray(v, dtype=np.float32)).ravel()
                for k, v in sorted(self._obs.items()) if not k.endswith("image")]
        return np.concatenate(flat).astype(np.float32)

    def hud(self) -> Hud:
        o = self._obs
        return Hud(score=self.score, time_left=self.time_left, streak=self.streak,
                   best_streak=self.best_streak,
                   grip="CLOSED" if self.closed else "OPEN",
                   ee=np.asarray(o.get("robot0_eef_pos", np.zeros(3))),
                   obj=np.zeros(3), task=self.task_name)

    def frame(self, width: int, height: int):
        img = self._obs.get("agentview_image")
        return None if img is None else np.flipud(np.asarray(img, np.uint8)).copy()

    def close(self) -> None:
        self.env.close()
