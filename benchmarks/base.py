"""The contract a teleoperable environment has to satisfy.

Skillcrane's own `Game` already worked this way; this module just names the
shape so benchmark environments can be driven by the same gamepad, the same
HUD and the same recorder.

The contract is deliberately small, and it is the *teleop* contract, not a
general RL one: an operator pushes a Cartesian direction and a gripper button,
and something either scores or does not. Everything else -- action spaces,
observation dicts, reward shaping -- stays behind the adapter.

Nothing here imports pygame. Adapters may import their benchmark lazily so that
`import benchmarks` costs nothing when none are installed.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Hud:
    """Everything the HUD draws. Envs report it; render.py only formats it."""
    score: int = 0
    time_left: float = 0.0
    streak: int = 0
    best_streak: int = 0
    grip: str = "OPEN"              # OPEN | CLOSED | HOLDING
    ee: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obj: np.ndarray = field(default_factory=lambda: np.zeros(3))
    task: str = ""


class TeleopEnv(ABC):
    """A world an operator can drive with four axes and a gripper button."""

    #: Column names for the recorder. Length must match observation()/step input.
    state_names: tuple = ()
    action_names: tuple = ("dx", "dy", "dz", "dyaw", "grip")
    #: Human-readable task string, written into the LeRobot task table.
    task: str = ""
    #: Seconds of simulated time per step() call. Benchmarks run their policies
    #: at 20-25 Hz, not our native 100 Hz, so main.py paces each env by this
    #: rather than assuming a single global control rate.
    control_dt: float = 0.01
    #: Camera azimuth in degrees, used to rotate the sticks into the world
    #: frame. Benchmarks have fixed cameras; only the native env orbits.
    azimuth: float = 0.0

    def orbit(self, degrees: float) -> None:
        """Rotate the camera, where the env has one that can move."""

    @abstractmethod
    def reset(self, full: bool = False) -> None:
        """Re-home. `full` also restarts the round (score, clock, streak)."""

    @abstractmethod
    def step(self, dx, dy, dz, dyaw, want_closed, dt) -> bool:
        """Advance one control tick. Returns True if this tick scored.

        dx/dy/dz/dyaw are deadzoned stick values in [-1, 1] already rotated into
        the world frame by the input layer. Implementations must rate-limit
        whatever target they drive -- never let it jump.
        """

    @abstractmethod
    def observation(self) -> np.ndarray:
        """Flat float state vector, matching `state_names`."""

    @abstractmethod
    def hud(self) -> Hud:
        """Current scoreboard, for the HUD."""

    def frame(self, width: int, height: int):
        """An RGB uint8 frame, or None if this env cannot render offscreen."""
        return None

    def close(self) -> None:
        pass
