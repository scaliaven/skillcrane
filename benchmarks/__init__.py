"""Benchmark environments the Claw Crew teleop rig can drive.

Importing this package is cheap and safe with no benchmarks installed: every
backend is imported lazily inside its factory.
"""
from .base import Hud, TeleopEnv
from .registry import FAMILIES, NATIVE, describe, installed, make, parse

__all__ = ["Hud", "TeleopEnv", "FAMILIES", "NATIVE", "describe", "installed",
           "make", "parse"]
