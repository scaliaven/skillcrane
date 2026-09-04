"""Benchmark environments the Skillcrane teleop rig can drive.

Importing this package is cheap and safe with no benchmarks installed: every
backend is imported lazily inside its factory.

Two nouns, kept apart on purpose: a **suite** is a benchmark (native,
robosuite, RoboCasa, LIBERO, ...) and a **task** is one setting inside it
(Lift, Stack, ...). `cycle_suite` changes the first, `cycle_task` the second,
and an environment spec -- "robosuite:Lift" -- names both.
"""
from .base import Hud, TeleopEnv
from .registry import (NATIVE, SUITES, cycle_suite, cycle_task, describe,
                       installed, make, parse, suites, tasks)

__all__ = ["Hud", "TeleopEnv", "SUITES", "NATIVE", "cycle_suite", "cycle_task",
           "describe", "installed", "make", "parse", "suites", "tasks"]
