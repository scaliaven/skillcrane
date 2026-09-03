"""The hard rule: game.py runs with no pygame and no display.

Checked in a subprocess with an import hook that makes `import pygame` raise,
because once pytest has imported pygame for the input tests a same-process
check would prove nothing.
"""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCK_PYGAME = '''
import sys

class _NoPygame:
    """Make pygame (and anything under it) un-importable."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pygame" or fullname.startswith("pygame."):
            raise ImportError("pygame is blocked for this test")
        return None

sys.meta_path.insert(0, _NoPygame())
for name in [m for m in sys.modules if m == "pygame" or m.startswith("pygame.")]:
    del sys.modules[name]
'''


def run_without_pygame(body):
    script = BLOCK_PYGAME + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=ROOT, SDL_VIDEODRIVER="dummy")
    return subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def test_pygame_really_is_blocked():
    """Guard the guard: the hook must actually break the import."""
    r = run_without_pygame("""
        try:
            import pygame
        except ImportError:
            print("BLOCKED")
    """)
    assert "BLOCKED" in r.stdout, r.stderr


def test_game_imports_and_runs_without_pygame():
    r = run_without_pygame("""
        import sys
        import numpy as np
        from game import Game, scripted_grasp

        g = Game(seed=5)
        assert g.d.ncon == 0
        for _ in range(50):
            g.step(0.5, -0.2, 0.1, 0.0, False)
        scripted_grasp(g)
        assert g.cube_pos()[2] > 0.15, "grasp failed headless"
        assert "pygame" not in sys.modules, "game.py pulled in pygame"
        print("OK cube_z=%.3f score=%d" % (g.cube_pos()[2], g.score))
    """)
    assert r.returncode == 0, f"stdout:\\n{r.stdout}\\nstderr:\\n{r.stderr}"
    assert "OK" in r.stdout, r.stdout


def test_scene_and_kin_are_also_pygame_free():
    r = run_without_pygame("""
        import sys, scene, kin
        assert "pygame" not in sys.modules
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_headless_cli_runs_without_pygame():
    """`python main.py --headless` must not need pygame either."""
    r = run_without_pygame("""
        import sys
        from main import main
        code = main(["--headless", "--seed", "2"])
        assert "pygame" not in sys.modules, "the headless path imported pygame"
        print("EXIT", code)
    """)
    assert r.returncode == 0, f"stdout:\\n{r.stdout}\\nstderr:\\n{r.stderr}"
    assert "EXIT 0" in r.stdout, r.stdout
