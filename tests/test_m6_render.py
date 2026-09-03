"""M6 -- the render/HUD layer.

This is the one part that cannot really be tested headless, so it holds no
logic and this test only checks that the offscreen path works end to end: the
window opens on SDL's dummy driver, MuJoCo renders into the declared offscreen
buffer, and the HUD draws without touching game state.
"""
import os

import numpy as np
import pytest

import scene
from game import Game

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # must precede set_mode


@pytest.fixture(scope="module")
def display():
    render = pytest.importorskip("render")
    g = Game(seed=3)
    try:
        d = render.Display(g.m)
    except Exception as exc:                        # no GL / no video device
        pytest.skip(f"no renderable display here: {exc}")
    yield g, d, render
    d.close()


def test_offscreen_buffer_is_big_enough_for_the_window():
    import render
    assert render.RENDER_W <= scene.OFF_W and render.RENDER_H <= scene.OFF_H, \
        "MuJoCo raises rather than downscaling when the window exceeds offwidth/offheight"


def test_renders_a_frame_of_the_declared_size(display):
    g, d, render = display
    rgb = d.frame(g.d)
    assert rgb.shape == (render.RENDER_H, render.RENDER_W, 3)
    assert rgb.dtype == np.uint8
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 50, "frame looks blank"


def test_hud_draws_without_mutating_the_game(display):
    g, d, _ = display
    before = (g.score, g.time_left, g.closed, g.cube_pos().copy())
    for i in range(5):
        d.draw(g, scored=(i == 2))
    assert (g.score, g.time_left, g.closed) == before[:3]
    assert np.array_equal(g.cube_pos(), before[3]), "render mutated the sim"


def test_orbit_moves_the_camera(display):
    _, d, render = display
    az = d.cam.azimuth
    d.orbit(1.0, 1 / 60)
    assert d.cam.azimuth == pytest.approx(az + render.CAM_SPEED / 60)
    d.orbit(-1.0, 1 / 60)
    assert d.cam.azimuth == pytest.approx(az)


def test_render_survives_a_live_step_loop(display):
    g, d, _ = display
    for i in range(20):
        g.step(0.4, 0.2, 0.0, 0.0, i > 10)
        d.draw(g)
    assert np.isfinite(g.arm.ee()).all()
