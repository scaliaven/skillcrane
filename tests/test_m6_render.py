"""M6 -- the render/HUD layer.

This is the one part that cannot really be tested headless, so it holds no
logic: it draws a `benchmarks.Hud` and blits whatever frame it is handed. These
tests check the offscreen path works end to end and that drawing has no side
effects on the simulation.
"""
import os

import numpy as np
import pytest

import scene

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # must precede set_mode


@pytest.fixture(scope="module")
def display():
    render = pytest.importorskip("render")
    from benchmarks.native import NativeEnv
    env = NativeEnv(seed=3)
    try:
        d = render.Display()
        env.frame(render.RENDER_W, render.RENDER_H)     # needs a GL context
    except Exception as exc:                            # no GL / no video device
        pytest.skip(f"no renderable display here: {exc}")
    yield env, d, render
    d.close()


def test_offscreen_buffer_is_big_enough_for_the_window():
    import render
    assert render.RENDER_W <= scene.OFF_W and render.RENDER_H <= scene.OFF_H, \
        "MuJoCo raises rather than downscaling when the window exceeds offwidth/offheight"


def test_renders_a_frame_of_the_declared_size(display):
    env, d, render = display
    rgb = env.frame(render.RENDER_W, render.RENDER_H)
    assert rgb.shape == (render.RENDER_H, render.RENDER_W, 3)
    assert rgb.dtype == np.uint8
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 50, "frame looks blank"


def test_hud_draws_without_mutating_the_sim(display):
    env, d, render = display
    g = env.game
    before = (g.score, g.time_left, g.closed, g.cube_pos().copy())
    frame = env.frame(render.RENDER_W, render.RENDER_H)
    for i in range(5):
        d.draw(env.hud(), frame, scored=(i == 2))
    assert (g.score, g.time_left, g.closed) == before[:3]
    assert np.array_equal(g.cube_pos(), before[3]), "render mutated the sim"


def test_a_differently_sized_frame_is_scaled_not_rejected(display):
    """Benchmark envs fix their own frame size; the window must still work."""
    _, d, render = display
    small = np.random.default_rng(0).integers(0, 255, (84, 84, 3), dtype=np.uint8)
    from benchmarks import Hud
    d.draw(Hud(task="scaled"), small)
    assert d.screen.get_size() == (render.WINDOW_W, render.WINDOW_H)


def test_a_missing_frame_still_draws_the_hud(display):
    """An env that cannot render offscreen must not take the HUD down with it."""
    _, d, _ = display
    from benchmarks import Hud
    d.draw(Hud(score=3, task="no camera"), None)


def test_orbit_moves_the_native_camera(display):
    env, d, render = display
    env.frame(render.RENDER_W, render.RENDER_H)
    az = env.azimuth
    env.orbit(render.CAM_SPEED / 60)
    assert env.azimuth == pytest.approx(az + render.CAM_SPEED / 60)


def test_render_survives_a_live_step_loop(display):
    env, d, render = display
    for i in range(20):
        env.step(0.4, 0.2, 0.0, 0.0, i > 10, scene.CTRL_DT)
        d.draw(env.hud(), env.frame(render.RENDER_W, render.RENDER_H))
    assert np.isfinite(env.hud().ee).all()
