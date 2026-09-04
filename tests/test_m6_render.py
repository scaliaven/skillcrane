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
        env.frame(*d.viewport)                          # needs a GL context
    except Exception as exc:                            # no GL / no video device
        pytest.skip(f"no renderable display here: {exc}")
    yield env, d, render
    d.close()


def test_offscreen_buffer_is_big_enough_for_the_default_window():
    import render
    assert render.WINDOW_W <= scene.OFF_W and render.WINDOW_H <= scene.OFF_H, \
        "MuJoCo raises rather than downscaling when the window exceeds offwidth/offheight"


def test_renders_a_frame_of_the_declared_size(display):
    env, d, render = display
    rgb = env.frame(*d.viewport)
    assert rgb.shape == (d.render_h, d.render_w, 3)
    assert rgb.dtype == np.uint8
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 50, "frame looks blank"


def test_hud_draws_without_mutating_the_sim(display):
    env, d, render = display
    g = env.game
    before = (g.score, g.time_left, g.closed, g.cube_pos().copy())
    frame = env.frame(*d.viewport)
    for i in range(5):
        d.draw(env.hud(), frame, scored=(i == 2))
    assert (g.score, g.time_left, g.closed) == before[:3]
    assert np.array_equal(g.cube_pos(), before[3]), "render mutated the sim"


def test_a_differently_sized_frame_is_scaled_not_rejected(display):
    """Benchmark envs fix their own frame size; the window must still work."""
    _, d, render = display
    small = np.random.default_rng(0).integers(0, 255, (84, 84, 3), dtype=np.uint8)
    from benchmarks import Hud
    size = d.screen.get_size()
    d.draw(Hud(task="scaled"), small)
    assert d.screen.get_size() == size


def test_a_missing_frame_still_draws_the_hud(display):
    """An env that cannot render offscreen must not take the HUD down with it."""
    _, d, _ = display
    from benchmarks import Hud
    d.draw(Hud(score=3, task="no camera"), None)


def test_orbit_moves_the_native_camera(display):
    env, d, render = display
    env.frame(*d.viewport)
    az = env.azimuth
    env.orbit(render.CAM_SPEED / 60)
    assert env.azimuth == pytest.approx(az + render.CAM_SPEED / 60)


# --- multi-view layouts -----------------------------------------------------

# Sizes the window can actually be: the minimum, the default, and a big one.
VIEWPORTS = [(720, 360), (1280, 638), (2400, 1200)]


@pytest.mark.parametrize("size", VIEWPORTS)
def test_panels_fit_inside_the_viewport(size):
    import render
    for layout in render.LAYOUTS:
        for n in range(1, 6):
            rects = render.panels(layout, n, size)
            assert 1 <= len(rects) <= max(1, min(n, 4))
            for x, y, w, h in rects:
                assert w > 0 and h > 0
                assert 0 <= x and x + w <= size[0]
                assert 0 <= y and y + h <= size[1]


@pytest.mark.parametrize("size", VIEWPORTS)
def test_one_camera_always_gets_the_whole_viewport(size):
    """Cycling layouts on a single-view benchmark must not shrink its frame."""
    import render
    for layout in render.LAYOUTS:
        assert render.panels(layout, 1, size) == [(0, 0, size[0], size[1])]


def test_grid_panels_do_not_overlap():
    import render
    rects = render.panels("grid", 4, (1280, 638))
    for i, (x, y, w, h) in enumerate(rects):
        for x2, y2, w2, h2 in rects[i + 1:]:
            apart = x + w <= x2 or x2 + w2 <= x or y + h <= y2 or y2 + h2 <= y
            assert apart, "grid cells must tile, not stack"


def test_view_sizes_ask_for_exactly_what_will_be_drawn(display):
    """The env renders panel-sized frames, so nothing is rendered and scaled away."""
    env, d, render = display
    d.layout = "grid"
    sizes = d.view_sizes(env.view_names)
    assert list(sizes) == list(env.view_names)[:4]
    rects = render.panels("grid", len(sizes), d.viewport)
    assert list(sizes.values()) == [(w, h) for _, _, w, h in rects]


def test_layout_cycles_through_every_layout(display):
    _, d, render = display
    seen = [d.layout]
    for _ in range(len(render.LAYOUTS)):
        seen.append(d.cycle_layout())
    assert set(seen) == set(render.LAYOUTS)
    assert seen[-1] == seen[0], "cycling the whole ring comes home"


def test_draws_every_view_of_a_multi_camera_env(display):
    env, d, render = display
    d.layout = "inset"
    views = env.frames(d.view_sizes(env.view_names))
    assert len(views) == len(env.view_names) >= 2
    for name, rgb in views.items():
        assert rgb is not None and rgb.dtype == np.uint8, f"{name} rendered nothing"
    size = d.screen.get_size()
    d.draw(env.hud(), views)
    assert d.screen.get_size() == size


def test_a_dark_panel_does_not_take_the_window_down(display):
    """An env whose extra camera returns None still draws the ones that worked."""
    env, d, render = display
    d.draw(env.hud(), {"scene": env.frame(320, 240), "wrist": None})


def test_render_survives_a_live_step_loop(display):
    env, d, render = display
    for i in range(20):
        env.step(0.4, 0.2, 0.0, 0.0, i > 10, scene.CTRL_DT)
        d.draw(env.hud(), env.frame(*d.viewport))
    assert np.isfinite(env.hud().ee).all()


# --- a resizable window -----------------------------------------------------

def test_resizing_moves_the_viewport_the_hud_and_the_type(display):
    env, d, render = display
    small = d.viewport, d.scale, d.f_big.get_height()
    try:
        d.resize(1600, 1000)
        assert d.viewport[0] == 1600
        assert d.render_h == 1000 - d.hud_h, "the HUD band sits under the viewport"
        assert d.scale > small[1] and d.f_big.get_height() > small[2], \
            "type has to grow with the window or it is unreadable on a big one"
        d.draw(env.hud(), env.frames(d.view_sizes(env.view_names)))
    finally:
        d.resize(render.WINDOW_W, render.WINDOW_H)


def test_a_window_smaller_than_the_hud_is_refused(display):
    env, d, render = display
    try:
        d.resize(100, 80)
        assert d.screen.get_size() == (render.MIN_W, render.MIN_H)
        assert d.render_h >= 120, "the viewport must never collapse to nothing"
    finally:
        d.resize(render.WINDOW_W, render.WINDOW_H)


def test_panels_are_capped_at_the_offscreen_buffer(display):
    """MuJoCo raises rather than downscaling, so an oversized window scales up."""
    env, d, render = display
    try:
        d.resize(scene.OFF_W + 600, render.WINDOW_H)
        for w, h in d.view_sizes(env.view_names).values():
            assert w <= scene.OFF_W and h <= scene.OFF_H
        env.frames(d.view_sizes(env.view_names))        # must not raise
    finally:
        d.resize(render.WINDOW_W, render.WINDOW_H)


def test_the_renderer_cache_does_not_grow_with_every_resize(display):
    """A window drag is a resize per mouse move, and each size is a GL context."""
    from benchmarks.native import MAX_RENDERERS
    env, d, render = display
    for w in range(900, 1200, 20):
        d.resize(w, 700)
        env.frames(d.view_sizes(env.view_names))
    assert len(env._renderers) <= MAX_RENDERERS
    d.resize(render.WINDOW_W, render.WINDOW_H)


# --- telling the operator what a key just did -------------------------------
#
# Switching suite or task can legitimately change nothing (one suite installed,
# one task in it). That used to be reported only on stdout, behind the window,
# so a correct no-op was indistinguishable from a dead key. These pin the HUD
# side of it.

def test_a_notice_is_shown_and_then_expires(display):
    env, d, render = display
    d.notify("suite -> robosuite")
    assert d.notice_t == pytest.approx(render.NOTICE_SECONDS)
    frame = env.frame(*d.viewport)
    d.draw(env.hud(), frame, dt=1.0)
    assert d.notice_t == pytest.approx(render.NOTICE_SECONDS - 1.0)
    d.draw(env.hud(), frame, dt=render.NOTICE_SECONDS)
    assert d.notice_t == 0.0, "a notice must not stay up forever"


def test_a_notice_never_outlives_the_hud_band(display):
    """It takes over the telemetry line rather than adding a fourth one."""
    env, d, render = display
    d.notify("x")
    line_y = d.render_h + round(130 * d.scale)
    assert line_y + d.f_tiny.get_height() <= d.screen.get_size()[1], \
        "the notice line must sit inside the HUD band"
    d.draw(env.hud(), env.frame(*d.viewport), dt=render.NOTICE_SECONDS)


def test_the_ring_position_says_when_there_is_nowhere_to_go():
    """The commonest 'the key does nothing' is a ring one entry long."""
    from main import _ring
    assert _ring(("native",), "native") == "1/1"
    assert _ring(("native", "robosuite", "fetch"), "fetch") == "3/3"
    # An unlisted current entry (a hand-typed --env task id) is position 1, not
    # a crash: cycle_task treats it the same way.
    assert _ring(("Lift", "Stack"), None) == "1/2"
    assert _ring((), None) == ""
