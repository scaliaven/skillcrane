"""M4 -- the scoring rules, exercised both end-to-end and condition by condition."""
import numpy as np
import mujoco
import pytest

import scene
from game import Game, scripted_pick_and_place


def place_cube(g, xy, z=0.024, vel=0.0):
    """Teleport the cube for a rules test (never used by the game itself)."""
    g.d.qpos[g.cube_q:g.cube_q + 3] = [xy[0], xy[1], z]
    g.d.qpos[g.cube_q + 3:g.cube_q + 7] = [1, 0, 0, 0]
    g.d.qvel[g.cube_v:g.cube_v + 6] = 0
    g.d.qvel[g.cube_v] = vel
    mujoco.mj_forward(g.m, g.d)


def test_full_cycle_scores():
    g = Game(seed=2)
    assert g.score == 0
    scored = scripted_pick_and_place(g)
    assert scored, "scripted pick-and-place did not trigger a score"
    assert g.score == 1
    assert g.streak == 1 and g.best_streak == 1


def test_scoring_respawns_the_cube_outside_the_zone():
    g = Game(seed=2)
    scripted_pick_and_place(g)
    d = float(np.linalg.norm(g.cube_pos()[:2] - g.target))
    assert d > scene.TARGET_R, "the cube respawned inside the zone it just scored in"


@pytest.mark.parametrize("seed", range(6))
def test_spawns_are_reachable_and_clear_of_the_zone(seed):
    g = Game(seed=seed)
    for _ in range(50):
        p = g.spawn_cube()
        assert np.linalg.norm(p - g.target) > scene.SPAWN_CLEAR
        r = float(np.linalg.norm(p))
        assert scene.REACH_MIN <= r <= scene.REACH_MAX, f"spawn at r={r:.3f} is unreachable"


def test_timer_decrements_and_floors_at_zero():
    g = Game(seed=0)
    assert g.time_left == pytest.approx(scene.ROUND_SECONDS)
    for _ in range(10):
        g.step(0, 0, 0, 0, False)
    assert g.time_left == pytest.approx(scene.ROUND_SECONDS - 10 * scene.CTRL_DT)

    g.time_left = scene.CTRL_DT / 2          # less than one tick left
    g.step(0, 0, 0, 0, False)
    assert g.time_left == 0.0, "timer went negative"


# --- the three conditions a score requires -----------------------------------

def test_scores_when_settled_in_zone_and_open():
    g = Game(seed=0)
    g.closed = False
    place_cube(g, g.target)
    assert g.check_score() is True
    assert g.score == 1


def test_no_score_while_the_gripper_is_closed():
    """Otherwise you could score by holding the cube over the zone."""
    g = Game(seed=0)
    g.closed = True
    place_cube(g, g.target)
    assert g.check_score() is False
    assert g.score == 0


def test_no_score_outside_the_zone():
    g = Game(seed=0)
    g.closed = False
    place_cube(g, g.target + np.array([scene.TARGET_R + 0.02, 0.0]))
    assert g.check_score() is False
    assert g.score == 0


def test_no_score_while_the_cube_is_still_moving():
    g = Game(seed=0)
    g.closed = False
    place_cube(g, g.target, vel=1.0)
    assert g.check_score() is False, "a cube skidding through the zone scored"


def test_no_score_while_the_cube_is_in_the_air():
    g = Game(seed=0)
    g.closed = False
    place_cube(g, g.target, z=0.20)
    assert g.check_score() is False


def test_streak_tracks_consecutive_scores():
    g = Game(seed=0)
    g.closed = False
    for i in range(3):
        place_cube(g, g.target)
        assert g.check_score() is True
        assert g.score == i + 1 and g.streak == i + 1
    assert g.best_streak == 3


def test_reset_clears_the_round_but_partial_reset_keeps_the_score():
    g = Game(seed=0)
    for _ in range(5):                        # burn some clock
        g.step(0, 0, 0, 0, False)
    g.closed = False
    place_cube(g, g.target)
    g.check_score()
    assert g.score == 1
    spent = g.time_left
    assert spent < scene.ROUND_SECONDS

    g.reset(full=False)                       # e.g. re-home mid-round
    assert g.score == 1, "a partial reset must not wipe the score"
    assert g.time_left == pytest.approx(spent), "a partial reset must not add time"

    g.reset(full=True)
    assert g.score == 0 and g.streak == 0
    assert g.time_left == pytest.approx(scene.ROUND_SECONDS)


def test_observation_shape_matches_the_recorder_schema():
    from recorder import STATE_NAMES
    g = Game(seed=0)
    assert g.observation().shape == (len(STATE_NAMES),)
