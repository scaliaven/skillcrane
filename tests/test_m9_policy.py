"""M9 -- a policy drives the env, and a recorded episode plays back into it.

The load-bearing test in here is the replay one. Everything else in the project
checks that the *simulation* is right; this is the only thing that checks the
*dataset* is right. A recording can be perfectly well-formed -- every column
present, every shape correct, every PNG on disk -- and still be untrainable
because the numbers in `action` are not the numbers that produced the motion in
`observation.state` beside them. That is not hypothetical: `--headless --record`
logged [0, 0, 0, 0, grip] for every tick of a completed pick-and-place, because
the scripted demo drove the Cartesian target directly instead of through the
sticks. Replaying it would have caught that on the first try.
"""
import numpy as np
import pytest

import scene
from game import Game
from policy import (MAX_TICKS, SETTLE_QVEL, ReplayPolicy, ScriptedPickPlace,
                    evaluate, rollout, success_rate, summarise)

SEEDS = [0, 1, 2, 3, 4, 5]


def collect(seed=0, policy=None):
    """One scripted round. Returns the game, the rollout, actions and states."""
    g = Game(seed=seed)
    acts, states = [], []
    r = rollout(g, policy or ScriptedPickPlace(), seed=seed,
                on_tick=lambda env, a, scored: (acts.append(a),
                                                states.append(env.observation())))
    return g, r, np.asarray(acts, dtype=float), np.asarray(states, dtype=float)


# --- the scripted policy is a real demonstrator ------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_the_scripted_policy_scores(seed):
    """Same waypoints as game.scripted_pick_and_place, driven through the sticks.

    Parametrised because one lucky spawn is not a pass -- the same rule the
    grasp milestone is held to.
    """
    g, r, _, _ = collect(seed)
    assert r.success, f"seed {seed} never scored in {r.ticks} ticks"
    assert g.score == r.score == 1
    assert r.ticks < MAX_TICKS, "hit the backstop instead of finishing"


def test_the_emitted_action_is_a_stick_value():
    """Inside [-1, 1], and never *on* it: a clipped action is a lying action.

    The script travels at 0.30 m/s against a 0.45 m/s full stick, so the four
    motion axes have headroom by construction. If a future waypoint speed ate
    that headroom the recorded action would silently stop describing the motion,
    which is exactly the failure this whole module exists to prevent.
    """
    _, _, acts, _ = collect(2)
    motion = acts[:, :4]
    assert np.abs(motion).max() <= 1.0
    assert np.abs(motion).max() < 0.99, "the stick saturated; the action is clipped"
    assert np.abs(motion).max() > 0.1, "nothing moved"
    assert set(np.unique(acts[:, 4])) <= {0.0, 1.0}, "grip is a bit, not a range"


def test_the_action_column_is_not_all_zeros():
    """The regression this milestone exists for.

    A demo that moves the target directly records a zero action beside a moving
    arm. Assert the two disagree in the only way that matters: the arm moved,
    and so did the numbers next to it.
    """
    _, _, acts, states = collect(1)
    assert np.abs(states[-1][:6] - states[0][:6]).max() > 0.1, "the arm never moved"
    moving = np.abs(acts[:, :4]).max(axis=1) > 1e-6
    assert moving.mean() > 0.5, "most ticks recorded no action while the arm moved"


def test_the_gripper_legs_wait_for_the_arm_to_settle():
    """Pins SETTLE_QVEL: arriving is not stopping.

    The commanded target is pure integration and gets to a waypoint first, with
    the arm still crossing the tolerance ball at 0.3 m/s. Without the velocity
    condition the fingers closed at |qvel| ~0.7 rad/s and opened at ~0.9; the
    cube was grasped and released from a moving gripper on every demonstration.
    """
    g = Game(seed=0)
    pol = ScriptedPickPlace()
    seen = {}
    rollout(g, pol, seed=0,
            on_tick=lambda env, a, s: seen.setdefault(
                pol.phase, float(np.linalg.norm(g.d.qvel[g.arm.dof]))))
    # Sampled on the first tick of the leg that follows, i.e. the instant the
    # gripper command changes.
    assert seen["close"] < 4 * SETTLE_QVEL, "grasped while still moving"
    assert seen["release"] < 4 * SETTLE_QVEL, "released while still moving"


# --- replay: does the dataset mean what it says? -----------------------------

def test_replay_reproduces_the_episode():
    """Same seed, same actions, same outcome -- to within float32 rounding.

    The actions come back through np.float32 because that is what the parquet
    stores, so this measures the round trip a training run would actually see.
    """
    g, r, acts, states = collect(3)
    replayed = ReplayPolicy(np.asarray(acts, dtype=np.float32))

    g2 = Game(seed=3)
    states2 = []
    r2 = rollout(g2, replayed, seed=3,
                 on_tick=lambda env, a, s: states2.append(env.observation()))

    assert r2.ticks == r.ticks, "replay ran a different number of ticks"
    assert r2.score == r.score == 1, "replay did not reproduce the score"
    assert r2.first_score_tick == r.first_score_tick
    diff = np.abs(np.asarray(states2) - states).max()
    assert diff < 1e-5, f"replayed trajectory diverged by {diff}"
    assert np.linalg.norm(g2.cube_pos() - g.cube_pos()) < 1e-6


def test_replay_in_a_different_world_does_not_score():
    """Teeth for the test above: it has to be able to fail.

    The actions are a fixed open-loop sequence aimed at where seed 3's cube
    spawned. Run them at a different spawn and the gripper closes on nothing.
    """
    _, _, acts, _ = collect(3)
    g = Game(seed=5)
    r = rollout(g, ReplayPolicy(acts), seed=5)
    assert not r.success, "the replay check is vacuous -- any seed scores"


def test_zero_actions_leave_the_cube_where_it_spawned():
    """The shape of the old bug: a dataset of zeros next to a moving arm.

    If this ever scored, the replay test above would pass on a broken recording.
    """
    g = Game(seed=0)
    spawn = g.cube_pos().copy()
    r = rollout(g, ReplayPolicy(np.zeros((400, 5))), seed=0)
    assert not r.success
    assert np.linalg.norm(g.cube_pos() - spawn) < 0.01, "nothing should have moved it"


def test_replay_stops_when_the_recording_runs_out():
    acts = np.zeros((37, 5))
    g = Game(seed=0)
    r = rollout(g, ReplayPolicy(acts), max_ticks=MAX_TICKS, seed=0)
    assert r.ticks == 37, "replay must not invent ticks past the episode"


def test_replay_round_trips_through_a_written_dataset(tmp_path):
    """The whole path a training run uses: recorder -> parquet -> ReplayPolicy."""
    pytest.importorskip("pyarrow", reason="the dataset round trip needs pyarrow")
    from recorder import EpisodeRecorder

    g = Game(seed=4)
    rec = EpisodeRecorder(tmp_path)
    r = rollout(g, ScriptedPickPlace(), seed=4,
                on_tick=lambda env, a, scored: rec.add(env.observation(), a,
                                                       reward=float(scored)))
    rec.save()

    loaded = ReplayPolicy.from_dataset(tmp_path, 0)
    assert len(loaded) == r.ticks == len(rec)
    g2 = Game(seed=4)
    r2 = rollout(g2, loaded, seed=4)
    assert r2.score == r.score == 1, "the episode on disk does not reproduce"
    assert np.linalg.norm(g2.cube_pos() - g.cube_pos()) < 1e-6


def test_replay_reports_a_missing_episode(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayPolicy.from_dataset(tmp_path, 7)


# --- eval --------------------------------------------------------------------

def test_rollout_reports_when_it_scored():
    _, r, _, _ = collect(0)
    assert r.first_score_tick is not None and 0 < r.first_score_tick <= r.ticks
    assert r.time_to_score == pytest.approx(r.first_score_tick * scene.CTRL_DT)
    assert r.seconds == pytest.approx(r.ticks * scene.CTRL_DT)


def test_evaluate_runs_one_clean_world_per_seed():
    """Each seed gets a fresh env, so the numbers are per seed and comparable."""
    seeds = [0, 1, 2]
    results = evaluate(lambda s: Game(seed=s), ScriptedPickPlace, seeds)
    assert [r.seed for r in results] == seeds
    assert all(r.score == 1 for r in results), summarise(results)
    assert success_rate(results) == 1.0
    # A shared env would carry the previous round's score into the next one.
    assert {r.score for r in results} == {1}


def test_evaluate_is_reproducible():
    a = evaluate(lambda s: Game(seed=s), ScriptedPickPlace, [1, 2])
    b = evaluate(lambda s: Game(seed=s), ScriptedPickPlace, [1, 2])
    assert [(r.ticks, r.score) for r in a] == [(r.ticks, r.score) for r in b]


def test_summarise_names_every_seed_and_the_rate():
    results = evaluate(lambda s: Game(seed=s), ScriptedPickPlace, [0, 1])
    text = summarise(results)
    assert "success 2/2 = 100%" in text
    assert text.count("ok") == 2
    assert "seed" in text.splitlines()[0]


def test_summarise_survives_an_empty_run():
    assert summarise([]) == "no episodes"
    assert success_rate([]) == 0.0


# --- the --policy spec -------------------------------------------------------

def test_policy_spec_picks_the_right_kind(tmp_path):
    from main import _make_policy

    assert isinstance(_make_policy("scripted")(), ScriptedPickPlace)
    with pytest.raises(SystemExit):
        _make_policy("nonsense")
    with pytest.raises(SystemExit):
        _make_policy("replay")          # no directory


def test_policy_spec_reads_the_episode_number(tmp_path):
    """"replay:DIR:3" is episode 3; "replay:DIR" is episode 0."""
    pytest.importorskip("pyarrow", reason="the dataset round trip needs pyarrow")
    from main import _make_policy
    from recorder import EpisodeRecorder

    for ep, n in ((0, 11), (3, 17)):
        rec = EpisodeRecorder(tmp_path, episode_index=ep)
        for i in range(n):
            rec.add(np.zeros(10), [0.1, 0, 0, 0, 0], reward=0.0)
        rec.save()

    assert len(_make_policy(f"replay:{tmp_path}")()) == 11
    assert len(_make_policy(f"replay:{tmp_path}:3")()) == 17


def test_the_scripted_policy_says_it_is_native_only():
    """It reads the cube and the drop zone; no benchmark suite has ours.

    Without this the first symptom is AttributeError several hundred ticks into
    what looked like a working eval.
    """
    class NotOurArm:
        pass

    with pytest.raises(TypeError, match="native-only"):
        ScriptedPickPlace().reset(NotOurArm())


def test_eval_refuses_the_scripted_policy_on_another_suite():
    from main import run_eval

    with pytest.raises(SystemExit, match="native-only"):
        run_eval(1, env_spec="robosuite:Lift")
