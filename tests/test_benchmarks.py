"""Benchmark adapter layer.

These run in both worlds: with no benchmark installed (the registry must degrade
politely) and with them installed (each adapter must satisfy the TeleopEnv
contract and actually move a robot). Anything needing a benchmark skips when it
is absent, so the core suite stays green on a bare checkout.
"""
import numpy as np
import pytest

import benchmarks
import scene
from benchmarks import Hud, TeleopEnv
from benchmarks.registry import NATIVE, SUITES

# Suites that can be teleoperated, i.e. have a factory.
DRIVABLE = [n for n, f in SUITES.items() if f.supported]
# LIBERO pins robosuite==1.4 and must live in its own env; never co-installed.
SHAREABLE = [n for n in DRIVABLE if n not in (NATIVE, "libero")]


# --- registry, with or without anything installed ---------------------------

def test_native_is_always_available():
    assert benchmarks.installed(NATIVE)


def test_parse_splits_suite_and_task():
    assert benchmarks.parse("robosuite:Lift") == ("robosuite", "Lift")
    assert benchmarks.parse("native") == ("native", None)
    assert benchmarks.parse("") == ("native", None)
    assert benchmarks.parse("libero:libero_spatial/3") == ("libero", "libero_spatial/3")


def test_describe_lists_every_suite_without_importing_them():
    text = benchmarks.describe()
    for name in SUITES:
        assert name in text
    assert "python main.py --env" in text


def test_unknown_suite_fails_with_a_useful_message():
    with pytest.raises(SystemExit) as e:
        benchmarks.make("nope:Task")
    assert "--list-envs" in str(e.value)


def test_missing_suite_names_the_install_command():
    for name in SHAREABLE:
        if benchmarks.installed(name):
            continue
        with pytest.raises(SystemExit) as e:
            benchmarks.make(name)
        assert "pip install" in str(e.value) or "conda env" in str(e.value)


def test_non_teleoperable_suite_explains_why():
    with pytest.raises(SystemExit) as e:
        benchmarks.make("aloha")
    assert "bimanual" in str(e.value).lower()


# --- the native adapter is the reference implementation ---------------------

@pytest.fixture
def native():
    return benchmarks.make(NATIVE, seed=2)


def test_native_satisfies_the_contract(native):
    assert isinstance(native, TeleopEnv)
    assert isinstance(native.hud(), Hud)
    assert native.observation().shape == (len(native.state_names),)
    assert len(native.action_names) == 5


def test_native_step_matches_the_underlying_game(native):
    from game import Game
    direct = Game(seed=2)
    for _ in range(20):
        native.step(0.5, -0.2, 0.1, 0.0, False, scene.CTRL_DT)
        direct.step(0.5, -0.2, 0.1, 0.0, False, scene.CTRL_DT)
    assert np.allclose(native.observation(), direct.observation()), \
        "the adapter changed the physics it is supposed to be forwarding"


def test_native_hud_reports_gripper_state(native):
    assert native.hud().grip == "OPEN"
    native.step(0, 0, 0, 0, True, scene.CTRL_DT)
    assert native.hud().grip in ("CLOSED", "HOLDING")


def test_native_reset_clears_the_round(native):
    for _ in range(5):
        native.step(0, 0, 0, 0, False, scene.CTRL_DT)
    native.reset(full=True)
    assert native.hud().score == 0
    assert native.hud().time_left == pytest.approx(scene.ROUND_SECONDS)


# --- installed benchmarks ---------------------------------------------------

@pytest.mark.parametrize("suite", SHAREABLE)
def test_installed_benchmark_drives(suite):
    if not benchmarks.installed(suite):
        pytest.skip(f"{suite} not installed")
    env = benchmarks.make(suite, seed=0)
    try:
        assert isinstance(env, TeleopEnv)
        assert 0 < env.control_dt <= 1.0
        obs = env.observation()
        assert obs.shape == (len(env.state_names),) and np.isfinite(obs).all()

        # Descend for a while; the end-effector must actually move.
        ee0 = np.asarray(env.hud().ee, dtype=float).copy()
        for _ in range(20):
            env.step(0.0, 0.0, -0.5, 0.0, False, env.control_dt)
        ee1 = np.asarray(env.hud().ee, dtype=float)
        assert np.linalg.norm(ee1 - ee0) > 1e-3, "commanding -z moved nothing"

        hud = env.hud()
        assert isinstance(hud, Hud)
        assert 0 < hud.time_left <= 90.0, "the round clock should be counting down"
        assert hud.grip in ("OPEN", "CLOSED", "HOLDING")
    finally:
        env.close()


@pytest.mark.parametrize("suite", SHAREABLE)
def test_installed_benchmark_renders(suite):
    if not benchmarks.installed(suite):
        pytest.skip(f"{suite} not installed")
    env = benchmarks.make(suite, seed=0)
    try:
        frame = env.frame(160, 120)
        if frame is None:
            pytest.skip(f"{suite} returned no frame (no GL context)")
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.max() > 0, "frame is entirely black"
    finally:
        env.close()


@pytest.mark.parametrize("suite", SHAREABLE)
def test_installed_benchmark_records_its_own_schema(suite, tmp_path):
    if not benchmarks.installed(suite):
        pytest.skip(f"{suite} not installed")
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from recorder import EpisodeRecorder

    env = benchmarks.make(suite, seed=0)
    try:
        rec = EpisodeRecorder(tmp_path, state_names=env.state_names,
                              action_names=env.action_names, task=env.task)
        for _ in range(5):
            env.step(0.0, 0.0, -0.3, 0.0, False, env.control_dt)
            rec.add(env.observation(), [0.0, 0.0, -0.3, 0.0, 0.0])
        table = pq.read_table(rec.save())
        assert table.num_rows == 5
        width = len(table.column("observation.state").to_pylist()[0])
        assert width == len(env.state_names), "recorded schema does not match the env"
    finally:
        env.close()


def test_gripper_signs_are_documented_per_suite():
    """The close sign differs between suites and was measured, not assumed."""
    from benchmarks.gym_env import FETCH, METAWORLD
    from benchmarks.robosuite_env import GRIP_CLOSE as RS_CLOSE
    assert METAWORLD.grip_close == +1.0
    assert FETCH.grip_close == -1.0, "Fetch closes on -1; +1 opens it"
    assert RS_CLOSE == +1.0


# --- LIBERO lives in its own environment ------------------------------------

def test_libero_drives_when_installed():
    """LIBERO pins robosuite==1.4, so this only runs inside its own env."""
    if not benchmarks.installed("libero"):
        pytest.skip("libero not installed (see environment-libero.yml)")
    env = benchmarks.make("libero:libero_spatial/0", seed=0)
    try:
        assert isinstance(env, TeleopEnv)
        assert env.task, "LIBERO tasks carry a natural-language description"
        obs = env.observation()
        assert obs.shape == (len(env.state_names),) and np.isfinite(obs).all()
        ee0 = np.asarray(env.hud().ee, dtype=float).copy()
        for _ in range(20):
            env.step(0.0, 0.0, -0.5, 0.0, False, env.control_dt)
        assert np.linalg.norm(np.asarray(env.hud().ee, float) - ee0) > 1e-3
        frame = env.frame(84, 84)
        assert frame is not None and frame.dtype == np.uint8
    finally:
        env.close()


# --- switching at runtime ---------------------------------------------------

def test_the_suite_ring_is_installed_teleoperable_and_constructible():
    ring = benchmarks.suites()
    assert NATIVE in ring, "the native arm is always in the suite ring"
    for name in ring:
        fam = SUITES[name]
        assert fam.supported and fam.factory is not None
        assert benchmarks.installed(name)
    assert "aloha" not in ring, "registered but has no factory -- cannot be built"


def test_cycle_suite_walks_the_ring_and_wraps():
    ring = benchmarks.suites()
    spec = NATIVE
    seen = [spec]
    for _ in range(len(ring)):
        spec = benchmarks.cycle_suite(spec, 1)
        seen.append(spec)
    assert seen[-1] == NATIVE, "stepping the length of the ring returns home"
    assert set(seen) == set(ring)


def test_cycle_suite_backwards_is_the_inverse():
    for name in benchmarks.suites():
        assert benchmarks.cycle_suite(benchmarks.cycle_suite(name, 1), -1) == name


def test_cycle_suite_drops_the_task_because_it_means_nothing_next_door():
    assert ":" not in benchmarks.cycle_suite("robosuite:Lift", 1)


def test_cycle_suite_from_an_unknown_suite_does_not_strand_the_operator():
    assert benchmarks.cycle_suite("nosuchthing:x", 1) in benchmarks.suites()


# --- switching the task inside a suite --------------------------------------

def test_cycle_task_stays_in_the_suite_and_walks_its_tasks():
    ring = list(SUITES["robosuite"].tasks)
    spec = f"robosuite:{ring[0]}"
    seen = [spec]
    for _ in range(len(ring)):
        spec = benchmarks.cycle_task(spec, 1)
        seen.append(spec)
    assert all(s.startswith("robosuite:") for s in seen)
    assert seen[-1] == seen[0], "stepping the whole list wraps back"
    assert {s.split(":")[1] for s in seen} == set(ring)


def test_cycle_task_backwards_is_the_inverse():
    spec = "metaworld:push-v3"
    assert benchmarks.cycle_task(benchmarks.cycle_task(spec, 1), -1) == spec


def test_cycle_task_does_nothing_for_a_suite_with_one_setting():
    """The native arm has one scene, so the caller can say so instead of
    silently rebuilding the identical environment."""
    assert benchmarks.cycle_task(NATIVE, 1) == NATIVE
    assert benchmarks.cycle_task("nosuchthing:x", 1) == "nosuchthing:x"


def test_cycle_task_from_a_hand_typed_task_lands_on_the_ring():
    spec = benchmarks.cycle_task("libero:libero_90/17", 1)
    assert spec.split(":")[1] in SUITES["libero"].tasks


def test_tasks_reports_what_the_task_ring_holds():
    assert benchmarks.tasks("robosuite:Lift") == SUITES["robosuite"].tasks
    assert benchmarks.tasks("nosuchthing") == ()


# --- multiple camera views --------------------------------------------------

def test_every_env_declares_at_least_one_view():
    assert TeleopEnv.view_names and len(TeleopEnv.view_names) >= 1


def test_native_offers_the_orbit_camera_plus_the_mjcf_ones(native):
    assert native.view_names[0] == "scene", "the operator's view comes first"
    assert set(scene.CAMERAS) <= set(native.view_names)


def test_native_renders_each_view_at_the_size_it_was_asked_for(native):
    sizes = {"scene": (96, 64), "wrist": (48, 32), "top": (64, 64)}
    try:
        views = native.frames(sizes)
    except Exception as exc:                    # no GL context on this box
        pytest.skip(f"no offscreen renderer here: {exc}")
    for name, (w, h) in sizes.items():
        assert views[name].shape == (h, w, 3), f"{name} came back the wrong size"
        assert views[name].max() > 0, f"{name} is entirely black"
    assert not np.array_equal(views["scene"][:32, :32], views["top"][:32, :32]), \
        "two cameras returned the same image"
    native.close()


def test_a_single_view_env_leaves_the_other_panels_dark():
    """The contract's default: one camera, and anything else comes back None."""
    class OneCamera(TeleopEnv):
        def reset(self, full=False): pass
        def step(self, *a): return False
        def observation(self): return np.zeros(1)
        def hud(self): return Hud()
        def frame(self, width, height):
            return np.zeros((height, width, 3), np.uint8)

    views = OneCamera().frames({"main": (8, 8), "wrist": (4, 4)})
    assert views["main"].shape == (8, 8, 3)
    assert views["wrist"] is None


# --- the operator's camera: zoom and follow ---------------------------------
# These are the controls that make the main view usable for close work, so they
# are asserted on numbers: where the camera is looking and how far away it is.
# None of it needs a GL context -- a camera is not a framebuffer.

def test_the_camera_opens_pointed_at_the_work(native):
    from benchmarks.native import HOME_CAM

    cam = native._ensure_cam()
    assert np.allclose(cam.lookat, native._focus())
    assert cam.distance == pytest.approx(HOME_CAM[0])


def test_focus_sits_between_the_gripper_and_what_it_is_reaching_for(native):
    ee = np.asarray(native.game.arm.ee(), float)
    cube = np.asarray(native.game.cube_pos(), float)
    focus = native._focus()
    assert np.linalg.norm(focus - ee) <= np.linalg.norm(cube - ee) + 1e-9
    assert np.linalg.norm(focus - cube) <= np.linalg.norm(cube - ee) + 1e-9
    assert focus[2] >= 0.06, "never look at a point under the floor"


def test_zoom_is_multiplicative_and_clamped_both_ways(native):
    from benchmarks.native import ZOOM_RANGE

    start = native._ensure_cam().distance
    native.zoom(0.1)
    assert native.cam.distance < start, "+ dollies in"
    native.zoom(-0.2)
    assert native.cam.distance > start, "- dollies out"
    for _ in range(200):
        native.zoom(1.0)
    assert native.cam.distance == pytest.approx(ZOOM_RANGE[0])
    for _ in range(200):
        native.zoom(-1.0)
    assert native.cam.distance == pytest.approx(ZOOM_RANGE[1])


def _run(env, ticks=60, dz=0.0):
    for _ in range(ticks):
        env.step(0.4, 0.0, dz, 0.0, False, env.control_dt)


def test_following_closes_on_the_work_and_can_be_switched_off(native):
    native.track()                              # first call only sets the clock
    _run(native)
    before = np.linalg.norm(native.cam.lookat - native._focus())
    native.track()
    after = np.linalg.norm(native.cam.lookat - native._focus())
    assert after < before, "the camera should have eased toward the work"

    assert native.toggle_follow() is False
    frozen = np.array(native.cam.lookat)
    _run(native)
    native.track()
    assert np.allclose(native.cam.lookat, frozen), "follow off means hold still"
    assert native.toggle_follow() is True


def test_tracking_advances_once_per_simulated_instant(native):
    """Recording renders the same tick twice; the camera must not move twice."""
    native.track()
    _run(native)
    native.track()
    once = np.array(native.cam.lookat)
    native.track()
    native.track()
    assert np.allclose(native.cam.lookat, once)


def test_a_benchmark_without_a_movable_camera_just_ignores_the_controls():
    class Fixed(TeleopEnv):
        def reset(self, full=False): pass
        def step(self, *a): return False
        def observation(self): return np.zeros(1)
        def hud(self): return Hud()

    env = Fixed()
    env.zoom(1.0)                               # must not raise
    env.orbit(10.0)
    assert env.toggle_follow() is False, "and it says so, rather than pretending"


# --- the live session: what main.py does with all of the above --------------

@pytest.fixture
def session_env():
    env = benchmarks.make(NATIVE, seed=0)
    yield env
    env.close()


def test_record_views_main_is_the_operators_view_only(session_env):
    import main
    sizes = main._record_sizes(session_env, "main")
    assert list(sizes) == [session_env.view_names[0]]
    assert set(sizes.values()) == {main.RECORD_SIZE}


def test_recorded_frames_ignore_the_window_size(session_env):
    """--record-size is the dataset's shape; the window is the operator's."""
    import main
    sizes = main._record_sizes(session_env, "all", (640, 480))
    assert set(sizes.values()) == {(640, 480)}
    assert main._size("1600x1000", "window") == (1600, 1000)
    with pytest.raises(SystemExit):
        main._size("huge", "window")


def test_record_views_all_takes_every_camera(session_env):
    import main
    assert list(main._record_sizes(session_env, "all")) == list(session_env.view_names)


def test_record_views_accepts_a_list_of_names(session_env):
    import main
    assert list(main._record_sizes(session_env, "wrist,top")) == ["wrist", "top"]


def test_record_views_says_so_when_a_camera_does_not_exist(session_env, capsys):
    """A typo must not look like a camera that quietly recorded nothing."""
    import main
    assert main._record_sizes(session_env, "wrst") == {}
    assert "no camera named wrst" in capsys.readouterr().out


def test_switching_nowhere_keeps_the_session_and_says_why(session_env, capsys):
    import main
    s = main.Session(session_env, NATIVE)
    same = main._switch(s, NATIVE, 0, None, "main", nothing="only native here")
    assert same is s
    assert "only native here" in capsys.readouterr().out


def test_a_backend_that_fails_to_build_does_not_end_the_session(session_env, capsys):
    import main
    s = main.Session(session_env, NATIVE)
    kept = main._switch(s, "nosuchthing:x", 0, None, "main", nothing="")
    assert kept is s, "a bad install must not take the live session down"
    assert "could not switch" in capsys.readouterr().out


def test_switching_closes_the_episode_and_opens_the_next(tmp_path, monkeypatch):
    """The two environments have different columns, so they cannot share one file."""
    pytest.importorskip("pyarrow")
    import main
    from benchmarks.registry import _native

    env = benchmarks.make(NATIVE, seed=0)
    s = main._session(env, NATIVE, tmp_path, "main")
    s.rec.add(env.observation(), np.zeros(5))
    monkeypatch.setattr(benchmarks, "make", lambda spec, seed=0: _native(None, seed))

    after = main._switch(s, "native:next", 0, tmp_path, "main", nothing="")
    assert after.spec == "native:next"
    assert (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert after.rec.episode_index == s.rec.episode_index + 1
    assert len(after.rec) == 0, "the new episode starts empty"
    after.env.close()


@pytest.mark.parametrize("suite", SHAREABLE)
def test_installed_benchmark_renders_every_view_it_declares(suite):
    if not benchmarks.installed(suite):
        pytest.skip(f"{suite} not installed")
    env = benchmarks.make(suite, seed=0)
    try:
        assert env.view_names, "an env must declare at least one view"
        views = env.frames({v: (96, 96) for v in env.view_names})
        assert set(views) == set(env.view_names)
        for name, rgb in views.items():
            assert rgb is not None, f"{suite} declares {name} but renders nothing"
            assert rgb.ndim == 3 and rgb.dtype == np.uint8
    finally:
        env.close()
