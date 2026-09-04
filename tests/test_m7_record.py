"""M7 (stretch) -- episodes land on disk in LeRobot's dataset layout."""
import json

import numpy as np
import pytest

import scene
from game import Game

pytest.importorskip("pyarrow", reason="--record needs pyarrow")
pytest.importorskip("PIL", reason="--record needs Pillow")

from recorder import (ACTION_NAMES, IMAGE_PREFIX, STATE_NAMES,  # noqa: E402
                      EpisodeRecorder, next_episode_index)

TICKS = 40


def renderer_or_skip(model, height, width):
    """MuJoCo needs a GL context; a CI box without one skips rather than errors."""
    import mujoco
    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:                     # no EGL/OSMesa/CGL here
        pytest.skip(f"no offscreen GL context available: {exc}")


def record_episode(root, seed=0, episode_index=0, with_frames=True):
    g = Game(seed=seed)
    rec = EpisodeRecorder(root, episode_index=episode_index)
    ren = renderer_or_skip(g.m, 48, 64) if with_frames else None
    for i in range(TICKS):
        action = [0.5, -0.25, 0.1, 0.0, float(i > TICKS // 2)]
        g.step(action[0], action[1], action[2], action[3], bool(action[4]))
        frame = None
        if ren is not None:
            ren.update_scene(g.d)
            frame = ren.render()
        rec.add(g.observation(), action, frame)
    return rec, rec.save()


def test_writes_parquet_with_one_row_per_control_tick(tmp_path):
    import pyarrow.parquet as pq
    rec, path = record_episode(tmp_path)
    assert path == tmp_path / "data" / "chunk-000" / "episode_000000.parquet"

    t = pq.read_table(path)
    assert t.num_rows == TICKS
    for col in ("observation.state", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index"):
        assert col in t.column_names, f"missing LeRobot column {col}"

    state = t.column("observation.state").to_pylist()
    action = t.column("action").to_pylist()
    assert len(state[0]) == len(STATE_NAMES)
    assert len(action[0]) == len(ACTION_NAMES)
    assert t.column("frame_index").to_pylist() == list(range(TICKS))
    ts = t.column("timestamp").to_pylist()
    assert ts[1] - ts[0] == pytest.approx(scene.CTRL_DT)


def test_state_and_action_carry_the_real_values(tmp_path):
    import pyarrow.parquet as pq
    rec, path = record_episode(tmp_path)
    t = pq.read_table(path)
    action = np.array(t.column("action").to_pylist())
    state = np.array(t.column("observation.state").to_pylist())

    assert action[0][:4] == pytest.approx([0.5, -0.25, 0.1, 0.0])
    assert action[-1][4] == 1.0, "gripper command should be recorded"
    # observation.state = 6 joints + gripper + cube xyz, and the arm moved
    assert state.shape == (TICKS, len(STATE_NAMES))
    assert np.abs(state[-1][:6] - state[0][:6]).max() > 1e-4, "joints never moved"
    assert np.isfinite(state).all()


def test_writes_one_png_per_tick(tmp_path):
    from PIL import Image
    rec, _ = record_episode(tmp_path)
    frames = sorted((tmp_path / "images" / rec.image_key / "episode_000000").glob("*.png"))
    assert len(frames) == TICKS
    with Image.open(frames[0]) as im:
        assert im.size == (64, 48)


def test_meta_files_describe_the_dataset(tmp_path):
    record_episode(tmp_path)
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == int(round(1 / scene.CTRL_DT))
    assert info["total_frames"] == TICKS
    assert info["features"]["observation.state"]["shape"] == [len(STATE_NAMES)]
    assert info["features"]["action"]["names"] == ACTION_NAMES

    tasks = [json.loads(x) for x in (tmp_path / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == 1 and tasks[0]["task_index"] == 0

    eps = [json.loads(x) for x in (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert eps[0]["episode_index"] == 0 and eps[0]["length"] == TICKS

    stats = [json.loads(x) for x in
             (tmp_path / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert set(stats[0]["stats"]) == {"observation.state", "action"}
    assert len(stats[0]["stats"]["action"]["mean"]) == len(ACTION_NAMES)


def test_second_episode_appends_without_duplicating_metadata(tmp_path):
    record_episode(tmp_path, episode_index=0)
    record_episode(tmp_path, seed=1, episode_index=1)

    assert (tmp_path / "data" / "chunk-000" / "episode_000001.parquet").exists()
    eps = [json.loads(x) for x in (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in eps] == [0, 1]
    tasks = [json.loads(x) for x in (tmp_path / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == 1, "the task row was duplicated"
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2


def test_recording_without_frames_still_writes_state_and_action(tmp_path):
    """A state-only run (no renderer) must not produce a broken image column."""
    import pyarrow.parquet as pq
    rec, path = record_episode(tmp_path, with_frames=False)
    t = pq.read_table(path)
    assert t.num_rows == TICKS
    assert rec.image_key not in t.column_names
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert rec.image_key not in info["features"]


def test_empty_recorder_refuses_to_save(tmp_path):
    with pytest.raises(ValueError):
        EpisodeRecorder(tmp_path).save()


# --- more than one camera ---------------------------------------------------

def frames(n: int, size=(12, 16)):
    """n distinguishable fake frames, so a mix-up between views is visible."""
    h, w = size
    return [np.full((h, w, 3), 10 * (i + 1), np.uint8) for i in range(n)]


def test_each_view_becomes_its_own_image_column(tmp_path):
    import pyarrow.parquet as pq
    rec = EpisodeRecorder(tmp_path)
    for i in range(4):
        scene_f, wrist_f = frames(2)
        rec.add(np.zeros(10), np.zeros(5), {"scene": scene_f, "wrist": wrist_f})
    t = pq.read_table(rec.save())

    assert f"{IMAGE_PREFIX}scene" in t.column_names
    assert f"{IMAGE_PREFIX}wrist" in t.column_names
    for key in (f"{IMAGE_PREFIX}scene", f"{IMAGE_PREFIX}wrist"):
        paths = t.column(key).to_pylist()
        assert len(paths) == 4 and all(key in p for p in paths)
        assert len(set(paths)) == 4, "every tick needs its own frame file"
        for rel in paths:
            assert (tmp_path / rel).exists()


def test_views_keep_their_own_size_and_pixels(tmp_path):
    from PIL import Image
    rec = EpisodeRecorder(tmp_path)
    rec.add(np.zeros(10), np.zeros(5),
            {"scene": np.full((24, 32, 3), 200, np.uint8),
             "wrist": np.full((12, 16, 3), 40, np.uint8)})
    rec.save()

    with Image.open(tmp_path / "images" / f"{IMAGE_PREFIX}scene"
                    / "episode_000000" / "frame_000000.png") as im:
        assert im.size == (32, 24) and im.getpixel((0, 0))[0] == 200
    with Image.open(tmp_path / "images" / f"{IMAGE_PREFIX}wrist"
                    / "episode_000000" / "frame_000000.png") as im:
        assert im.size == (16, 12) and im.getpixel((0, 0))[0] == 40

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"][f"{IMAGE_PREFIX}scene"]["shape"] == [24, 32, 3]
    assert info["features"][f"{IMAGE_PREFIX}wrist"]["shape"] == [12, 16, 3]


def test_a_tick_without_new_frames_repeats_each_view(tmp_path):
    """Frames can be logged at a lower rate than the control loop, per camera."""
    import pyarrow.parquet as pq
    from PIL import Image
    rec = EpisodeRecorder(tmp_path)
    first, second = frames(2)
    rec.add(np.zeros(10), np.zeros(5), {"scene": first, "wrist": second})
    rec.add(np.zeros(10), np.zeros(5), None)
    t = pq.read_table(rec.save())

    assert t.num_rows == 2
    paths = t.column(f"{IMAGE_PREFIX}scene").to_pylist()
    assert len(paths) == 2 and paths[0] != paths[1]
    with Image.open(tmp_path / paths[1]) as im:
        assert im.getpixel((0, 0))[0] == 10, "the repeat should be the last frame"


def test_a_view_that_never_rendered_gets_no_column(tmp_path):
    """A camera that comes back None must not invent an image path."""
    import pyarrow.parquet as pq
    rec = EpisodeRecorder(tmp_path)
    rec.add(np.zeros(10), np.zeros(5), {"scene": frames(1)[0], "wrist": None})
    t = pq.read_table(rec.save())
    assert f"{IMAGE_PREFIX}wrist" not in t.column_names
    assert f"{IMAGE_PREFIX}scene" in t.column_names


def test_a_view_that_starts_late_pads_with_nulls(tmp_path):
    import pyarrow.parquet as pq
    rec = EpisodeRecorder(tmp_path)
    rec.add(np.zeros(10), np.zeros(5), {"scene": frames(1)[0]})
    rec.add(np.zeros(10), np.zeros(5), {"scene": frames(1)[0],
                                        "wrist": frames(1)[0]})
    t = pq.read_table(rec.save())
    wrist = t.column(f"{IMAGE_PREFIX}wrist").to_pylist()
    assert wrist[0] is None and wrist[1] is not None


def test_a_single_frame_still_uses_the_default_image_key(tmp_path):
    """The one-camera API is unchanged: an array goes under image_key."""
    import pyarrow.parquet as pq
    rec = EpisodeRecorder(tmp_path)
    rec.add(np.zeros(10), np.zeros(5), frames(1)[0])
    t = pq.read_table(rec.save())
    assert rec.image_key in t.column_names
    assert rec.image_key.startswith(IMAGE_PREFIX)


def test_image_column_names_a_view_the_way_lerobot_does():
    assert EpisodeRecorder.image_column("wrist") == "observation.images.wrist"
    already = "observation.images.agentview"
    assert EpisodeRecorder.image_column(already) == already


# --- episode numbering ------------------------------------------------------

def test_next_episode_index_starts_at_zero_in_an_empty_directory(tmp_path):
    assert next_episode_index(tmp_path) == 0
    assert next_episode_index(tmp_path / "does-not-exist") == 0


def test_next_episode_index_follows_what_is_on_disk(tmp_path):
    """The bug this prevents: every run writing episode_000000 over the last."""
    rec = EpisodeRecorder(tmp_path, episode_index=next_episode_index(tmp_path))
    rec.add(np.zeros(10), np.zeros(5))
    rec.save()
    assert next_episode_index(tmp_path) == 1

    rec2 = EpisodeRecorder(tmp_path, episode_index=next_episode_index(tmp_path))
    rec2.add(np.zeros(10), np.zeros(5))
    out = rec2.save()
    assert out.name == "episode_000001.parquet"
    assert len(list((tmp_path / "data" / "chunk-000").glob("*.parquet"))) == 2


def test_next_episode_index_ignores_files_that_are_not_episodes(tmp_path):
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    (data / "episode_000004.parquet").touch()
    (data / "notes.parquet").touch()
    (data / "episode_bogus.parquet").touch()
    assert next_episode_index(tmp_path) == 5
