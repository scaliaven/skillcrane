"""M7 (stretch) -- episodes land on disk in LeRobot's dataset layout."""
import json

import numpy as np
import pytest

import scene
from game import Game

pytest.importorskip("pyarrow", reason="--record needs pyarrow")
pytest.importorskip("PIL", reason="--record needs Pillow")

from recorder import ACTION_NAMES, STATE_NAMES, EpisodeRecorder  # noqa: E402

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
