"""M7 (stretch): log teleop episodes in LeRobot's dataset layout.

One row per control tick -- observation.state, the action that was applied, and
the rendered camera frame -- written as LeRobot v2.1: parquet under data/, PNG
frames under images/, and the meta/ sidecars a LeRobotDataset needs to load.
That is the shape `lerobot/scripts/train.py --policy.type=act` expects.

pyarrow and Pillow are only imported here, and only when --record is used, so
the core game keeps its three dependencies.
"""
import json
from pathlib import Path

import numpy as np

import scene

FPS = int(round(1.0 / scene.CTRL_DT))
TASK = "Pick up the cube and place it in the target zone."

# observation.state = 6 arm joints + gripper opening + cube xyz  (see Game.observation)
STATE_NAMES = [f"j{i}" for i in range(1, 7)] + ["grip", "cube_x", "cube_y", "cube_z"]
# action = what the operator asked for this tick, not what the joints did
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "grip"]


class EpisodeRecorder:
    """Accumulates one episode in memory, writes it out on save().

    Frames are kept as uint8 arrays; a 90 s round at 100 Hz is ~9000 frames, so
    callers that record long sessions should pass `frame_every` > 1.
    """

    def __init__(self, root, episode_index: int = 0, image_key: str = "observation.images.cam"):
        self.root = Path(root)
        self.episode_index = episode_index
        self.image_key = image_key
        self.states: list = []
        self.actions: list = []
        self.frames: list = []

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state, action, frame=None) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.frames.append(None if frame is None else np.asarray(frame, dtype=np.uint8))

    # -- output ---------------------------------------------------------------
    def save(self) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self.states:
            raise ValueError("nothing recorded")

        ep = self.episode_index
        n = len(self.states)
        data_dir = self.root / "data" / "chunk-000"
        meta_dir = self.root / "meta"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        img_paths = self._write_frames(ep)

        cols = {
            "observation.state": [s.tolist() for s in self.states],
            "action": [a.tolist() for a in self.actions],
            "timestamp": [i * scene.CTRL_DT for i in range(n)],
            "frame_index": list(range(n)),
            "episode_index": [ep] * n,
            "index": list(range(n)),
            "task_index": [0] * n,
        }
        if img_paths:
            cols[self.image_key] = img_paths
        table = pa.table(cols)
        out = data_dir / f"episode_{ep:06d}.parquet"
        pq.write_table(table, out)

        self._write_meta(ep, n, bool(img_paths))
        return out

    def _write_frames(self, ep: int) -> list:
        """PNG per tick. Returns dataset-relative paths, or [] if none captured."""
        if not any(f is not None for f in self.frames):
            return []
        from PIL import Image

        rel_dir = Path("images") / self.image_key / f"episode_{ep:06d}"
        (self.root / rel_dir).mkdir(parents=True, exist_ok=True)
        paths = []
        last = None
        for i, f in enumerate(self.frames):
            if f is not None:
                last = f
            rel = rel_dir / f"frame_{i:06d}.png"
            # A tick with no new frame reuses the previous one, so the image
            # column stays aligned 1:1 with the control ticks.
            Image.fromarray(last).save(self.root / rel)
            paths.append(str(rel))
        return paths

    def _write_meta(self, ep: int, n: int, has_images: bool) -> None:
        meta = self.root / "meta"
        h, w, c = (0, 0, 0)
        for f in self.frames:
            if f is not None:
                h, w, c = f.shape
                break

        features = {
            "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)],
                                  "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)],
                       "names": ACTION_NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        if has_images:
            features[self.image_key] = {
                "dtype": "image", "shape": [h, w, c],
                "names": ["height", "width", "channel"]}

        info = {
            "codebase_version": "v2.1",
            "robot_type": "clawcrew",
            "total_episodes": ep + 1,
            "total_frames": n,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": FPS,
            "splits": {"train": f"0:{ep + 1}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": features,
        }
        _write_json(meta / "info.json", info)
        _append_jsonl(meta / "tasks.jsonl", {"task_index": 0, "task": TASK}, key="task_index")
        _append_jsonl(meta / "episodes.jsonl",
                      {"episode_index": ep, "tasks": [TASK], "length": n},
                      key="episode_index")
        _append_jsonl(meta / "episodes_stats.jsonl",
                      {"episode_index": ep, "stats": self._stats()},
                      key="episode_index")

    def _stats(self) -> dict:
        out = {}
        for key, rows in (("observation.state", self.states), ("action", self.actions)):
            a = np.stack(rows)
            out[key] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
                        "count": [len(a)]}
        return out


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _append_jsonl(path: Path, obj, key: str) -> None:
    """Append unless a row with the same `key` is already there (idempotent reruns)."""
    rows = []
    if path.exists():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        rows = [r for r in rows if r.get(key) != obj.get(key)]
    rows.append(obj)
    rows.sort(key=lambda r: r.get(key, 0))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
