"""M7 (stretch): log teleop episodes in LeRobot's dataset layout.

One row per control tick -- observation.state, the action that was applied, and
the rendered camera frames -- written as LeRobot v2.1: parquet under data/, PNG
frames under images/, and the meta/ sidecars a LeRobotDataset needs to load.
That is the shape `lerobot/scripts/train.py --policy.type=act` expects.

A tick can carry more than one camera: pass `add()` a {view: frame} dict and
each view becomes its own `observation.images.<view>` column, which is how
LeRobot names a multi-camera rig. One array is still one camera, under
`image_key`.

pyarrow and Pillow are only imported here, and only when --record is used, so
the core game keeps its three dependencies.
"""
import json
from pathlib import Path

import numpy as np

import scene

FPS = int(round(1.0 / scene.CTRL_DT))
TASK = "Pick up the cube and place it in the target zone."

IMAGE_PREFIX = "observation.images."

# observation.state = 6 arm joints + gripper opening + cube xyz  (see Game.observation)
STATE_NAMES = [f"j{i}" for i in range(1, 7)] + ["grip", "cube_x", "cube_y", "cube_z"]
# action = what the operator asked for this tick, not what the joints did
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "grip"]


def next_episode_index(root) -> int:
    """First unused episode index in `root`.

    Without this every run writes episode_000000 and silently overwrites the
    last one, which is fatal for a directory that is supposed to accumulate a
    dataset. Reads the directory rather than any counter, so it stays right
    across separate runs of the program.
    """
    data = Path(root) / "data" / "chunk-000"
    if not data.is_dir():
        return 0
    used = []
    for f in data.glob("episode_*.parquet"):
        digits = f.stem.split("_")[-1]
        if digits.isdigit():
            used.append(int(digits))
    return max(used) + 1 if used else 0


class EpisodeRecorder:
    """Accumulates one episode in memory, writes it out on save().

    Frames are kept as uint8 arrays; a 90 s round at 100 Hz is ~9000 frames, so
    callers that record long sessions should pass `frame_every` > 1.
    """

    def __init__(self, root, episode_index: int = 0, image_key: str = "observation.images.cam",
                 state_names=None, action_names=None, task: str = TASK):
        self.root = Path(root)
        self.episode_index = episode_index
        self.image_key = image_key
        # Benchmark environments have their own observation widths, so the
        # schema is per-recorder. The module constants stay the Skillcrane
        # defaults so existing callers and tests are unaffected.
        self.state_names = list(state_names or STATE_NAMES)
        self.action_names = list(action_names or ACTION_NAMES)
        self.task = task
        self.states: list = []
        self.actions: list = []
        self.frames: list = []

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state, action, frame=None) -> None:
        """One control tick. `frame` is an RGB array, or {view: RGB}, or None."""
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.frames.append(self._views(frame))

    def _views(self, frame) -> dict:
        """Normalise a tick's frames to {image key: array}, dropping empties."""
        if frame is None:
            return {}
        if isinstance(frame, dict):
            return {self.image_column(k): np.asarray(v, dtype=np.uint8)
                    for k, v in frame.items() if v is not None}
        return {self.image_key: np.asarray(frame, dtype=np.uint8)}

    @staticmethod
    def image_column(view: str) -> str:
        """LeRobot column for a camera: "wrist" -> observation.images.wrist."""
        return view if view.startswith(IMAGE_PREFIX) else IMAGE_PREFIX + view

    def image_keys(self) -> list:
        """Every image column recorded, in the order the views first appeared."""
        keys = {}
        for tick in self.frames:
            keys.update(dict.fromkeys(tick))
        return list(keys)

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
        cols.update(img_paths)
        table = pa.table(cols)
        out = data_dir / f"episode_{ep:06d}.parquet"
        pq.write_table(table, out)

        self._write_meta(ep, n, list(img_paths))
        return out

    def _write_frames(self, ep: int) -> dict:
        """PNG per tick per camera. Returns {column: paths}, empty if none."""
        keys = self.image_keys()
        if not keys:
            return {}
        from PIL import Image

        cols = {}
        for key in keys:
            rel_dir = Path("images") / key / f"episode_{ep:06d}"
            (self.root / rel_dir).mkdir(parents=True, exist_ok=True)
            paths, last = [], None
            for i, tick in enumerate(self.frames):
                last = tick.get(key, last)
                if last is None:
                    # This camera has not produced anything yet, so there is no
                    # image to point at. Leave the cell null rather than
                    # inventing one; the column still lines up with the ticks.
                    paths.append(None)
                    continue
                rel = rel_dir / f"frame_{i:06d}.png"
                # A tick with no new frame reuses the previous one, so the image
                # column stays aligned 1:1 with the control ticks.
                Image.fromarray(last).save(self.root / rel)
                paths.append(str(rel))
            cols[key] = paths
        return cols

    def _write_meta(self, ep: int, n: int, image_keys: list) -> None:
        meta = self.root / "meta"
        shapes = {}
        for tick in self.frames:
            for key, f in tick.items():
                shapes.setdefault(key, f.shape)

        features = {
            "observation.state": {"dtype": "float32", "shape": [len(self.state_names)],
                                  "names": self.state_names},
            "action": {"dtype": "float32", "shape": [len(self.action_names)],
                       "names": self.action_names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for key in image_keys:
            h, w, c = shapes.get(key, (0, 0, 0))
            features[key] = {"dtype": "image", "shape": [h, w, c],
                             "names": ["height", "width", "channel"]}

        info = {
            "codebase_version": "v2.1",
            "robot_type": "skillcrane",
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
        _append_jsonl(meta / "tasks.jsonl", {"task_index": 0, "task": self.task}, key="task_index")
        _append_jsonl(meta / "episodes.jsonl",
                      {"episode_index": ep, "tasks": [self.task], "length": n},
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
