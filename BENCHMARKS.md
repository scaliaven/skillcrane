# Benchmark environments

**Yes, it's possible — and four benchmark suites now work**, driven by the same
gamepad, HUD and LeRobot recorder as the native arm.

Vocabulary, because the two are switched separately at runtime: a **suite** is a
benchmark (robosuite, LIBERO, RoboCasa), a **task** is one setting inside it
(`Lift`, `Stack`). `--env` takes `SUITE[:TASK]`.

Everything below was verified by running it on this machine (macOS 15, Apple
Silicon, Python 3.12), not read off a README. Where something was only looked up
rather than executed, it says so.

```sh
python main.py --list-envs                       # what's installed
python main.py --env robosuite:Lift              # teleop a benchmark
python main.py --env metaworld:pick-place-v3 --record runs/
```

## Verdict

| suite | status | verified by | env |
|---|---|---|---|
| **robosuite** 1.5.2 | ported | Lift/Panda, 84×84 offscreen, 20 steps/s | shared |
| **Meta-World** 3.1.1 | ported | `pick-place-v3`, 4-D action, 480×480 | shared |
| **Fetch** (Gymnasium-Robotics 1.4.2) | ported | `FetchPickAndPlace-v4`, 4-D action | shared |
| **LIBERO** 0.1.1 | ported | all 6 suites, 130 tasks, 7-D action | **its own** |
| **RoboCasa** | registered, **unverified** | nothing — not installed here | shared (robosuite) |

"shared" = installable together via `requirements-benchmarks.txt` or
`environment-benchmarks.yml`.

## Everything surveyed

The full search, including what was rejected and why. The **evidence** column is
the important one — most of this field cannot be judged from a README, and only
the top five rows were actually executed here.

### Ported

| package | version | evidence | notes |
|---|---|---|---|
| `robosuite` | 1.5.2 | **ran it** | OSC/BASIC controller takes our exact action. The anchor of this whole group. |
| `metaworld` | 3.1.1 | **ran it** | 50 tasks, 4-D action. Pins `mujoco==3.3.0`, which happens to satisfy the robosuite constraint. |
| `gymnasium-robotics` | 1.4.2 | **ran it** | Fetch pick-place / reach / push / slide. Loosest pin of the group (`mujoco>=2.2`). |
| `libero` | 0.1.1 | **ran it** | Hugging Face's LeRobot-aligned repackaging (`huggingface/lerobot-libero`), original authors credited. 130 tasks with language descriptions. Needs its own env. |

### Rejected after running it

| package | version | evidence | why not |
|---|---|---|---|
| `gym-aloha` | 0.1.4 | **ran it** | Installs and steps fine, but it is **14-D bimanual joint-position** control, not a Cartesian delta. One gamepad cannot drive it without an IK layer. Registered as not-teleoperable with the reason attached, so it isn't rediscovered. |

### Rejected on inspection

| package | version | evidence | why not |
|---|---|---|---|
| `gym-xarm` | 0.1.1 | PyPI metadata | Pins `mujoco>=2.3.7,<3.0`. Every other suite here needs `mujoco>=3.0`, so it is mutually exclusive with all of them — a third environment for one small task suite. |
| `gym-pusht` | 0.1.6 | PyPI metadata | 2-D pusher on pymunk, no MuJoCo and **no gripper**. Our contract's gripper bit has nothing to map to. |
| `panda-gym` | 3.0.7 | PyPI metadata | A different physics engine (PyBullet) for tasks robosuite and Fetch already cover. Not worth a second engine. |
| `dm-control` | 1.0.45 | PyPI metadata | The interesting near-miss: it wants `mujoco>=3.12.0`, making it the **only** suite compatible with this project's current stack. But it is a control/RL suite, not a Cartesian-delta manipulation interface. Best future candidate if the `mujoco<3.12` pin ever becomes painful. |
| `mani-skill` | 3.0.1 | PyPI metadata | SAPIEN does ship `macosx_12_0_universal2` wheels (3.0.2+), so it is not obviously impossible — but its renderer is Vulkan-based and macOS is not a supported target. Not attempted. |
| `robocasa` | — | not on PyPI | GitHub only, and built on robosuite — which we already support, so it is the **most likely next addition**. Not attempted. |
| `rlbench` / `pyrep` | — / 3.2.0 | not on PyPI / metadata | Needs a separate CoppeliaSim install and is Linux-centric. |
| `calvin-env` | — | not on PyPI | GitHub only, PyBullet, Linux-oriented. |
| `omnigibson` | 1.1.1 | PyPI metadata | Requires NVIDIA Isaac Sim. |
| `isaacsim` | 6.0.1.0 | PyPI metadata | NVIDIA RTX on Linux/Windows. Not macOS, at all. |

### Complementary — tooling, not environments

These sit *on top of* the environments and are worth knowing about, but there is
nothing to teleoperate in them:

| package | version | what it is |
|---|---|---|
| `robomimic` | 0.3.0 | Policy learning + demonstration datasets over robosuite. LIBERO pulls 0.2.0. |
| `mimicgen` | 1.0.0 | Generates demonstrations from a handful of human ones, robosuite-based. A natural fit for this rig's output. |
| `lerobot` | 0.6.1 | The training framework whose dataset format this project already writes. Requires Python ≥3.12 and `numpy<2.3`. |

## Why the adapters are thin

These benchmarks already take the action this project produces. robosuite's
BASIC/OSC controller and LIBERO want a Cartesian delta plus a gripper bit;
Meta-World and Fetch want the same thing minus the wrist:

```
robosuite / LIBERO   [dx, dy, dz, drx, dry, drz, grip]   7-D   (our dyaw -> drz)
Meta-World / Fetch   [dx, dy, dz,                grip]   4-D   (dyaw unused)
Skillcrane sticks     [dx, dy, dz,          dyaw, grip]
```

So `benchmarks/` is a remapping layer, not a re-implementation. Everything
routes through `TeleopEnv` (`benchmarks/base.py`), which `Game` already
satisfied — the native arm goes through the same path, which is what keeps the
adapters honest.

## Three landmines, all found by running it

**1. mujoco ≥ 3.12 breaks every robosuite-based benchmark.** robosuite's
mujoco-py compatibility shim does:

```python
assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
```

`joint_type` is a `numpy.int32`. Tuple membership compares *element == needle*,
i.e. `enum == np.int32`, and that reflected direction started returning False:

| mujoco | `jt == HINGE` | `HINGE == jt` | `jt in (…)` |
|---|---|---|---|
| 3.3.0 – 3.11.0 | True | True | **True** |
| 3.12.0 | True | **False** | **False** |

Every robosuite env then dies in `_setup_references` with a bare
`AssertionError`. Bisected to 3.12.0 exactly; hence `mujoco<3.12`. The core
Skillcrane suite passes on both sides of that line (measured at 103 tests, 8.5 s
either way; the suite is 230 tests now and the pin is unchanged),
so sharing an environment costs nothing but the pin.

**2. `egl_probe` won't build on CMake 4.** LIBERO → `robomimic==0.2.0` →
`egl_probe`, whose `CMakeLists.txt` declares `cmake_minimum_required(<3.5)`.
CMake 4 removed that compatibility:

```
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

Fix: `cmake<4` (what `environment-libero.yml` does) or
`export CMAKE_POLICY_VERSION_MINIMUM=3.5` before `pip install`. Notably HF's
`hf-egl-probe` fork builds fine — it's only the legacy transitive dep that fails.

**3. Importing robosuite breaks gymnasium's renderer.** robosuite sets
`MUJOCO_GL=cgl` on macOS at import time. gymnasium's MuJoCo renderer only knows
`glfw`/`egl`/`osmesa`, so any Fetch or Meta-World env created afterwards dies
with `KeyError: 'cgl'`. This only shows up when two suites are used in one
process — which is exactly what the test suite does.
`benchmarks/gym_env.py` clears the variable when it holds a backend gymnasium
cannot use.

## Gripper conventions were measured, not assumed

They disagree, and getting one backwards makes a task quietly unsolvable while
looking like bad teleop:

| suite | `action[-1]` that closes | evidence |
|---|---|---|
| robosuite (Panda) | **+1** | finger joints 0.0417 → 0.0012 |
| Meta-World | **+1** | gripper obs 1.000 → 0.296 |
| Fetch | **−1** | on +1 the fingers *open* 0.000 → 0.100 |
| LIBERO | **+1** | robosuite Panda gripper |

`tests/test_benchmarks.py::test_gripper_signs_are_documented_per_suite` pins
these so a refactor can't quietly flip one.

## Installing

**Shared suites** (robosuite, Meta-World, Fetch):

```sh
pip install -r requirements.txt -r requirements-benchmarks.txt
# or:  CONDA_SUBDIR=osx-arm64 conda env create -f environment-benchmarks.yml
```

**LIBERO** needs its own environment because it pins `robosuite==1.4.0`, whose
controller API is not the 1.5 one our adapter uses:

```sh
CONDA_SUBDIR=osx-arm64 conda env create -f environment-libero.yml
conda activate skillcrane-libero
printf 'N\n' | python -c "import libero.libero"    # one-time config prompt
python main.py --env libero:libero_spatial/0
```

That first import writes `~/.libero/config.yaml` and downloads assets to
`~/.cache/libero`. It **prompts on stdin**, so it hangs a non-interactive
script — answer it once up front. `benchmarks/libero_env.py` raises with these
instructions rather than hanging if the config is missing.

`benchmarks/registry.py` version-checks robosuite, so inside the LIBERO
environment the `robosuite` suite correctly reports as unavailable instead of
failing later at `make()`.

**RoboCasa** (untested here):

```sh
pip install git+https://github.com/robocasa/robocasa.git
python -m robocasa.scripts.download_kitchen_assets      # several GB, once
python main.py --env robocasa:PnPCounterToCab
```

## Per-suite notes

- **robosuite** — `Lift`, `Stack`, `PickPlaceCan`, `Door`, `NutAssemblyRound`.
  Runs at 20 Hz; `main.py` paces each env by its own `control_dt` rather than
  assuming the native 100 Hz.
- **Meta-World / Fetch** — no wrist joint, so `dyaw` is still recorded (keeping
  one action schema) but does nothing. That's the benchmark, not a bug.
- **LIBERO** — task ids take an index: `libero_spatial/3`. Tasks carry a
  natural-language description, which is written into the LeRobot task table.
- **RoboCasa** — kitchen tasks that *are* robosuite environments: importing
  `robocasa` registers them in robosuite's registry, so it reuses
  `benchmarks/robosuite_env.py` whole. Two things are guesses rather than
  measurements, because the package is not installed on this machine: the mobile
  robot's name (the factory tries `PandaOmron`, then `PandaMobile`, which is
  what it was called in v0.1) and the default task. Its kitchens have no
  `agentview`, which is why the adapter now *probes* for a main camera instead
  of naming one.
- **gym-aloha** — installs and steps fine, but it is 14-D bimanual *joint
  position* control, not a Cartesian delta. One gamepad can't drive it without
  an IK layer, so the registry lists it as not teleoperable with the reason
  attached rather than leaving the next person to rediscover it.

## Recording

Identical for every suite. The recorder takes its schema from the environment,
so observation width and task string follow whatever you're driving:

```sh
python main.py --env robosuite:Lift --record runs/lift
```

## Test coverage

`tests/test_benchmarks.py` runs in all three worlds and skips what isn't there:

| environment | result |
|---|---|
| core only (no benchmarks) | 242 passed, 17 skipped |
| + robosuite / Meta-World / Fetch | 125 passed, 1 skipped † |
| LIBERO env | 117 passed, 9 skipped † |

† measured when the adapters landed and **not re-measured since** — none of these
suites is installed on this machine, so the two rows below the first are the last
numbers actually observed, not current ones. The core row is current.
