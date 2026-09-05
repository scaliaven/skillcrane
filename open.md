# Open items

What is known to be undone, and what was deliberately not done. Everything here
was raised while building something else and consciously left; nothing in this
file is broken, and the suite is green with and without the optional deps.

Last reviewed 2026-09-05, at commit `0431167`.

Two rules for this file: an item leaves when it is done or when it is decided
against (moving to **Decided against**, not deleted), and anything with a measured
number cites where the number came from. Finished work lives in git and in
`DEVLOG.md`, not here.

---

## 1. Generate demonstrations at scale

**Why it is the top item.** The rig can now collect demonstrations, prove they
are trainable (`--policy replay:`), and score a policy (`--eval`). What it cannot
do is produce *enough* of them — a human will not sit and drive 500 rounds on a
gamepad, and 50 human episodes is not an imitation-learning dataset.

**What it takes.** `--eval N --record` already writes one episode per seed
through the same path a played round uses, so the collection half exists. What is
missing is variety: the cube spawns on an arc but everything else is fixed. Add
randomisation of cube size, friction and colour, light position, and small camera
jitter, behind a flag so the deterministic path the tests rely on is unaffected.

**What it unlocks.** The comparison worth running: does 50 human episodes beat
5000 scripted ones, and at what ratio do they cross? That question is one script
away and currently unanswerable.

## 2. Install the benchmark suites and re-measure

`registry.suites()` returns `('native',)` on this machine, so both switch rings
are one entry long, `--eval` can only run the native arm, and two rows of every
test-count table are marked `†` — last observed when the adapters landed, not
current.

`pip install -r requirements-benchmarks.txt` adds robosuite, Meta-World and
Fetch. Then: confirm live suite/task switching with `[` `]` and `,` `.`, re-run
the suite in that environment, and drop the `†` from the rows in `BENCHMARKS.md`
and `DEVLOG.md` that it makes current again.

Watch the two constraints that bite here — `mujoco<3.12` for anything
robosuite-based, and never install LIBERO beside the robosuite backend
(`CLAUDE.md`, constraints 10 and 11).

## 3. Run a real policy through `LeRobotPolicy`

`policy.LeRobotPolicy` loads an ACT checkpoint and has **never been executed** —
`lerobot` is not installed here. It is written against the documented
`from_pretrained` / `select_action` interface, the same standing as the RoboCasa
adapter, and both assumptions are stated in its docstring.

Until it runs once end to end, the "learned" third of the policy layer is
unverified. The check is cheap once lerobot is installed: train on a handful of
scripted episodes, then `--eval 12 --policy act:<checkpoint>` and read the
success rate. It does not need to be a *good* policy to prove the wiring.

## 4. Video encoding for recorded frames

`--record` writes one PNG per camera per control tick. A 90 s round at 320x240 is
about 9,000 files and ~360 MB per camera; rendering, not physics, is what makes
`--eval --record` roughly 7x the wall time of `--eval` (one 320x240 view costs
2366 µs against a 107 µs control tick — measured, `DEVLOG.md` Stage 6).

LeRobot v2.1 supports an mp4 per view with `video_path` in `info.json`, which is
what the format expects for volume. `recorder.py` already writes `"video_path":
None` and would need an encoder beside `_write_frames`.

Not urgent: it is a volume problem, and the volume is not there yet. It becomes
urgent the moment item 1 lands.

## 5. Record each benchmark's native action vector

Every backend receives the same 5-D teleop action, which the adapter then remaps
to whatever that suite wants — 7-D for robosuite and LIBERO, 4-D for Meta-World
and Fetch. Only the 5-D one is recorded.

A dataset collected on robosuite therefore cannot be replayed through robosuite's
own API, only through this rig's adapter. Logging the exact array passed to
`env.step` as a second column would make the recordings portable, and would make
a wrong remapping visible in the data rather than only in the behaviour.

## 6. Decide what `arm_game.py` is for

The original single-file prototype, 19 KB at the repo root. Nothing imports it,
no test covers it, and it is absent from the architecture in `CLAUDE.md`.
`README.md` says it is "kept for reference", so its presence is deliberate — but
it is the one file in the tree that can rot without anything noticing.

Either keep it and say so somewhere a reader will look, or move it under a
`prototype/` directory, or delete it and let `DEVLOG.md` Stage 0 be the record.
A decision, not a task.

## 7. Gravity compensation for the static droop

The arm sits ~1.4 mm below its commanded pose at rest. That is gravity droop
against a position servo, and no amount of gain fixes it — removing it needs
gravity compensation in the controller (`DEVLOG.md`, Known limits).

Small, self-contained, and worth almost nothing to the data pipeline: 1.4 mm is
well inside the 10 mm tracking budget and far inside the tolerances the grasp
depends on. Listed so it stops being rediscovered.

---

## Decided against

Recorded so they are not re-litigated. Each was raised by a review, measured, and
declined for the reason given.

**Reusing one environment across `--eval` seeds.** Rebuilding costs 14.5 ms per
seed — about 13% of a non-recording eval, 0.2% once `--record` is on. A shared
env needs a reseed hook on `Game` and reintroduces exactly the cross-episode
contamination the per-seed world exists to prevent: score, clock and cube would
carry into the next seed. Revisit only if seed counts grow large *and* recording
is off.

**Optimising the per-tick policy path.** Profiled at 4% of a rollout against
`mj_step`'s 43%. Every candidate fix together saves under 1.5 ms of a 112 ms
episode — rebuilding the 3-element `full` array each tick is 0.28 ms per episode.
The lever for eval throughput is `--record-size` and the view count, not the code.

**Making `reward` required on `EpisodeRecorder.add`.** The argument was real: an
omitted reward writes `"success": false`, a positive false claim rather than a
missing field. But the fix adds `reward=0.0` to ~16 test call sites that are
about image columns and episode numbering, where `0.0` is the honest value, and
all four production call sites already pass it. Reconsider if a fifth call site
appears that does not.
