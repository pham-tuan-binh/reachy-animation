# AGENTS.md

Instructions for AI coding agents (and the humans driving them) working on `reachy-animation`.
Read this and `README.md` before writing code.

## What this project is

The control loop that animates Reachy Mini. It turns two high-level inputs, **speech audio** and **motions to
play**, into one full-body pose per control tick. It does **not** talk to the robot. A caller's `sink` does
that, so keep robot I/O out of the core.

## Map

```
src/reachy_animation/
  pose.py       9-DoF pose vector <-> SDK set_target arguments           (numpy only)
  motion.py     Motion protocol, Clip (keyframes), Breathing (idle)      (numpy only)
  speech.py     SpeechSway: speech audio -> loudness envelopes -> head offset   (internal, numpy only)
  animator.py   Animator: layers + crossfades, hooks, tick(), the fps thread
  sim.py        CLI: scripted scenario -> simulated ticks -> MuJoCo -> mp4   (sim extra only)
tests/          mirrors src/, runs without the sim extra
```

Per tick, bottom to top: idle motion, the played motion (entered and left by crossfade), and speech sway added
on top. Public API = the names in `__init__.py` `__all__`. Keep it small, and add a name only when a caller
can't do without it.

## Invariants: don't break these

1. **`tick(t)` is deterministic** in the hook calls made so far and `t`. No wall-clock reads inside it, no
   randomness beyond seeded init. The real-time thread and offline simulation must run identical code.
   Anything time-based goes through the injected `clock`.
2. **One rate in the library.** `Animator.fps` is both the control rate and the animation frame rate.
   Ingested clips keep the rate they were recorded at (`Clip.fps`) until `play` resamples them to the
   animator's fps (`Clip.resample`, anti-aliased). Motions are still sampled by *time* (`sample(t_seconds)`),
   so procedural motions need no resampling. Never index a motion by tick count, and never assume a
   source's fps equals the animator's.
3. **The output never jumps.** Every source change goes through `_Crossfade` between *live* sources.
   A new transition during a blend blends from the blend. Don't snap, and don't freeze the old pose.
4. **Hooks are thread-safe and cheap.** They take `_lock` briefly. Motion hooks are queued and applied on the
   next tick, at that tick's time. Nothing slow (I/O, model inference) runs under the lock or in `tick`.
5. **Core stays light.** `pose`, `motion`, `speech` and `animator` import only numpy and the stdlib.
   MuJoCo, the SDK, imageio-ffmpeg and PIL belong in `sim.py`.
6. **Units:** metres and radians everywhere. Euler angles are extrinsic xyz, matching the SDK's
   `create_head_pose` and the motion generator's trajectories. Antenna right droops with negative angles,
   left with positive.
7. **Reachability is out of scope.** The animator doesn't clamp or project. The robot output stage does.

## Common tasks

- **New motion source** (procedural, a generator stream): implement `Motion` (`name`, `duration`,
  `sample(t)`, holding the last pose past `duration`; `math.inf` for open-ended). If it can be built from
  keyframes, add a `Clip` constructor instead of a new class.
- **Streamed generator output:** a `Motion` whose frames arrive over time. `sample` must stay valid for any
  `t` (hold the latest frame while waiting), and it must be safe to append from another thread.
- **Tuning the speech sway:** the constants at the top of `speech.py`. Check by ear and eye with
  `python -m reachy_animation.sim --speech` and a speech-only render.
- **Robot link:** a sink `lambda pose: robot.set_target(*to_target(pose))`, with reachability projection
  in front of it. It lives in the caller, or in a separate module that keeps the core free of SDK imports.

## Validate

Unit tests are the floor. Motion changes also need a look in simulation.

```bash
uv sync --extra sim
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python -m reachy_animation.sim --out out/check.mp4 --duration 8 \
    --speech out/speech.wav@0.5 --play <clip.json>@2.0
```

- Watch the mp4: it has sound and a caption showing what's playing.
- For numbers, drive the animator with an injected clock (as `sim.main` does), collect `tick(t)` poses, and
  check for jumps with `np.abs(np.diff(poses, axis=0)).max(0)`.
- The sim logs unreachable frames. Compare against the clip alone before blaming your change, since some
  library clips are unreachable by themselves.
- Clips to test with: the emotion library (`pollen-robotics/reachy-mini-emotions-library` on the HF Hub, 50 fps)
  and motion-generator output (`../reachy-motion-generator/runs/*/motions/*.json`, 25 fps).

## Code conventions

- Match the surrounding code. Short names that carry meaning, typed signatures (mypy strict passes), and
  one-line docstrings on public APIs.
- Comments explain *why*, never *what*.
- Log with the module `logger` and lazy `%` args. Never swallow errors silently. The one broad `except` is the
  sink call in the tick thread, where a failing robot link must not kill the loop.
- Tests exercise behaviour through the public API (plus `speech.SpeechSway` directly). A bug fix comes with
  a regression test, and a feature with at least a happy-path test.
- Minimal diffs, no dead code, no speculative parameters. Keep `README.md` in sync with any API change.
- The package is installed from the git URL (`pip install git+https://github.com/pham-tuan-binh/reachy-animation`),
  so `main` must always be installable and green.
