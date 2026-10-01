"""The animator: composes idle motion, played motions and speech sway into one pose per tick.

Layers, bottom to top:
- **idle** (breathing by default) runs whenever nothing is played;
- **played motion** replaces idle, entered and left through a crossfade so no command ever jumps;
- **speech sway** is added on top of whatever the body is doing.

``tick(t)`` is a pure function of the hook calls made so far and ``t``, so the same animator drives the
real-time loop (``start``) and offline simulation (calling ``tick`` yourself). Hooks are thread-safe;
motion hooks take effect on the next tick.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reachy_animation.motion import Breathing, Clip, Motion
from reachy_animation.pose import Pose
from reachy_animation.speech import SpeechSway

logger = logging.getLogger(__name__)


def _smoothstep(t: float, start: float, duration: float) -> float:
    a = 1.0 if duration <= 0 else min(max((t - start) / duration, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


class _Ramp:
    """A value that eases to each new target from wherever it is, so changing it mid-ramp never jumps."""

    def __init__(self, value: float) -> None:
        self.source = self.target = value
        self.start = self.duration = 0.0

    def at(self, t: float) -> float:
        return self.source + (self.target - self.source) * _smoothstep(t, self.start, self.duration)

    def to(self, value: float, t: float, duration: float) -> None:
        self.source = self.at(t)
        self.target = value
        self.start = t
        self.duration = duration


class _Source(Protocol):
    def sample(self, t: float) -> Pose: ...

    def settle(self, t: float) -> _Source: ...

    def moving(self, t: float) -> float: ...


class _Playing:
    """A motion anchored on the animator's clock."""

    def __init__(self, motion: Motion, start: float, idle: bool = False) -> None:
        self.motion = motion
        self.start = start
        self.end = start + motion.duration
        self.idle = idle

    def sample(self, t: float) -> Pose:
        return self.motion.sample(t - self.start)

    def settle(self, t: float) -> _Source:
        return self

    def moving(self, t: float) -> float:
        return 0.0 if self.idle else 1.0


class _Crossfade:
    """Smoothstep blend between two live sources; collapses into the target once done."""

    def __init__(self, source: _Source, target: _Source, start: float, duration: float) -> None:
        self.source = source
        self.target = target
        self.start = start
        self.duration = duration

    def sample(self, t: float) -> Pose:
        w = _smoothstep(t, self.start, self.duration)
        blended: Pose = (1.0 - w) * self.source.sample(t) + w * self.target.sample(t)
        return blended

    def moving(self, t: float) -> float:
        w = _smoothstep(t, self.start, self.duration)
        return (1.0 - w) * self.source.moving(t) + w * self.target.moving(t)

    def settle(self, t: float) -> _Source:
        if t >= self.start + self.duration:
            return self.target.settle(t)
        self.source = self.source.settle(t)
        return self


class Animator:
    """Compose idle, played motions and speech sway into a full-body pose, ``fps`` times a second.

    ``fps`` is both the control rate and the animation frame rate: clips recorded at other rates are resampled
    to it when played, so every tick lands on a frame.
    """

    def __init__(
        self,
        fps: float = 60.0,
        idle: Motion | None = None,
        blend_s: float = 0.4,
        speech_latency_s: float = 0.0,
        speech_sway: float = 1.0,
        speech_sway_in_motion: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an animator; ``clock`` timestamps hooks and schedules ticks (inject one to simulate).

        ``speech_sway`` scales the speech head wobble, and ``speech_sway_in_motion`` is the fraction of it kept while
        a motion plays (faded in and out with the motion's crossfade). Change them live with ``set_speech_sway``.
        """
        self.fps = fps
        self.blend_s = blend_s
        self._sway = _Ramp(speech_sway)
        self._sway_in_motion = _Ramp(speech_sway_in_motion)
        self._idle = idle or Breathing()
        self._speech = SpeechSway(latency_s=speech_latency_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._source: _Source = _Playing(self._idle, 0.0, idle=True)
        self._current: _Playing | None = None
        self._queue: deque[Motion] = deque()
        self._interrupt = False  # leave the current motion on the next tick
        self._generation = 0  # bumped by every play-now and stop; a motion made for an older one is dropped
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._pose_callbacks: list[Callable[[Pose], None]] = []

    def play(self, motion: Motion | Callable[[], Motion], *, queue: bool = False) -> None:
        """Play ``motion`` now, dropping anything playing or queued; with ``queue``, play it after them.

        ``motion`` can also be a function that makes one, such as a request to a motion generator. It runs on a
        background thread and its motion plays when it returns, unless ``play`` or ``stop`` was called since.
        """
        with self._lock:
            if not queue:
                self._generation += 1
            generation = self._generation
        if isinstance(motion, Motion):
            self._schedule(motion, queue, generation)
        else:
            threading.Thread(target=self._make, args=(motion, queue, generation), daemon=True).start()

    def stop(self) -> None:
        """Drop the playing and queued motions, and any being made, and blend back to idle."""
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._interrupt = True

    @property
    def playing(self) -> Motion | None:
        """The played motion as of the last tick, ``None`` when idle."""
        with self._lock:
            return self._current.motion if self._current else None

    def feed_speech(self, pcm: NDArray[np.generic], sample_rate: int) -> None:
        """Feed audio as it is sent to the speaker; chunks play back to back."""
        with self._lock:
            self._speech.feed(pcm, sample_rate, at=self._clock())

    def interrupt_speech(self) -> None:
        """Speech playback was cut (barge-in): drop the audio not yet played."""
        with self._lock:
            self._speech.interrupt(self._clock())

    def set_speech_sway(self, sway: float | None = None, *, in_motion: float | None = None) -> None:
        """Change ``speech_sway`` and/or ``speech_sway_in_motion``, easing to the new values over ``blend_s``."""
        with self._lock:
            now = self._clock()
            if sway is not None:
                self._sway.to(sway, now, self.blend_s)
            if in_motion is not None:
                self._sway_in_motion.to(in_motion, now, self.blend_s)

    def tick(self, t: float) -> Pose:
        """Return the pose to command at time ``t``; call with increasing ``t``."""
        with self._lock:
            finished = self._current is not None and t >= self._current.end
            if self._interrupt or finished or (self._current is None and self._queue):
                self._interrupt = False
                self._enter(self._queue.popleft() if self._queue else None, t)
            self._source = self._source.settle(t)
            moving = self._source.moving(t)
            gain = self._sway.at(t) * (1.0 - moving + moving * self._sway_in_motion.at(t))
            pose: Pose = self._source.sample(t) + gain * self._speech.sample(t)
            return pose

    def on_pose(self, callback: Callable[[Pose], None]) -> Callable[[Pose], None]:
        """Call ``callback(pose)`` on every tick of ``start()``; usable as a decorator, and callable more than once."""
        with self._lock:
            self._pose_callbacks.append(callback)
        return callback

    def start(self) -> None:
        """Tick ``fps`` times a second on a background thread, passing each pose to the ``on_pose`` callbacks."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Animator already running; start() ignored")
            return
        self._closing.clear()
        self._thread = threading.Thread(target=self._run, name="reachy-animation", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the tick thread and wait for it to exit."""
        self._closing.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        """Tick on a drift-free grid; after an overrun, skip missed ticks rather than replay them in a burst."""
        period = 1.0 / self.fps
        next_tick = self._clock()
        last_overrun_log = last_error_log = -1.0
        while not self._closing.is_set():
            pose = self.tick(next_tick)
            with self._lock:
                callbacks = tuple(self._pose_callbacks)
            for callback in callbacks:
                try:
                    callback(pose.copy())
                except Exception as e:  # noqa: BLE001 - a failing robot link must not kill the loop
                    if next_tick - last_error_log >= 1.0:  # a dropped link fails every tick: log once a second
                        logger.error("on_pose callback failed: %s", e)
                        last_error_log = next_tick
            next_tick += period
            now = self._clock()
            if now - next_tick > period:
                skipped = int((now - next_tick) / period)
                next_tick += skipped * period
                if now - last_overrun_log >= 1.0:
                    logger.warning("Animator behind schedule, skipped %d ticks", skipped)
                    last_overrun_log = now
            self._closing.wait(max(0.0, next_tick - self._clock()))

    def _make(self, make_motion: Callable[[], Motion], queue: bool, generation: int) -> None:
        try:
            motion = make_motion()
        except Exception as e:  # noqa: BLE001 - a failed generator must not take the animator down
            logger.warning("Could not make a motion to play: %s", e)
            return
        self._schedule(motion, queue, generation)

    def _schedule(self, motion: Motion, queue: bool, generation: int) -> None:
        if isinstance(motion, Clip):
            motion = motion.resample(self.fps)
        with self._lock:
            if generation != self._generation:
                logger.info("Dropped %s: superseded while it was being made", motion.name)
                return
            if not queue:
                self._queue.clear()
                self._interrupt = True
            self._queue.append(motion)

    def _enter(self, motion: Motion | None, t: float) -> None:
        """Crossfade from whatever is showing into ``motion``, or back into idle."""
        if motion is None:
            target = _Playing(self._idle, 0.0, idle=True)  # idle keeps its own phase across interruptions
            if self._current:
                logger.debug("Finished %s, back to idle", self._current.motion.name)
            self._current = None
        else:
            target = self._current = _Playing(motion, t)
            logger.debug("Playing %s (%.2fs)", motion.name, motion.duration)
        self._source = _Crossfade(self._source, target, t, self.blend_s)
