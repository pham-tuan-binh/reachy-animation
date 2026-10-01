"""Make Reachy Mini dance to the music its microphone hears: groove on the beat, bust a move on the drop.

  python examples/dance_to_music.py                  # then play some music near the robot (ctrl-C to quit)
  python examples/dance_to_music.py --intensity 3    # go wilder (1 is subtle, the default is 2)

A beat tracker finds the tempo and where the beats fall (onsets from spectral flux, tempo from their
autocorrelation), and an idle motion phase-locked to that grid nods on every beat and sways across bars, a little
ahead of the beat so the motors land on it. The mic audio also goes to ``feed_speech`` for a loudness wobble on top.
When the music gets much louder than it has been lately (the drop after a breakdown), a random clip from Pollen's
dance library plays, then the robot goes back to grooving. The mic is shared through the daemon's media backend, so
other apps keep it too.
"""

import argparse
import logging
import math
import random
import threading
import time
from collections import deque
from functools import partial

import numpy as np
from numpy.typing import NDArray
from reachy_mini import ReachyMini

from reachy_animation import Animator, Breathing, Pose, clips, to_target

logger = logging.getLogger(__name__)

DANCES = [
    "chicken_peck",
    "chin_lead",
    "dizzy_spin",
    "grid_snap",
    "groovy_sway_and_roll",
    "head_tilt_roll",
    "interwoven_spirals",
    "jackson_square",
    "neck_recoil",
    "pendulum_swing",
    "polyrhythm_combo",
    "sharp_side_tilt",
    "side_glance_flick",
    "side_peekaboo",
    "side_to_side_sway",
    "simple_nod",
    "stumble_and_recover",
    "uh_huh_tilt",
    "yeah_nod",
]
# Listening
HOP_S = 0.01
WINDOW = 512  # 32 ms at 16 kHz: short enough to keep onsets sharp
HISTORY_S = 8.0  # onsets kept for the tempo estimate
MIN_HISTORY_S = 3.0
ESTIMATE_EVERY_S = 0.25
BPM_RANGE = (70.0, 180.0)
BPM_PRIOR = 120.0  # ties between half and double tempo go to the one nearer this
LOCK_ACF = 0.3  # weaker periodicity than this (normalized autocorrelation) is not a beat
MUSIC_FLOOR_DB = -60.0  # quieter (RMS, dBFS) is silence: no beat, no drop
DROP_DB = 5.0  # the drop: this much louder (over ~0.5 s) than over the last ~8 s
WARMUP_S = 8.0  # no drop until the music has been on this long
DROP_COOLDOWN_S = 6.0
LEAD_S = 0.1  # move this far ahead of the heard beat: mic buffering plus the motors' lag
# Grooving, at intensity 1
PLL_TAU_S = 0.5  # how fast the groove's phase pulls onto a new beat grid
FADE_S = 1.0  # groove fade in on lock, out on losing the beat
NOD_PITCH_RAD = math.radians(8.0)  # head dips on every beat
NOD_Z_M = 0.004
ROLL_RAD = math.radians(6.0)  # side to side over two beats
BODY_YAW_RAD = math.radians(8.0)  # body swings over a bar of four
ANTENNA_RAD = math.radians(25.0)  # antennas flick on every beat
SPEECH_SWAY = 0.25  # the loudness wobble, per unit of intensity: the beat should lead, not the wobble


class Groove:
    """Idle motion phase-locked to a beat grid: a nod on every beat, a sway across two, a body swing across four."""

    name, duration = "groove", math.inf

    def __init__(self, intensity: float) -> None:
        self.intensity = intensity
        self._breathing = Breathing()
        self._grid: tuple[float, float] | None = None  # (animator time of a beat, period s); swapped whole
        self._t: float | None = None
        self._phase = 0.0  # in beats
        self._period = 0.5
        self._level = 0.0

    def lock(self, grid: tuple[float, float] | None) -> None:
        """Follow the beat grid ``(time of a beat, period)``, or fade out with ``None``."""
        self._grid = grid

    def sample(self, t: float) -> Pose:
        # Called once per tick. The phase is integrated, not read off the grid, so a grid update never jumps the pose.
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        grid = self._grid
        if grid is not None:
            anchor, self._period = grid
            error = ((t - anchor) / self._period - self._phase + 0.5) % 1.0 - 0.5
            self._phase += error * min(1.0, dt / PLL_TAU_S)
        self._phase += dt / self._period
        self._level += ((grid is not None) - self._level) * min(1.0, dt / FADE_S)

        pose = self._breathing.sample(t)
        gain = self.intensity * self._level
        hit = ((1.0 + math.cos(2 * math.pi * self._phase)) / 2) ** 4  # peaks on the beat
        pose[2] -= gain * NOD_Z_M * hit
        pose[3] += gain * ROLL_RAD * math.sin(math.pi * self._phase)
        pose[4] += gain * NOD_PITCH_RAD * hit
        pose[6] -= gain * ANTENNA_RAD * hit
        pose[7] += gain * ANTENNA_RAD * hit
        pose[8] += gain * BODY_YAW_RAD * math.sin(math.pi * self._phase / 2)
        return pose


class Ears:
    """Tempo and beat phase of a mono audio stream, plus drops in its loudness."""

    def __init__(self, sample_rate: int) -> None:
        self.hop = round(HOP_S * sample_rate)
        self.time = 0.0  # stream time: seconds of audio heard
        self.beat: tuple[float, float] | None = None  # (stream time of the latest beat, period s), None if unsure
        self._band = np.fft.rfftfreq(WINDOW, 1.0 / sample_rate) >= 30.0
        self._taper = np.hanning(WINDOW)
        self._buffer = np.zeros(WINDOW, dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._spectrum: NDArray[np.float64] | None = None
        self._flux: deque[float] = deque(maxlen=round(HISTORY_S / HOP_S))
        self._next_estimate = 0.0
        self._short = self._long = 0.0  # mean square, over ~0.5 s and ~8 s
        self._music_since = math.inf
        self._last_drop = -math.inf

    def hear(self, audio: NDArray[np.float32]) -> bool:
        """Feed mono audio, updating ``beat``; return whether the drop hit."""
        samples = np.concatenate([self._pending, audio])
        n_hops = samples.size // self.hop
        self._pending = samples[n_hops * self.hop :]
        drop = False
        for k in range(n_hops):
            hop = samples[k * self.hop : (k + 1) * self.hop]
            self._buffer = np.concatenate([self._buffer[self.hop :], hop])
            self.time += HOP_S
            # Log magnitudes make the flux independent of how loud the mic hears the music.
            spectrum = np.log10(np.abs(np.fft.rfft(self._buffer * self._taper))[self._band] ** 2 + 1e-10)
            if self._spectrum is not None:
                self._flux.append(float(np.maximum(spectrum - self._spectrum, 0.0).sum()))
            self._spectrum = spectrum

            power = float(np.mean(hop * hop))
            self._short += (power - self._short) * HOP_S / 0.5
            self._long += (power - self._long) * HOP_S / HISTORY_S
            if _db(self._long) < MUSIC_FLOOR_DB:
                self._music_since = math.inf
            elif _db(self._short) > MUSIC_FLOOR_DB:
                self._music_since = min(self._music_since, self.time)
            if (
                self.beat is not None
                and _db(self._short) - _db(self._long) > DROP_DB
                and self.time - self._music_since > WARMUP_S
                and self.time - self._last_drop > DROP_COOLDOWN_S
            ):
                drop = True
                self._last_drop = self.time
            if self.time >= self._next_estimate:
                self._next_estimate = self.time + ESTIMATE_EVERY_S
                self.beat = self._estimate()
        return drop

    def _estimate(self) -> tuple[float, float] | None:
        """The beat grid that best explains the recent onsets, if the music is on and periodic enough."""
        if len(self._flux) * HOP_S < MIN_HISTORY_S or self.time - self._music_since < MIN_HISTORY_S:
            return None
        flux = np.array(self._flux)
        onsets = flux - flux.mean()
        acf = np.correlate(onsets, onsets, "full")[onsets.size - 1 :] / (onsets @ onsets + 1e-12)
        lo, hi = round(60.0 / BPM_RANGE[1] / HOP_S), round(60.0 / BPM_RANGE[0] / HOP_S)
        lags = np.arange(lo, hi + 1)
        prior = np.exp(-0.5 * (np.log2(60.0 / (lags * HOP_S) / BPM_PRIOR) / 0.6) ** 2)
        lag = int(lags[np.argmax(acf[lags] * prior)])
        if acf[lag] < LOCK_ACF:
            return None
        a, b, c = acf[lag - 1], acf[lag], acf[lag + 1]
        period = lag + 0.5 * (a - c) / (
            a - 2 * b + c + 1e-12
        )  # parabolic peak: sub-hop tempo, so the grid doesn't drift
        # Phase: the offset whose comb of beats, walking back from now, lands on the most onset energy.
        beats = np.arange(int((flux.size - lag) / period)) * period
        scores = [flux[(flux.size - 1 - phase - beats).astype(int)].sum() for phase in range(lag)]
        return self.time - int(np.argmax(scores)) * HOP_S, period * HOP_S


def _db(power: float) -> float:
    return 10.0 * math.log10(power + 1e-12)


def main() -> None:
    """Listen through the robot's mic and dance until ctrl-C."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intensity", type=float, default=2.0, help="scales the groove and the wobble (1 = subtle)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Download the dances in the background now, so the first drop doesn't wait on the Hub.
    threading.Thread(target=lambda: [clips.get(name) for name in DANCES], daemon=True).start()

    with ReachyMini() as robot:  # the default media backend: the mic is shared through the daemon
        groove = Groove(args.intensity)
        animator = Animator(idle=groove, speech_sway=SPEECH_SWAY * args.intensity, speech_sway_in_motion=0.5)
        animator.on_pose(lambda pose: robot.set_target(*to_target(pose)))
        animator.start()
        robot.media.start_recording()
        rate = robot.media.get_input_audio_samplerate()
        ears = Ears(rate)
        bpm = None
        print("Listening. Play some music! (ctrl-C to quit)")
        try:
            while True:
                chunk = robot.media.get_audio_sample()  # (samples, 2) float32, or None after a 20 ms wait
                if chunk is None:
                    continue
                audio = chunk.mean(axis=1)
                animator.feed_speech(audio, rate)
                drop = ears.hear(audio)
                # The animator's default clock is time.monotonic: map the heard beat onto it, ahead by LEAD_S.
                if ears.beat is None:
                    groove.lock(None)
                else:
                    beat, period = ears.beat
                    groove.lock((time.monotonic() - (ears.time - beat) - LEAD_S, period))
                heard = None if ears.beat is None else round(60.0 / ears.beat[1])
                if heard != bpm and (heard is None or bpm is None or abs(heard - bpm) > 3):
                    logger.info("Beat: %s", f"{heard} bpm" if heard else "lost")
                    bpm = heard
                if drop and animator.playing is None:
                    name = random.choice(DANCES)
                    logger.info("Drop! %s", name)
                    animator.play(partial(clips.get, name))  # fetched off the loop if not cached yet
        except KeyboardInterrupt:
            pass
        finally:
            robot.media.stop_recording()
            animator.close()


if __name__ == "__main__":
    main()
