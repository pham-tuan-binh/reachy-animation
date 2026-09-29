"""Speech sway: the loudness of the audio being spoken, turned into an additive head offset.

Fed audio is laid on a playback timeline (chunks play back to back, like a speaker queue) and reduced
to loudness envelopes on arrival. Sampling is then a cheap interpolation at any tick rate, and the sway
oscillators run on continuous time instead of stepping once per audio hop.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from reachy_animation.pose import Pose, zero_pose

HOP_S = 0.01
DB_FLOOR = -50.0  # at or below: no sway
DB_CEIL = -22.0  # at or above: full sway
LOUDNESS_GAMMA = 0.9
SWAY_GAIN = 1.5
SWAY_ATTACK_S = 0.05
SWAY_RELEASE_S = 0.25
ACCENT_ATTACK_S = 0.015
ACCENT_RELEASE_S = 0.08
ACCENT_PITCH_RAD = math.radians(3.0)  # head dips on syllable onsets
HISTORY_S = 30.0
_RELEASES_S = np.array([SWAY_RELEASE_S, ACCENT_RELEASE_S])
# (pose index, frequency Hz, amplitude m or rad): tuned in the conversation app's head wobbler.
SWAY = (
    (0, 0.35, 0.0045),
    (1, 0.45, 0.00375),
    (2, 0.25, 0.00225),
    (3, 1.3, math.radians(2.25)),
    (4, 2.2, math.radians(4.5)),
    (5, 0.6, math.radians(7.5)),
)


def _to_mono_float(pcm: NDArray[np.generic]) -> NDArray[np.float32]:
    """Convert int or float PCM, mono or (samples, channels), to float32 mono in [-1, 1]."""
    audio = np.asarray(pcm)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[1] <= audio.shape[0] else audio.mean(axis=0)
    if np.issubdtype(audio.dtype, np.integer):
        return (audio / float(np.iinfo(audio.dtype).max)).astype(np.float32)
    return audio.astype(np.float32)


class SpeechSway:
    """Loudness envelopes on a playback timeline, sampled into a head-sway offset."""

    def __init__(self, latency_s: float = 0.0, seed: int = 7) -> None:
        """Delay the sway by ``latency_s`` to match the audio output path."""
        self.latency_s = latency_s
        self._phases = np.random.default_rng(seed).uniform(0.0, 2 * math.pi, len(SWAY))
        # Columns: time, sway envelope, fast envelope (its excess over sway marks syllable onsets).
        self._envelopes = np.empty((0, 3))
        self._state = np.zeros(2)
        self._end = -math.inf
        self._carry = np.empty(0, dtype=np.float32)

    @property
    def playing_until(self) -> float:
        """Time at which the fed audio finishes playing."""
        return self._end

    def feed(self, pcm: NDArray[np.generic], sample_rate: int, at: float) -> None:
        """Schedule ``pcm`` to play at ``at``, or right after the audio already fed if that ends later."""
        audio = _to_mono_float(pcm)
        if audio.size == 0:
            return
        start = max(at, self._end)
        if start > self._end:
            # Write the release into the gap, else sampling would interpolate linearly across the silence.
            last = self._envelopes[-1, 0] if self._envelopes.size else start
            times = np.append(last + np.linspace(0.0, 5 * SWAY_RELEASE_S, 21)[1:], start)
            times = times[times <= start]
            self._append(np.column_stack([times, self._state * np.exp(-(times[:, None] - last) / _RELEASES_S)]))
            self._state = self._envelopes[-1, 1:].copy()
            self._carry = np.empty(0, dtype=np.float32)
        hop = max(1, round(HOP_S * sample_rate))
        stream_start = start - self._carry.size / sample_rate
        samples = np.concatenate([self._carry, audio])
        n_hops = samples.size // hop
        self._carry = samples[n_hops * hop :]
        self._end = start + audio.size / sample_rate
        if n_hops == 0:
            return

        frames = samples[: n_hops * hop].reshape(n_hops, hop)
        db = 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-9)
        loudness = np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0) ** LOUDNESS_GAMMA
        rows = np.empty((n_hops, 3))
        rows[:, 0] = stream_start + (np.arange(n_hops) + 1) * hop / sample_rate
        dt = hop / sample_rate
        rates = [
            (1 - math.exp(-dt / SWAY_ATTACK_S), 1 - math.exp(-dt / SWAY_RELEASE_S)),
            (1 - math.exp(-dt / ACCENT_ATTACK_S), 1 - math.exp(-dt / ACCENT_RELEASE_S)),
        ]
        state = self._state
        for k, level in enumerate(loudness):
            for j, (attack, release) in enumerate(rates):
                state[j] += (level - state[j]) * (attack if level > state[j] else release)
            rows[k, 1:] = state
        self._append(rows)

    def interrupt(self, at: float) -> None:
        """Stop the speech at ``at``: drop audio scheduled after it and let the sway decay."""
        if self._envelopes.size == 0 or at >= self._end:
            return
        self._state = self._envelopes_at(at)
        self._envelopes = self._envelopes[self._envelopes[:, 0] < at]
        self._append(np.array([[at, *self._state]]))
        self._end = at
        self._carry = np.empty(0, dtype=np.float32)

    def loudness(self, t: float) -> float:
        """Smoothed speech loudness in [0, 1] driving the sway at ``t``."""
        return float(self._envelopes_at(t - self.latency_s)[0])

    def sample(self, t: float) -> Pose:
        """Additive head offset at ``t``."""
        offset = zero_pose()
        sway, fast = self._envelopes_at(t - self.latency_s)
        if sway <= 0.0 and fast <= 0.0:
            return offset
        audio_t = t - self.latency_s
        for (index, frequency, amplitude), phase in zip(SWAY, self._phases):
            offset[index] = SWAY_GAIN * sway * amplitude * math.sin(2 * math.pi * frequency * audio_t + phase)
        offset[4] += ACCENT_PITCH_RAD * max(0.0, fast - sway)
        return offset

    def _envelopes_at(self, t: float) -> NDArray[np.float64]:
        """Interpolated envelopes at audio time ``t``, decaying past the last known point."""
        env = self._envelopes
        if env.size == 0 or t < env[0, 0]:
            return np.zeros(2)
        if t >= env[-1, 0]:
            decay = np.exp(-(t - env[-1, 0]) / _RELEASES_S)
            tail: NDArray[np.float64] = env[-1, 1:] * decay
            return tail
        return np.array([np.interp(t, env[:, 0], env[:, j]) for j in (1, 2)])

    def _append(self, rows: NDArray[np.float64]) -> None:
        env = np.concatenate([self._envelopes, rows])
        self._envelopes = env[env[:, 0] >= env[-1, 0] - HISTORY_S]
