import numpy as np
import pytest

from reachy_animation.speech import SpeechSway

RATE = 24_000


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def _peak(sway: SpeechSway, start: float, end: float) -> float:
    return max(float(np.abs(sway.sample(t)).max()) for t in np.arange(start, end, 1 / 60))


def test_silence_does_not_move_the_head() -> None:
    sway = SpeechSway()
    sway.feed(np.zeros(RATE, dtype=np.int16), RATE, at=0.0)
    assert _peak(sway, 0.0, 1.0) == 0.0


def test_chunks_fed_at_once_play_back_to_back() -> None:
    sway = SpeechSway()
    for chunk in np.array_split(_tone(1.0), 10):
        sway.feed(chunk, RATE, at=0.0)
    assert sway.playing_until == pytest.approx(1.0)
    assert sway.loudness(0.9) > 0.9
    assert sway.loudness(2.5) < 0.01


def test_speech_waits_for_its_playback_time() -> None:
    sway = SpeechSway(latency_s=0.2)
    sway.feed(_tone(1.0), RATE, at=1.0)
    assert _peak(sway, 0.0, 1.2) == 0.0
    assert _peak(sway, 1.3, 2.0) > 0.01


def test_interrupt_drops_unplayed_audio() -> None:
    sway = SpeechSway()
    sway.feed(_tone(3.0), RATE, at=0.0)
    sway.interrupt(at=1.0)
    assert sway.playing_until == 1.0
    assert sway.loudness(0.99) > 0.9
    assert sway.loudness(2.5) < 0.01


def test_sway_decays_through_a_gap_between_utterances() -> None:
    sway = SpeechSway()
    sway.feed(_tone(0.5), RATE, at=0.0)
    sway.feed(_tone(0.5), RATE, at=3.0)
    assert sway.loudness(2.9) < 0.01
    assert sway.loudness(3.4) > 0.9
