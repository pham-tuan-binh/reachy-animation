import time

import numpy as np
import pytest

from reachy_animation import Animator, Clip
from reachy_animation.pose import zero_pose

FPS = 60.0


class Hold:
    """Constant pose for ``duration`` seconds."""

    def __init__(self, name: str, pose: np.ndarray, duration: float) -> None:
        self.name, self.pose, self.duration = name, pose, duration

    def sample(self, t: float) -> np.ndarray:
        return self.pose.copy()


class Still:
    name = "still"
    duration = float("inf")

    def sample(self, t: float) -> np.ndarray:
        return zero_pose()


def _pitch(value: float) -> np.ndarray:
    pose = zero_pose()
    pose[4] = value
    return pose


def _run(animator: Animator, start: float, end: float) -> np.ndarray:
    return np.array([animator.tick(k / FPS) for k in range(int(start * FPS), int(end * FPS))])


def test_played_motion_blends_in_then_back_to_idle() -> None:
    animator = Animator(idle=Still(), blend_s=0.4)
    animator.play(Hold("nod", _pitch(0.3), 1.0))
    poses = _run(animator, 0.0, 2.0)
    assert poses[0, 4] == 0.0
    np.testing.assert_allclose(poses[int(0.5 * FPS) : int(1.0 * FPS), 4], 0.3)
    assert animator.playing is None
    np.testing.assert_allclose(poses[-1], 0.0)


def test_transitions_never_jump() -> None:
    animator = Animator(idle=Still(), blend_s=0.4)
    animator.play(Hold("up", _pitch(0.4), 2.0))
    first = _run(animator, 0.0, 0.2)
    animator.play(Hold("down", _pitch(-0.4), 2.0))  # interrupts mid-blend
    second = _run(animator, 0.2, 1.0)
    animator.stop()
    third = _run(animator, 1.0, 2.0)
    steps = np.abs(np.diff(np.concatenate([first, second, third])[:, 4]))
    # Smoothstep over 0.4 s peaks at 1.5x the mean slope: 1.5 * 0.8 rad / 24 ticks.
    assert steps.max() < 1.5 * 0.8 / (0.4 * FPS) + 1e-9


def test_queued_motions_play_in_order() -> None:
    animator = Animator(idle=Still(), blend_s=0.0)
    animator.play(Hold("a", _pitch(0.1), 0.5))
    animator.play(Hold("b", _pitch(0.2), 0.5), queue=True)
    names = []
    for k in range(int(1.5 * FPS)):
        animator.tick(k / FPS)
        names.append(animator.playing.name if animator.playing else None)
    assert names[0] == "a" and names[45] == "b" and names[-1] is None


def test_play_replaces_the_queue() -> None:
    animator = Animator(idle=Still(), blend_s=0.0)
    animator.play(Hold("a", _pitch(0.1), 0.5))
    animator.play(Hold("b", _pitch(0.2), 0.5), queue=True)
    animator.play(Hold("c", _pitch(0.3), 0.5))
    _run(animator, 0.0, 0.7)
    assert animator.playing is None


def test_speech_adds_on_top_of_a_played_motion() -> None:
    now = [0.0]
    animator = Animator(idle=Still(), blend_s=0.0, clock=lambda: now[0])
    clip = Clip(times=[0.0, 5.0], poses=[_pitch(0.2), _pitch(0.2)])
    animator.play(clip)
    rate = 24_000
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 220 * np.arange(2 * rate) / rate)).astype(np.int16)
    animator.feed_speech(tone, rate)
    poses = _run(animator, 0.0, 2.0)
    swayed = poses[int(0.5 * FPS) :, 5]  # clip keeps yaw at 0, only speech moves it
    assert np.abs(swayed).max() > 0.01


@pytest.mark.parametrize("fps", [30.0, 60.0, 100.0])
def test_clips_at_any_recorded_rate_play_in_real_time(fps: float) -> None:
    poses = np.zeros((26, 9))
    poses[:, 4] = np.linspace(0.0, 0.5, 26)
    animator = Animator(fps=fps, idle=Still(), blend_s=0.0)
    animator.play(Clip.from_frames(poses, fps=25))
    ticks = [animator.tick(k / fps) for k in range(round(0.5 * fps) + 1)]
    np.testing.assert_allclose(ticks[-1][4], 0.25)


def test_start_ticks_at_fps_until_closed() -> None:
    poses: list[np.ndarray] = []
    animator = Animator(fps=50.0)
    animator.start(poses.append)
    time.sleep(0.5)
    animator.close()
    count = len(poses)
    time.sleep(0.05)
    assert 20 <= count <= 27
    assert len(poses) == count


def test_play_resamples_clips_to_the_animator_fps() -> None:
    animator = Animator(fps=60.0, idle=Still(), blend_s=0.0)
    animator.play(Clip.from_frames(np.zeros((26, 9)), fps=25))
    animator.tick(0.0)
    assert animator.playing is not None
    assert animator.playing.fps == pytest.approx(60)
