import math
from typing import Any

import numpy as np
import pytest

from reachy_animation import Clip, from_target, to_target


def _move(times: list[float], yaws: list[float]) -> dict[str, Any]:
    frames = [
        {
            "head": to_target(np.array([0, 0, 0, 0, 0, yaw, 0, 0, 0]))[0].tolist(),
            "antennas": [0.1, -0.1],
            "body_yaw": 0.0,
        }
        for yaw in yaws
    ]
    return {"description": "turn", "time": times, "set_target_data": frames}


def test_clip_loads_a_move_interpolates_and_holds_ends() -> None:
    clip = Clip.load(_move([1.0, 2.0], [0.0, 0.4]))
    assert clip.name == "turn"
    assert clip.duration == 1.0
    assert clip.sample(0.5)[5] == np.float64(0.2)
    np.testing.assert_allclose(clip.sample(-1.0)[5], 0.0)
    np.testing.assert_allclose(clip.sample(5.0)[5], 0.4)
    np.testing.assert_allclose(clip.sample(0.5)[6:8], [0.1, -0.1])


def test_clip_interpolates_across_the_yaw_wrap_the_short_way() -> None:
    clip = Clip.load(_move([0.0, 1.0], [math.pi - 0.1, -math.pi + 0.1]))
    mid = from_target(*to_target(clip.sample(0.5)))
    np.testing.assert_allclose(abs(mid[5]), math.pi, atol=1e-9)


def test_clip_from_frames_keeps_its_own_fps() -> None:
    poses = np.zeros((26, 9))
    poses[:, 4] = np.linspace(0.0, 0.5, 26)
    clip = Clip.from_frames(poses, fps=25)
    assert clip.fps == pytest.approx(25)
    assert clip.duration == 1.0
    np.testing.assert_allclose(clip.sample(0.5)[4], 0.25)


def test_resample_keeps_timing_and_ends() -> None:
    poses = np.zeros((101, 9))
    poses[:, 4] = np.linspace(0.0, 0.5, 101)
    clip = Clip.from_frames(poses, fps=100).resample(30)
    assert clip.fps == pytest.approx(30)
    assert clip.duration == pytest.approx(1.0)
    np.testing.assert_allclose(clip.sample(0.5)[4], 0.25, atol=1e-9)
    np.testing.assert_allclose([clip.sample(0)[4], clip.sample(1)[4]], [0.0, 0.5], atol=1e-2)


def test_downsampling_filters_detail_the_new_rate_cannot_hold() -> None:
    t = np.arange(200) / 100
    poses = np.zeros((200, 9))
    poses[:, 4] = 0.1 * np.sin(2 * np.pi * 45 * t)  # 45 Hz: above 30 fps Nyquist, would alias to 15 Hz
    resampled = Clip.from_frames(poses, fps=100).resample(30)
    assert np.abs(resampled.poses[:, 4]).max() < 0.1 / 3


def test_resample_is_a_no_op_at_the_same_rate() -> None:
    clip = Clip.from_frames(np.zeros((10, 9)), fps=60)
    assert clip.resample(60) is clip
