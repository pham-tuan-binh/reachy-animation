"""Motions: anything the animator can play, sampled on its own local time axis."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reachy_animation.pose import DOF, Pose, from_target, zero_pose


class Motion(Protocol):
    """A pose over time: a recorded clip, a procedural motion, or streamed generator output.

    ``sample`` must accept any ``t >= 0``, holding the last pose past ``duration``. An open-ended
    motion reports ``math.inf`` and only stops when interrupted.
    """

    @property
    def name(self) -> str: ...

    @property
    def duration(self) -> float: ...

    def sample(self, t: float) -> Pose: ...


class Clip:
    """Keyframed motion, linearly interpolated between keyframes and held at both ends.

    A clip keeps the rate it was recorded at (``fps``); the animator resamples it to its own fps on ``play``.
    """

    def __init__(self, times: ArrayLike, poses: ArrayLike, name: str = "clip") -> None:
        """Build from keyframe times (s, increasing) and matching (N, 9) poses."""
        self.times: NDArray[np.float64] = np.asarray(times, dtype=np.float64) - float(np.asarray(times)[0])
        self.poses: NDArray[np.float64] = np.asarray(poses, dtype=np.float64).reshape(len(self.times), len(DOF)).copy()
        if len(self.times) > 1 and np.any(np.diff(self.times) <= 0):
            raise ValueError(f"clip {name!r}: keyframe times must be strictly increasing")
        # Angles decoded from matrices wrap at +-pi; interpolating across the wrap would spin the head.
        self.poses[:, 3:6] = np.unwrap(self.poses[:, 3:6], axis=0)
        self.poses[:, 8] = np.unwrap(self.poses[:, 8])
        self.name = name

    @classmethod
    def load(cls, move: str | Path | Mapping[str, Any]) -> Clip:
        """Load a recorded move (the emotion/dance libraries' and motion-gen API's format), as a file or dict."""
        if isinstance(move, Mapping):
            recorded, name = move, move.get("description") or "clip"
        else:
            recorded, name = json.loads(Path(move).read_text()), Path(move).stem
        frames = recorded["set_target_data"]
        poses = [from_target(f["head"], f["antennas"], f.get("body_yaw", 0.0)) for f in frames]
        return cls(recorded["time"], poses, name)

    @classmethod
    def from_frames(cls, poses: ArrayLike, fps: float, name: str = "clip") -> Clip:
        """Build from evenly spaced (N, 9) poses recorded at ``fps``, e.g. motion-generator arrays."""
        poses = np.asarray(poses, dtype=np.float64)
        return cls(np.arange(len(poses)) / fps, poses, name)

    @classmethod
    def from_sdk(cls, move: Any, name: str = "sdk_move", fps: float = 100.0) -> Clip:
        """Sample an SDK ``Move`` (anything with ``duration`` and ``evaluate(t)``, e.g. ``DanceMove``) at ``fps``."""
        times = np.linspace(0.0, move.duration, max(2, round(move.duration * fps) + 1))
        poses = []
        for t in times:
            head, antennas, body_yaw = move.evaluate(t)
            poses.append(
                from_target(
                    np.eye(4) if head is None else head,
                    (0.0, 0.0) if antennas is None else antennas,
                    0.0 if body_yaw is None else body_yaw,
                )
            )
        return cls(times, poses, name)

    @property
    def duration(self) -> float:
        """Time of the last keyframe."""
        return float(self.times[-1])

    @property
    def fps(self) -> float:
        """The keyframe rate it was recorded at (median, as recorded clips jitter)."""
        return 1.0 / float(np.median(np.diff(self.times))) if len(self.times) > 1 else 0.0

    def resample(self, fps: float) -> Clip:
        """This clip on an ``fps`` frame grid, low-passed first when downsampling so fast detail can't alias."""
        if len(self.times) < 2 or math.isclose(self.fps, fps, rel_tol=1e-3):
            return self
        poses = self.poses
        ratio = self.fps / fps
        if ratio > 1.25:
            half = max(1, round(ratio / 2))
            kernel = np.ones(2 * half + 1) / (2 * half + 1)
            padded = np.pad(poses, ((half, half), (0, 0)), mode="edge")
            poses = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(len(DOF))], axis=1)
        grid = np.arange(max(2, round(self.duration * fps) + 1)) / fps
        resampled = np.stack([np.interp(grid, self.times, poses[:, j]) for j in range(len(DOF))], axis=1)
        return Clip(grid, resampled, self.name)

    def sample(self, t: float) -> Pose:
        """Interpolate the pose at ``t`` seconds."""
        if len(self.times) == 1 or t <= 0.0 or t >= self.times[-1]:
            held: Pose = self.poses[0 if t <= 0.0 else -1].copy()
            return held
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        a = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        blended: Pose = (1.0 - a) * self.poses[i] + a * self.poses[i + 1]
        return blended


@dataclass(frozen=True)
class Breathing:
    """Idle motion: a slow head rise and fall with a gentle antenna sway, forever."""

    z_amplitude_m: float = 0.005
    z_frequency_hz: float = 0.1
    antenna_amplitude_rad: float = math.radians(15.0)
    antenna_frequency_hz: float = 0.5
    name: str = "breathing"
    duration: float = math.inf

    def sample(self, t: float) -> Pose:
        """Breathing pose at ``t``."""
        pose = zero_pose()
        pose[2] = self.z_amplitude_m * math.sin(2 * math.pi * self.z_frequency_hz * t)
        sway = self.antenna_amplitude_rad * math.sin(2 * math.pi * self.antenna_frequency_hz * t)
        pose[6], pose[7] = sway, -sway
        return pose
