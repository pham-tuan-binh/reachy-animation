"""Full-body pose as one 9-DoF vector: the unit every animation layer produces, blends and adds.

Layout matches the motion generator's trajectories: ``x y z`` in metres, ``roll pitch yaw`` in radians
(extrinsic xyz, as the SDK's ``create_head_pose``), then ``antenna_right antenna_left body_yaw`` in radians.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

DOF = ("x", "y", "z", "roll", "pitch", "yaw", "antenna_right", "antenna_left", "body_yaw")
Pose = NDArray[np.float64]


def zero_pose() -> Pose:
    """Return the neutral pose."""
    return np.zeros(len(DOF))


def head_matrix(pose: Pose) -> NDArray[np.float64]:
    """Return the 4x4 head pose the SDK's ``set_target(head=...)`` expects."""
    x, y, z, roll, pitch, yaw = pose[:6]
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def from_target(head: ArrayLike, antennas: ArrayLike, body_yaw: float) -> Pose:
    """Build a pose from SDK target arguments (4x4 head, two antennas, body yaw)."""
    h = np.asarray(head, dtype=np.float64)
    roll = math.atan2(h[2, 1], h[2, 2])
    pitch = math.atan2(-h[2, 0], math.hypot(h[0, 0], h[1, 0]))
    yaw = math.atan2(h[1, 0], h[0, 0])
    right, left = np.asarray(antennas, dtype=np.float64)
    return np.array([h[0, 3], h[1, 3], h[2, 3], roll, pitch, yaw, right, left, float(body_yaw)])


def to_target(pose: Pose) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return ``(head, antennas, body_yaw)``, the positional arguments of the SDK's ``set_target``."""
    return head_matrix(pose), pose[6:8].copy(), float(pose[8])
