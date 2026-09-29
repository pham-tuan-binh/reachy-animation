import numpy as np

from reachy_animation import from_target, to_target


def test_to_target_round_trips_through_from_target() -> None:
    pose = np.array([0.01, -0.02, 0.015, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7])
    back = from_target(*to_target(pose))
    np.testing.assert_allclose(back, pose, atol=1e-12)
