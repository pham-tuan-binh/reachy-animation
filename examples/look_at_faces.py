"""Make Reachy Mini look at the person in front of it, while it keeps breathing and playing whatever it plays.

  pip install opencv-python
  python examples/look_at_faces.py            # walk around in front of the robot (ctrl-C to quit)

A face detector runs on each camera frame, and the biggest face becomes the ``look`` target. The camera sits in the
head, so a face's pixel position is relative to where the head already points: the SDK's ``look_at_image`` turns it
into a direction in the robot's frame using the head pose measured right now, which is what ``look`` wants. When no
face has been seen for a moment, the robot eases back to its plain animation. The camera is shared through the
daemon's media backend, so other apps keep it too.
"""

import argparse
import logging
import time

import cv2
from reachy_mini import ReachyMini

from reachy_animation import Animator, from_target, to_target

logger = logging.getLogger(__name__)

LOST_AFTER_S = 1.0  # a detector misses faces for a frame or two; only let go after this long


def main() -> None:
    """Track the biggest face until ctrl-C."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-motion", type=float, default=0.5, help="share of the look kept while a clip plays")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    with ReachyMini() as robot:  # the default media backend: the camera is shared through the daemon
        animator = Animator(look_weight_in_motion=args.in_motion)
        animator.on_pose(lambda pose: robot.set_target(*to_target(pose)))
        animator.start()
        tracking, last_seen = False, 0.0
        try:
            while True:
                frame = robot.media.get_frame()  # BGR, or None before the first frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
                now = time.monotonic()
                if len(faces):
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    head = robot.look_at_image(int(x + w / 2), int(y + h / 2), perform_movement=False)
                    pose = from_target(head, [0.0, 0.0], 0.0)
                    animator.look(yaw=pose[5], pitch=pose[4])
                    tracking, last_seen = True, now
                elif tracking and now - last_seen > LOST_AFTER_S:
                    animator.look(None)
                    tracking = False
        except KeyboardInterrupt:
            pass
        finally:
            animator.close()


if __name__ == "__main__":
    main()
