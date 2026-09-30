"""Keep Reachy Mini animated in the background and turn text prompts into motion as they arrive.

  ssh -N -L 8001:localhost:8000 <gpu-host>   # the motion API, tunnelled to 8001: 8000 is the robot daemon's port
  python examples/generate_dense.py          # then type prompts, e.g. "surprised. A door slams behind you."

``play`` takes a function that makes the motion: it runs on a background thread, so the robot keeps breathing (and
swaying with any speech fed to ``feed_speech``) until the motion arrives and crossfades in. The newest prompt wins.
Only motion goes to the robot: the connection takes no audio or camera, so other apps keep them.
"""

import argparse
import json
import logging
import sys
import urllib.request
from functools import partial

from reachy_mini import ReachyMini

from reachy_animation import Animator, Clip, to_target


def generate(api: str, prompt: str) -> Clip:
    """Ask the motion generator for one motion, from whichever planner the server has loaded (~0.2-0.8 s)."""
    request = urllib.request.Request(
        f"{api}/generate-dense",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return Clip.load(json.load(response)["moves"][0])


def main() -> None:
    """Animate in the background; each line on stdin becomes a generated motion."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8001")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with ReachyMini(media_backend="no_media") as robot:
        animator = Animator()
        animator.on_pose(lambda pose: robot.set_target(*to_target(pose)))
        animator.start()
        print("Type a prompt and press enter (ctrl-D to quit).")
        for line in sys.stdin:
            if line.strip():
                animator.play(partial(generate, args.api, line.strip()))
        animator.close()


if __name__ == "__main__":
    main()
